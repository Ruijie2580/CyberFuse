import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6, affine=False):
        super().__init__()
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x = x * self.weight + self.bias

        return x


class BaseFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fusion = nn.Conv2d(dim * 2, dim, 1, 1, 0)

    def forward(self, x, y):
        return self.fusion(torch.cat([x, y], dim=1))


class B1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.stripconv = nn.Sequential(
            nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
            nn.Conv2d(dim, dim, (1, 29), 1, (0, 29 // 2), groups=dim),
            nn.Conv2d(dim, dim, (29, 1), 1, (29 // 2, 0), groups=dim),
            nn.Conv2d(dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        x2 = self.stripconv(x)
        return x * x2


class B2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.stripconv = nn.Sequential(
            nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
            nn.Conv2d(dim, dim, (1, 19), 1, (0, 19 // 2), groups=dim),
            nn.Conv2d(dim, dim, (19, 1), 1, (19 // 2, 0), groups=dim),
            nn.Conv2d(dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        x2 = self.stripconv(x)
        return x * x2


class B3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.middle_conv = nn.Conv2d(dim, dim, 9, 1, 9 // 2, groups=dim)
        self.pconv = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        x2 = self.middle_conv(x)
        x2 = self.pconv(x2)
        return x * x2


class B4(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local_conv = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)
        self.pconv = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        x2 = self.local_conv(x)
        x2 = self.pconv(x2)
        return x * x2


class GroupLKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 4 == 0, f"dim must be divisible by 4, but got {dim}"
        group_dim = dim // 4

        self.conv = nn.Conv2d(dim, dim, 1, 1, 0)

        self.pwconv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.pwconv2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.B1 = B1(group_dim)
        self.B2 = B2(group_dim)
        self.B3 = B3(group_dim)
        self.B4 = B4(group_dim)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

    def forward(self, x):
        x1 = x2 = x
        x1 = self.conv(x1)

        x11, x12, x13, x14 = torch.chunk(x1, 4, dim=1)
        x11 = self.B1(x11)
        x12 = self.B2(x12)
        x13 = self.B3(x13)
        x14 = self.B4(x14)
        x1 = torch.cat([x11, x12, x13, x14], dim=1)
        x1 = self.pwconv1(x1)
        x = x1 * x2
        x = x * self.sca(x)
        x = self.pwconv2(x)
        return x


class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pwconv1 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.pwconv2 = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.pwconv1(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = x1 * x2
        x = self.pwconv2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.GLKA = GroupLKA(dim)
        self.ffn = FFN(dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.GLKA(x)
        x = res + self.beta * x
        res = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = res + self.gamma * x
        return x
