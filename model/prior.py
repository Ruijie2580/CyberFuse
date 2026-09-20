import torch
import torch.nn as nn
from .blocks import Block, BaseFusion


class Blocks(nn.Module):
    def __init__(self, dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim) for _ in range(num_blocks)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class Prior(nn.Module):
    def __init__(
        self,
        dim_h,
        dim_m,
        embed_dim=64,
        num_blocks=[4, 4, 4, 8],
        num_multiplyer=[1, 1, 1, 1],
    ):
        super().__init__()

        self.init_hsi = nn.Conv2d(dim_h, embed_dim, 3, 1, 1)
        self.init_msi = nn.Conv2d(dim_m, embed_dim, 3, 1, 1)

        depth = len(num_blocks)
        dims = [embed_dim * m for m in num_multiplyer]

        self.down_msi = nn.ModuleList()
        self.down_hsi = nn.ModuleList()

        for i in range(depth - 1):
            self.down_msi.append(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1))
            self.down_hsi.append(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1))

        self.ups = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.fusers = nn.ModuleList()
        self.skip_fuse = nn.ModuleList()

        for i in reversed(range(depth - 1)):
            self.ups.append(nn.ConvTranspose2d(dims[i + 1], dims[i], 3, 2, 1, 1))
            self.skip_fuse.append(BaseFusion(dims[i]))
            self.decoder.append(Blocks(dims[i], num_blocks[i]))

        self.bottle = Blocks(dims[-1], num_blocks[-1])

        self.out = nn.Conv2d(embed_dim, dim_h, 3, 1, 1)

    def run_block(self, block, x):
        x = block(x)
        if isinstance(x, tuple):
            x = x[0]
        return x

    def forward(self, hs, ms):
        x_hs = self.init_hsi(hs)
        x_ms = self.init_msi(ms)

        skip_hs = [x_hs]
        skip_ms = [x_ms]

        for down_hs, down_ms in zip(self.down_hsi, self.down_msi):
            x_hs = down_hs(x_hs)
            x_ms = down_ms(x_ms)

            skip_hs.append(x_hs)
            skip_ms.append(x_ms)

        x = x_hs + x_ms
        x = self.run_block(self.bottle, x)

        for i, (up, dec, skip_fus) in enumerate(
            zip(self.ups, self.decoder, self.skip_fuse)
        ):
            x = up(x)

            hs_skip = skip_hs[-(i + 2)]
            ms_skip = skip_ms[-(i + 2)]
            skip = hs_skip + ms_skip

            x = skip_fus(x, skip)

            x = self.run_block(dec, x)

        out = self.out(x)

        return out + hs


if __name__ == "__main__":
    hs = torch.randn(1, 16, 64, 64).to("cuda")
    ms = torch.randn(1, 8, 64, 64).to("cuda")
    prior = Prior(16, 8).to("cuda")
    print(prior(hs, ms).shape)
