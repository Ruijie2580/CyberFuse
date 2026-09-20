import torch
import torch.nn as nn
import torch.nn.functional as F
from .prior import Prior


class SpectralDegradation(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.conv(x)

    def transpose(self, x):
        return F.conv_transpose2d(x, self.conv.weight, bias=None)


class SpatialDegradation(nn.Module):
    """
    空间退化算子 (PSF 模糊 + 降采样) 及其严格伴随 (转置)。

    transpose() 必须是 forward() 的数学伴随算子 <A x, y> == <x, A^T y>，
    展开网络里的保真项更新才与梯度下降一致。为了让伴随精确成立，
    下采样卷积取 kernel_size == stride == scale 且 padding=0：

        forward   :  H        --blur(k=5,p=2, 保尺寸)-->  H   --down(k=s=scale,p=0)-->  H/scale
        transpose :  H/scale  --down^T---------------->  H   --blur^T---------------->  H

    两个方向的尺寸都精确对齐，不再需要 output_padding 凑尺寸。
    注意 blur 取 padding=2，其伴随同样是 padding=2 的转置卷积（尺寸不变）。
    要求 H 能被 scale 整除；Net 中的 X 由 LRHS 上采样得到，天然满足。
    """

    def __init__(self, in_c, scale):
        super().__init__()

        self.scale = scale

        self.blur = nn.Conv2d(
            in_c,
            in_c,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=in_c,
            bias=False,
        )

        self.down = nn.Conv2d(
            in_c,
            in_c,
            kernel_size=scale,
            stride=scale,
            padding=0,
            groups=in_c,
            bias=False,
        )

    def forward(self, x):
        x = self.blur(x)
        x = self.down(x)
        return x

    def transpose(self, x):
        # padding/output_padding 均为 0，与 forward 中的 down(k=s=scale, p=0) 严格对应
        x = F.conv_transpose2d(
            x,
            self.down.weight,
            stride=self.scale,
            padding=0,
            groups=self.down.groups,
        )

        x = F.conv_transpose2d(
            x,
            self.blur.weight,
            stride=1,
            padding=2,
            groups=self.blur.groups,
        )

        return x


class GainNet(nn.Module):
    def __init__(self, channels, hidden_dim=48):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, hidden_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 9, 1, 9 // 2, groups=hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, channels, 3, 1, 1)

        self.act = nn.SiLU()

    def forward(self, error):
        feat = self.conv1(error)
        feat = self.act(feat)

        feat = self.conv2(feat)
        feat = self.act(feat)

        out = F.sigmoid(self.conv3(feat))

        return out


class PIDController(nn.Module):
    """
    PID 控制器 (逐通道)。

    leaky 是积分项的泄漏系数：integral = leaky * integral + error。
    leaky == 1.0 时退化为普通累加（无抗积分饱和）；
    leaky < 1.0 时旧误差按几何级数衰减，可抑制长序列展开中的积分饱和。
    默认保持 1.0，即与原始实现数值完全一致。
    """

    def __init__(self, channels, leaky=1.0):
        super().__init__()

        self.kp_raw = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        self.ki_raw = nn.Parameter(torch.zeros(1, channels, 1, 1) - 2.0)
        self.kd_raw = nn.Parameter(torch.zeros(1, channels, 1, 1) - 2.0)

        self.leaky = float(leaky)

    def forward(self, error, prev_error, integral):
        derivative = error - prev_error

        integral = self.leaky * integral + error

        kp = F.softplus(self.kp_raw)
        ki = F.softplus(self.ki_raw)
        kd = F.softplus(self.kd_raw)

        control = kp * error + ki * integral + kd * derivative

        return control, integral


class KalmanPIDFidelity(nn.Module):
    def __init__(
        self,
        hsi_channels,
        msi_channels,
        spe_de,
        spa_de,
        total_steps,
        pid_leaky=1.0,
        collect_vis=False,
    ):
        super().__init__()
        self.spe_de = spe_de
        self.spa_de = spa_de

        self.gain_s = GainNet(msi_channels)
        self.gain_p = GainNet(hsi_channels)

        self.pid_s = PIDController(msi_channels, leaky=pid_leaky)
        self.pid_p = PIDController(hsi_channels, leaky=pid_leaky)

        self.alpha = nn.Parameter(torch.full((total_steps,), 0.1))

        # 可视化缓存：默认关闭。开启后 eval 阶段每步都会保存一份 detach 的
        # CPU 张量，对大图 (如 512x512x128) 单个就有 ~128MB，整个展开下来会吃光内存。
        self.collect_vis = collect_vis
        self.vis_update_s_all = []
        self.vis_update_p_all = []
        self.vis_update_all = []

    def init_state(self, X, Y_h, Y_m):
        B = X.shape[0]

        H_hr, W_hr = Y_m.shape[2:]
        H_lr, W_lr = Y_h.shape[2:]

        prev_e_s = torch.zeros(B, Y_m.shape[1], H_hr, W_hr, device=X.device)
        prev_e_p = torch.zeros(B, Y_h.shape[1], H_lr, W_lr, device=X.device)

        int_s = torch.zeros_like(prev_e_s)
        int_p = torch.zeros_like(prev_e_p)

        return (prev_e_s, prev_e_p, int_s, int_p)

    def forward(self, X, Y_h, Y_m, state, stage_idx):
        prev_e_s, prev_e_p, int_s, int_p = state

        e_s = Y_m - self.spe_de(X)
        e_p = Y_h - self.spa_de(X)

        K_s = self.gain_s(e_s)
        K_p = self.gain_p(e_p)

        u_s, int_s = self.pid_s(e_s, prev_e_s, int_s)
        u_p, int_p = self.pid_p(e_p, prev_e_p, int_p)

        update_s = self.spe_de.transpose(K_s * u_s)
        update_p = self.spa_de.transpose(K_p * u_p)

        update = update_s + update_p

        if self.collect_vis and not self.training:
            self.vis_update_s_all.append(u_s.detach().cpu())
            self.vis_update_p_all.append(u_p.detach().cpu())
            self.vis_update_all.append(update.detach().cpu())

        X_new = X + self.alpha[stage_idx] * update

        new_state = (e_s, e_p, int_s, int_p)
        return X_new, new_state


class Net(nn.Module):
    def __init__(
        self,
        hsi_channels=31,
        msi_channels=3,
        scale=4,
        ntier=2,
        n_sub_iters=3,
        pid_leaky=1.0,
        return_intermediate=False,
        collect_vis=False,
    ):
        super(Net, self).__init__()

        self.scale = scale
        self.ntier = ntier
        self.n_sub_iters = n_sub_iters
        self.total_steps = ntier * n_sub_iters

        self.hsi_channels = hsi_channels
        self.msi_channels = msi_channels

        # 默认 False：只返回最后一步结果。开启后返回每个 tier 的中间输出列表，
        # 但这些输出会一直挂在计算图上，训练时会额外占显存，
        # 只有确实要做深监督 / 中间结果可视化时才需要打开。
        self.return_intermediate = return_intermediate

        self.spa_de = SpatialDegradation(hsi_channels, scale)
        self.spe_de = SpectralDegradation(hsi_channels, msi_channels)

        self.fidelity = KalmanPIDFidelity(
            hsi_channels,
            msi_channels,
            self.spe_de,
            self.spa_de,
            self.total_steps,
            pid_leaky=pid_leaky,
            collect_vis=collect_vis,
        )

        self.prior = nn.ModuleList(
            [Prior(hsi_channels, msi_channels) for _ in range(ntier + 1)]
        )

    def forward(self, LRHS, HRMS):
        if self.fidelity.collect_vis:
            self.fidelity.vis_update_s_all = []
            self.fidelity.vis_update_p_all = []
            self.fidelity.vis_update_all = []

        X = F.interpolate(
            LRHS,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )

        outputs = []

        X = self.prior[0](X, HRMS)

        state = self.fidelity.init_state(X, LRHS, HRMS)

        if self.return_intermediate:
            outputs.append(X)
        idx = 0

        for i in range(self.ntier):

            for sub_step in range(self.n_sub_iters):
                X, state = self.fidelity(X, LRHS, HRMS, state, idx)
                idx += 1

            X = self.prior[i + 1](X, HRMS)

            if self.return_intermediate:
                outputs.append(X)

        # 保持向后兼容：调用方统一用 `isinstance(pred, (list, tuple))` 兼容两种返回
        return outputs if self.return_intermediate else X


if __name__ == "__main__":
    model = Net(hsi_channels=31, msi_channels=3, scale=4, ntier=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")

    lrhs = torch.randn(1, 31, 32, 32)
    hrms = torch.randn(1, 3, 128, 128)

    out = model(lrhs, hrms)
    print("return_intermediate=False ->", tuple(out.shape))

    model.return_intermediate = True
    outs = model(lrhs, hrms)
    print("return_intermediate=True  ->", [tuple(o.shape) for o in outs])
