import numpy as np
import torch
from skimage.metrics import structural_similarity


class Metrics:
    """
    HSI 融合常用指标。

    口径约定：
      * 数据和预测都先 clip 到 [0, data_range]，再用 data_range 作为 PSNR/SSIM 的
        动态范围。数据集按 /65535 归一化时 data_range=1.0 是正确的（即使实际
        观测最大值只有 ~0.7）。若换了归一化方式，务必同步修改 data_range。
      * ERGAS 的 ratio 就是空间放大倍数，必须与 cfg.model.scale 一致，
        不要沿用默认值 4。
    """

    def __init__(self, ratio=4, data_range=1.0):
        self.ratio = ratio
        self.data_range = data_range

    def init(self):
        return {
            "count": 0,
            "PSNR": 0.0,
            "SSIM": 0.0,
            "SAM": 0.0,
            "ERGAS": 0.0,
        }

    def update(self, state, preds, targets):
        B = preds.shape[0]

        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        preds = np.clip(preds, 0, self.data_range)
        targets = np.clip(targets, 0, self.data_range)

        psnr = float(self._psnr(targets, preds))
        ssim = float(self._ssim(targets, preds))
        sam = float(self._sam(targets, preds))
        ergas = float(self._ergas(targets, preds))

        state["PSNR"] += psnr * B
        state["SSIM"] += ssim * B
        state["SAM"] += sam * B
        state["ERGAS"] += ergas * B
        state["count"] += B

        return state

    def compute(self, state):
        count = max(state["count"], 1)
        return {
            "PSNR": state["PSNR"] / count,
            "SSIM": state["SSIM"] / count,
            "SAM": state["SAM"] / count,
            "ERGAS": state["ERGAS"] / count,
        }

    def _psnr(self, targets, preds, data_range=None):
        data_range = self.data_range if data_range is None else data_range
        mse = np.mean((preds - targets) ** 2, axis=(1, 2, 3))
        mse = np.maximum(mse, 1e-12)
        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
        return psnr.mean()

    def _ssim(self, targets, preds):
        B = targets.shape[0]
        ssim_val = 0.0
        for t, p in zip(targets, preds):
            t = t.transpose(1, 2, 0)
            p = p.transpose(1, 2, 0)
            ssim_val += structural_similarity(
                t, p, data_range=self.data_range, channel_axis=-1
            )
        return ssim_val / B

    def _sam(self, targets, preds):
        dot = np.sum(targets * preds, axis=1)
        norm_t = np.linalg.norm(targets, axis=1)
        norm_p = np.linalg.norm(preds, axis=1)

        cos = dot / (norm_t * norm_p + 1e-12)
        cos = np.clip(cos, -1, 1)

        sam_map = np.arccos(cos)
        return (sam_map * (180.0 / np.pi)).mean()

    def _ergas(self, targets, preds):
        diff = targets - preds
        mse_per_band = np.mean(diff**2, axis=(2, 3))
        mu_ref_per_band = np.mean(targets, axis=(2, 3))

        term_per_band = mse_per_band / (mu_ref_per_band**2 + 1e-12)

        mean_term = np.mean(term_per_band, axis=1)

        root_term = np.sqrt(mean_term)

        ergas_score = (100 / self.ratio) * root_term
        return ergas_score.mean()
