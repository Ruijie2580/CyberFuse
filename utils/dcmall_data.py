import h5py
import os
import numpy as np
import torch.utils.data as data
import scipy.io as sio
from scipy.ndimage import convolve1d
from typing import Tuple

# ============================================================
# 退化算子
# ============================================================


def gaussian_kernel_1d(kernel_size: int, sigma: float):
    """
    生成一维高斯核
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"

    radius = kernel_size // 2

    x = np.arange(-radius, radius + 1)

    kernel = np.exp(-(x**2) / (2 * sigma**2))

    kernel /= kernel.sum()

    return kernel.astype(np.float32)


def blur(
    img: np.ndarray,
    kernel_size: int = 7,
    sigma_h: float = 2.0,
    sigma_w: float = 2.0,
) -> np.ndarray:
    """
    Gaussian blur

    参数
    ----------
    img : (B,C,H,W)

    返回
    -------
    blurred : (B,C,H,W)
    """

    assert img.ndim == 4

    # 生成 1D kernel
    kernel_h = gaussian_kernel_1d(kernel_size, sigma_h)
    kernel_w = gaussian_kernel_1d(kernel_size, sigma_w)

    out = img.astype(np.float32).copy()

    # W方向卷积
    out = convolve1d(out, weights=kernel_w, axis=-1, mode="reflect")

    # H方向卷积
    out = convolve1d(out, weights=kernel_h, axis=-2, mode="reflect")

    return out


def downsample_stride(img: np.ndarray, stride: int = 4) -> np.ndarray:
    """
    等间隔降采样。
    """
    return img[:, :, ::stride, ::stride]


# ============================================================
# 数据集类
# ============================================================


class LoadData(data.Dataset):
    """
    H5 数据集，要求含 'HRHS' (N,C,H,W) 与 'HRMS' (N,c,H,W) 两个 key，
    'LRHS' 可选（存在则直接使用，否则用固定高斯模糊 + 降采样实时生成）。

    in_memory=True（默认）
        启动时把全部数据读进内存并关闭文件句柄。读取最快，但内存占用等于
        数据集大小；且配合 num_workers>0 时，Windows(spawn) 会把整份数据
        pickle 给每个 worker，内存会成倍膨胀 —— 这种组合请用 in_memory=False。

    in_memory=False
        只读元信息，按需惰性读取单个样本。内存友好，且可安全配合
        num_workers>0（每个 worker 进程各自打开文件句柄）。
    """

    def __init__(
        self,
        dataname: str,
        file_path: str,
        scaling_factor: int = 4,
        kernel_size: int = 7,
        sigma: float = 2.0,
        in_memory: bool = True,
    ):
        super(LoadData, self).__init__()
        self.dataname = dataname
        self.file_path = file_path
        self.scaling_factor = scaling_factor
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.in_memory = in_memory

        self._h5 = None
        self._GT = self._MSI = self._LRHSI = None

        self._load_meta()

        if self.in_memory:
            self._GT, self._MSI, self._LRHSI = self._load_in_memory()

        self._print_data_source()

    # ---- 文件句柄管理 ----

    def _ensure_handle(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.file_path, "r")
        return self._h5

    def close(self):
        """关闭底层 H5 句柄（幂等）。"""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self):
        # DataLoader 多进程：h5py 句柄无法跨进程传递，置空让各 worker 自行重开
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5 = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---- 内部方法 ----

    def _load_meta(self):
        """只读元信息：key 是否存在、样本数与各张量形状。"""
        with h5py.File(self.file_path, "r") as h:
            for key in ("HRHS", "HRMS"):
                if key not in h:
                    raise KeyError(f"H5 文件中未找到 '{key}' key: {self.file_path}")

            self._has_precomputed_lrhs = "LRHS" in h

            self._len = h["HRHS"].shape[0]
            self._gt_shape = tuple(h["HRHS"].shape[1:])
            self._msi_shape = tuple(h["HRMS"].shape[1:])

            if self._has_precomputed_lrhs:
                self._lrhs_shape = tuple(h["LRHS"].shape[1:])
            else:
                c, hh, ww = self._gt_shape
                s = self.scaling_factor
                self._lrhs_shape = (c, hh // s, ww // s)

    def _load_in_memory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(self.file_path, "r") as h:
            gt = np.array(h["HRHS"], dtype=np.float32)
            msi = np.array(h["HRMS"], dtype=np.float32)
            lrhs = (
                np.array(h["LRHS"], dtype=np.float32)
                if self._has_precomputed_lrhs
                else None
            )

        if lrhs is None:
            lrhs = self._get_lrhsi(gt)

        return gt, msi, lrhs

    def _get_lrhsi(self, gt: np.ndarray) -> np.ndarray:
        """固定高斯模糊 + 等间隔降采样。同时接受 (C,H,W) 与 (N,C,H,W)。"""
        squeeze = gt.ndim == 3
        x = gt[None] if squeeze else gt

        out = downsample_stride(
            blur(x, self.kernel_size, self.sigma, self.sigma), self.scaling_factor
        )

        return out[0] if squeeze else out

    def _sample(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回原始 (gt, msi, lrhs)，统一 float32 numpy。"""
        if self.in_memory:
            return self._GT[index], self._MSI[index], self._LRHSI[index]

        h = self._ensure_handle()

        gt = np.asarray(h["HRHS"][index], dtype=np.float32)
        msi = np.asarray(h["HRMS"][index], dtype=np.float32)

        if self._has_precomputed_lrhs:
            lrhs = np.asarray(h["LRHS"][index], dtype=np.float32)
        else:
            lrhs = self._get_lrhsi(gt)

        return gt, msi, lrhs

    def _print_data_source(self):
        source = (
            "预生成 (H5)"
            if self._has_precomputed_lrhs
            else "实时生成 (高斯模糊+降采样)"
        )
        mode = "全量载入内存" if self.in_memory else "惰性读取"
        # 不要用 ℹ (U+2139)：GBK 控制台无法编码，会直接 UnicodeEncodeError
        print(f"  [i] LRHS 来源: {source}  |  读取模式: {mode}")

    # ---- 公共接口 ----

    @property
    def GT(self) -> np.ndarray:
        """HRHS 全量张量。仅在 in_memory=True 时可用。"""
        if self._GT is None:
            raise RuntimeError("in_memory=False 模式下无全量 GT，请用 _sample(i) 读取")
        return self._GT

    @property
    def MSI(self) -> np.ndarray:
        if self._MSI is None:
            raise RuntimeError("in_memory=False 模式下无全量 MSI，请用 _sample(i) 读取")
        return self._MSI

    @property
    def LRHSI(self) -> np.ndarray:
        if self._LRHSI is None:
            raise RuntimeError("in_memory=False 模式下无全量 LRHSI，请用 _sample(i) 读取")
        return self._LRHSI

    def summary(self):
        """打印数据集摘要信息"""
        _, _, lrhs0 = self._sample(0)
        gt0, msi0, _ = self._sample(0)

        print(f"\n[数据集: {self.dataname}]")
        print(f"  - 文件:        {self.file_path}")
        print(f"  - 样本数:      {len(self)}")
        print(f"  - 缩放因子:    ×{self.scaling_factor}")
        print(f"  - HRHS 形状:   {self._gt_shape}    (C, H, W)")
        print(f"  - HRMS 形状:   {self._msi_shape}   (C, H, W)")
        print(f"  - LRHS 形状:   {self._lrhs_shape}  (C, H//s, W//s)")

        # 数据统计
        print(f"\n  数据统计 (第一个样本):")
        print(
            f"    HRHS - 范围: [{gt0.min():.4f}, {gt0.max():.4f}]  "
            f"均值: {gt0.mean():.4f}  标准差: {gt0.std():.4f}"
        )
        print(
            f"    HRMS - 范围: [{msi0.min():.4f}, {msi0.max():.4f}]  "
            f"均值: {msi0.mean():.4f}  标准差: {msi0.std():.4f}"
        )
        print(
            f"    LRHS - 范围: [{lrhs0.min():.4f}, {lrhs0.max():.4f}]  "
            f"均值: {lrhs0.mean():.4f}  标准差: {lrhs0.std():.4f}"
        )

    def export_in_one_mat(self):
        """将数据导出为 MATLAB .mat 文件（每个样本一个文件）"""
        out_dir = f"{self.dataname}_data"
        os.makedirs(out_dir, exist_ok=True)
        for i in range(len(self)):
            gt, msi, lrhs = self._sample(i)
            sio.savemat(
                f"./{out_dir}/{self.dataname}_{i}.mat",
                {
                    "GT": gt.transpose(1, 2, 0),
                    "MSI": msi.transpose(1, 2, 0),
                    "LRHSI": lrhs.transpose(1, 2, 0),
                },
            )
        print(f"✓ 已导出 {len(self)} 组样本 → ./{out_dir}/")

    def export_data(self):
        """将数据导出为 MATLAB .mat 文件（按类别分目录）"""
        for subdir in ["GT", "MSI", "LRHSI"]:
            os.makedirs(subdir, exist_ok=True)
        for i in range(len(self)):
            gt, msi, lrhs = self._sample(i)
            sio.savemat(
                f"./GT/{self.dataname}_{i}.mat",
                {"GT": gt.transpose(1, 2, 0)},
            )
            sio.savemat(
                f"./MSI/{self.dataname}_{i}.mat",
                {"MSI": msi.transpose(1, 2, 0)},
            )
            sio.savemat(
                f"./LRHSI/{self.dataname}_{i}.mat",
                {"LRHSI": lrhs.transpose(1, 2, 0)},
            )
        print(f"✓ 已导出 {len(self)} 组样本 → ./GT/  ./MSI/  ./LRHSI/")

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取单个样本

        返回
        -------
        msi : (C_msi, H, W)
        lrhsi : (C_hsi, H//s, W//s)
        gt : (C_hsi, H, W)
        """
        gt, msi, lrhs = self._sample(index)
        return msi, lrhs, gt

    def __len__(self) -> int:
        return self._len
