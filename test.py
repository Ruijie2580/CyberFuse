import argparse
import os
import time

import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import get_opt, load_config
from model.dun import Net
from utils.dcmall_data import LoadData
from utils.metrics import Metrics


def test_chop(model, lrhs, hrms, scale=4, patch_size=256, shave=16):
    assert patch_size % scale == 0, f"patch_size({patch_size}) 必须能被 scale({scale}) 整除"
    assert patch_size % 8 == 0, f"patch_size({patch_size}) 必须能被 8 整除 (Prior 下采样 2^3)"
    assert 0 <= shave < patch_size // 2, f"shave({shave}) 必须落在 [0, {patch_size // 2})"

    b, c, h_lr, w_lr = lrhs.size()
    b, c_hr, h_hr, w_hr = hrms.size()

    assert h_lr * scale == h_hr and w_lr * scale == w_hr, (
        f"LRHS {tuple(lrhs.shape[2:])} 与 HRMS {tuple(hrms.shape[2:])} 不满足 ×{scale} 关系"
    )

    h_pad = (-h_hr) % patch_size
    w_pad = (-w_hr) % patch_size

    hrms_pad = F.pad(hrms, (0, w_pad, 0, h_pad), mode="reflect")
    lrhs_pad = F.pad(lrhs, (0, w_pad // scale, 0, h_pad // scale), mode="reflect")

    output_pad = torch.zeros(
        (b, c, hrms_pad.shape[2], hrms_pad.shape[3]), device=lrhs.device
    )

    stride = patch_size - 2 * shave
    h_idx_list = list(range(0, hrms_pad.shape[2] - patch_size + 1, stride))
    w_idx_list = list(range(0, hrms_pad.shape[3] - patch_size + 1, stride))

    if (hrms_pad.shape[2] - patch_size) not in h_idx_list:
        h_idx_list.append(hrms_pad.shape[2] - patch_size)
    if (hrms_pad.shape[3] - patch_size) not in w_idx_list:
        w_idx_list.append(hrms_pad.shape[3] - patch_size)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            hr_patch = hrms_pad[
                ..., h_idx : h_idx + patch_size, w_idx : w_idx + patch_size
            ]

            lr_h_idx = h_idx // scale
            lr_w_idx = w_idx // scale
            lr_patch_size = patch_size // scale

            lr_patch = lrhs_pad[
                ...,
                lr_h_idx : lr_h_idx + lr_patch_size,
                lr_w_idx : lr_w_idx + lr_patch_size,
            ]

            with torch.no_grad():
                out_patch = model(lr_patch, hr_patch)
                if isinstance(out_patch, (list, tuple)):
                    out_patch = out_patch[-1]

            s_h_start = shave if h_idx != 0 else 0
            s_h_end = patch_size - shave if h_idx != h_idx_list[-1] else patch_size

            s_w_start = shave if w_idx != 0 else 0
            s_w_end = patch_size - shave if w_idx != w_idx_list[-1] else patch_size

            output_pad[
                ...,
                h_idx + s_h_start : h_idx + s_h_end,
                w_idx + s_w_start : w_idx + s_w_end,
            ] = out_patch[..., s_h_start:s_h_end, s_w_start:s_w_end]

    output = output_pad[..., :h_hr, :w_hr]
    return output


def build_model(cfg):
    return Net(
        hsi_channels=cfg.dataset.hsi_channels,
        msi_channels=cfg.dataset.msi_channels,
        scale=cfg.model.scale,
        ntier=cfg.model.ntier,
        n_sub_iters=cfg.model.n_sub_iters,
        pid_leaky=get_opt(cfg, "model.pid_leaky", 1.0),
        return_intermediate=get_opt(cfg, "model.return_intermediate", False),
        collect_vis=get_opt(cfg, "model.collect_vis", False),
    )


def resolve_checkpoint(cfg, ckpt_arg=None, use_ema=False):
    if ckpt_arg:
        return ckpt_arg

    suffix = "_best_model_ema.pth" if use_ema else "_best_model.pth"
    return os.path.join(cfg.logging.save_dir, f"{cfg.dataset.name}{suffix}")


def load_checkpoint(model, ckpt_path, device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"权重文件不存在: {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        # 老版本 torch 保存的权重可能带 optimizer 等非张量对象
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 兼容纯 state_dict 与 {'model': ..., 'epoch': ...} 两种格式
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)

    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    print(f"✓ 已加载权重: {ckpt_path}" + (f" (epoch {epoch})" if epoch else ""))
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="CyberFuse 推理与评测")
    parser.add_argument("--config", default="configs/chikusei.yaml")
    parser.add_argument("--checkpoint", default=None, help="不指定则按配置推导")
    parser.add_argument("--ema", action="store_true", help="加载 EMA 权重")
    parser.add_argument("--output-dir", default="./output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device)

    in_memory = bool(get_opt(cfg, "dataset.in_memory", True))
    num_workers = int(get_opt(cfg, "inference.num_workers", 0))
    pin_memory = bool(get_opt(cfg, "inference.pin_memory", True))

    test_dataset = LoadData(
        dataname=cfg.dataset.dataname,
        file_path=cfg.dataset.test_path,
        scaling_factor=cfg.dataset.scaling_factor,
        in_memory=in_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(cfg).to(device)
    ckpt_path = resolve_checkpoint(cfg, args.checkpoint, args.ema)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    scale = cfg.model.scale
    patch_size = cfg.inference.patch_size
    shave = get_opt(cfg, "inference.shave", 16)

    cal_metrics = Metrics(ratio=scale)
    total_metrics = cal_metrics.init()
    inference_times = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            msi, hsi, gt = batch
            msi, hsi, gt = msi.to(device), hsi.to(device), gt.to(device)
            _, _, h, w = msi.shape

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if h > patch_size or w > patch_size:
                pred = test_chop(
                    model,
                    hsi,
                    msi,
                    scale=scale,
                    patch_size=patch_size,
                    shave=shave,
                )
            else:
                pred = model(hsi, msi)
                if isinstance(pred, (list, tuple)):
                    pred = pred[-1]

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            inference_times.append(elapsed)
            print(f"  Sample {i}: {elapsed:.4f}s")

            total_metrics = cal_metrics.update(total_metrics, pred, gt)

            # 输出文件名跟随数据集，避免不同数据集互相覆盖
            sio.savemat(
                os.path.join(args.output_dir, f"{cfg.dataset.dataname}_{i}.mat"),
                {"HSI": pred.squeeze(0).cpu().numpy()},
            )

    total_metrics = cal_metrics.compute(total_metrics)
    print(total_metrics)

    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    trimmed_avg = (
        (sum(inference_times) - min(inference_times) - max(inference_times))
        / max(len(inference_times) - 2, 1)
        if len(inference_times) >= 3
        else avg_time
    )
    print(f"\n推理时间统计:")
    print(f"  总样本数: {len(inference_times)}")
    print(f"  总推理时间: {sum(inference_times):.4f}s")
    print(f"  平均推理时间: {avg_time:.4f}s")
    print(f"  去极值平均推理时间: {trimmed_avg:.4f}s (去掉1个最大+1个最小)")


if __name__ == "__main__":
    main()
