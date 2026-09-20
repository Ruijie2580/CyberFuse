import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.ema import EMA
from utils.loss import CharbonnierLoss
from config import get_opt, namespace_to_dict
import os

import swanlab
import numpy as np
import random


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self,
        config,
        loss_fn,
        metrics_fn,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
    ):
        self.cfg = config
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        seed_everything(seed=config.training.seed)

        self.device = torch.device(config.device)
        self.mode = config.logging.metrics_mode
        if self.mode == "min":
            self.best_metric = float("inf")
        else:
            self.best_metric = float("-inf")
        self.save_dir = config.logging.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.use_ema = config.ema.enabled
        self.ema = None

        # 所有 swanlab 调用都以这个开关为前提，关掉日志功能时不会炸
        self.use_swanlab = bool(get_opt(config, "logging.use_swanlab", False))

        if self.use_swanlab:
            swanlab.init(
                project=get_opt(config, "logging.project", "CyberFuse"),
                name=self.cfg.dataset.name,
                config=namespace_to_dict(self.cfg),
            )

    def fit(self, model):
        self.model = model.to(self.device)

        if self.use_ema:
            start_epoch = self.cfg.ema.start_epoch
            print(f"EMA enabled. Tracking will start at Epoch {start_epoch}.")
            self.ema = EMA(
                self.model,
                decay=self.cfg.ema.decay,
                start_epoch=start_epoch,
                device=self.device,
            )

        for epoch in range(1, self.cfg.training.epochs + 1):
            print(f"\nEpoch {epoch}/{self.cfg.training.epochs}")

            train_loss = self._train_one_epoch(epoch)

            if self.scheduler:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # 逐 epoch 记录即可，逐 iteration 记录会把训练拖慢一个量级
            if self.use_swanlab:
                swanlab.log(
                    {"train/loss": train_loss, "train/lr": current_lr}, step=epoch
                )

            is_check_point = (epoch % 5 == 0) or (epoch == self.cfg.training.epochs)

            if is_check_point:
                val_loss, val_metrics = self._validate(epoch)
                self._log_epoch(epoch, train_loss, val_loss, val_metrics)
                self._save_if_best(epoch, val_metrics)
            else:
                print(f"Train Loss: {train_loss:.4f} (Skipping Val)")

        print("Training finished.")

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        pbar = tqdm(self.train_loader, desc="Train", leave=False)

        for batch in pbar:
            HRMS, LRHS, HRHS = [x.to(self.device) for x in batch]

            pred = self.model(LRHS, HRMS)

            if isinstance(pred, (list, tuple)):
                pred = pred[-1]

            loss = self.loss_fn(pred, HRHS)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.use_ema:
                self.ema.update(self.model, current_epoch=epoch)

            # 按样本数加权，最后一个不满的 batch 不会被高估权重
            bs = HRHS.shape[0]
            total_loss += loss.item() * bs
            total_samples += bs

            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        return total_loss / max(total_samples, 1)

    def _validate(self, epoch=None):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        total_metrics = self.metrics_fn.init()

        def run_validation():
            nonlocal total_loss, total_samples, total_metrics
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Val", leave=False):
                    HRMS, LRHS, HRHS = [x.to(self.device) for x in batch]

                    pred = self.model(LRHS, HRMS)

                    if isinstance(pred, (list, tuple)):
                        pred = pred[-1]

                    loss = self.loss_fn(pred, HRHS)

                    bs = HRHS.shape[0]
                    total_loss += loss.item() * bs
                    total_samples += bs
                    total_metrics = self.metrics_fn.update(total_metrics, pred, HRHS)

        if self.use_ema:
            with self.ema.swap(self.model):
                run_validation()
        else:
            run_validation()

        avg_loss = total_loss / max(total_samples, 1)
        result = self.metrics_fn.compute(total_metrics)

        if self.use_swanlab:
            swanlab.log(
                {
                    "val/loss": avg_loss,
                    "val/PSNR": result["PSNR"],
                    "val/SSIM": result["SSIM"],
                    "val/SAM": result["SAM"],
                    "val/ERGAS": result["ERGAS"],
                },
                step=epoch,
            )
        return avg_loss, result

    def _log_epoch(self, epoch, train_loss, val_loss, metrics):
        print(f"Epoch {epoch}:")
        print(f"  Train Loss = {train_loss:.4f}")
        print(f"  Val Loss   = {val_loss:.4f}")
        print(f"  Metrics: {metrics}")

    def _save_if_best(self, epoch, metrics):
        key = self.cfg.logging.key_metric
        score = metrics[key]
        is_best = False

        if self.mode == "min":
            if score < self.best_metric:
                self.best_metric = score
                is_best = True
        else:
            if score > self.best_metric:
                self.best_metric = score
                is_best = True

        if is_best:
            path = os.path.join(
                self.save_dir,
                (
                    f"{self.cfg.dataset.name}_best_model_ema.pth"
                    if self.use_ema
                    else f"{self.cfg.dataset.name}_best_model.pth"
                ),
            )

            model_state = (
                self.ema.state_dict() if self.use_ema else self.model.state_dict()
            )

            state = {
                "epoch": epoch,
                "model": model_state,
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(state, path)
            print(f"Saved best model at epoch {epoch} (best {key}={score:.4f})")


def build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]

    opt_cls = getattr(optim, cfg.optimizer.name)

    optimizer = opt_cls(params, lr=cfg.optimizer.lr, **cfg.optimizer.params.__dict__)

    print(f"Initialized Optimizer: {cfg.optimizer.name}")
    return optimizer


def build_scheduler(optimizer, cfg):
    if not cfg.scheduler.name:
        return None

    sche_cls = getattr(lr_scheduler, cfg.scheduler.name)

    scheduler = sche_cls(optimizer, **cfg.scheduler.params.__dict__)

    print(f"Initialized Scheduler: {cfg.scheduler.name}")
    return scheduler


if __name__ == "__main__":
    import argparse

    from model.dun import Net
    from utils.metrics import Metrics
    from utils.dcmall_data import LoadData
    from config import load_config

    parser = argparse.ArgumentParser(description="CyberFuse 训练")
    parser.add_argument("--config", default="configs/chikusei.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model = Net(
        hsi_channels=cfg.dataset.hsi_channels,
        msi_channels=cfg.dataset.msi_channels,
        scale=cfg.model.scale,
        ntier=cfg.model.ntier,
        n_sub_iters=cfg.model.n_sub_iters,
        pid_leaky=get_opt(cfg, "model.pid_leaky", 1.0),
        return_intermediate=get_opt(cfg, "model.return_intermediate", False),
        collect_vis=get_opt(cfg, "model.collect_vis", False),
    ).to(cfg.device)

    in_memory = bool(get_opt(cfg, "dataset.in_memory", True))
    num_workers = int(get_opt(cfg, "training.num_workers", 0))
    pin_memory = bool(get_opt(cfg, "training.pin_memory", True))

    # Windows(spawn) 下，全量载入内存的数据集会被 pickle 给每个 worker，
    # 内存成倍膨胀。要并行加载请改用 dataset.in_memory: false
    if in_memory and num_workers > 0:
        print(
            f"[Warning] dataset.in_memory=true 配合 num_workers={num_workers} 会让"
            f"每个 worker 各持一份数据副本；建议改为 in_memory: false 或 num_workers: 0"
        )

    train_data = LoadData(
        dataname=cfg.dataset.dataname,
        file_path=cfg.dataset.train_path,
        scaling_factor=cfg.dataset.scaling_factor,
        in_memory=in_memory,
    )
    val_data = LoadData(
        dataname=cfg.dataset.dataname,
        file_path=cfg.dataset.val_path,
        scaling_factor=cfg.dataset.scaling_factor,
        in_memory=in_memory,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    loss_fn = CharbonnierLoss()
    # ERGAS 的 ratio 必须跟随放大倍数，否则换 scale 后指标会算错
    metrics_fn = Metrics(ratio=cfg.model.scale)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    trainer = Trainer(
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
    )
    trainer.fit(model)
