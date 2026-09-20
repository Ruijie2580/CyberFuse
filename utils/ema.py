import torch
import copy
from contextlib import contextmanager

class EMA:
    def __init__(self, model, decay=0.999, start_epoch=1, device=None):
        self.decay = decay
        self.start_epoch = start_epoch
        self.device = device if device else next(model.parameters()).device
        
        # 创建影子模型
        self.shadow = copy.deepcopy(model).to(self.device)
        self.shadow.eval()
        # 冻结参数
        for param in self.shadow.parameters():
            param.requires_grad_(False)

        self.collected = False

    @torch.no_grad()
    def update(self, model, current_epoch):
        if current_epoch < self.start_epoch:
            return

        if not self.collected:
            self.shadow.load_state_dict(model.state_dict())
            self.collected = True
            # 第一次直接复制，不需要平滑
            return

        model_params = dict(model.named_parameters())
        shadow_params = dict(self.shadow.named_parameters())

        for name, param in model_params.items():
            if name in shadow_params:
                # shadow = decay * shadow + (1 - decay) * new_param
                shadow_params[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

        model_buffers = dict(model.named_buffers())
        shadow_buffers = dict(self.shadow.named_buffers())
        for name, buffer in model_buffers.items():
            if name in shadow_buffers:
                shadow_buffers[name].copy_(buffer)

    @contextmanager
    def swap(self, model):
        if not self.collected:
            yield
            return

        backup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        model.load_state_dict(self.shadow.state_dict(), strict=False)
        
        try:
            yield
        finally:
            model.load_state_dict(backup_state, strict=False)

    def state_dict(self):
        return self.shadow.state_dict()