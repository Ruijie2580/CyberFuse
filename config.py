import yaml
import torch
from types import SimpleNamespace


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(v) for v in d]
    return d


def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(v) for v in obj]
    return obj


def get_opt(cfg, path: str, default=None):
    """
    按 "a.b.c" 路径读取可选配置项，路径上任意一环缺失时返回 default。

    用于新增的可选配置：老配置文件不加这些 key 也能正常跑。
    """
    cur = cfg
    for key in path.split("."):
        if not hasattr(cur, key):
            return default
        cur = getattr(cur, key)
    return cur


def load_config(path: str) -> SimpleNamespace:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 空文件 / 只有注释的文件会让 safe_load 返回 None，
    # 若不拦住，后面 cfg['device'] 会抛一个很难定位的 TypeError
    if not isinstance(cfg, dict):
        raise ValueError(
            f"配置文件为空或顶层不是字典: {path}（safe_load 返回 {type(cfg).__name__}）"
        )

    cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg['device'] == 'cpu':
        print("[Warning] CUDA not available, training will be slow!")
    return _dict_to_namespace(cfg)
