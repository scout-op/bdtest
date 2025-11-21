import os
import time
import torch

# Rich is optional; fallback to plain prints if missing
try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # noqa: BLE001
    Console = None
    Table = None

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import MODELS

# Ensure our package is registered
import seq_grow_graph  # noqa: F401


def _get_console():
    if Console is None:
        class _Dummy:
            def print(self, *args, **kwargs):  # type: ignore
                print(*args)

            def print_exception(self):  # type: ignore
                import traceback
                traceback.print_exc()
        return _Dummy()
    return Console()


def _make_table(title: str):
    if Table is None:
        return None
    t = Table(title=title)
    t.add_column("Key", style="magenta")
    t.add_column("Shape", style="cyan")
    t.add_column("Dtype")
    t.add_column("Device")
    return t


def _shape_dtype_device(x):
    if x is None:
        return ("None", "N/A", "N/A")
    try:
        shape = " x ".join(map(str, x.shape))
        return (shape, str(x.dtype), str(x.device))
    except Exception:
        try:
            return (str(type(x)), "N/A", "N/A")
        except Exception:
            return ("?", "?", "?")


def main():
    console = _get_console()

    cfg_path = os.path.join("configs", "seq_grow_graph", "seq_grow_graph_default.py")
    console.print("🚀 GCAD 健全性检查 (Sanity Check)")
    console.print(f"加载配置: {cfg_path}")

    cfg = Config.fromfile(cfg_path)

    # Optional: make it lighter for a dry run
    # cfg.train_dataloader.batch_size = 1

    console.print("🔧 构建 Dataloader ...")
    train_dataloader = Runner.build_dataloader(cfg.train_dataloader)
    console.print("✅ Dataloader 构建成功!")

    console.print("🔧 构建 GCAD 模型 ...")
    model = MODELS.build(cfg.model)
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()
    console.print("✅ 模型构建成功!")

    console.print("\nFetching 1 个 Batch ...")
    start = time.time()
    try:
        data_batch = next(iter(train_dataloader))
    except Exception:
        console.print("❌ Dataloader Fetch 失败!")
        console.print_exception()
        return
    console.print(f"✅ Batch Fetch 成功 (用时: {time.time()-start:.2f}s)")

    # Preprocess (move to device, stack, etc.)
    processed = model.data_preprocessor(data_batch, True)
    inputs = processed["inputs"]

    # Summary
    table = _make_table("📦 GCAD Batch 张量形状检查")
    def add_row(name, tensor):
        if table is None:
            shape, dtype, device = _shape_dtype_device(tensor)
            console.print(f"{name:<36} | shape={shape:<20} | {dtype:<14} | {device}")
        else:
            shape, dtype, device = _shape_dtype_device(tensor)
            table.add_row(name, shape, dtype, device)

    # Basic sizes
    B = inputs.get('img').shape[0] if 'img' in inputs else 'N/A'
    L = inputs.get('gt_topo_seq_input').shape[1] if 'gt_topo_seq_input' in inputs else 'N/A'
    Nmax = inputs.get('gt_nodes_xy').shape[1] if 'gt_nodes_xy' in inputs else 'N/A'
    V = inputs.get('gt_logits_mask').shape[2] if 'gt_logits_mask' in inputs else 'N/A'
    console.print(f"[Info] B={B}, L={L}, N_max={Nmax}, V={V}")

    # Vision
    add_row("inputs['img']", inputs.get('img'))

    # Topo-AR
    console.print("--- Topo-AR ---")
    add_row("inputs['gt_topo_seq_input']", inputs.get('gt_topo_seq_input'))
    add_row("inputs['gt_topo_seq_target']", inputs.get('gt_topo_seq_target'))
    add_row("inputs['gt_logits_mask']", inputs.get('gt_logits_mask'))

    # Geom
    console.print("--- Geom ---")
    add_row("inputs['gt_nodes_xy']", inputs.get('gt_nodes_xy'))
    add_row("inputs['gt_nodes_mask']", inputs.get('gt_nodes_mask'))
    add_row("inputs['gt_add_node_indices']", inputs.get('gt_add_node_indices'))
    add_row("inputs['gt_geom_blocks']", inputs.get('gt_geom_blocks'))
    add_row("inputs['gt_geom_blocks_mask']", inputs.get('gt_geom_blocks_mask'))

    if table is not None:
        console.print(table)

    # Forward one loss pass
    console.print("\n🔥 执行一次 model.loss() ...")
    try:
        losses = model.loss(inputs, processed['data_samples'])
        console.print("✅ 前向传播成功! ")
    except Exception:
        console.print("❌ model.loss() 执行失败!")
        console.print("请检查对齐步骤 (torch.gather) 与 MLP 维度。")
        console.print_exception()
        return

    # Pretty print losses
    if isinstance(losses, dict):
        if Table is not None:
            lt = Table(title="Loss 输出")
            lt.add_column("Name")
            lt.add_column("Value")
            for k, v in losses.items():
                try:
                    val = v.item() if torch.is_tensor(v) else float(v)
                except Exception:
                    val = str(v)
                lt.add_row(k, f"{val}")
            console.print(lt)
        else:
            for k, v in losses.items():
                try:
                    val = v.item() if torch.is_tensor(v) else float(v)
                except Exception:
                    val = str(v)
                console.print(f"{k}: {val}")

    console.print("\n🎉 GCAD 训练闭环验证通过! 🎉")


if __name__ == "__main__":
    main()
