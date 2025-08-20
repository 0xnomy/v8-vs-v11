import argparse
import time
from pathlib import Path
from typing import Dict, Any

from ultralytics import YOLO
from ultralytics.utils.checks import check_file
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

console = Console()

def fmt_ms(ms: float) -> str:
    return f"{ms:.1f} ms/img"

def fmt_pct(p: float) -> str:
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f}%"

def get_weight_size_mb(weights_name: str) -> float:
    path = Path(check_file(weights_name))  # resolves cache path if not in CWD
    return path.stat().st_size / (1024 * 1024)

def get_dataset_yaml(arg: str) -> str:
    # shorthand choices
    if arg.lower() in {"coco", "coco2017"}:
        return "coco.yaml"
    if arg.lower() in {"coco128", "coco-128"}:
        return "coco128.yaml"
    # otherwise assume custom yaml path
    return arg

def pretty_dataset_info(model: YOLO, data_yaml: str, imgsz: int, batch: int) -> Dict[str, Any]:
    metrics = model.val(data=data_yaml, imgsz=imgsz, batch=1, verbose=False, plots=False, save_json=False)
    names = getattr(metrics, "names", None) or {}
    ncls = len(names)
    return {"names": names, "num_classes": ncls}

def evaluate_model(model: YOLO, data_yaml: str, imgsz: int, batch: int) -> Dict[str, float]:
    t0 = time.time()
    m = model.val(
        data=data_yaml, imgsz=imgsz, batch=batch, verbose=False,
        plots=False, save_json=False
    )
    total_s = time.time() - t0
    # Speed metrics are in ms per image
    inf_ms = float(m.speed["inference"])
    # Metrics
    return {
        "map50": float(m.box.map50),
        "map5095": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "inf_ms": inf_ms,
        "fps": 1000.0 / inf_ms if inf_ms > 0 else 0.0,
        "total_s": total_s,
        "num_classes": len(getattr(m, "names", {})),
    }

def diff_pct(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0

def render_header(title: str, subtitle: str):
    console.print(Panel.fit(
        f"[bold white]{title}[/bold white]\n[dim]{subtitle}[/dim]",
        border_style="cyan", padding=(1,2)
    ))

def render_setup_info(dataset_label: str, imgsz: int, batch: int, device: str | None):
    t = Table(box=box.SIMPLE_HEAVY, show_edge=True)
    t.add_column("Setting", style="bold")
    t.add_column("Value")
    t.add_row("Dataset", dataset_label)
    t.add_row("imgsz", str(imgsz))
    t.add_row("batch", str(batch))
    t.add_row("device", device or "auto")
    console.print(t)

def render_model_sizes(sz_v8: float, sz_v11: float):
    t = Table(box=box.MINIMAL_HEAVY_HEAD, show_edge=True, title="Model Files")
    t.add_column("Model", style="bold")
    t.add_column("Size (MB)", justify="right")
    t.add_row("YOLOv8n", f"{sz_v8:.1f}")
    t.add_row("YOLOv11n", f"{sz_v11:.1f}")
    console.print(t)

def render_results_table(r8: Dict[str, float], r11: Dict[str, float]):
    t = Table(title="Validation Metrics", box=box.SIMPLE_HEAVY, show_edge=True)
    t.add_column("Metric", style="bold")
    t.add_column("YOLOv8n", justify="right")
    t.add_column("YOLOv11n", justify="right")
    t.add_column("Δ (YOLOv11n - YOLOv8n)", justify="right")

    # accuracy
    t.add_row("mAP@0.5", f"{r8['map50']:.3f}", f"{r11['map50']:.3f}", fmt_pct(diff_pct(r11['map50'], r8['map50'])))
    t.add_row("mAP@0.5:0.95", f"{r8['map5095']:.3f}", f"{r11['map5095']:.3f}", fmt_pct(diff_pct(r11['map5095'], r8['map5095'])))
    t.add_row("Precision", f"{r8['precision']:.3f}", f"{r11['precision']:.3f}", fmt_pct(diff_pct(r11['precision'], r8['precision'])))
    t.add_row("Recall", f"{r8['recall']:.3f}", f"{r11['recall']:.3f}", fmt_pct(diff_pct(r11['recall'], r8['recall'])))

    # speed
    t.add_row("Speed (ms/img)", fmt_ms(r8["inf_ms"]), fmt_ms(r11["inf_ms"]), fmt_pct(diff_pct(r11["inf_ms"], r8["inf_ms"])))
    t.add_row("FPS (↑ better)", f"{r8['fps']:.0f}", f"{r11['fps']:.0f}", fmt_pct(diff_pct(r11["fps"], r8["fps"])))

    # dataset classes (informational)
    t.add_row("Dataset classes", str(int(r8["num_classes"])), str(int(r11["num_classes"])), "—")
    console.print(t)

def render_takeaways(r8: Dict[str, float], r11: Dict[str, float]):
    acc = diff_pct(r11['map50'], r8['map50'])
    acc9595 = diff_pct(r11['map5095'], r8['map5095'])
    speed = diff_pct(r11['inf_ms'], r8['inf_ms'])  # ms/img (↓ better)
    fps = diff_pct(r11['fps'], r8['fps'])
    bullets = [
        f"🎯  Accuracy (mAP@0.5): [bold]{fmt_pct(acc)}[/bold]",
        f"📐  Accuracy strict (mAP@0.5:0.95): [bold]{fmt_pct(acc9595)}[/bold]",
        f"⚡  Speed (ms/img): [bold]{fmt_pct(speed)}[/bold] (negative = faster)",
        f"🎞️  Throughput (FPS): [bold]{fmt_pct(fps)}[/bold]",
    ]
    console.print(Panel("\n".join(bullets), title="Quick Takeaways", border_style="green"))

def maybe_save_csv(args, r8: Dict[str, float], r11: Dict[str, float], sz8: float, sz11: float, dataset_label: str):
    if not args.save_csv:
        return
    import csv
    out = Path(args.save_csv)
    rows = [
        {"model": "yolov8n", "dataset": dataset_label, "imgsz": args.imgsz, "batch": args.batch,
         "map50": r8["map50"], "map50_95": r8["map5095"], "precision": r8["precision"], "recall": r8["recall"],
         "inf_ms": r8["inf_ms"], "fps": r8["fps"], "size_mb": sz8, "total_s": r8["total_s"]},
        {"model": "yolo11n", "dataset": dataset_label, "imgsz": args.imgsz, "batch": args.batch,
         "map50": r11["map50"], "map50_95": r11["map5095"], "precision": r11["precision"], "recall": r11["recall"],
         "inf_ms": r11["inf_ms"], "fps": r11["fps"], "size_mb": sz11, "total_s": r11["total_s"]},
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"💾 Saved CSV → [bold]{out}[/bold]")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8n vs YOLOv11n pretty evaluation")
    parser.add_argument("--dataset", default="coco128", help="coco128 | coco | path/to/your.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help='e.g. "cuda:0" or "cpu" (None=auto)')
    parser.add_argument("--save-csv", default="", help="path to save results as CSV (optional)")
    args = parser.parse_args()

    dataset_yaml = get_dataset_yaml(args.dataset)
    dataset_label = dataset_yaml

    render_header("🚀 YOLOv8n vs YOLOv11n Evaluation", f"Dataset: {dataset_label}")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        progress.add_task(description="Loading models…", total=None)
        model_v8 = YOLO("yolov8n.pt")
        model_v11 = YOLO("yolo11n.pt")

        sz8 = get_weight_size_mb("yolov8n.pt")
        sz11 = get_weight_size_mb("yolo11n.pt")

    render_setup_info(dataset_label, args.imgsz, args.batch, args.device)
    render_model_sizes(sz8, sz11)

    # Quick metadata pull (names/classes) using v8 (cheap)
    _ = pretty_dataset_info(model_v8, dataset_yaml, args.imgsz, batch=1)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        progress.add_task(description=f"Validating YOLOv8n on {dataset_label}…", total=None)
        r8 = evaluate_model(model_v8, dataset_yaml, args.imgsz, args.batch)
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        progress.add_task(description=f"Validating YOLOv11n on {dataset_label}…", total=None)
        r11 = evaluate_model(model_v11, dataset_yaml, args.imgsz, args.batch)

    render_results_table(r8, r11)
    render_takeaways(r8, r11)
    maybe_save_csv(args, r8, r11, sz8, sz11, dataset_label)

    console.print(Panel.fit("✅ Evaluation Complete", border_style="cyan"))

if __name__ == "__main__":
    main()
