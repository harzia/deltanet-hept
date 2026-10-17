"""Unified train/test runner for the DeltaNet-on-HEPT comparison.

Handles both HEPT benchmark tasks with the *same* backbone:
  * tracking  -> InfoNCE metric learning, metric = accuracy@0.9
  * pileup    -> node binary classification, metric = auc (average precision)

The token-mixer (hept | deltanet | ...) is selected purely by ``model_name`` in
the yaml config, so HEPT and DeltaNet run through identical code -> apples to
apples. This script deliberately avoids the ``nni`` dependency of the original
trainers and adds structured logging (see utils/run_logger.py).

Usage (run from the ``src`` directory):
    python run.py --config ../configs/tracking/deltanet_smoke.yaml
    python run.py --config ../configs/tracking/deltanet_smoke.yaml --mode test \
                  --ckpt ../logs/<run>/best_model.pt
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from torchmetrics import MeanMetric
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR
from torch_geometric.utils import unbatch, to_undirected

from utils import set_seed, get_optimizer, get_lr_scheduler, get_loss
from utils.get_data import get_data_loader, get_dataset
from utils.get_model import get_model
from utils.metrics import acc_and_pr_at_k, point_filter
from utils.run_logger import RunLogger, make_log_dir
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]


# ===========================================================================
# Task helpers
# ===========================================================================
def infer_task(dataset_name: str) -> str:
    return "tracking" if "tracking" in dataset_name else "pileup"


def resolve_dataset_dir(data_dir: str, dataset_name: str) -> Path:
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    sub = "tracking" if "tracking" in dataset_name else dataset_name
    return data_dir / sub


# ---- tracking (InfoNCE) ---------------------------------------------------
def tracking_process(data, phase, device, p=0.2):
    data = data.to(device)
    if phase == "train":
        num_aug_pairs = int(data.point_pairs_index.size(1) * p / 2)
        pairs_to_add = to_undirected(torch.randint(0, data.num_nodes, (2, num_aug_pairs), device=device))
        data.point_pairs_index = torch.cat([data.point_pairs_index, pairs_to_add], dim=1)
    return data


def tracking_update_metrics(metrics, data, emb, cluster_ids, dist_metric):
    embeddings = unbatch(emb, data.batch)
    cluster_ids_u = unbatch(cluster_ids, data.batch)
    acc_09 = 0.0
    for pt in metrics["pt_thres"]:
        batch_mask = point_filter(cluster_ids, data.reconstructable, data.pt, pt_thres=pt)
        mask = unbatch(batch_mask, data.batch)
        res = [acc_and_pr_at_k(embeddings[i], cluster_ids_u[i], mask[i], dist_metric) for i in range(len(embeddings))]
        res = torch.tensor(res)
        metrics[f"accuracy@{pt}"].update(res[:, 0])
        metrics[f"precision@{pt}"].update(res[:, 1])
        metrics[f"recall@{pt}"].update(res[:, 2])
        if pt == 0.9:
            acc_09 = res[:, 0].mean().item()
    return acc_09


def tracking_metrics_dict():
    pt_thres = [0, 0.5, 0.9]
    names = ["accuracy", "precision", "recall"]
    m = {f"{n}@{p}": MeanMetric(nan_strategy="error") for n in names for p in pt_thres}
    m["loss"] = MeanMetric(nan_strategy="error")
    m["pt_thres"] = pt_thres
    return m


def tracking_compute(metrics):
    out = {
        f"{n}@{p}": metrics[f"{n}@{p}"].compute().item()
        for n in ["accuracy", "precision", "recall"]
        for p in metrics["pt_thres"]
    }
    out["loss"] = metrics["loss"].compute().item()
    return out


# ---- pileup (classification) ---------------------------------------------
def pileup_update_metrics(metrics, data, emb):
    pred = (emb > 0.5).int()[data.is_neu].cpu()
    label = data.y[data.is_neu].cpu()
    scores = emb[data.is_neu].cpu()
    ap = average_precision_score(label, scores)
    metrics["auc"].update(ap)
    metrics["f1"].update(f1_score(label, pred))
    metrics["roc"].update(roc_auc_score(label, scores))
    return ap


def pileup_metrics_dict():
    m = {n: MeanMetric() for n in ["auc", "f1", "roc"]}
    m["loss"] = MeanMetric()
    return m


def pileup_compute(metrics):
    return {n: metrics[n].compute().item() for n in ["auc", "f1", "roc"]} | {"loss": metrics["loss"].compute().item()}


def reset_metrics(metrics):
    for m in metrics.values():
        if isinstance(m, MeanMetric):
            m.reset()


# ===========================================================================
# Epoch loop
# ===========================================================================
def run_one_epoch(task, model, optimizer, criterion, loader, phase, epoch, device, metrics, lr_s):
    is_train = phase == "train"
    model.train() if is_train else model.eval()
    metric_res = None
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for idx, data in enumerate(loader):
            if task == "tracking":
                data = tracking_process(data, phase, device)
                emb = model(data)
                loss = criterion(emb, data.point_pairs_index, data.particle_id, data.reconstructable, data.pt)
            else:
                data = data.to(device)
                emb = model(data)
                loss = criterion(emb[data.is_neu], data.y[data.is_neu].unsqueeze(-1).float())

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if lr_s is not None and isinstance(lr_s, LambdaLR):
                    lr_s.step()

            with torch.no_grad():
                if task == "tracking":
                    tracking_update_metrics(metrics, data, emb.detach(), data.particle_id.detach(), criterion.dist_metric)
                else:
                    pileup_update_metrics(metrics, data, emb.detach())
                metrics["loss"].update(loss.item())

            if idx == len(loader) - 1:
                metric_res = tracking_compute(metrics) if task == "tracking" else pileup_compute(metrics)
                reset_metrics(metrics)
    return metric_res


# ===========================================================================
# Driver
# ===========================================================================
def build(config):
    if config["model_name"] in ("trans_hept", "trans_deltanet") and config["batch_size"] != 1:
        raise ValueError("HEPT and DeltaNet currently require batch_size=1 (one event per forward)")
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(config.get("num_threads", 4))
    set_seed(config["seed"])

    dataset_name = config["dataset_name"]
    task = infer_task(dataset_name)
    dataset_dir = resolve_dataset_dir(config["data_dir"], dataset_name)

    dataset = get_dataset(dataset_name, dataset_dir, use_sharded=config.get("use_sharded", False))
    idx_split = deepcopy(dataset.idx_split)

    # Memory / time cap: keep only the first `max_events` of each split.
    max_events = config.get("max_events", None)
    if max_events:
        for k in ("train", "valid", "test"):
            s = idx_split[k]
            idx_split[k] = s[: max_events] if not torch.is_tensor(s) else s[: max_events]

    if any(len(idx_split[k]) == 0 for k in ("train", "valid", "test")):
        raise ValueError("Dataset must have nonempty train, valid and test splits")
    loaders = get_data_loader(dataset, idx_split, batch_size=config["batch_size"],
                              num_workers=config.get("num_workers", 0))
    # Legacy dataset constructors may seed preprocessing globally (Pileup=42).
    # Restore the requested model/loader seed after loading stored data.
    set_seed(config["seed"])
    model = get_model(config["model_name"], config["model_kwargs"], dataset,
                      count_flops=config.get("count_flops", False)).to(device)
    criterion = get_loss(config["loss_name"], config["loss_kwargs"])
    return device, task, dataset, loaders, model, criterion


def train(config, logger: RunLogger):
    device, task, dataset, loaders, model, criterion = build(config)
    logger.info(f"device={device} task={task} model={config['model_name']} "
                f"dataset={config['dataset_name']} n_train={len(loaders['train'].dataset)}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.event("meta", model=config["model_name"], dataset=config["dataset_name"],
                 task=task, device=str(device), num_params=num_params,
                 model_kwargs=config["model_kwargs"])

    opt = get_optimizer(model.parameters(), config["optimizer_name"], config["optimizer_kwargs"])
    config["lr_scheduler_kwargs"]["num_training_steps"] = config["num_epochs"] * max(1, len(loaders["train"]))
    lr_s = get_lr_scheduler(opt, config["lr_scheduler_name"], config["lr_scheduler_kwargs"])
    metrics = tracking_metrics_dict() if task == "tracking" else pileup_metrics_dict()

    main_metric = config["main_metric"]
    coef = 1 if config["mode"] == "max" else -1
    best = {"epoch": -1, "valid": -coef * float("inf"), "test": None}

    for epoch in range(config["num_epochs"]):
        tr = run_one_epoch(task, model, opt, criterion, loaders["train"], "train", epoch, device, metrics, lr_s)
        va = run_one_epoch(task, model, opt, criterion, loaders["valid"], "valid", epoch, device, metrics, lr_s)
        te = run_one_epoch(task, model, opt, criterion, loaders["test"], "test", epoch, device, metrics, lr_s)

        if lr_s is not None:
            if isinstance(lr_s, ReduceLROnPlateau):
                lr_s.step(va[config["lr_scheduler_metric"]])
            elif isinstance(lr_s, StepLR):
                lr_s.step()

        lr_now = opt.param_groups[0]["lr"]
        logger.epoch(epoch, "train", tr, lr=lr_now)
        logger.epoch(epoch, "valid", va, lr=lr_now)
        logger.epoch(epoch, "test", te, lr=lr_now)

        if va[main_metric] * coef > best["valid"] * coef:
            best = {"epoch": epoch, "valid": va[main_metric], "test": te[main_metric], "test_full": te}
            torch.save(model.state_dict(), logger.log_dir / "best_model.pt")
            logger.info(f"** new best @ epoch {epoch}: valid {main_metric}={va[main_metric]:.4f} "
                        f"test {main_metric}={te[main_metric]:.4f} (checkpoint saved)")

    logger.event("final", best_epoch=best["epoch"], main_metric=main_metric,
                 best_valid=best["valid"], best_test=best.get("test"),
                 best_test_full=best.get("test_full"))
    logger.info(f"DONE. best epoch {best['epoch']}: valid {main_metric}={best['valid']:.4f} "
                f"test {main_metric}={best.get('test')}")
    return best


def test(config, ckpt_path, logger: RunLogger):
    device, task, dataset, loaders, model, criterion = build(config)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    logger.info(f"loaded checkpoint {ckpt_path}")
    metrics = tracking_metrics_dict() if task == "tracking" else pileup_metrics_dict()
    te = run_one_epoch(task, model, None, criterion, loaders["test"], "test", 0, device, metrics, None)
    logger.epoch(0, "test", te)
    logger.event("final", main_metric=config["main_metric"], best_test=te[config["main_metric"]], best_test_full=te)
    logger.info(f"TEST {config['main_metric']}={te[config['main_metric']]:.4f}")
    return te


def main():
    ap = argparse.ArgumentParser(description="DeltaNet/HEPT train+test runner")
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--mode", choices=["train", "test"], default="train")
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint for --mode test")
    ap.add_argument("--kernel", choices=["hept", "deltanet", "deltanet-hept"],
                    help="select the token mixer while keeping the configured backbone and task")
    ap.add_argument("--note", type=str, default=None, help="override config note")
    ap.add_argument("--override", "-o", action="append", default=[],
                    help="dotted-key override, e.g. -o model_kwargs.deltanet_sort_key=phi "
                         "(repeatable). Values are parsed as YAML scalars.")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if args.kernel:
        config["model_name"] = "trans_" + ("deltanet" if args.kernel == "deltanet-hept" else args.kernel)
        if args.kernel in ("deltanet", "deltanet-hept"):
            config["model_kwargs"]["deltanet_kernel"] = "hept" if args.kernel == "deltanet-hept" else "standard"
        if args.kernel == "deltanet-hept":
            config["model_kwargs"].update(deltanet_impl="torch", deltanet_qk_activation="identity", deltanet_qk_norm="none")
            for key,value in dict(deltanet_hept_backend="rff", deltanet_kernel_features=128, num_w_per_dist=10,
                                  block_size=100, n_hashes=3,
                                  num_regions=15 if config["dataset_name"] == "tracking-6k" else (150 if "tracking" in config["dataset_name"] else 140)).items():
                config["model_kwargs"].setdefault(key,value)
        if args.kernel == "hept":
            defaults = dict(block_size=100, n_hashes=3, num_regions=15 if config["dataset_name"] == "tracking-6k" else (150 if "tracking" in config["dataset_name"] else 140),
                            pe_type="none", num_w_per_dist=10)
            for key, value in defaults.items():
                config["model_kwargs"].setdefault(key, value)
    if args.note:
        config["note"] = args.note
    for ov in args.override:
        key, _, val = ov.partition("=")
        if "=" not in ov or not key or any(not p for p in key.split(".")):
            ap.error(f"invalid override {ov!r}; expected dotted.key=value")
        parsed = yaml.safe_load(val)
        node = config
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parsed

    if args.mode == "test" and not args.ckpt:
        ap.error("--mode test requires --ckpt")
    log_root = REPO_ROOT / "logs"
    log_dir = make_log_dir(log_root, config["model_name"], config["dataset_name"],
                           config.get("note", "run"), config["seed"])
    logger = RunLogger(log_dir, config=config)
    logger.info(f"config: {args.config}  mode: {args.mode}")
    try:
        if args.mode == "train":
            train(config, logger)
        else:
            assert args.ckpt, "--mode test requires --ckpt"
            test(config, args.ckpt, logger)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
