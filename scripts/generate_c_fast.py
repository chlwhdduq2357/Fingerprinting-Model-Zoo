"""Generate the fast C-stage descendants: FT5 and PRUNE50.

This script is designed for an interruptible Google Colab workflow.  It saves a
completed descendant after every parent and safely skips matching completed
outputs on the next invocation.  Adversarial fine-tuning and quantization are
intentionally outside this first fast batch.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_zoo.core import ROOT, models, save_models, sha256, tensor_hash, write_json
from model_zoo.loaders.unified import Classifier, build_network, load_model


PARENTS = ["A001", "A002", "A003", "A006", "A008", "A010", "A013", "A015", "A027", "A029"]
TRANSFORM_SLOT = {"ft5": 1, "prune50": 2, "adv_ft270": 3, "ptq_int8": 4}
PAPER_URL = (
    "https://openaccess.thecvf.com/content/CVPR2022/papers/"
    "Peng_Fingerprinting_Deep_Neural_Networks_Globally_via_Universal_"
    "Adversarial_Perturbations_CVPR_2022_paper.pdf"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def c_model_id(parent_id: str, transform: str) -> str:
    """Map the fixed ten-parent/four-transform design onto C001..C040."""
    if parent_id not in PARENTS:
        raise ValueError(f"C parent is not in the fixed selection: {parent_id}")
    if transform not in TRANSFORM_SLOT:
        raise ValueError(f"Unknown transform: {transform}")
    number = PARENTS.index(parent_id) * 4 + TRANSFORM_SLOT[transform]
    return f"C{number:03d}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def parameter_hash(network: nn.Module) -> str:
    return tensor_hash({name: value.detach() for name, value in network.named_parameters()})


def state_stats(network: nn.Module) -> tuple[int, int]:
    parameters = list(network.parameters())
    total = sum(value.numel() for value in parameters)
    nonzero = sum(torch.count_nonzero(value.detach()).item() for value in parameters)
    return total, int(nonzero)


def make_loaders(data_root: Path, batch_size: int, workers: int, seed: int):
    # The model wrapper performs native normalization.  These transforms must
    # therefore return unnormalized RGB tensors in [0, 1].
    train_set = CIFAR10(
        root=str(data_root),
        train=True,
        download=True,
        transform=Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor()]),
    )
    test_set = CIFAR10(root=str(data_root), train=False, download=True, transform=ToTensor())
    generator = torch.Generator().manual_seed(seed)
    common = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, test_loader


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct = total = 0
    started = time.perf_counter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != 10 or not torch.isfinite(logits).all():
            raise RuntimeError("Invalid logits during verification")
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return {
        "samples": total,
        "correct": correct,
        "accuracy_percent": 100.0 * correct / total,
        "seconds": round(time.perf_counter() - started, 3),
    }


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_ft5(
    parent_id: str,
    device: torch.device,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    amp: bool,
    log_path: Path,
) -> tuple[nn.Module, dict]:
    seed = 31000 + int(parent_id[1:])
    set_seed(seed)
    model = load_model(parent_id, device=str(device), preprocessing="native")
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = amp and device.type == "cuda"
    scaler = make_scaler(use_amp)
    history = []
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        started = time.perf_counter()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with autocast:
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().item() * labels.numel()
            correct += (logits.detach().argmax(1) == labels).sum().item()
            total += labels.numel()
        scheduler.step()
        record = {
            "epoch": epoch,
            "loss": loss_sum / total,
            "train_accuracy_percent": 100.0 * correct / total,
            "learning_rate_after_epoch": scheduler.get_last_lr()[0],
            "seconds": round(time.perf_counter() - started, 3),
            "date": utc_now(),
        }
        history.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"  epoch {epoch}/{epochs}: loss={record['loss']:.4f}, "
            f"train_acc={record['train_accuracy_percent']:.2f}%, {record['seconds']:.1f}s",
            flush=True,
        )
    return model.network.cpu(), {
        "seed": seed,
        "epochs": epochs,
        "optimizer": "SGD",
        "momentum": 0.9,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": "CosineAnnealingLR",
        "batch_size": train_loader.batch_size,
        "augmentation": ["RandomCrop(32,padding=4)", "RandomHorizontalFlip"],
        "amp": use_amp,
        "history": history,
    }


def prune_network(network: nn.Module, amount: float = 0.5) -> tuple[nn.Module, dict]:
    """Apply global L1 weight pruning and remove PyTorch's reparameterization."""
    if not 0.0 < amount < 1.0:
        raise ValueError("Pruning amount must be between zero and one")
    targets = [
        (module, "weight")
        for module in network.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None
    ]
    if not targets:
        raise ValueError("Network has no Conv2d or Linear weights to prune")
    eligible = sum(module.weight.numel() for module, _ in targets)
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)
    masked_zero = sum((module.weight_mask == 0).sum().item() for module, _ in targets)
    for module, name in targets:
        prune.remove(module, name)
    return network.cpu(), {
        "method": "global_unstructured_L1",
        "amount": amount,
        "eligible_weight_elements": eligible,
        "masked_weight_elements": int(masked_zero),
        "actual_mask_sparsity": masked_zero / eligible,
        "recovery_fine_tuning": False,
        "affected_modules": len(targets),
    }


def make_prune50(parent_id: str) -> tuple[nn.Module, dict]:
    model = load_model(parent_id, device="cpu", preprocessing="native")
    return prune_network(model.network, amount=0.5)


def verify_round_trip(network: nn.Module, parent_row: dict) -> None:
    state = network.state_dict()
    rebuilt = build_network(parent_row)
    rebuilt.load_state_dict(state, strict=True)
    rebuilt.eval()
    with torch.inference_mode():
        logits = rebuilt(torch.zeros(2, 3, 32, 32))
    if logits.shape != (2, 10) or not torch.isfinite(logits).all():
        raise RuntimeError("Checkpoint round-trip smoke test failed")


def completed_output(row: dict | None, path: Path, expected_config: dict) -> bool:
    return bool(
        row
        and path.exists()
        and row.get("sha256") == sha256(path)
        and row.get("transform_config") == expected_config
        and row.get("status") == "verified"
    )


def descendant_row(
    parent: dict,
    model_id: str,
    transform: str,
    checkpoint_path: Path,
    network: nn.Module,
    config: dict,
    metrics: dict,
    evaluation: dict,
    device: torch.device,
) -> dict:
    row = copy.deepcopy(parent)
    relative_path = checkpoint_path.relative_to(ROOT).as_posix()
    total, nonzero = state_stats(network)
    state = network.state_dict()
    row.update(
        model_id=model_id,
        group="C",
        parent_id=parent["model_id"],
        lineage_id=parent["lineage_id"],
        transform_type=transform,
        lineage_mechanism="parameter_inheritance",
        parent_checkpoint_sha256=parent["sha256"],
        derivation_reference=PAPER_URL,
        derivation_date=utc_now(),
        original_filename=checkpoint_path.name,
        local_checkpoint_path=relative_path,
        checkpoint_url=None,
        download_spec={"kind": "local_derivative"},
        sha256=sha256(checkpoint_path),
        state_dict_sha256=tensor_hash(state),
        parameter_sha256=parameter_hash(network),
        checkpoint_bytes=checkpoint_path.stat().st_size,
        checkpoint_format="state_dict",
        framework="PyTorch",
        framework_version=torch.__version__,
        num_parameters=total,
        num_trainable_parameters=total,
        num_nontrainable_parameters=0,
        num_nonzero_parameters=nonzero,
        parameter_sparsity=1.0 - nonzero / total,
        cifar10_test_accuracy_reported=None,
        cifar10_test_accuracy_verified=evaluation["accuracy_percent"],
        cifar10_smoke_accuracy=None,
        status="verified",
        transform_config=config,
        transform_metrics=metrics,
        training_seed=config.get("seed"),
        optimizer=config.get("optimizer"),
        learning_rate=config.get("learning_rate"),
        weight_decay=config.get("weight_decay"),
        augmentation=config.get("augmentation"),
        training_recipe=(
            "Continue training the parent checkpoint for five epochs"
            if transform == "ft5"
            else "Global unstructured L1 weight pruning without recovery fine-tuning"
        ),
        notes=(
            f"Local same-lineage descendant of {parent['model_id']}; generated by {transform}."
        ),
        download_date=None,
        verification={
            "model_id": model_id,
            "passed": True,
            "samples": evaluation["samples"],
            "correct": evaluation["correct"],
            "accuracy_percent": evaluation["accuracy_percent"],
            "full_test_set": evaluation["samples"] == 10000,
            "preprocessing": "native",
            "device": str(device),
            "torch_version": torch.__version__,
            "seconds": evaluation["seconds"],
            "date": utc_now(),
        },
    )
    row.pop("full_verification", None)
    row.pop("reported_verified_gap_pp", None)
    row.pop("accuracy_warning", None)
    return row


def upsert(rows: list[dict], new_row: dict) -> None:
    for index, row in enumerate(rows):
        if row["model_id"] == new_row["model_id"]:
            rows[index] = new_row
            return
    rows.append(new_row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", nargs="+", choices=PARENTS, default=PARENTS)
    parser.add_argument("--transforms", nargs="+", choices=["ft5", "prune50"], default=["ft5", "prune50"])
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--data-root", type=Path, default=ROOT / "work/data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is false")
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers must be non-negative")

    print(f"device={device}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    rows = models()
    by_id = {row["model_id"]: row for row in rows}
    missing = [parent for parent in args.model_id if not (ROOT / by_id[parent]["local_checkpoint_path"]).exists()]
    if missing:
        parser.error(
            "Missing parent checkpoints: " + ", ".join(missing) +
            ". Run scripts/download_models.py --model-id ... first."
        )

    train_loader = test_loader = None
    if "ft5" in args.transforms:
        train_loader, test_loader = make_loaders(
            args.data_root, args.batch_size, args.workers, seed=31000
        )
    else:
        _, test_loader = make_loaders(args.data_root, args.batch_size, args.workers, seed=31000)

    run_summary = []
    for parent_id in args.model_id:
        parent = by_id[parent_id]
        for transform in args.transforms:
            model_id = c_model_id(parent_id, transform)
            checkpoint_path = ROOT / "checkpoints/C" / f"{model_id}_{parent_id}_{transform}.pth"
            log_path = ROOT / "runs/C" / f"{model_id}.jsonl"
            seed = 31000 + int(parent_id[1:]) if transform == "ft5" else 33000 + int(parent_id[1:])
            expected_config = (
                {
                    "seed": seed,
                    "epochs": args.epochs,
                    "optimizer": "SGD",
                    "momentum": 0.9,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "scheduler": "CosineAnnealingLR",
                    "batch_size": args.batch_size,
                    "augmentation": ["RandomCrop(32,padding=4)", "RandomHorizontalFlip"],
                    "amp": bool(args.amp and device.type == "cuda"),
                }
                if transform == "ft5"
                else {
                    "method": "global_unstructured_L1",
                    "amount": 0.5,
                    "eligible_modules": ["Conv2d.weight", "Linear.weight"],
                    "recovery_fine_tuning": False,
                    "seed": seed,
                }
            )
            existing = next((row for row in rows if row["model_id"] == model_id), None)
            if completed_output(existing, checkpoint_path, expected_config):
                print(f"{model_id} {parent_id} {transform}: already verified, skip", flush=True)
                run_summary.append({"model_id": model_id, "status": "skipped"})
                continue

            print(f"{model_id} {parent_id} {transform}: start", flush=True)
            started = time.perf_counter()
            if transform == "ft5":
                # Give every parent the documented parent-specific shuffle and
                # augmentation seed instead of reusing another run's generator state.
                train_loader, _ = make_loaders(
                    args.data_root, args.batch_size, args.workers, seed=seed
                )
                if log_path.exists():
                    log_path.unlink()
                network, details = train_ft5(
                    parent_id,
                    device,
                    train_loader,
                    args.epochs,
                    args.learning_rate,
                    args.weight_decay,
                    args.amp,
                    log_path,
                )
                config = {key: details[key] for key in expected_config}
                metrics = {
                    "final_train_loss": details["history"][-1]["loss"],
                    "final_train_accuracy_percent": details["history"][-1]["train_accuracy_percent"],
                    "total_epoch_seconds": sum(item["seconds"] for item in details["history"]),
                }
            else:
                network, details = make_prune50(parent_id)
                config = expected_config
                metrics = details

            verify_round_trip(network, parent)
            atomic_torch_save(network.state_dict(), checkpoint_path)
            wrapper = Classifier(network, parent, preprocessing="native").to(device).eval()
            evaluation = evaluate(wrapper, test_loader, device)
            new_row = descendant_row(
                parent, model_id, transform, checkpoint_path, network, config, metrics, evaluation, device
            )
            if new_row["sha256"] == parent["sha256"] or new_row["state_dict_sha256"] == parent.get("state_dict_sha256"):
                raise RuntimeError(f"{model_id} is identical to its parent")
            upsert(rows, new_row)
            save_models(rows)
            elapsed = round(time.perf_counter() - started, 3)
            result = {
                "model_id": model_id,
                "parent_id": parent_id,
                "transform": transform,
                "accuracy_percent": evaluation["accuracy_percent"],
                "checkpoint_sha256": new_row["sha256"],
                "seconds": elapsed,
                "status": "verified",
            }
            run_summary.append(result)
            print(
                f"{model_id}: verified accuracy={evaluation['accuracy_percent']:.2f}%, "
                f"total={elapsed:.1f}s",
                flush=True,
            )
            del wrapper, network
            if device.type == "cuda":
                torch.cuda.empty_cache()

    registered = sorted(
        (
            {
                "model_id": row["model_id"],
                "parent_id": row["parent_id"],
                "transform": row["transform_type"],
                "accuracy_percent": row.get("cifar10_test_accuracy_verified"),
                "checkpoint_sha256": row.get("sha256"),
                "status": row.get("status"),
            }
            for row in rows
            if row.get("group") == "C" and row.get("transform_type") in {"ft5", "prune50"}
        ),
        key=lambda item: item["model_id"],
    )
    summary_path = ROOT / "reports/c_fast_run.json"
    write_json(
        summary_path,
        {
            "date": utc_now(),
            "device": str(device),
            "torch_version": torch.__version__,
            "parents": args.model_id,
            "transforms": args.transforms,
            "invocation_results": run_summary,
            "registered_results": registered,
            "registered_count": len(registered),
        },
    )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
