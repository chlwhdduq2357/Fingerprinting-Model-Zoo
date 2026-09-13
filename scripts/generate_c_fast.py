"""Generate C-stage descendants: FT5, PRUNE50, PTQ_INT8, and PRUNE20.

This script is designed for an interruptible Google Colab workflow.  It saves a
completed descendant after every parent and safely skips matching completed
outputs on the next invocation.  Adversarial fine-tuning is intentionally
outside this practical C-stage design.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_zoo.core import ROOT, models, save_models, sha256, tensor_hash, write_json
from model_zoo.loaders.unified import Classifier, build_network, load_model


PARENTS = ["A001", "A002", "A003", "A006", "A008", "A010", "A013", "A015", "A027", "A029"]
TRANSFORM_SLOT = {"ft5": 1, "prune50": 2, "ptq_int8": 3, "prune20": 4}
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


def atomic_jit_save(module: torch.jit.ScriptModule, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.jit.save(module, str(temporary))
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


def make_calibration_loader(
    data_root: Path, batch_size: int, workers: int, seed: int = 32000, samples: int = 1024
):
    dataset = CIFAR10(root=str(data_root), train=True, download=True, transform=ToTensor())
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:samples].tolist()
    index_bytes = json.dumps(indices, separators=(",", ":")).encode("utf-8")
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    return loader, hashlib.sha256(index_bytes).hexdigest()


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


def make_prune20(parent_id: str) -> tuple[nn.Module, dict]:
    model = load_model(parent_id, device="cpu", preprocessing="native")
    return prune_network(model.network, amount=0.2)


def choose_quant_backend() -> str:
    supported = list(torch.backends.quantized.supported_engines)
    for candidate in ("x86", "fbgemm", "onednn"):
        if candidate in supported:
            return candidate
    raise RuntimeError(f"No supported x86/fbgemm/onednn static quantization backend: {supported}")


def quantize_network_static(
    network: nn.Module,
    calibration_loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    backend: str | None = None,
) -> tuple[torch.jit.ScriptModule, dict]:
    """Calibrate and convert both supported weights and activations to static INT8."""
    from torch.ao.quantization import QConfigMapping, get_default_qconfig
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    supported = list(torch.backends.quantized.supported_engines)
    backend = choose_quant_backend() if backend is None else backend
    if backend not in supported:
        raise RuntimeError(f"Unsupported static quantization backend {backend}: {supported}")
    torch.backends.quantized.engine = backend
    network = network.cpu().eval()
    mean, std = mean.cpu(), std.cpu()

    iterator = iter(calibration_loader)
    first_images, _ = next(iterator)
    first_normalized = (first_images - mean) / std
    example = first_normalized[:1].contiguous()
    qconfig_mapping = QConfigMapping().set_global(get_default_qconfig(backend))
    prepared = prepare_fx(network, qconfig_mapping, (example,))

    calibrated = 0
    with torch.inference_mode():
        prepared(first_normalized)
        calibrated += first_images.shape[0]
        for images, _ in iterator:
            prepared((images - mean) / std)
            calibrated += images.shape[0]
        converted = convert_fx(prepared).eval()
        output = converted(example)

    quantized_modules = []
    weight_qschemes = set()
    for name, module in converted.named_modules():
        module_path = module.__class__.__module__
        if ".quantized" not in module_path:
            continue
        if hasattr(module, "weight") and callable(module.weight):
            try:
                weight = module.weight()
            except (RuntimeError, TypeError):
                continue
            if isinstance(weight, torch.Tensor) and weight.is_quantized:
                quantized_modules.append(name or "<root>")
                weight_qschemes.add(str(weight.qscheme()))
    activation_quantizers = sum(
        node.op == "call_function" and "quantize_per_tensor" in str(node.target)
        for node in converted.graph.nodes
    )
    if not quantized_modules or activation_quantizers < 1:
        raise RuntimeError("Conversion did not create both quantized weights and activation boundaries")
    if output.shape != (1, 10) or output.dtype != torch.float32 or not torch.isfinite(output).all():
        raise RuntimeError("Converted INT8 model violated the float [N,10] output contract")

    # A state_dict cannot reconstruct an FX-converted graph.  Store the locally
    # generated graph as a SHA256-verified TorchScript archive instead.
    traced = torch.jit.trace(converted, example, check_trace=False).eval()
    scripted = torch.jit.freeze(traced)
    with torch.inference_mode():
        batch_output = scripted(first_normalized[: min(3, len(first_normalized))])
    if batch_output.ndim != 2 or batch_output.shape[1] != 10 or not torch.isfinite(batch_output).all():
        raise RuntimeError("TorchScript INT8 round-trip precheck failed")
    return scripted, {
        "backend": backend,
        "calibration_samples": calibrated,
        "weight_dtype": "torch.qint8",
        "activation_dtype": "torch.quint8",
        "weight_qschemes": sorted(weight_qschemes),
        "quantized_weight_module_count": len(quantized_modules),
        "quantized_weight_modules": quantized_modules,
        "activation_quantizer_count": activation_quantizers,
        "float_input_output": True,
    }


def make_ptq_int8(parent_id: str, calibration_loader: DataLoader) -> tuple[torch.jit.ScriptModule, dict]:
    model = load_model(parent_id, device="cpu", preprocessing="native")
    return quantize_network_static(
        model.network,
        calibration_loader,
        model.mean,
        model.std,
    )


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


def verify_jit_round_trip(path: Path, parent_row: dict) -> torch.jit.ScriptModule:
    network = torch.jit.load(str(path), map_location="cpu").eval()
    wrapper = Classifier(network, parent_row, preprocessing="native").eval()
    with torch.inference_mode():
        logits = wrapper(torch.rand(3, 3, 32, 32))
    if logits.shape != (3, 10) or logits.dtype != torch.float32 or not torch.isfinite(logits).all():
        raise RuntimeError("Saved TorchScript INT8 checkpoint failed round-trip verification")
    return network


def quantized_descendant_row(
    parent: dict,
    model_id: str,
    checkpoint_path: Path,
    config: dict,
    metrics: dict,
    evaluation: dict,
) -> dict:
    row = copy.deepcopy(parent)
    relative_path = checkpoint_path.relative_to(ROOT).as_posix()
    parent_loader_spec = copy.deepcopy(parent["loader_spec"])
    row.update(
        model_id=model_id,
        group="C",
        parent_id=parent["model_id"],
        lineage_id=parent["lineage_id"],
        transform_type="ptq_int8",
        lineage_mechanism="parameter_inheritance",
        parent_checkpoint_sha256=parent["sha256"],
        derivation_reference=PAPER_URL,
        derivation_date=utc_now(),
        original_filename=checkpoint_path.name,
        local_checkpoint_path=relative_path,
        checkpoint_url=None,
        download_spec={"kind": "local_derivative"},
        loader_spec={"adapter": "local_torchscript_int8", "parent": parent_loader_spec},
        sha256=sha256(checkpoint_path),
        state_dict_sha256=None,
        parameter_sha256=None,
        quantized_graph_sha256=sha256(checkpoint_path),
        checkpoint_bytes=checkpoint_path.stat().st_size,
        checkpoint_format="torchscript_int8",
        framework="PyTorch",
        framework_version=torch.__version__,
        num_parameters=parent["num_parameters"],
        num_trainable_parameters=0,
        num_nontrainable_parameters=parent["num_parameters"],
        num_nonzero_parameters=None,
        parameter_sparsity=None,
        cifar10_test_accuracy_reported=None,
        cifar10_test_accuracy_verified=evaluation["accuracy_percent"],
        cifar10_smoke_accuracy=None,
        status="verified",
        transform_config=config,
        transform_metrics=metrics,
        training_seed=None,
        optimizer=None,
        learning_rate=None,
        weight_decay=None,
        augmentation=None,
        training_recipe="Static post-training weight-and-activation INT8 quantization",
        notes=f"Local same-lineage PTQ descendant of {parent['model_id']}.",
        download_date=None,
        verification={
            "model_id": model_id,
            "passed": True,
            "samples": evaluation["samples"],
            "correct": evaluation["correct"],
            "accuracy_percent": evaluation["accuracy_percent"],
            "full_test_set": evaluation["samples"] == 10000,
            "preprocessing": "native",
            "device": "cpu",
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
    parser.add_argument(
        "--transforms",
        nargs="+",
        choices=["ft5", "prune50", "ptq_int8", "prune20"],
        default=["ft5", "prune50", "ptq_int8", "prune20"],
    )
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
    calibration_loader = calibration_indices_sha256 = None
    if "ptq_int8" in args.transforms:
        calibration_loader, calibration_indices_sha256 = make_calibration_loader(
            args.data_root, min(args.batch_size, 128), args.workers, seed=32000, samples=1024
        )
    quant_backend = choose_quant_backend() if "ptq_int8" in args.transforms else None

    run_summary = []
    for parent_id in args.model_id:
        parent = by_id[parent_id]
        for transform in args.transforms:
            model_id = c_model_id(parent_id, transform)
            suffix = ".pt" if transform == "ptq_int8" else ".pth"
            checkpoint_path = ROOT / "checkpoints/C" / f"{model_id}_{parent_id}_{transform}{suffix}"
            log_path = ROOT / "runs/C" / f"{model_id}.jsonl"
            parent_number = int(parent_id[1:])
            if transform == "ft5":
                seed = 31000 + parent_number
                expected_config = {
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
            elif transform in {"prune50", "prune20"}:
                seed = (33000 if transform == "prune50" else 34000) + parent_number
                expected_config = {
                    "method": "global_unstructured_L1",
                    "amount": 0.5 if transform == "prune50" else 0.2,
                    "eligible_modules": ["Conv2d.weight", "Linear.weight"],
                    "recovery_fine_tuning": False,
                    "seed": seed,
                }
            else:
                seed = 32000
                expected_config = {
                    "method": "fx_graph_static_ptq",
                    "weight_dtype": "torch.qint8",
                    "activation_dtype": "torch.quint8",
                    "calibration_samples": 1024,
                    "calibration_seed": seed,
                    "calibration_indices_sha256": calibration_indices_sha256,
                    "backend": quant_backend,
                    "float_input_output": True,
                }
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
            elif transform == "prune50":
                network, details = make_prune50(parent_id)
                config = expected_config
                metrics = details
            elif transform == "prune20":
                network, details = make_prune20(parent_id)
                config = expected_config
                metrics = details
            else:
                network, details = make_ptq_int8(parent_id, calibration_loader)
                config = expected_config
                metrics = details

            if transform == "ptq_int8":
                atomic_jit_save(network, checkpoint_path)
                network = verify_jit_round_trip(checkpoint_path, parent)
                wrapper = Classifier(network, parent, preprocessing="native").eval()
                evaluation = evaluate(wrapper, test_loader, torch.device("cpu"))
                new_row = quantized_descendant_row(
                    parent, model_id, checkpoint_path, config, metrics, evaluation
                )
                if new_row["sha256"] == parent["sha256"]:
                    raise RuntimeError(f"{model_id} is identical to its parent")
            else:
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
            if device.type == "cuda" and transform != "ptq_int8":
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
            if row.get("group") == "C"
            and row.get("transform_type") in {"ft5", "prune50", "ptq_int8", "prune20"}
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
