"""Idempotently register the curated B2 hyperparameter-diverse checkpoint set."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_zoo.core import ROOT, models, save_models, write_json


CLASS_ORDER = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SOURCE_ID = "sharpness_cifar10"
REPOSITORY = "https://github.com/tml-epfl/sharpness-vs-generalization"
REVISION = "6d73be94eb88dae6d3096647bb24b92244fae18f"
SOURCE_FOLDER = "https://drive.google.com/drive/folders/1dwfb2Iqw6BTMi57SeG44aDebuu-TBzo5"


def field(name, text, cast=str):
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", text)
    if not match:
        raise ValueError(f"Missing {name} in {text}")
    return cast(match.group(1))


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError(value)
    return value == "True"


def main():
    selected = json.loads((ROOT / "configs/b2_source_selection.json").read_text(encoding="utf-8"))
    if len(selected) != 30 or len({item["id"] for item in selected}) != 30:
        raise ValueError("B2 selection must contain 30 unique Drive files")
    current = models()
    existing = [row for row in current if row["group"] != "B2"]
    previous_b2 = {row["model_id"]: row for row in current if row["group"] == "B2"}
    b2 = []
    for index, item in enumerate(selected, 1):
        name = item["name"]
        model_id = f"B2_{index:03d}"
        lr = field("lr_max", name, float)
        sam = field("sam_rho", name, float)
        augm = field("augm", name, boolean)
        randaug = field("randaug", name, boolean)
        seed = field("seed", name, int)
        recipe = {
            "optimizer": "sgd",
            "max_learning_rate": lr,
            "momentum": None,
            "weight_decay": field("l2_reg", name, float),
            "sam_rho": sam,
            "epochs": field("epochs", name, int),
            "batch_size": field("batch_size", name, int),
            "lr_schedule": field("lr_schedule", name),
            "standard_augmentation": augm,
            "randaugment": randaug,
            "model_width": field("model_width", name, int),
            "training_fraction": field("frac_train", name, float),
            "label_noise_probability": field("p_label_noise", name, float),
            "seed": seed,
            "checkpoint_state_key": "last",
        }
        augmentations = []
        if randaug:
            augmentations.append("RandAugment(2,14)")
        if augm:
            augmentations.extend(["RandomCrop(32,padding=4)", "RandomHorizontalFlip"])
        registered = {
            "model_id": model_id,
            "group": "B2",
            "lineage_id": f"original_{60 + index:03d}",
            "parent_id": None,
            "architecture": "PreActResNet18-CIFAR-3x3",
            "architecture_family": "ResNet",
            "topology_id": "tml-epfl:PreActResNet18-CIFAR-width64",
            "source_id": SOURCE_ID,
            "source_repository": REPOSITORY,
            "source_url": SOURCE_FOLDER,
            "source_revision": REVISION,
            "checkpoint_url": f"https://drive.usercontent.google.com/download?id={item['id']}&export=download&confirm=t",
            "original_filename": name,
            "local_checkpoint_path": f"checkpoints/B2/{model_id}_preact_resnet18_lr{lr:g}_sam{sam:g}_aug{int(augm)}.pth",
            "sha256": None,
            "state_dict_sha256": None,
            "parameter_sha256": None,
            "framework": "PyTorch",
            "framework_version": None,
            "checkpoint_format": "PyTorch ZIP container with last/best/SWA state_dicts",
            "num_parameters": None,
            "input_size": [3, 32, 32],
            "normalization_mean": [0.4914, 0.4822, 0.4465],
            "normalization_std": [0.2023, 0.1994, 0.2010],
            "cifar10_test_accuracy_reported": None,
            "cifar10_test_accuracy_verified": None,
            "training_seed": seed,
            "optimizer": "sgd",
            "learning_rate": lr,
            "weight_decay": recipe["weight_decay"],
            "augmentation": augmentations,
            "training_recipe": recipe,
            "notes": "Published final epoch-200 training run. Loader uses the source evaluation protocol's 'last' state_dict; best and SWA variants remain preserved in the original file. All B2 runs use seed=0, so B2 controls hyperparameter changes rather than initialization changes.",
            "download_date": None,
            "status": "planned",
            "dataset": "CIFAR-10",
            "class_order": CLASS_ORDER,
            "loader_spec": {"adapter": "sharpness_preact_resnet18", "state_key": "last"},
            "download_spec": {"kind": "direct"},
            "training_run_id": item["id"],
            "checkpoint_iteration": 200,
            "reported_accuracy_iteration": None,
            "hyperparameter_signature": f"lr_max={lr:g}|sam_rho={sam:g}|augm={augm}|randaug={randaug}",
            "independence_evidence": "Unique public Drive file and training run; identical seed=0 but distinct hyperparameter signature. Filename records the complete varied recipe.",
            "preprocessing_evidence": f"{REPOSITORY}/blob/{REVISION}/data.py#L46-L69",
            "normalization_location": "wrapper",
            "input_range": [0.0, 1.0],
            "input_color_order": "RGB",
            "input_layout": "NCHW",
            "output_type": "logits",
            "num_classes": 10,
        }
        old = previous_b2.get(model_id, {})
        runtime_fields = {
            "sha256", "state_dict_sha256", "parameter_sha256", "framework_version",
            "num_parameters", "num_trainable_parameters", "num_nontrainable_parameters",
            "cifar10_test_accuracy_verified", "cifar10_smoke_accuracy", "download_date",
            "status", "checkpoint_bytes", "verification", "full_verification",
            "reported_verified_gap_pp", "accuracy_warning", "model_definition_packages",
        }
        registered.update({key: old[key] for key in runtime_fields if key in old})
        b2.append(registered)
    if len({row["hyperparameter_signature"] for row in b2}) != 30:
        raise ValueError("B2 hyperparameter signatures are not unique")
    save_models(existing + b2)

    sources = json.loads((ROOT / "metadata/sources.json").read_text(encoding="utf-8"))
    sources = [source for source in sources if source["source_id"] != SOURCE_ID]
    sources.append({
        "source_id": SOURCE_ID,
        "repository": REPOSITORY,
        "revision": REVISION,
        "paper": "A Modern Look at the Relationship between Sharpness and Generalization (ICML 2023)",
        "paper_url": "https://proceedings.mlr.press/v202/andriushchenko23a.html",
        "checkpoint_folder": SOURCE_FOLDER,
        "strategy": "Thirty individually addressable Google Drive files; no folder/archive bulk download",
        "license": "Checkpoint license not stated; repository has no detected license file",
        "selected_bytes_expected": sum(
            next((value for value in item.get("large_ints", []) if 100_000_000 <= value <= 500_000_000), 0)
            for item in selected
        ),
    })
    write_json(ROOT / "metadata/sources.json", sources)
    selection_path = ROOT / "configs/selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection.update({
        "B2_target": 30,
        "B2": [row["model_id"] for row in b2],
        "B2_selection_rule": "Five learning-rate representatives in each SAM-rho x augmentation stratum; unique hyperparameter signature; no accuracy selection",
        "B2_checkpoint_state_key": "last",
        "independent_lineages": len(existing) + len(b2),
    })
    write_json(selection_path, selection)
    print(f"registered {len(b2)} B2 models")


if __name__ == "__main__":
    main()
