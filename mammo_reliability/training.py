"""Train on fit patients; select checkpoints using tune patients only."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .audit import BREAST_KEY, environment, file_sha256, require, write_json
from .model import BreastDataset, MammographyModel, collate_breasts, seed_everything


def source_fingerprint() -> str:
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def loader(dataset, config: dict, training: bool, seed: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=training,
                      num_workers=config["num_workers"], collate_fn=collate_breasts,
                      pin_memory=torch.cuda.is_available(), persistent_workers=False,
                      generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def predict(model: MammographyModel, dataset: BreastDataset, config: dict, device: torch.device) -> pd.DataFrame:
    model.eval()
    records = []
    for batch in loader(dataset, config, False):
        images, slots = batch["images"].to(device), batch["slots"].to(device)
        _, view_logits = model(images, slots, len(batch["labels"]))
        values = view_logits.cpu().numpy()
        require(np.isfinite(values).all(), "Model emitted nonfinite predictions.")
        records.extend({"patient_id": p, "laterality": l, "z_cc": float(z[0]), "z_mlo": float(z[1])}
                       for p, l, z in zip(batch["patient_id"], batch["laterality"], values))
    return pd.DataFrame(records)


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train_member(manifest: pd.DataFrame, breasts: pd.DataFrame, roles: pd.DataFrame, config: dict,
                 seed: int, output: str | Path, metadata_sha256: str) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    require(roles["patient_id"].is_unique, "Patients occur in more than one role.")
    labeled = breasts.merge(roles[["patient_id", "role"]], on="patient_id", validate="many_to_one")
    fit, tune = labeled.loc[labeled["role"] == "fit"], labeled.loc[labeled["role"] == "tune"]
    require(len(fit) > 0 and len(tune) > 0, "Empty fitting or tuning partition.")
    require(set(fit["patient_id"]).isdisjoint(set(tune["patient_id"])), "Patient leakage between fitting and tuning.")
    require(set(fit["cancer"].unique()) == {0, 1} and set(tune["cancer"].unique()) == {0, 1}, "Both partitions need both classes.")
    role_hash = hashlib.sha256(roles.sort_values("patient_id").to_csv(index=False).encode()).hexdigest()
    signature = {"config": config, "seed": int(seed), "metadata_sha256": metadata_sha256,
                 "roles_sha256": role_hash, "source_sha256": source_fingerprint(),
                 "manifest_sha256": hashlib.sha256(manifest.sort_values("image_id").to_csv(index=False).encode()).hexdigest()}
    signature_path = output / "signature.json"
    if signature_path.exists():
        require(json.loads(signature_path.read_text()) == signature, "Run configuration/code changed; use a new run directory.")
    else:
        write_json(signature_path, signature)
    if (output / "completed.json").exists():
        finished = json.loads((output / "completed.json").read_text())
        require(file_sha256(output / "best.pt") == finished["checkpoint_sha256"], "Completed model checkpoint changed.")
        return finished

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get("require_gpu", True):
        require(device.type == "cuda", "GPU unavailable. CPU is supported for software tests; enable a Kaggle GPU for research training.")
    model = MammographyModel(pretrained=config["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    amp_enabled = bool(config.get("mixed_precision", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    n_pos = int(fit["cancer"].sum())
    positive_weight = (len(fit) - n_pos) / n_pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    train_data = BreastDataset(manifest, fit, True, config["height"], config["width"])
    tune_data = BreastDataset(manifest, tune, False, config["height"], config["width"])
    start_epoch, best_ap, stale, history = 0, -1.0, 0, []
    last = output / "last.pt"
    if last.exists():
        # Only load checkpoints created by this run in a trusted private directory.
        state = torch.load(last, map_location=device, weights_only=False)
        require(state["signature"] == signature, "Resume checkpoint signature mismatch.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_ap, stale, history = state["next_epoch"], state["best_ap"], state["stale"], state["history"]

    require(config["batch_size"] >= 1 and config["accumulation_steps"] >= 1, "Invalid batch settings.")
    for epoch in range(start_epoch, config["epochs"]):
        if stale >= config["patience"]:
            break
        epoch_seed = seed + 1000 * epoch
        seed_everything(epoch_seed)
        model.train()
        batches = loader(train_data, config, True, epoch_seed)
        optimizer.zero_grad(set_to_none=True)
        total_loss, n_seen, start_time = 0.0, 0, time.monotonic()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        accumulation = config["accumulation_steps"]
        for i, batch in enumerate(batches):
            images = batch["images"].to(device, non_blocking=True)
            slots, labels = batch["slots"].to(device), batch["labels"].to(device)
            group_start = (i // accumulation) * accumulation * config["batch_size"]
            group_breasts = min(accumulation * config["batch_size"], len(train_data) - group_start)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits, _ = model(images, slots, len(labels))
                loss = loss_fn(logits, labels)
            require(torch.isfinite(loss).item(), "Nonfinite training loss.")
            scaler.scale(loss * len(labels) / group_breasts).backward()
            if (i + 1) % accumulation == 0 or (i + 1) == len(batches):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * len(labels)
            n_seen += len(labels)
            if (i + 1) % 256 == 0:
                print(json.dumps({'epoch': epoch + 1, 'fit_breasts_processed': n_seen,
                                  'fit_breasts_total': len(train_data)}), flush=True)
        training_seconds = time.monotonic() - start_time
        validation_start = time.monotonic()
        predictions = predict(model, tune_data, config, device).merge(tune[BREAST_KEY + ["cancer"]], on=BREAST_KEY, validate="one_to_one")
        logits = (predictions["z_cc"].to_numpy() + predictions["z_mlo"].to_numpy()) / 2
        from scipy.special import expit
        probabilities = expit(logits)
        ap = float(average_precision_score(predictions["cancer"], probabilities))
        record = {"epoch": epoch + 1, "weighted_training_loss": total_loss / n_seen,
                  "tune_average_precision": ap, "tune_roc_auc": float(roc_auc_score(predictions["cancer"], probabilities)),
                  "seconds": time.monotonic() - start_time, "learning_rate": optimizer.param_groups[0]["lr"]}
        record.update(training_seconds=training_seconds, validation_seconds=time.monotonic() - validation_start,
                      fit_breasts_per_second=n_seen / training_seconds,
                      peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else None)
        history.append(record)
        print(json.dumps(record), flush=True)
        if ap > best_ap:
            best_ap, stale = ap, 0
            save_checkpoint(output / "best.pt", {"state_dict": model.state_dict(), "epoch": epoch + 1, "seed": seed})
        else:
            stale += 1
        scheduler.step(ap)
        pd.DataFrame(history).to_csv(output / "learning_curve.csv", index=False)
        save_checkpoint(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                              "signature": signature, "next_epoch": epoch + 1, "best_ap": best_ap,
                              "stale": stale, "history": history})
    require((output / "best.pt").exists(), "No model checkpoint was produced.")
    result = {"status": "training_completed", "best_tune_average_precision": best_ap,
              "checkpoint_sha256": file_sha256(output / "best.pt"), "positive_class_weight": positive_weight,
              "fit_breasts": len(fit), "tune_breasts": len(tune), "device": str(device), "environment": environment()}
    write_json(output / "completed.json", result)
    return result


def predict_members(manifest: pd.DataFrame, breasts: pd.DataFrame, config: dict, seeds: list[int],
                    fold_dir: str | Path) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BreastDataset(manifest, breasts, False, config["height"], config["width"])
    frames = []
    for seed in seeds:
        folder = Path(fold_dir) / f"seed_{seed}"
        completed = json.loads((folder / "completed.json").read_text())
        require(file_sha256(folder / "best.pt") == completed["checkpoint_sha256"], "Prediction checkpoint changed.")
        signature = json.loads((folder / "signature.json").read_text())
        require(signature["source_sha256"] == source_fingerprint() and signature["config"] == config,
                "Prediction code/config differs from training.")
        require(signature["manifest_sha256"] == hashlib.sha256(manifest.sort_values("image_id").to_csv(index=False).encode()).hexdigest(),
                "The image manifest differs from the one used for training.")
        model = MammographyModel(pretrained=False).to(device)
        model.load_state_dict(torch.load(folder / "best.pt", map_location=device, weights_only=True)["state_dict"])
        member = predict(model, dataset, config, device)
        member["seed"] = seed
        frames.append(member)
        del model
    return pd.concat(frames, ignore_index=True)
