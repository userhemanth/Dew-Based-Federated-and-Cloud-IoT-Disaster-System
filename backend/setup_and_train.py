# setup_and_train.py
"""
All-in-one script: Download disaster images from Bing via icrawler,
organise images into our 9-class folder structure, then train the
EfficientNet-B4 + ResNet50 Ensemble model to produce models/global_model.pth.

Datasets used:
  1. QCRI/MEDIC              - 71,198 images (earthquake, flood, fire, landslide, etc.)
  2. TheNetherWatcher/DisasterClassification - covers Drought, Urban_Fire, Infrastructure
  3. QCRI/CrisisMMD          - additional fire, infrastructure, flood images
  4. kevincluo/structure_wildfire_damage_classification - wildfire structural damage

Usage:
    python setup_and_train.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))  # train_model lives in backend/

import os, sys, time, csv, psutil
from pathlib import Path

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
DATA_DIR      = Path("data")
MODELS_DIR    = Path("models")
MODEL_SAVE    = MODELS_DIR / "global_model.pth"
METRICS_FILE  = DATA_DIR / "round_metrics.csv"

EPOCHS        = 12
BATCH_SIZE    = 16
IMG_SIZE      = 380    # EfficientNet-B4 optimal input size
LR_BACKBONE   = 1e-4
LR_HEAD       = 5e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 0        # safe on Windows
MAX_PER_CLASS = 2000     # set None to use all images

OUR_CLASSES = [
    "Drought", "Earthquake",
    "Land_Slide", "Water_Disaster", "Wild_Fire"
]

# -----------------------------------------------------------------------
# LABEL MAPPINGS per dataset
# -----------------------------------------------------------------------

# Dataset 1: QCRI/MEDIC
# disaster_types: earthquake(0), flood(1), hurricane(2), fire(3),
#                 landslide(4), not_disaster(5), other_disaster(6)
MEDIC_LABEL_MAP = {
    "earthquake":     "Earthquake",
    "flood":          "Water_Disaster",
    "hurricane":      "Water_Disaster",
    "fire":           "Wild_Fire",
    "landslide":      "Land_Slide",
    "not_disaster":   "Non_Damage",
    "other_disaster": "Human_Damage",
}

# Dataset 2: TheNetherWatcher/DisasterClassification
# Labels are string class names we can map directly
NETHER_LABEL_MAP = {
    # exact label names from the dataset card
    "Drought":              "Drought",
    "drought":              "Drought",
    "Earthquake":           "Earthquake",
    "earthquake":           "Earthquake",
    "Flood":                "Water_Disaster",
    "flood":                "Water_Disaster",
    "Wildfire":             "Wild_Fire",
    "wildfire":             "Wild_Fire",
    "Wild_Fire":            "Wild_Fire",
    "Fire":                 "Urban_Fire",
    "Urban_Fire":           "Urban_Fire",
    "urban_fire":           "Urban_Fire",
    "Infrastructure":       "Infrastructure",
    "infrastructure":       "Infrastructure",
    "Infrastructure_Damage":"Infrastructure",
    "Land_Slide":           "Land_Slide",
    "Landslide":            "Land_Slide",
    "landslide":            "Land_Slide",
    "Human_Damage":         "Human_Damage",
    "Non_Damage":           "Non_Damage",
    "non_damage":           "Non_Damage",
    "Non_disaster":         "Non_Damage",
}

# Dataset 3: QCRI/CrisisMMD
# label column is 'label' with values like 'informative' / 'not_informative'
# and 'event_type' with values: earthquake, flood, hurricane, wildfire, etc.
CRISISMMD_EVENT_MAP = {
    "earthquake":    "Earthquake",
    "flood":         "Water_Disaster",
    "hurricane":     "Water_Disaster",
    "wildfire":      "Wild_Fire",
    "fire":          "Urban_Fire",
    "cyclone":       "Water_Disaster",
    "typhoon":       "Water_Disaster",
    "landslide":     "Land_Slide",
    "infrastructure":"Infrastructure",
}

# Dataset 4: kevincluo/structure_wildfire_damage_classification
# label column: 'label' -> 'no_damage', 'minor_damage', 'major_damage', 'destroyed'
WILDFIRE_STRUCT_MAP = {
    "no_damage":    "Non_Damage",
    "minor_damage": "Infrastructure",
    "major_damage": "Urban_Fire",
    "destroyed":    "Urban_Fire",
}


# -----------------------------------------------------------------------
# 1. DEPENDENCY CHECK
# -----------------------------------------------------------------------
def check_dependencies():
    missing = []
    for pkg in ["torch", "torchvision", "datasets", "PIL", "tqdm"]:
        try:
            __import__(pkg if pkg != "PIL" else "PIL.Image")
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing: {', '.join(missing)}")
        print("Run:  pip install torch torchvision datasets Pillow tqdm")
        sys.exit(1)
    print("[OK] All dependencies present")


# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------
def count_existing():
    return sum(
        len(list((DATA_DIR / c).glob("*.jpg"))) +
        len(list((DATA_DIR / c).glob("*.png")))
        for c in OUR_CLASSES if (DATA_DIR / c).exists()
    )

def count_per_class():
    counts = {}
    for c in OUR_CLASSES:
        p = DATA_DIR / c
        counts[c] = len(list(p.glob("*.jpg"))) + len(list(p.glob("*.png"))) if p.exists() else 0
    return counts

def save_image(pil_img, our_class, index, prefix):
    """Save a PIL image to the correct class folder."""
    import io as _io
    out = DATA_DIR / our_class / f"{prefix}_{index:07d}.jpg"
    if not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    pil_img.convert("RGB").save(out, "JPEG", quality=88)

def to_pil(raw):
    """Convert raw HF image field -> PIL Image."""
    from PIL import Image as PILImage
    import io as _io
    if isinstance(raw, PILImage.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict) and "bytes" in raw:
        return PILImage.open(_io.BytesIO(raw["bytes"])).convert("RGB")
    if isinstance(raw, bytes):
        return PILImage.open(_io.BytesIO(raw)).convert("RGB")
    return None

def print_class_summary(counts_before, counts_after, source_name):
    added = {c: counts_after[c] - counts_before.get(c, 0) for c in OUR_CLASSES}
    total_added = sum(added.values())
    print(f"   [{source_name}] +{total_added} images added:")
    for c in OUR_CLASSES:
        if added[c] > 0:
            print(f"     {c:<22} +{added[c]}")


# -----------------------------------------------------------------------
# 2a. DOWNLOAD: QCRI/MEDIC
# -----------------------------------------------------------------------
def download_medic(class_counts):
    from datasets import load_dataset
    from tqdm import tqdm

    print("\n[1/4] QCRI/MEDIC (~71K images: earthquake, flood, fire, landslide...)")
    try:
        from datasets import Image as HFImage
        dataset = load_dataset("QCRI/MEDIC", split="train+test+dev", verification_mode="no_checks")
        dataset = dataset.cast_column("image", HFImage(decode=False))
    except Exception as e:
        print(f"      [SKIP] MEDIC failed: {e}")
        return class_counts

    saved = skipped = 0
    iterator = iter(dataset)
    for i in tqdm(range(len(dataset)), desc="MEDIC"):
        try:
            sample = next(iterator)
        except Exception:
            skipped += 1
            continue
        label_int = sample.get("disaster_types")
        if label_int is None:
            skipped += 1; continue
        try:
            label_str = dataset.features["disaster_types"].int2str(label_int)
        except Exception:
            label_str = str(label_int)

        our_class = MEDIC_LABEL_MAP.get(label_str)
        if not our_class:
            skipped += 1; continue
        if MAX_PER_CLASS and class_counts.get(our_class, 0) >= MAX_PER_CLASS:
            continue

        img = to_pil(sample.get("image"))
        if img is None:
            skipped += 1; continue
        try:
            save_image(img, our_class, i, "medic")
            class_counts[our_class] = class_counts.get(our_class, 0) + 1
            saved += 1
        except Exception:
            skipped += 1
    print(f"      Saved: {saved} | Skipped: {skipped}")
    return class_counts


# -----------------------------------------------------------------------
# 2b. DOWNLOAD: TheNetherWatcher/DisasterClassification
#     Covers: Drought, Urban_Fire, Infrastructure + all others
# -----------------------------------------------------------------------
def download_nether(class_counts):
    from datasets import load_dataset
    from tqdm import tqdm

    print("\n[2/4] TheNetherWatcher/DisasterClassification (Drought, Urban_Fire, Infrastructure...)")
    try:
        from datasets import Image as HFImage
        dataset = load_dataset("TheNetherWatcher/DisasterClassification", split="train", verification_mode="no_checks")
        if "image" in dataset.column_names:
            dataset = dataset.cast_column("image", HFImage(decode=False))
    except Exception as e:
        # Try without split
        try:
            from datasets import Image as HFImage
            ds = load_dataset("TheNetherWatcher/DisasterClassification", verification_mode="no_checks")
            dataset = list(ds.values())[0]
            if "image" in dataset.column_names:
                dataset = dataset.cast_column("image", HFImage(decode=False))
        except Exception as e2:
            print(f"      [SKIP] DisasterClassification failed: {e2}")
            return class_counts

    saved = skipped = 0

    # Detect the label column
    sample0 = dataset[0] if hasattr(dataset, '__getitem__') else None
    label_col = None
    if sample0:
        for col in ["label", "class", "category", "disaster_type", "disaster", "labels"]:
            if col in sample0:
                label_col = col
                break

    # If dataset has feature ClassLabel, get names
    label_names = []
    try:
        feat = dataset.features.get(label_col) if label_col else None
        if feat and hasattr(feat, "names"):
            label_names = feat.names
    except Exception:
        pass

    iterator = iter(dataset)
    for i in tqdm(range(len(dataset)), desc="DisasterClass"):
        try:
            sample = next(iterator)
        except Exception:
            skipped += 1
            continue

        # Get label string
        raw_label = sample.get(label_col) if label_col else None
        if raw_label is None:
            # Try any key that looks like a label
            for k, v in sample.items():
                if k != "image" and isinstance(v, (str, int)):
                    raw_label = v
                    break

        if raw_label is None:
            skipped += 1; continue

        # Convert int label to string if we have names
        if isinstance(raw_label, int) and label_names:
            label_str = label_names[raw_label] if raw_label < len(label_names) else str(raw_label)
        else:
            label_str = str(raw_label)

        our_class = NETHER_LABEL_MAP.get(label_str) or NETHER_LABEL_MAP.get(label_str.lower())
        if not our_class:
            skipped += 1; continue
        if MAX_PER_CLASS and class_counts.get(our_class, 0) >= MAX_PER_CLASS:
            continue

        # Find image field
        img_raw = sample.get("image") or sample.get("img") or sample.get("pixel_values")
        img = to_pil(img_raw)
        if img is None:
            skipped += 1; continue
        try:
            save_image(img, our_class, i, "nether")
            class_counts[our_class] = class_counts.get(our_class, 0) + 1
            saved += 1
        except Exception:
            skipped += 1
    print(f"      Saved: {saved} | Skipped: {skipped}")
    return class_counts


# -----------------------------------------------------------------------
# 2c. DOWNLOAD: QCRI/CrisisMMD
#     Additional infrastructure, fire, flood images from social media
# -----------------------------------------------------------------------
def download_crisismmd(class_counts):
    from datasets import load_dataset
    from tqdm import tqdm

    print("\n[3/4] QCRI/CrisisMMD (infrastructure, fire, flood from crisis tweets...)")
    try:
        from datasets import Image as HFImage
        dataset = load_dataset("QCRI/CrisisMMD", split="train", verification_mode="no_checks")
        if "image" in dataset.column_names:
            dataset = dataset.cast_column("image", HFImage(decode=False))
    except Exception:
        try:
            from datasets import Image as HFImage
            ds = load_dataset("QCRI/CrisisMMD", verification_mode="no_checks")
            dataset = list(ds.values())[0]
            if "image" in dataset.column_names:
                dataset = dataset.cast_column("image", HFImage(decode=False))
        except Exception as e:
            print(f"      [SKIP] CrisisMMD failed: {e}")
            return class_counts

    saved = skipped = 0

    iterator = iter(dataset)
    for i in tqdm(range(len(dataset)), desc="CrisisMMD"):
        try:
            sample = next(iterator)
        except Exception:
            skipped += 1
            continue

        # Try event_name or event column
        event = (
            sample.get("event_name") or
            sample.get("event") or
            sample.get("disaster_type") or ""
        ).lower()

        our_class = None
        for key, cls in CRISISMMD_EVENT_MAP.items():
            if key in event:
                our_class = cls
                break

        # Also check humanitarian label for infrastructure
        if our_class is None:
            hum = str(sample.get("humanitarian", "")).lower()
            if "infrastructure" in hum or "utility" in hum:
                our_class = "Infrastructure"
            elif "not_humanitarian" in hum:
                our_class = "Non_Damage"

        if not our_class:
            skipped += 1; continue
        if MAX_PER_CLASS and class_counts.get(our_class, 0) >= MAX_PER_CLASS:
            continue

        img_raw = sample.get("image") or sample.get("img")
        img = to_pil(img_raw)
        if img is None:
            skipped += 1; continue
        try:
            save_image(img, our_class, i, "crisis")
            class_counts[our_class] = class_counts.get(our_class, 0) + 1
            saved += 1
        except Exception:
            skipped += 1
    print(f"      Saved: {saved} | Skipped: {skipped}")
    return class_counts


# -----------------------------------------------------------------------
# 2d. DOWNLOAD: kevincluo/structure_wildfire_damage_classification
#     Structural wildfire damage -> Urban_Fire & Infrastructure
# -----------------------------------------------------------------------
def download_wildfire_struct(class_counts):
    from datasets import load_dataset
    from tqdm import tqdm

    print("\n[4/4] kevincluo/structure_wildfire_damage_classification (Urban_Fire, Infrastructure...)")
    try:
        from datasets import Image as HFImage
        dataset = load_dataset("kevincluo/structure_wildfire_damage_classification", split="train", verification_mode="no_checks")
        if "image" in dataset.column_names:
            dataset = dataset.cast_column("image", HFImage(decode=False))
    except Exception:
        try:
            from datasets import Image as HFImage
            ds = load_dataset("kevincluo/structure_wildfire_damage_classification", verification_mode="no_checks")
            dataset = list(ds.values())[0]
            if "image" in dataset.column_names:
                dataset = dataset.cast_column("image", HFImage(decode=False))
        except Exception as e:
            print(f"      [SKIP] wildfire_struct failed: {e}")
            return class_counts

    saved = skipped = 0

    # Detect label column
    sample0 = dataset[0] if hasattr(dataset, '__getitem__') else None
    label_col = None
    label_names = []
    if sample0:
        for col in ["label", "damage_level", "class", "category"]:
            if col in sample0:
                label_col = col
                break
        if label_col:
            try:
                feat = dataset.features.get(label_col)
                if feat and hasattr(feat, "names"):
                    label_names = feat.names
            except Exception:
                pass

    iterator = iter(dataset)
    for i in tqdm(range(len(dataset)), desc="WildfireStruct"):
        try:
            sample = next(iterator)
        except Exception:
            skipped += 1
            continue

        raw_label = sample.get(label_col) if label_col else None
        if raw_label is None:
            skipped += 1; continue

        if isinstance(raw_label, int) and label_names:
            label_str = label_names[raw_label] if raw_label < len(label_names) else str(raw_label)
        else:
            label_str = str(raw_label).lower()

        our_class = WILDFIRE_STRUCT_MAP.get(label_str) or WILDFIRE_STRUCT_MAP.get(label_str.replace(" ", "_"))
        if not our_class:
            skipped += 1; continue
        if MAX_PER_CLASS and class_counts.get(our_class, 0) >= MAX_PER_CLASS:
            continue

        img_raw = sample.get("image") or sample.get("img")
        img = to_pil(img_raw)
        if img is None:
            skipped += 1; continue
        try:
            save_image(img, our_class, i, "wfstruct")
            class_counts[our_class] = class_counts.get(our_class, 0) + 1
            saved += 1
        except Exception:
            skipped += 1
    print(f"      Saved: {saved} | Skipped: {skipped}")
    return class_counts


# -----------------------------------------------------------------------
# 2. MAIN DOWNLOAD ORCHESTRATOR
# -----------------------------------------------------------------------
def download_all():
    print("\n[DOWNLOAD] Setting up all class folders...")
    MODELS_DIR.mkdir(exist_ok=True)
    for cls in OUR_CLASSES:
        (DATA_DIR / cls).mkdir(parents=True, exist_ok=True)

    existing = count_existing()
    print(f"[INFO] Currently have {existing} images total across all classes.")

    counts = count_per_class()

    # Run all 4 downloaders
    counts = download_medic(counts)
    counts = download_nether(counts)
    counts = download_crisismmd(counts)
    counts = download_wildfire_struct(counts)

    # Final summary
    print("\n[SUMMARY] Images per class after all downloads:")
    total = 0
    for cls in OUR_CLASSES:
        n = counts.get(cls, 0)
        total += n
        bar = "#" * min(n // 25, 40)
        status = "[OK]" if n >= 50 else "[LOW]" if n > 0 else "[EMPTY]"
        print(f"  {status} {cls:<22}  {n:>5}  {bar}")

    print(f"\n  Total images: {total}")

    empties = [c for c in OUR_CLASSES if counts.get(c, 0) == 0]
    if empties:
        print(f"\n[WARN] Still 0 images for: {', '.join(empties)}")
        print("  Add images manually from Kaggle for best results.")
    return counts


# -----------------------------------------------------------------------
# 3. BUILD DATALOADERS
# -----------------------------------------------------------------------
def build_loaders():
    import torch, random
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    from collections import defaultdict

    available = [
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name in OUR_CLASSES
        and (len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))) >= 5
    ]
    if not available:
        print("[ERROR] No images found. Run download step first.")
        sys.exit(1)
    print(f"\n[INFO] Training classes ({len(available)}): {sorted(available)}")

    tfm_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.RandomGrayscale(0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds_train = datasets.ImageFolder(str(DATA_DIR), transform=tfm_train)
    ds_eval  = datasets.ImageFolder(str(DATA_DIR), transform=tfm_eval)

    by_class = defaultdict(list)
    for idx, (_, lbl) in enumerate(ds_train.samples):
        by_class[lbl].append(idx)

    train_idx, test_idx = [], []
    for idxs in by_class.values():
        random.shuffle(idxs)
        n = int(0.8 * len(idxs))
        train_idx.extend(idxs[:n])
        test_idx.extend(idxs[n:])

    trl = DataLoader(Subset(ds_train, train_idx), batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    tel = DataLoader(Subset(ds_eval,  test_idx),  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"   Train: {len(train_idx)} | Test: {len(test_idx)} | Classes: {ds_train.classes}\n")
    return trl, tel, ds_train.classes


# -----------------------------------------------------------------------
# 4. TRAIN
# -----------------------------------------------------------------------
def train_model(trl, tel, class_names):
    import torch, torch.nn as nn, torch.optim as optim
    from train_model import DisasterEnsemble

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = f" | GPU: {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""
    print(f"[INFO] Device: {device}{gpu_info}")

    model = DisasterEnsemble(num_classes=len(class_names), pretrained=True).to(device)

    backbone_p = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n and "fc" not in n]
    head_p     = [p for n, p in model.named_parameters() if p.requires_grad and ("classifier" in n or "fc" in n)]
    groups = []
    if backbone_p: groups.append({"params": backbone_p, "lr": LR_BACKBONE})
    if head_p:     groups.append({"params": head_p,     "lr": LR_HEAD})
    if not groups: groups = [{"params": list(model.parameters()), "lr": LR_BACKBONE}]

    opt  = optim.AdamW(groups, weight_decay=1e-2)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # Compute class weights
    class_counts = []
    for c in class_names:
        c_count = len(list((DATA_DIR / c).glob("*.jpg"))) + len(list((DATA_DIR / c).glob("*.png")))
        class_counts.append(max(c_count, 1))
    total_samples = sum(class_counts)
    weights = [total_samples / (len(class_names) * count) for count in class_counts]
    class_weights = torch.FloatTensor(weights).to(device)

    crit = nn.CrossEntropyLoss(weight=class_weights)

    DATA_DIR.mkdir(exist_ok=True)
    with open(METRICS_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["round", "avg_accuracy", "num_clients"])

    best = 0.0
    print()
    for ep in range(1, EPOCHS + 1):
        # Battery-Awareness Check
        battery = psutil.sensors_battery()
        if battery is not None and not battery.power_plugged and battery.percent < 20:
            print("\n" + "!"*60)
            print("🔋 [CRITICAL] Battery level dropped below 20%.")
            print("   Suspending Federated Learning to reserve power for critical disaster alerts.")
            print("!"*60 + "\n")
            MODELS_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), str(MODEL_SAVE))
            break

        model.train()
        loss_sum = cor = tot = 0
        t0 = time.time()
        for imgs, lbls in trl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            out  = model(imgs)
            loss = crit(out, lbls)
            loss.backward(); opt.step()
            loss_sum += loss.item()
            _, pred = torch.max(out, 1)
            cor += (pred == lbls).sum().item(); tot += lbls.size(0)

        tr_acc  = 100.0 * cor / max(tot, 1)
        tr_loss = loss_sum / max(len(trl), 1)

        model.eval()
        tc = tt = 0
        with torch.no_grad():
            for imgs, lbls in tel:
                imgs, lbls = imgs.to(device), lbls.to(device)
                _, pred = torch.max(model(imgs), 1)
                tc += (pred == lbls).sum().item(); tt += lbls.size(0)

        te_acc = 100.0 * tc / max(tt, 1)
        sch.step()
        print(f"Epoch {ep:>2}/{EPOCHS}  Loss: {tr_loss:.4f}  Train: {tr_acc:.2f}%  Test: {te_acc:.2f}%  ({time.time()-t0:.0f}s)")

        with open(METRICS_FILE, "a", newline="") as f:
            csv.writer(f).writerow([ep, round(te_acc, 4), 1])

        if te_acc > best:
            best = te_acc
            MODELS_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), str(MODEL_SAVE))
            print(f"           [SAVED] Best model: {best:.2f}%")

    return best


# -----------------------------------------------------------------------
# 5. PER-CLASS ACCURACY REPORT
# -----------------------------------------------------------------------
def per_class_report(tel, class_names):
    import torch
    from train_model import DisasterEnsemble

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DisasterEnsemble(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(torch.load(str(MODEL_SAVE), map_location=device))
    model.to(device).eval()

    cor = {c: 0 for c in class_names}
    tot = {c: 0 for c in class_names}
    with torch.no_grad():
        for imgs, lbls in tel:
            imgs, lbls = imgs.to(device), lbls.to(device)
            _, pred = torch.max(model(imgs), 1)
            for p, l in zip(pred, lbls):
                c = class_names[l.item()]
                tot[c] += 1
                cor[c] += int(p.item() == l.item())

    print("\nPer-Class Accuracy on Test Set:")
    print(f"  {'Class':<22} {'Correct':>7} {'Total':>7} {'Acc':>7}")
    print(f"  {'-'*55}")
    for c in class_names:
        acc = 100.0 * cor[c] / max(tot[c], 1)
        print(f"  {c:<22} {acc:>6.1f}%  {'*' * int(acc // 5)}")


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("  Dew-FDL | Multi-Dataset Setup & Standalone Training")
    print("=" * 65)
    print("  Datasets: MEDIC + DisasterClassification + CrisisMMD + WildfireStruct")
    print("=" * 65)

    check_dependencies()
    download_all()
    # trl, tel, class_names = build_loaders()

    print("[INFO] Download completed. Training skipped as requested.")
    # best = train_model(trl, tel, class_names)

    # per_class_report(tel, class_names)

    print("\n" + "=" * 65)
    print(f"  Training skipped!")
    # print(f"  Best test accuracy : {best:.2f}%")
    print(f"  Model saved        : {MODEL_SAVE.resolve()}")
    print(f"  Metrics CSV        : {METRICS_FILE.resolve()}")
    print("=" * 65)
    print("\n  Next:  streamlit run streamlit_dashboard.py")
