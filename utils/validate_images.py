from PIL import Image
from pathlib import Path
from tqdm import tqdm

data_dir = Path("data")
bad_files = []

for img_path in tqdm(list(data_dir.rglob("*.*"))):
    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        try:
            Image.open(img_path).verify()
        except Exception:
            bad_files.append(img_path)

print(f"Found {len(bad_files)} corrupted files:")
for f in bad_files:
    print(f)
