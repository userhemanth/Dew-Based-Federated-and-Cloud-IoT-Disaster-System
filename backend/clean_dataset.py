import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm

DATA_DIR = "data"
TARGET_COUNT = 1000

CLASSES = [
    "Drought", "Earthquake",
    "Land_Slide", "Water_Disaster", "Wild_Fire"
]

NEGATIVE_PROMPTS = [
    "a photo of a normal day without damage",
    "a cartoon, drawing, or illustration",
    "a portrait of a person's face",
    "a screenshot of a document, news article, or text"
]

# Build the complete list of text prompts (all disaster classes + negative prompts)
ALL_CLASS_PROMPTS = [f"a photo of a {c.replace('_', ' ').lower()} disaster" for c in CLASSES]
ALL_TEXTS = ALL_CLASS_PROMPTS + NEGATIVE_PROMPTS

def main():
    print("=" * 65)
    print("  Dew-FDL | Strict Multi-Class AI Cleaner (CLIP)")
    print("=" * 65)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading CLIP model on {device}...")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        print(f"[ERROR] Failed to load CLIP model: {e}")
        return

    total_deleted = 0
    total_kept = 0

    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Directory not found: {cls_dir}")
            continue

        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            continue

        print(f"\n[CLEANING] Class: {cls} | Found {len(images)} images")
        
        # Determine the index of the expected class in our texts array
        expected_class_idx = CLASSES.index(cls)
        
        image_scores = []
        
        for img_name in tqdm(images, desc=f"Scoring {cls}"):
            img_path = os.path.join(cls_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = processor(text=ALL_TEXTS, images=img, return_tensors="pt", padding=True).to(device)
                outputs = model(**inputs)
                
                # Image-text similarity scores
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()[0]
                
                # Find the index of the highest scoring prompt
                best_idx = probs.argmax()
                
                # Strict Rule: Is the highest scoring prompt EXACTLY the folder it's in?
                # Also ensure the confidence is at least 30%
                expected_prob = probs[expected_class_idx]
                is_accurate = (best_idx == expected_class_idx) and (expected_prob >= 0.3)
                
                if not is_accurate:
                    # It's cross-contaminated or inaccurate. Delete it immediately.
                    img.close()
                    os.remove(img_path)
                    total_deleted += 1
                else:
                    image_scores.append({"path": img_path, "score": expected_prob})
                    img.close()
                    
            except Exception as e:
                # Corrupted image, delete it
                try:
                    os.remove(img_path)
                except: pass
                total_deleted += 1

        # Now sort the highly accurate images by score and strictly keep top TARGET_COUNT
        image_scores.sort(key=lambda x: x["score"], reverse=True)
        
        kept = image_scores[:TARGET_COUNT]
        discarded = image_scores[TARGET_COUNT:]
        
        for d in discarded:
            try:
                os.remove(d["path"])
                total_deleted += 1
            except: pass
            
        final_count = len(kept)
        total_kept += final_count
        print(f"  -> Kept: {final_count} | Deleted/Discarded for {cls}: {len(images) - final_count}")

    print("\n" + "=" * 65)
    print(f"[DONE] Strict Multi-Class Cleanup Complete.")
    print(f"       Total Highly Accurate Images Kept: {total_kept}")
    print(f"       Total Inaccurate/Contaminated Deleted: {total_deleted}")
    print("=" * 65)

if __name__ == "__main__":
    main()
