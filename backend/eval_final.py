# backend/eval_final.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))  # data_utils + train_model live in backend/

import torch
from data_utils import get_dataloader
from train_model import DisasterEnsemble

def eval_best_model():
    print("Evaluating final model...")
    device = torch.device("cpu")
    model = DisasterEnsemble(num_classes=6)
    
    state_dict = torch.load("models/global_model.pth", map_location=device, weights_only=True)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    class_names = ['Earthquake', 'Human_Damage', 'Land_Slide', 'Non_Damage', 'Water_Disaster', 'Wild_Fire']
    
    _, test_loader = get_dataloader(data_dir="data", batch_size=32, val_split=0.2)
    
    correct = 0
    total = 0
    cor_c = {c: 0 for c in class_names}
    tot_c = {c: 0 for c in class_names}
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for l, p in zip(labels, predicted):
                c = class_names[l.item()]
                tot_c[c] += 1
                cor_c[c] += int(p.item() == l.item())
                
    print(f"\nFinal Test Accuracy: {100.0 * correct / total:.2f}%")
    print("\n--- Per-Class Accuracy (Final) ---")
    for c in class_names:
        acc = 100.0 * cor_c[c] / max(tot_c[c], 1)
        print(f"  {c:<22} {acc:>6.1f}%  {'#' * int(acc // 5)}")
    print("="*40)
        
if __name__ == "__main__":
    eval_best_model()
