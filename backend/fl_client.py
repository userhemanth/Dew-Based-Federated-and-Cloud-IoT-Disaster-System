import sys, os
import torch
import torch.nn as nn
import flwr as fl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from train_model import DisasterEnsemble

# Number of classes
CLASS_NAMES = [
    "Drought", "Earthquake",
    "Land_Slide", "Water_Disaster", "Wild_Fire", "Non_Damage"
]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "offline_images")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "global_model.pth")

def load_data():
    """Load local offline images for training."""
    if not os.path.exists(DATA_DIR) or not any(os.path.isdir(os.path.join(DATA_DIR, d)) for d in os.listdir(DATA_DIR)):
        return None, None

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    try:
        ds = datasets.ImageFolder(DATA_DIR, transform=tfm)
        if len(ds) == 0:
            return None, None
        
        # Split into train and test
        # If very few images, just use all for train
        if len(ds) < 3:
            train_loader = DataLoader(ds, batch_size=2, shuffle=True)
            test_loader = DataLoader(ds, batch_size=2)
        else:
            train_size = int(0.8 * len(ds))
            test_size = len(ds) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size])
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=16)
            
        return train_loader, test_loader
    except Exception as e:
        print(f"[Client] Could not load dataset: {e}")
        return None, None

def train(model, trainloader, epochs, proximal_mu=0.0, global_params=None):
    """Train the model on the local dataset with FedProx support."""
    if not trainloader:
        return
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    # Store initial global parameters for Proximal term
    if proximal_mu > 0.0 and global_params is not None:
        global_weight_collector = [param.detach().clone() for param in global_params]

    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Add Proximal Term
            if proximal_mu > 0.0 and global_params is not None:
                proximal_term = 0.0
                for local_weights, global_weights in zip(model.parameters(), global_weight_collector):
                    proximal_term += torch.square(local_weights - global_weights).sum()
                loss += (proximal_mu / 2) * proximal_term

            loss.backward()
            
            # Differential Privacy: Inject Gaussian noise to gradients
            noise_multiplier = 0.01
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.normal(mean=0.0, std=noise_multiplier, size=param.grad.shape).to(DEVICE)
                    param.grad += noise
                    
            optimizer.step()

def test(model, testloader):
    """Evaluate the model on the local dataset."""
    if not testloader:
        return 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

class DewFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        proximal_mu = config.get("proximal_mu", 0.0)
        global_params = list(self.model.parameters()) if proximal_mu > 0.0 else None

        if self.trainloader:
            print(f"[Client] Training locally (FedProx mu={proximal_mu})...")
            train(self.model, self.trainloader, epochs=1, proximal_mu=proximal_mu, global_params=global_params)
            num_examples = len(self.trainloader.dataset)
        else:
            print("[Client] No offline images found. Skipping local training.")
            num_examples = 0
            
        return self.get_parameters(config={}), num_examples, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if self.testloader:
            loss, accuracy = test(self.model, self.testloader)
            num_examples = len(self.testloader.dataset)
        else:
            loss, accuracy, num_examples = 0.0, 0.0, 0
            
        return loss, num_examples, {"accuracy": accuracy}

def main():
    print("=" * 60)
    print("  Dew-FDL | Federated Learning Client (Fog Layer)")
    print("=" * 60)
    
    # Load model
    model = DisasterEnsemble(num_classes=len(CLASS_NAMES), pretrained=False).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded starting global model from {MODEL_PATH}")
    
    # Load data
    trainloader, testloader = load_data()
    if not trainloader:
        print("[WARN] No offline images available for training. Client will just sync weights.")
    
    # Start client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=DewFlowerClient(model, trainloader, testloader).to_client(),
    )

if __name__ == "__main__":
    main()
