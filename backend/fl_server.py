import sys, os
import flwr as fl
import torch
from logging import INFO
from flwr.common.logger import log
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(__file__))
from train_model import DisasterEnsemble

# Number of classes
CLASS_NAMES = [
    "Drought", "Earthquake",
    "Land_Slide", "Water_Disaster", "Wild_Fire", "Non_Damage"
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "global_model.pth")

class SaveModelStrategy(fl.server.strategy.FedProx):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        
        # Call aggregate_fit from base class (FedProx) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"[Server] Round {server_round} aggregation complete. Saving global model...")
            
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Initialize model
            model = DisasterEnsemble(num_classes=len(CLASS_NAMES), pretrained=False)
            
            # Set parameters
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            
            # Save the model
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"[Server] Global model saved to {MODEL_PATH}")

        return aggregated_parameters, aggregated_metrics

def main():
    print("=" * 60)
    print("  Dew-FDL | Federated Learning Server (Cloud Layer)")
    print("=" * 60)
    
    # Create strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=1, # Never start training without at least 1 client
        min_evaluate_clients=1,
        min_available_clients=1, # Wait until at least 1 client is available
        proximal_mu=0.1, # FedProx regularization term
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
