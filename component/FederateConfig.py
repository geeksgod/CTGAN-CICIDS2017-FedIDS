
from component.imp import *
@dataclass
class FederatedConfig:
    n_organizations: int = 5
    global_rounds: int = 20
    local_epochs: int = 3
    learning_rate: float = 0.001
    batch_size: int = 128
    test_size: float = 0.2
    feature_selection_k: int = 50
    model_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    random_seed: int = 42
    experiment_name: str = "federated_ids_ctgan_eval"
   