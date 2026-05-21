"""
==============================================================================
  FEDERATED IDS WITH CTGAN SYNTHETIC DATA EVALUATION
  Built on top of CICIDS-2017.py architecture
  
  New additions:
    - SyntheticAugmentedOrg: organization variant that trains on real + CTGAN data
    - FederatedIDSExperiment: orchestrates Real-only vs CTGAN-augmented comparison
    - Comprehensive evaluation suite with all plots saved to /results/
==============================================================================
"""

from component.imp import *
from component.FederatedIDSExperiment import FederatedIDSExperiment

def main():
    config = FederatedConfig(
        n_organizations    = 5,
        global_rounds      = 20,
        local_epochs       = 3,
        learning_rate      = 0.001,
        batch_size         = 128,
        feature_selection_k= 50,
        model_hidden_dims  = [256, 128, 64],
        dp_epsilon         = 4.0,
        dp_delta           = 1e-5,
        random_seed        = 42,
        experiment_name    = "federated_ids_ctgan_eval",
    )

    # ── Update these paths to point to your actual files ──────────────────────
    REAL_CSV      = "datasets/cicids2017_train.csv"      # full CICIDS-2017 CSV (with Label column)
    SYNTHETIC_CSV = "datasets/cicids2017_synthetic.csv"     # CTGAN-generated CSV (same column schema)
    OUTPUT_DIR    = f"results/results_{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    experiment = FederatedIDSExperiment(config, REAL_CSV, SYNTHETIC_CSV, OUTPUT_DIR)
    results    = experiment.run()
    return results


if __name__ == "__main__":
    main()
