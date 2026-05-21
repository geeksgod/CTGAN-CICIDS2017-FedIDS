from component.imp import *
from component.CybersecurityNet import CybersecurityNet
from component.DataManager import DataManager
from component.FederateConfig import FederatedConfig
from component.FederateServer import FederatedServer
from component.OrganizationAgent import OrganizationAgent
from component.Visualizer import Visualizer

ORG_PROFILES = {
    "financial_bank"         : {"name":"Global Financial Bank",    "data_quality":0.95, "privacy_level":"high"},
    "tech_company"           : {"name":"Technology Corporation",   "data_quality":0.90, "privacy_level":"medium"},
    "healthcare_system"      : {"name":"Healthcare Network",       "data_quality":0.85, "privacy_level":"high"},
    "government_agency"      : {"name":"Government Agency",        "data_quality":0.88, "privacy_level":"maximum"},
    "educational_institution": {"name":"University Network",       "data_quality":0.80, "privacy_level":"low"},
}

class FederatedIDSExperiment:
    """
    Orchestrates two parallel federated IDS runs:
      A) Real-only   (5 organizations, standard FedAvg + DP)
      B) CTGAN-augmented (same orgs, each gets real + CTGAN data)
    then produces a full comparison report.
    """

    def __init__(self, config: FederatedConfig,
                 real_csv: str, synthetic_csv: str,
                 out_dir: str = "results_federated_ids"):
        self.config       = config
        self.real_csv     = real_csv
        self.synthetic_csv = synthetic_csv
        self.out_dir      = out_dir
        self.viz          = Visualizer(out_dir)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)

    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        t0 = time.time()
        print("\n" + "═"*70)
        print("  🛡️  FEDERATED IDS WITH CTGAN SYNTHETIC DATA EVALUATION")
        print("═"*70)

        # 1. Load data ─────────────────────────────────────────────────────────
        dm = DataManager(self.config)
        X_real, y_real, feat_names = dm.load_data(self.real_csv)
        X_synth, y_synth ,feat_names = dm.load_data(self.synthetic_csv)

       

        # 3. Train/test split ──────────────────────────────────────────────────
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_real, y_real, test_size=self.config.test_size,
            random_state=self.config.random_seed, stratify=y_real)
        
        X_tr_sy, X_te_sy, y_tr_sy, y_te_sy = train_test_split(
            X_synth, y_synth, test_size=self.config.test_size,
            random_state=self.config.random_seed, stratify=y_real)

        # distribute real data across organizations
        org_keys   = list(ORG_PROFILES.keys())
        splits_tr  = np.array_split(np.random.permutation(len(X_tr)), self.config.n_organizations)
        splits_te  = np.array_split(np.random.permutation(len(X_te)), self.config.n_organizations)
        test_sets  = [(X_te[idx], y_te[idx]) for idx in splits_te]

        input_dim = X_real.shape[1]

        print("\n" + "─"*60)
        print("Base Line: Real only")
        print("─"*60)
        real_metrics, real_orgs, real_gm = self._run_federation(
            input_dim, X_tr, y_tr, X_te, y_te, splits_tr, test_sets,
            synthetic=False)

        
        print("\n" + "─"*60)
        print("TRTS: Train Real Test Synthetic")
        print("─"*60)
        trts_metrics, trts_orgs, trts_gm = self._run_federation(
            input_dim, X_tr, y_tr, X_te_sy, y_te_sy, splits_tr, test_sets,
            synthetic=False)


        print("\n" + "─"*60)
        print("TSTR: Train Synthetic Test Real")
        print("─"*60)
        tstr_metrics, tstr_orgs, tstr_gm = self._run_federation(
            input_dim, X_tr_sy, y_tr_sy, X_te, y_te, splits_tr, test_sets,
            synthetic=True)

        # 6. Collect per-org ROC data ──────────────────────────────────────────
        org_roc_real  = self._collect_org_roc(real_orgs)
        org_roc_trts = self._collect_org_roc(trts_orgs)
        org_roc_tstr = self._collect_org_roc(tstr_orgs)

        # 7. All visualizations ────────────────────────────────────────────────
        self.viz.plot_convergence(real_metrics,trts_metrics, tstr_metrics)
        self.viz.plot_confusion_matrices(real_gm, trts_gm, tstr_gm)
        self.viz.plot_roc_curves(real_gm, trts_gm,trts_gm, org_roc_real, org_roc_trts, org_roc_tstr)
        self.viz.plot_org_breakdown(real_orgs, trts_orgs, tstr_orgs)
        self.viz.plot_summary_panel(real_gm, trts_gm, tstr_gm)
        self.viz.plot_privacy_budget(real_orgs, trts_orgs, tstr_orgs)

        # 8. Save JSON results ────────────────────────────────────────────────
        results = {
            'config'          : vars(self.config),
            'real_only'       : {'final': {k:v for k,v in real_gm.items()
                                           if k not in ('all_probs','all_labels')}},
            'TRTS' : {'final': {k:v for k,v in trts_gm.items()
                                           if k not in ('all_probs','all_labels')}},
            'TSTR' : {'final': {k:v for k,v in tstr_gm.items()
                                           if k not in ('all_probs','all_labels')}},
            'duration_seconds': round(time.time() - t0, 1),
            'timestamp'       : datetime.now().isoformat(),
        }
        json_path = f"{self.out_dir}/experiment_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    # ─────────────────────────────────────────────────────────────────────────
    def _run_federation(self, input_dim, X_tr, y_tr, X_te, y_te,
                        splits_tr, test_sets,
                        synthetic=False):
        config = self.config
        global_model = CybersecurityNet(input_dim, config.model_hidden_dims)
        server       = FederatedServer(global_model, config)

        org_keys = list(ORG_PROFILES.keys())
        orgs = []
        for i in range(config.n_organizations):
            key = org_keys[i % len(org_keys)]
            org = OrganizationAgent(str(i), ORG_PROFILES[key], global_model, config)
            idx = splits_tr[i]
            org.set_data(X_tr[idx], y_tr[idx], X_te, y_te)
            orgs.append(org)
            mode = "CTGAN-Synthetic data" if synthetic else "Real-only"
            synth_n = org.data_stats.get('synth_added', 0)
            print(f"   Org {i} [{ORG_PROFILES[key]['name']:25s}] "
                  f"train={org.data_stats['train_samples']:,}  "
                  f"(+{synth_n} synth)  mode={mode}")

        round_metrics = []
        for rnd in range(1, config.global_rounds + 1):
            print(f"\n  🔄 Round {rnd}/{config.global_rounds}")
            gw = server.global_model.state_dict()
            updates = [org.local_train(gw, rnd) for org in orgs]
            server.aggregate(updates)
            gm = server.evaluate_global(test_sets)
            round_metrics.append({k: gm[k] for k in
                                   ('accuracy','precision','recall','f1_score','auc_roc')})
            print(f"     Global → Acc={gm['accuracy']:.4f}  F1={gm['f1_score']:.4f}  AUC={gm['auc_roc']:.4f}")
        model_path = f"./model/{'global_synthetic.pt' if synthetic else 'global_real.pt'}"
        server.save_model(model_path)
        final_gm = server.evaluate_global(test_sets)
        return round_metrics, orgs, final_gm

    # ─────────────────────────────────────────────────────────────────────────
    def _collect_org_roc(self, orgs: List[OrganizationAgent]) -> List[Dict]:
        records = []
        for org in orgs:
            org.model.eval()
            probs, labels = [], []
            with torch.no_grad():
                for Xb, yb in org.test_loader:
                    out = org.model(Xb.to(DEVICE))
                    probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
                    labels.extend(yb.numpy().flatten())
            try:
                fpr, tpr, _ = roc_curve(labels, probs)
                auc = roc_auc_score(labels, probs)
            except:
                fpr, tpr, auc = [0,1], [0,1], 0.5
            records.append({'name': org.profile['name'], 'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': auc})
        return records
