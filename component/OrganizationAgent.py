from component.imp import *
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager


class OrganizationAgent:
    def __init__(self, org_id: str, profile: Dict, model: nn.Module,
                 config: FederatedConfig, use_synthetic: bool = False):
        self.org_id        = org_id
        self.profile       = profile
        self.config        = config
        self.use_synthetic = use_synthetic
        self.metrics_hist  = []
        self._dp_enabled   = config.dp_epsilon != float('inf')

        # Validate before doing anything — fail fast with a clear message
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            raise ValueError(
                "Model is incompatible with Opacus. Make sure CybersecurityNet "
                "uses GroupNorm instead of BatchNorm.\n"
                f"Errors: {errors}"
            )

        self.model        = copy.deepcopy(model).to(DEVICE)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion    = nn.BCEWithLogitsLoss()
        self.scheduler    = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        self.privacy_engine = PrivacyEngine(accountant="rdp")

        # These are set in set_data() once we have a DataLoader
        self._private_model     = None
        self._private_optimizer = None
        self._private_loader    = None

    # ── data loading ──────────────────────────────────────────────────────────
    def set_data(self, X_train, y_train, X_test, y_test,
                 X_synth=None, y_synth=None):
        n   = int(len(X_train) * self.profile['data_quality'])
        idx = np.random.choice(len(X_train), n, replace=False)
        Xtr, ytr = X_train[idx], y_train[idx]

        if self.use_synthetic and X_synth is not None:
            n_add = int(len(Xtr) * self.config.ctgan_augment_ratio)
            idx_s = np.random.choice(len(X_synth), min(n_add, len(X_synth)), replace=False)
            Xtr = np.vstack([Xtr, X_synth[idx_s]])
            ytr = np.concatenate([ytr, y_synth[idx_s]])

        train_ds = TensorDataset(
            torch.FloatTensor(Xtr),
            torch.FloatTensor(ytr).view(-1, 1),
        )
        test_ds = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test),
        )

        self.train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size
        )

        self.data_stats = {
            'train_samples'     : len(Xtr),
            'test_samples'      : len(X_test),
            'attack_ratio_train': float(np.mean(ytr)),
            'synth_added'       : int(len(Xtr) - n) if self.use_synthetic else 0,
        }

        # Attach PrivacyEngine now that we have the DataLoader
        if self._dp_enabled:
            (
                self._private_model,
                self._private_optimizer,
                self._private_loader,
            ) = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                epochs=self.config.local_epochs,
                target_epsilon=self.config.dp_epsilon,
                target_delta=self.config.dp_delta,
                max_grad_norm=1.0,
            )
        else:
            self._private_model     = self.model
            self._private_optimizer = self.optimizer
            self._private_loader    = self.train_loader

    # ── local training ────────────────────────────────────────────────────────
    def local_train(self, global_weights: Dict, round_num: int) -> Tuple[Dict, Dict]:
        # Opacus wraps the model as GradSampleModule; load with strict=False
        # to tolerate the "_module." prefix difference
        self._private_model.load_state_dict(global_weights, strict=False)
        self._private_model.train()

        losses = []
        for _ in range(self.config.local_epochs):
            batch_losses = []
            with BatchMemoryManager(
                data_loader=self._private_loader,
                max_physical_batch_size=self.config.batch_size,
                optimizer=self._private_optimizer,
            ) as memory_safe_loader:
                for Xb, yb in memory_safe_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    self._private_optimizer.zero_grad()
                    loss = self.criterion(self._private_model(Xb), yb)
                    loss.backward()
                    self._private_optimizer.step()
                    batch_losses.append(loss.item())
            losses.append(np.mean(batch_losses))

        self.scheduler.step()

        epsilon = (
            self.privacy_engine.get_epsilon(self.config.dp_delta)
            if self._dp_enabled else float('inf')
        )

        meta = {
            'org_id'             : self.org_id,
            'round'              : round_num,
            'profile'            : self.profile['name'],
            'avg_loss'           : float(np.mean(losses)),
            'learning_rate'      : self._private_optimizer.param_groups[0]['lr'],
            'privacy_budget_used': epsilon,
            'data_contribution'  : self.data_stats['train_samples'],
            'use_synthetic'      : self.use_synthetic,
        }
        self.metrics_hist.append(meta)

        # Unwrap to plain state_dict for federated aggregation
        raw_weights = (
            self._private_model._module.state_dict()
            if self._dp_enabled
            else self._private_model.state_dict()
        )
        return raw_weights, meta

    # ── evaluation ───────────────────────────────────────────────────────────
    def evaluate(self) -> Dict:
        self._private_model.eval()
        preds, labels, probs = [], [], []
        with torch.no_grad():
            for Xb, yb in self.test_loader:
                out  = self._private_model(Xb.to(DEVICE))
                prob = torch.sigmoid(out)
                preds.extend((prob > 0.5).float().cpu().numpy().flatten())
                labels.extend(yb.numpy().flatten())
                probs.extend(prob.cpu().numpy().flatten())

        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )
        try:    auc = roc_auc_score(labels, probs)
        except: auc = 0.5

        return {
            'accuracy' : acc,
            'precision': prec,
            'recall'   : rec,
            'f1_score' : f1,
            'auc_roc'  : auc,
        }