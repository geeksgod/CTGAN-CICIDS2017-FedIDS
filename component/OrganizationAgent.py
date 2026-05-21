from component.imp import *

class OrganizationAgent:
    def __init__(self, org_id: str, profile: Dict, model: nn.Module,
                 config: FederatedConfig, use_synthetic: bool = False):
        self.org_id        = org_id
        self.profile       = profile
        self.model         = copy.deepcopy(model).to(DEVICE)
        self.config        = config
        self.use_synthetic = use_synthetic
        self.optimizer     = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion     = nn.BCEWithLogitsLoss()
        self.scheduler     = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        self.noise_mult    = self._noise_mult()
        self.privacy_used  = 0.0
        self.metrics_hist  = []

    def _noise_mult(self) -> float:
        if self.config.dp_epsilon == float('inf'):
            return 0.0
        return np.sqrt(2 * np.log(1.25 / self.config.dp_delta)) / self.config.dp_epsilon

    # ── data loading ──────────────────────────────────────────────────────────
    def set_data(self, X_train, y_train, X_test, y_test,
                 X_synth=None, y_synth=None):
        n = int(len(X_train) * self.profile['data_quality'])
        idx = np.random.choice(len(X_train), n, replace=False)
        Xtr, ytr = X_train[idx], y_train[idx]

        # CTGAN augmentation
        if self.use_synthetic and X_synth is not None:
            n_add = int(len(Xtr) * self.config.ctgan_augment_ratio)
            idx_s = np.random.choice(len(X_synth), min(n_add, len(X_synth)), replace=False)
            Xtr = np.vstack([Xtr, X_synth[idx_s]])
            ytr = np.concatenate([ytr, y_synth[idx_s]])

        train_ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr).view(-1,1))
        test_ds  = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        self.train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        self.test_loader  = DataLoader(test_ds,  batch_size=self.config.batch_size)
        self.data_stats   = {
            'train_samples'       : len(Xtr),
            'test_samples'        : len(X_test),
            'attack_ratio_train'  : float(np.mean(ytr)),
            'synth_added'         : int(len(Xtr) - n) if self.use_synthetic else 0,
        }

    # ── local training ────────────────────────────────────────────────────────
    def local_train(self, global_weights: Dict, round_num: int) -> Tuple[Dict, Dict]:
        self.model.load_state_dict(global_weights)
        self.model.train()
        losses = []
        for _ in range(self.config.local_epochs):
            batch_losses = []
            for Xb, yb in self.train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(Xb), yb)
                loss.backward()
                self._dp_noise()
                self.optimizer.step()
                batch_losses.append(loss.item())
            losses.append(np.mean(batch_losses))
        self.scheduler.step()
        meta = {
            'org_id'             : self.org_id,
            'round'              : round_num,
            'profile'            : self.profile['name'],
            'avg_loss'           : float(np.mean(losses)),
            'learning_rate'      : self.optimizer.param_groups[0]['lr'],
            'privacy_budget_used': self.privacy_used,
            'data_contribution'  : self.data_stats['train_samples'],
            'use_synthetic'      : self.use_synthetic,
        }
        self.metrics_hist.append(meta)
        return self.model.state_dict(), meta

    def _dp_noise(self):
        if self.noise_mult == 0:
            return
        max_norm = 1.0
        total_norm = sum(p.grad.data.norm(2).item()**2
                         for p in self.model.parameters() if p.grad is not None) ** 0.5
        clip = max_norm / (total_norm + 1e-6)
        for p in self.model.parameters():
            if p.grad is not None:
                if clip < 1:
                    p.grad.data.mul_(clip)
                noise = torch.normal(0, self.noise_mult * max_norm,
                                     size=p.grad.shape, device=p.grad.device)
                p.grad.data.add_(noise)
        self.privacy_used += self.noise_mult * np.sqrt(2 * self.config.local_epochs) / self.config.dp_epsilon

    # ── evaluation ───────────────────────────────────────────────────────────
    def evaluate(self) -> Dict:
        self.model.eval()
        preds, labels, probs = [], [], []
        with torch.no_grad():
            for Xb, yb in self.test_loader:
                out  = self.model(Xb.to(DEVICE))
                prob = torch.sigmoid(out)
                preds.extend((prob > 0.5).float().cpu().numpy().flatten())
                labels.extend(yb.numpy().flatten())
                probs.extend(prob.cpu().numpy().flatten())
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        try:    auc = roc_auc_score(labels, probs)
        except: auc = 0.5
        return {'accuracy':acc, 'precision':prec, 'recall':rec, 'f1_score':f1, 'auc_roc':auc}