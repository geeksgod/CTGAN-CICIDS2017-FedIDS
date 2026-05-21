from component.imp import *
class FederatedServer:
    def __init__(self, global_model: nn.Module, config: FederatedConfig):
        self.global_model = global_model.to(DEVICE)
        self.config       = config
        self.round_metrics: List[Dict] = []

    def aggregate(self, updates: List[Tuple[Dict, Dict]]) -> Dict:
        if not updates:
            return self.global_model.state_dict()

        client_ws  = [u[0] for u in updates]
        client_mts = [u[1] for u in updates]

        sizes     = np.array([m['data_contribution'] for m in client_mts])
        losses    = np.array([m['avg_loss']           for m in client_mts])
        sz_w      = sizes / sizes.sum()
        inv_loss  = 1.0 / (losses + 1e-8); inv_loss /= inv_loss.sum()
        weights   = 0.7 * sz_w + 0.3 * inv_loss
        weights  /= weights.sum()

        gw = self.global_model.state_dict()
        for key in gw:
            gw[key] = torch.zeros_like(gw[key])
        for w, cw in zip(weights, client_ws):
            for key in gw:
                if key in cw:
                    if torch.is_floating_point(gw[key]):
                        gw[key] += cw[key].to(DEVICE) * w
                    else:
                        gw[key] = cw[key].to(DEVICE)
        self.global_model.load_state_dict(gw)
        return gw

    def evaluate_global(self, test_sets: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        self.global_model.eval()
        all_preds, all_labels, all_probs = [], [], []
        for Xt, yt in test_sets:
            ds = TensorDataset(torch.FloatTensor(Xt), torch.FloatTensor(yt))
            dl = DataLoader(ds, batch_size=256)
            with torch.no_grad():
                for Xb, yb in dl:
                    out   = self.global_model(Xb.to(DEVICE))
                    prob  = torch.sigmoid(out)
                    preds = (prob > 0.5).float()
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_labels.extend(yb.numpy().flatten())
                    all_probs.extend(prob.cpu().numpy().flatten())
        acc  = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0)
        try:    auc = roc_auc_score(all_labels, all_probs)
        except: auc = 0.5
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        return dict(accuracy=acc, precision=prec, recall=rec, f1_score=f1,
                    auc_roc=auc, true_positives=int(tp), true_negatives=int(tn),
                    false_positives=int(fp), false_negatives=int(fn),
                    specificity=tn/(tn+fp+1e-9), sensitivity=tp/(tp+fn+1e-9),
                    all_probs=all_probs, all_labels=all_labels)
    
    def save_model(self, path: str = "global_model.pt"):
        torch.save(self.global_model.state_dict(), path)
        print(f"✅ Model weights saved → {path}")

    def load_model(self, path: str = "global_model.pt"):
        self.global_model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.global_model.eval()
        print(f"✅ Model weights loaded ← {path}")
