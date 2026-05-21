"""
==============================================================================
  FEDERATED IDS WITH CTGAN SYNTHETIC DATA EVALUATION
  Built on top of CICIDS-2017.py architecture
  
  New additions:
    - CTGANDataEvaluator  : quality tests (KS, AUC-ROC discriminator, feature importance)
    - SyntheticAugmentedOrg: organization variant that trains on real + CTGAN data
    - FederatedIDSExperiment: orchestrates Real-only vs CTGAN-augmented comparison
    - Comprehensive evaluation suite with all plots saved to /results/
==============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import json
import time
import copy
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import matplotlib
matplotlib.use('Agg')          # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# ─── Style ────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
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
    # CTGAN settings
    ctgan_augment_ratio: float = 0.30   # add 30 % synthetic on top of real
    ctgan_eval_sample: int = 5000       # rows used for quality evaluation

# ══════════════════════════════════════════════════════════════════════════════
#  CTGAN DATA EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
class CTGANDataEvaluator:
    """
    Evaluate quality of CTGAN-generated synthetic data vs real data.
    Tests:
      1. KS-test per numeric feature (p > 0.05 → distributions match)
      2. Discriminator AUC-ROC (should be ≈ 0.5 for indistinguishable data)
      3. Discriminator accuracy
      4. Top giveaway features (feature importance from discriminator)
      5. Statistical moments comparison (mean, std, skew, kurtosis)
    """

    def __init__(self, config: FederatedConfig):
        self.config = config
        self.results: Dict = {}

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    # ── main evaluation entry point ───────────────────────────────────────────
    def evaluate(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                 label: str = "CTGAN Evaluation") -> Dict:
        """Run full quality evaluation suite and return result dict."""
        print(f"\n{'='*70}")
        print(f"  🧪 {label}")
        print(f"{'='*70}")
        print(f"  Real samples    : {len(real_df):,}")
        print(f"  Synthetic samples: {len(synthetic_df):,}")

        n = self.config.ctgan_eval_sample
        real_s  = self._sample(real_df,  n)
        synth_s = self._sample(synthetic_df, n)

        # keep only shared numeric columns
        common_cols = [c for c in real_s.columns
                       if c in synth_s.columns
                       and pd.api.types.is_numeric_dtype(real_s[c])]
        real_s  = real_s[common_cols].copy()
        synth_s = synth_s[common_cols].copy()

        # fill NaN so sklearn doesn't complain
        for df_ in (real_s, synth_s):
            df_.fillna(df_.median(), inplace=True)

        ks_results      = self._ks_tests(real_s, synth_s, common_cols)
        disc_results    = self._discriminator_test(real_s, synth_s)
        moment_results  = self._moment_comparison(real_s, synth_s, common_cols)

        self.results = {
            'label'          : label,
            'n_real'         : len(real_df),
            'n_synthetic'    : len(synthetic_df),
            'ks'             : ks_results,
            'discriminator'  : disc_results,
            'moments'        : moment_results,
        }

        self._print_summary()
        return self.results

    # ── KS tests ──────────────────────────────────────────────────────────────
    def _ks_tests(self, real: pd.DataFrame, synth: pd.DataFrame,
                  cols: List[str]) -> Dict:
        passed, failed = [], []
        details = {}
        for col in cols:
            stat, p = scipy_stats.ks_2samp(real[col].values, synth[col].values)
            details[col] = {'stat': float(stat), 'p': float(p), 'pass': p > 0.05}
            (passed if p > 0.05 else failed).append(col)

        pct_pass = len(passed) / len(cols) * 100 if cols else 0
        status = "✅" if pct_pass > 50 else "🚨"
        print(f"\n  {status} KS Test: {pct_pass:.1f}% of features passed (p > 0.05)")
        print(f"     Passed {len(passed)}/{len(cols)} features")
        return {'pct_pass': pct_pass, 'n_pass': len(passed),
                'n_fail': len(failed), 'details': details}

    # ── Discriminator test ────────────────────────────────────────────────────
    def _discriminator_test(self, real: pd.DataFrame, synth: pd.DataFrame) -> Dict:
        X = pd.concat([real, synth], ignore_index=True).values
        y = np.array([0]*len(real) + [1]*len(synth))

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        auc_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
        acc_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

        auc = float(auc_scores.mean())
        acc = float(acc_scores.mean())

        # Feature importance for giveaway features
        clf.fit(X, y)
        importances = clf.feature_importances_
        feat_imp = pd.DataFrame({
            'Feature'   : real.columns.tolist(),
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        auc_ok = auc < 0.65
        status = "✅" if auc_ok else "🚨"
        print(f"\n  {status} Discriminator AUC-ROC  : {auc:.4f}  (target ≈ 0.50)")
        print(f"  {status} Discriminator Accuracy : {acc:.4f}  (target ≈ 0.50)")
        print(f"\n  🔍 Top 5 giveaway features:")
        print(feat_imp.head(5).to_string(index=False))

        return {
            'auc_roc'    : auc,
            'accuracy'   : acc,
            'feature_imp': feat_imp.head(20).to_dict('records'),
        }

    # ── Statistical moments ───────────────────────────────────────────────────
    def _moment_comparison(self, real: pd.DataFrame, synth: pd.DataFrame,
                           cols: List[str]) -> Dict:
        records = []
        for col in cols[:30]:   # cap at 30 for speed
            r, s = real[col], synth[col]
            records.append({
                'feature'   : col,
                'mean_diff' : abs(r.mean() - s.mean()),
                'std_diff'  : abs(r.std()  - s.std()),
                'skew_diff' : abs(float(scipy_stats.skew(r)) - float(scipy_stats.skew(s))),
                'kurt_diff' : abs(float(scipy_stats.kurtosis(r)) - float(scipy_stats.kurtosis(s))),
            })
        df_ = pd.DataFrame(records)
        avg = df_[['mean_diff','std_diff','skew_diff','kurt_diff']].mean()
        print(f"\n  📐 Avg moment differences  mean={avg['mean_diff']:.4f}  "
              f"std={avg['std_diff']:.4f}  skew={avg['skew_diff']:.4f}  "
              f"kurt={avg['kurt_diff']:.4f}")
        return {'summary': avg.to_dict(), 'per_feature': records}

    # ── Console summary ───────────────────────────────────────────────────────
    def _print_summary(self):
        r = self.results
        auc  = r['discriminator']['auc_roc']
        ks   = r['ks']['pct_pass']
        grade = "EXCELLENT" if auc < 0.55 and ks > 70 \
           else "GOOD"      if auc < 0.65 and ks > 40 \
           else "FAIR"      if auc < 0.75 \
           else "POOR"
        print(f"\n  Overall Quality Grade: [{grade}]")
        print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  NEURAL NETWORK (same as original)
# ══════════════════════════════════════════════════════════════════════════════
class CybersecurityNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3), nn.BatchNorm1d(h)]
            prev = h
        layers += [nn.Linear(prev, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)]
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  ORGANIZATION AGENT (extended to support CTGAN augmentation)
# ══════════════════════════════════════════════════════════════════════════════
ORG_PROFILES = {
    "financial_bank"         : {"name":"Global Financial Bank",    "data_quality":0.95, "privacy_level":"high"},
    "tech_company"           : {"name":"Technology Corporation",   "data_quality":0.90, "privacy_level":"medium"},
    "healthcare_system"      : {"name":"Healthcare Network",       "data_quality":0.85, "privacy_level":"high"},
    "government_agency"      : {"name":"Government Agency",        "data_quality":0.88, "privacy_level":"maximum"},
    "educational_institution": {"name":"University Network",       "data_quality":0.80, "privacy_level":"low"},
}

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


# ══════════════════════════════════════════════════════════════════════════════
#  FEDERATED SERVER
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════
class DataManager:
    def __init__(self, config: FederatedConfig):
        self.config   = config
        self.scaler   = StandardScaler()
        self.selector = SelectKBest(mutual_info_classif, k=config.feature_selection_k)

    def load_real(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        print(f"📂 Loading real data from: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes('object').columns:
            df[col] = df[col].str.strip()
        return self._preprocess(df)

    def load_synthetic(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load CTGAN-generated CSV. Expects same columns as real data."""
        print(f"🤖 Loading CTGAN synthetic data from: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes('object').columns:
            df[col] = df[col].str.strip()
        return self._preprocess_synthetic(df)

    def _find_label(self, cols) -> Optional[str]:
        for c in ['Label',' Label','label',' label']:
            if c in cols: return c
        return None

    def _preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        label_col = self._find_label(df.columns)
        if label_col is None:
            raise ValueError("No 'Label' column found in real data!")
        labels = df[label_col].copy()

        # Identify top 5 attacks
        attack_counts = labels[labels != 'BENIGN'].value_counts()
        self.top_attacks = attack_counts.head(5).index.tolist()
        valid = ['BENIGN'] + self.top_attacks
        df = df[labels.isin(valid)].copy()
        labels = labels[labels.isin(valid)]

        # Store raw df for CTGAN evaluator
        self.raw_real_df = df.copy()

        feat_cols = [c for c in df.columns if c not in [label_col, 'source_file']]
        X_df = df[feat_cols].apply(pd.to_numeric, errors='coerce')
        X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_df.fillna(X_df.median(), inplace=True)
        X_df = X_df.loc[:, X_df.var() > 1e-6]

        y = (labels != 'BENIGN').astype(int).values
        X = X_df.values

        k = min(self.config.feature_selection_k, X.shape[1])
        self.selector = SelectKBest(mutual_info_classif, k=k)
        X_sel = self.selector.fit_transform(X, y)
        self.selected_cols = [feat_cols[i] for i in self.selector.get_support(indices=True)]
        X_scaled = self.scaler.fit_transform(X_sel)

        print(f"✅ Real data: {X_scaled.shape}  |  Attack ratio: {y.mean():.1%}")
        return X_scaled, y, self.selected_cols

    def _preprocess_synthetic(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the SAME selector + scaler fitted on real data."""
        label_col = self._find_label(df.columns)
        if label_col is None:
            raise ValueError("No 'Label' column found in synthetic data!")
        labels = df[label_col].copy()
        y = (labels.str.strip() != 'BENIGN').astype(int).values

        feat_cols = [c for c in df.columns if c not in [label_col, 'source_file']]
        X_df = df[feat_cols].apply(pd.to_numeric, errors='coerce')
        X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_df.fillna(X_df.median(), inplace=True)

        # Use only the features selected on real data
        available = [c for c in self.selected_cols if c in X_df.columns]
        missing   = [c for c in self.selected_cols if c not in X_df.columns]
        if missing:
            print(f"⚠️  {len(missing)} features missing in synthetic data – zero-filled")
            for c in missing:
                X_df[c] = 0.0
        X = X_df[self.selected_cols].values
        X_scaled = self.scaler.transform(X)

        print(f"✅ Synthetic data: {X_scaled.shape}  |  Attack ratio: {y.mean():.1%}")
        return X_scaled, y


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════
class Visualizer:
    def __init__(self, out_dir: str):
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)

    # ── 1. CTGAN quality dashboard ────────────────────────────────────────────
    def plot_ctgan_quality(self, eval_results: Dict, out_name: str = "01_ctgan_quality.png"):
        fig = plt.figure(figsize=(20, 14))
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        # --- KS pass/fail pie
        ax0 = fig.add_subplot(gs[0, 0])
        n_pass = eval_results['ks']['n_pass']
        n_fail = eval_results['ks']['n_fail']
        ax0.pie([n_pass, n_fail], labels=[f'Pass ({n_pass})', f'Fail ({n_fail})'],
                colors=['#2ecc71','#e74c3c'], autopct='%1.0f%%', startangle=90)
        ax0.set_title('KS Test Pass/Fail\n(p > 0.05 = real-like)', fontweight='bold')

        # --- KS p-value distribution
        ax1 = fig.add_subplot(gs[0, 1:])
        details = eval_results['ks']['details']
        p_vals  = [v['p'] for v in details.values()]
        ax1.hist(p_vals, bins=30, color='#3498db', edgecolor='white', alpha=0.85)
        ax1.axvline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05 threshold')
        ax1.set_xlabel('KS p-value'); ax1.set_ylabel('Count')
        ax1.set_title('Distribution of KS p-values Across Features', fontweight='bold')
        ax1.legend()

        # --- Discriminator AUC gauge
        ax2 = fig.add_subplot(gs[1, 0])
        auc = eval_results['discriminator']['auc_roc']
        color = '#2ecc71' if auc < 0.6 else '#f39c12' if auc < 0.75 else '#e74c3c'
        ax2.barh(['AUC-ROC'], [auc], color=color, height=0.4)
        ax2.barh(['AUC-ROC'], [1.0], color='#ecf0f1', height=0.4, zorder=0)
        ax2.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Target ≈ 0.5')
        ax2.set_xlim(0, 1); ax2.set_title(f'Discriminator AUC\n{auc:.4f}', fontweight='bold')
        ax2.text(auc + 0.02, 0, f'{auc:.3f}', va='center', fontweight='bold')
        ax2.legend(fontsize=9)

        # --- Top giveaway features
        ax3 = fig.add_subplot(gs[1, 1:])
        feat_imp = pd.DataFrame(eval_results['discriminator']['feature_imp']).head(15)
        colors_fi = ['#e74c3c' if i < 3 else '#e67e22' if i < 6 else '#3498db'
                     for i in range(len(feat_imp))]
        ax3.barh(feat_imp['Feature'][::-1], feat_imp['Importance'][::-1],
                 color=colors_fi[::-1], alpha=0.85, edgecolor='white')
        ax3.set_xlabel('Feature Importance'); ax3.set_title('Top Giveaway Features (Red = Highest Risk)', fontweight='bold')
        ax3.grid(True, axis='x', alpha=0.3)

        # --- Statistical moments comparison
        ax4 = fig.add_subplot(gs[2, :])
        moments = eval_results['moments']
        pf = pd.DataFrame(moments['per_feature'][:25])
        x  = np.arange(len(pf))
        w  = 0.2
        ax4.bar(x - 1.5*w, pf['mean_diff'], w, label='Mean diff',  color='#3498db', alpha=0.8)
        ax4.bar(x - 0.5*w, pf['std_diff'],  w, label='Std diff',   color='#e74c3c', alpha=0.8)
        ax4.bar(x + 0.5*w, pf['skew_diff'], w, label='Skew diff',  color='#2ecc71', alpha=0.8)
        ax4.bar(x + 1.5*w, pf['kurt_diff'], w, label='Kurt diff',  color='#9b59b6', alpha=0.8)
        ax4.set_xticks(x); ax4.set_xticklabels(pf['feature'], rotation=90, fontsize=7)
        ax4.set_ylabel('Absolute Difference'); ax4.set_title('Statistical Moments: |Real − Synthetic|', fontweight='bold')
        ax4.legend(); ax4.grid(True, axis='y', alpha=0.3)

        fig.suptitle('CTGAN Synthetic Data Quality Dashboard', fontsize=16, fontweight='bold', y=1.01)
        path = f"{self.out}/{out_name}"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"💾 Saved: {path}")

    # ── 2. Training convergence ───────────────────────────────────────────────
    def plot_convergence(self, real_metrics: List[Dict], synth_metrics: List[Dict],
                         out_name: str = "02_training_convergence.png"):
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))
        rounds = range(1, len(real_metrics) + 1)
        kpi_map = [
            ('accuracy',  'Accuracy',  axes[0,0]),
            ('f1_score',  'F1-Score',  axes[0,1]),
            ('auc_roc',   'AUC-ROC',   axes[0,2]),
            ('precision', 'Precision', axes[1,0]),
            ('recall',    'Recall',    axes[1,1]),
        ]
        for key, title, ax in kpi_map:
            rv = [m[key] for m in real_metrics]
            sv = [m[key] for m in synth_metrics]
            ax.plot(rounds, rv, 'o-', linewidth=2.5, color='#2E86AB', label='Real only')
            ax.plot(rounds, sv, 's--', linewidth=2.5, color='#E63946', label='+ CTGAN')
            ax.fill_between(rounds, rv, sv, alpha=0.15, color='gray')
            ax.set_title(title, fontweight='bold'); ax.set_ylim(0, 1)
            ax.set_xlabel('Round'); ax.set_ylabel(title)
            ax.legend(); ax.grid(True, alpha=0.3)

        # Final round bar comparison
        ax = axes[1, 2]
        metrics_names = ['accuracy','precision','recall','f1_score','auc_roc']
        real_final  = [real_metrics[-1][m]  for m in metrics_names]
        synth_final = [synth_metrics[-1][m] for m in metrics_names]
        x = np.arange(len(metrics_names)); w = 0.35
        ax.bar(x - w/2, real_final,  w, label='Real only', color='#2E86AB', alpha=0.85, edgecolor='white')
        ax.bar(x + w/2, synth_final, w, label='+ CTGAN',   color='#E63946', alpha=0.85, edgecolor='white')
        ax.set_xticks(x); ax.set_xticklabels(metrics_names, rotation=30)
        ax.set_ylim(0, 1.1); ax.set_title('Final Round: Real vs CTGAN-Augmented', fontweight='bold')
        ax.legend(); ax.grid(True, axis='y', alpha=0.3)
        for i, (rv, sv) in enumerate(zip(real_final, synth_final)):
            ax.text(i - w/2, rv + 0.01, f'{rv:.3f}', ha='center', fontsize=8, fontweight='bold')
            ax.text(i + w/2, sv + 0.01, f'{sv:.3f}', ha='center', fontsize=8, fontweight='bold')

        fig.suptitle('Federated IDS: Training Convergence  (Real-only  vs  CTGAN-augmented)',
                     fontsize=15, fontweight='bold')
        path = f"{self.out}/{out_name}"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"💾 Saved: {path}")

    # ── 3. Confusion matrices ─────────────────────────────────────────────────
    def plot_confusion_matrices(self, real_gm: Dict, synth_gm: Dict,
                                out_name: str = "03_confusion_matrices.png"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, gm, title, cmap in zip(
                axes,
                [real_gm, synth_gm],
                ['Global Model – Real Only', 'Global Model – CTGAN Augmented'],
                ['Blues', 'Reds']):
            cm = np.array([[gm['true_negatives'],  gm['false_positives']],
                           [gm['false_negatives'], gm['true_positives']]])
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                        xticklabels=['Benign','Attack'], yticklabels=['Benign','Attack'],
                        linewidths=1, linecolor='white', cbar_kws={'label':'Count'})
            ax.set_title(f'{title}\nAcc={gm["accuracy"]:.4f}  F1={gm["f1_score"]:.4f}  AUC={gm["auc_roc"]:.4f}',
                         fontweight='bold')
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        fig.suptitle('Confusion Matrix Comparison', fontsize=15, fontweight='bold')
        path = f"{self.out}/{out_name}"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"💾 Saved: {path}")

    # ── 4. ROC curves ─────────────────────────────────────────────────────────
    def plot_roc_curves(self, real_gm: Dict, synth_gm: Dict,
                        org_results_real: List[Dict], org_results_synth: List[Dict],
                        out_name: str = "04_roc_curves.png"):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, gm, org_res, title in zip(
                axes,
                [real_gm, synth_gm],
                [org_results_real, org_results_synth],
                ['Real-Only Federated IDS', 'CTGAN-Augmented Federated IDS']):
            colors = plt.cm.tab10(np.linspace(0, 1, len(org_res) + 1))
            for i, om in enumerate(org_res):
                if 'fpr' in om:
                    ax.plot(om['fpr'], om['tpr'], linewidth=1.5,
                            color=colors[i], alpha=0.7, label=f"{om['name']} (AUC={om['auc']:.3f})")
            # Global ROC from stored probs
            if 'all_probs' in gm and 'all_labels' in gm:
                fpr, tpr, _ = roc_curve(gm['all_labels'], gm['all_probs'])
                ax.plot(fpr, tpr, linewidth=3, color=colors[-1],
                        label=f'Global Model (AUC={gm["auc_roc"]:.3f})')
            ax.plot([0,1],[0,1],'k--',linewidth=1,alpha=0.5,label='Random')
            ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
            ax.set_title(title, fontweight='bold'); ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle('ROC Curve Comparison', fontsize=15, fontweight='bold')
        path = f"{self.out}/{out_name}"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"💾 Saved: {path}")

    # ── 5. Org-level breakdown ─────────────────────────────────────────────────
    def plot_org_breakdown(self, orgs_real: List, orgs_synth: List,
                           out_name: str = "05_org_breakdown.png"):
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        names = [o.profile['name'].split()[0] for o in orgs_real]
        x = np.arange(len(names)); w = 0.35

        for ax, metric, title in zip(
                axes.flatten(),
                ['accuracy','f1_score','auc_roc','recall'],
                ['Accuracy','F1-Score','AUC-ROC','Recall']):
            rv = [o.evaluate()[metric] for o in orgs_real]
            sv = [o.evaluate()[metric] for o in orgs_synth]
            ax.bar(x-w/2, rv, w, label='Real only', color='#2E86AB', alpha=0.85, edgecolor='white')
            ax.bar(x+w/2, sv, w, label='+ CTGAN',   color='#E63946', alpha=0.85, edgecolor='white')
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=30)
            ax.set_ylim(0, 1.1); ax.set_title(f'Per-Org {title}', fontweight='bold')
            ax.legend(); ax.grid(True, axis='y', alpha=0.3)
            for i, (r, s) in enumerate(zip(rv, sv)):
                ax.text(i-w/2, r+0.01, f'{r:.3f}', ha='center', fontsize=8)
                ax.text(i+w/2, s+0.01, f'{s:.3f}', ha='center', fontsize=8)

        fig.suptitle('Per-Organization Performance: Real vs CTGAN-Augmented',
                     fontsize=15, fontweight='bold')
        path = f"{self.out}/{out_name}"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"💾 Saved: {path}")

    # ── 6. Comprehensive summary panel ────────────────────────────────────────
    def plot_summary_panel(self, real_gm: Dict, synth_gm: Dict, ctgan_eval: Dict,
                           out_name: str = "06_summary_panel.png"):
        fig = plt.figure(figsize=(22, 16))
        gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

        metric_keys   = ['accuracy','precision','recall','f1_score','auc_roc','specificity','sensitivity']
        metric_labels = ['Acc','Prec','Recall','F1','AUC','Spec','Sens']

        # 1. Spider / radar chart
        ax0 = fig.add_subplot(gs[0, :2])
        real_v  = [real_gm.get(k,0)  for k in metric_keys]
        synth_v = [synth_gm.get(k,0) for k in metric_keys]
        x_pos = np.arange(len(metric_labels))
        w = 0.3
        ax0.bar(x_pos-w/2, real_v,  w, color='#2E86AB', label='Real only', alpha=0.85, edgecolor='white')
        ax0.bar(x_pos+w/2, synth_v, w, color='#E63946', label='+ CTGAN',   alpha=0.85, edgecolor='white')
        ax0.set_xticks(x_pos); ax0.set_xticklabels(metric_labels)
        ax0.set_ylim(0, 1.15); ax0.set_title('All Metrics: Global Model Comparison', fontweight='bold')
        ax0.legend(); ax0.grid(True, axis='y', alpha=0.3)

        # 2. Delta (improvement from CTGAN)
        ax1 = fig.add_subplot(gs[0, 2:])
        deltas = [synth_gm.get(k,0) - real_gm.get(k,0) for k in metric_keys]
        colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in deltas]
        ax1.bar(metric_labels, deltas, color=colors, alpha=0.85, edgecolor='white')
        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_ylabel('Δ (CTGAN − Real)'); ax1.set_title('Performance Delta from CTGAN Augmentation', fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.3)
        for i, d in enumerate(deltas):
            ax1.text(i, d + (0.002 if d >= 0 else -0.005), f'{d:+.4f}',
                     ha='center', fontsize=9, fontweight='bold')

        # 3. TP/TN/FP/FN comparison
        ax2 = fig.add_subplot(gs[1, :2])
        cm_cats = ['TP','TN','FP','FN']
        rv_ = [real_gm['true_positives'], real_gm['true_negatives'],
               real_gm['false_positives'], real_gm['false_negatives']]
        sv_ = [synth_gm['true_positives'], synth_gm['true_negatives'],
               synth_gm['false_positives'], synth_gm['false_negatives']]
        x2 = np.arange(4)
        ax2.bar(x2-w/2, rv_, w, color='#2E86AB', label='Real only', alpha=0.85, edgecolor='white')
        ax2.bar(x2+w/2, sv_, w, color='#E63946', label='+ CTGAN',   alpha=0.85, edgecolor='white')
        ax2.set_xticks(x2); ax2.set_xticklabels(cm_cats)
        ax2.set_title('Confusion Matrix Counts', fontweight='bold'); ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)

        # 4. CTGAN quality summary
        ax3 = fig.add_subplot(gs[1, 2:])
        ks_pct = ctgan_eval['ks']['pct_pass']
        disc_auc = ctgan_eval['discriminator']['auc_roc']
        disc_acc = ctgan_eval['discriminator']['accuracy']
        q_metrics = ['KS Pass %', 'Disc AUC\n(↓ better)', 'Disc Acc\n(↓ better)']
        q_values  = [ks_pct/100, disc_auc, disc_acc]
        q_colors  = ['#2ecc71' if v < 0.6 else '#f39c12' if v < 0.75 else '#e74c3c'
                     for v in q_values]
        q_colors[0] = '#2ecc71' if ks_pct > 50 else '#e74c3c'
        bars = ax3.bar(q_metrics, q_values, color=q_colors, alpha=0.85, edgecolor='white')
        ax3.set_ylim(0, 1.1); ax3.set_title('CTGAN Quality Metrics', fontweight='bold')
        ax3.axhline(0.5, color='green', linestyle='--', alpha=0.7, label='Ideal ~0.5 / ~50%')
        ax3.legend(fontsize=9)
        for bar, v in zip(bars, q_values):
            ax3.text(bar.get_x()+bar.get_width()/2, v+0.02, f'{v:.3f}', ha='center', fontweight='bold')

        # 5. Key findings text box
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        best = "CTGAN-Augmented" if synth_gm['f1_score'] > real_gm['f1_score'] else "Real-Only"
        f1_delta = synth_gm['f1_score'] - real_gm['f1_score']
        auc_delta = synth_gm['auc_roc'] - real_gm['auc_roc']
        grade = ("EXCELLENT" if disc_auc < 0.55 and ks_pct > 70 else
                 "GOOD" if disc_auc < 0.65 and ks_pct > 40 else
                 "FAIR" if disc_auc < 0.75 else "POOR")
        summary = (
            f"KEY FINDINGS\n"
            f"{'─'*80}\n"
            f"  Best performing system : {best}\n"
            f"  F1 delta (CTGAN−Real)  : {f1_delta:+.4f}  |  AUC delta: {auc_delta:+.4f}\n"
            f"  CTGAN data quality     : {grade}  (Disc AUC={disc_auc:.4f}, KS Pass={ks_pct:.1f}%)\n"
            f"  Real-only final metrics: Acc={real_gm['accuracy']:.4f}  F1={real_gm['f1_score']:.4f}  AUC={real_gm['auc_roc']:.4f}\n"
            f"  CTGAN final metrics    : Acc={synth_gm['accuracy']:.4f}  F1={synth_gm['f1_score']:.4f}  AUC={synth_gm['auc_roc']:.4f}\n"
            f"{'─'*80}\n"
            f"  Interpretation: {'CTGAN augmentation IMPROVED detection' if f1_delta > 0.005 else 'CTGAN augmentation had NEUTRAL effect' if abs(f1_delta) <= 0.005 else 'CTGAN augmentation HURT detection (check synthetic data quality)'}"
        )
        ax4.text(0.01, 0.98, summary, transform=ax4.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8, edgecolor='#adb5bd'))

        fig.suptitle('Federated IDS – Comprehensive Evaluation Summary',
                     fontsize=16, fontweight='bold', y=1.01)
        path = f"{self.out}/{out_name}"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"💾 Saved: {path}")

    # ── 7. Privacy budget ────────────────────────────────────────────────────
    def plot_privacy_budget(self, orgs_real: List, orgs_synth: List,
                            out_name: str = "07_privacy_budget.png"):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, orgs, title, color in zip(
                axes, [orgs_real, orgs_synth],
                ['Real-Only', 'CTGAN-Augmented'], ['#2E86AB', '#E63946']):
            for org in orgs:
                rounds_  = [m['round'] for m in org.metrics_hist]
                privs    = [m['privacy_budget_used'] for m in org.metrics_hist]
                ax.plot(rounds_, privs, 'o-', linewidth=2,
                        label=org.profile['name'].split()[0], alpha=0.8)
            ax.set_xlabel('Round'); ax.set_ylabel('Cumulative ε Used')
            ax.set_title(f'DP Budget – {title}', fontweight='bold')
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        fig.suptitle('Differential Privacy Budget Consumption', fontsize=14, fontweight='bold')
        path = f"{self.out}/{out_name}"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"💾 Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════
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
        X_real, y_real, feat_names = dm.load_real(self.real_csv)
        X_synth, y_synth           = dm.load_synthetic(self.synthetic_csv)

        # 2. CTGAN quality evaluation ──────────────────────────────────────────
        evaluator = CTGANDataEvaluator(self.config)
        # build small DataFrames with feature names for evaluator
        real_eval_df  = pd.DataFrame(X_real[:self.config.ctgan_eval_sample], columns=feat_names)
        synth_eval_df = pd.DataFrame(X_synth[:self.config.ctgan_eval_sample], columns=feat_names)
        ctgan_quality = evaluator.evaluate(real_eval_df, synth_eval_df, "CTGAN Quality vs Real CICIDS-2017")
        self.viz.plot_ctgan_quality(ctgan_quality)

        # 3. Train/test split ──────────────────────────────────────────────────
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_real, y_real, test_size=self.config.test_size,
            random_state=self.config.random_seed, stratify=y_real)

        # distribute real data across organizations
        org_keys   = list(ORG_PROFILES.keys())
        splits_tr  = np.array_split(np.random.permutation(len(X_tr)), self.config.n_organizations)
        splits_te  = np.array_split(np.random.permutation(len(X_te)), self.config.n_organizations)
        test_sets  = [(X_te[idx], y_te[idx]) for idx in splits_te]

        input_dim = X_real.shape[1]

        # 4. Run A: Real-only ──────────────────────────────────────────────────
        print("\n" + "─"*60)
        print("  🏃 Experiment A: Real-Only Federated IDS")
        print("─"*60)
        real_metrics, real_orgs, real_gm = self._run_federation(
            input_dim, X_tr, y_tr, X_te, y_te, splits_tr, test_sets,
            use_synthetic=False)

        # 5. Run B: CTGAN augmented ────────────────────────────────────────────
        print("\n" + "─"*60)
        print("  🤖 Experiment B: CTGAN-Augmented Federated IDS")
        print("─"*60)
        synth_metrics, synth_orgs, synth_gm = self._run_federation(
            input_dim, X_tr, y_tr, X_te, y_te, splits_tr, test_sets,
            use_synthetic=True, X_synth=X_synth, y_synth=y_synth)

        # 6. Collect per-org ROC data ──────────────────────────────────────────
        org_roc_real  = self._collect_org_roc(real_orgs)
        org_roc_synth = self._collect_org_roc(synth_orgs)

        # 7. All visualizations ────────────────────────────────────────────────
        self.viz.plot_convergence(real_metrics, synth_metrics)
        self.viz.plot_confusion_matrices(real_gm, synth_gm)
        self.viz.plot_roc_curves(real_gm, synth_gm, org_roc_real, org_roc_synth)
        self.viz.plot_org_breakdown(real_orgs, synth_orgs)
        self.viz.plot_summary_panel(real_gm, synth_gm, ctgan_quality)
        self.viz.plot_privacy_budget(real_orgs, synth_orgs)

        # 8. Save JSON results ────────────────────────────────────────────────
        results = {
            'config'          : vars(self.config),
            'ctgan_quality'   : {k:v for k,v in ctgan_quality.items() if k != 'moments'},
            'real_only'       : {'final': {k:v for k,v in real_gm.items()
                                           if k not in ('all_probs','all_labels')}},
            'ctgan_augmented' : {'final': {k:v for k,v in synth_gm.items()
                                           if k not in ('all_probs','all_labels')}},
            'duration_seconds': round(time.time() - t0, 1),
            'timestamp'       : datetime.now().isoformat(),
        }
        json_path = f"{self.out_dir}/experiment_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self._print_final_summary(results, time.time() - t0)
        return results

    # ─────────────────────────────────────────────────────────────────────────
    def _run_federation(self, input_dim, X_tr, y_tr, X_te, y_te,
                        splits_tr, test_sets,
                        use_synthetic=False, X_synth=None, y_synth=None):
        config = self.config
        global_model = CybersecurityNet(input_dim, config.model_hidden_dims)
        server       = FederatedServer(global_model, config)

        org_keys = list(ORG_PROFILES.keys())
        orgs = []
        for i in range(config.n_organizations):
            key = org_keys[i % len(org_keys)]
            org = OrganizationAgent(str(i), ORG_PROFILES[key], global_model, config, use_synthetic)
            idx = splits_tr[i]
            org.set_data(X_tr[idx], y_tr[idx], X_te, y_te, X_synth, y_synth)
            orgs.append(org)
            mode = "CTGAN-aug" if use_synthetic else "Real-only"
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

    # ─────────────────────────────────────────────────────────────────────────
    def _print_final_summary(self, results: Dict, duration: float):
        r  = results['real_only']['final']
        s  = results['ctgan_augmented']['final']
        cq = results['ctgan_quality']
        print("\n" + "═"*70)
        print("  🏆  EXPERIMENT COMPLETE – FINAL SUMMARY")
        print("═"*70)
        print(f"  {'Metric':<18} {'Real-Only':>12} {'CTGAN-Aug':>12} {'Delta':>10}")
        print(f"  {'─'*18} {'─'*12} {'─'*12} {'─'*10}")
        for key, label in [('accuracy','Accuracy'),('precision','Precision'),
                            ('recall','Recall'),('f1_score','F1-Score'),
                            ('auc_roc','AUC-ROC'),('specificity','Specificity'),
                            ('sensitivity','Sensitivity')]:
            rv, sv = r.get(key, 0), s.get(key, 0)
            d = sv - rv
            arrow = "↑" if d > 0.001 else "↓" if d < -0.001 else "→"
            print(f"  {label:<18} {rv:>12.4f} {sv:>12.4f} {arrow}{d:>+8.4f}")
        print(f"\n  CTGAN Quality : KS={cq['ks']['pct_pass']:.1f}%  Disc-AUC={cq['discriminator']['auc_roc']:.4f}")
        print(f"  Duration      : {duration:.1f}s")
        print(f"  Output dir    : {self.out_dir}/")
        print("═"*70)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    config = FederatedConfig(
        n_organizations    = 5,
        global_rounds      = 20,
        local_epochs       = 3,
        learning_rate      = 0.001,
        batch_size         = 128,
        feature_selection_k= 50,
        model_hidden_dims  = [256, 128, 64],
        dp_epsilon         = 1.0,
        dp_delta           = 1e-5,
        random_seed        = 42,
        ctgan_augment_ratio= 0.30,
        ctgan_eval_sample  = 5000,
        experiment_name    = "federated_ids_ctgan_eval",
    )

    # ── Update these paths to point to your actual files ──────────────────────
    REAL_CSV      = "cicids2017_real.csv"      # full CICIDS-2017 CSV (with Label column)
    SYNTHETIC_CSV = "cicids2017_ctgan.csv"     # CTGAN-generated CSV (same column schema)
    OUTPUT_DIR    = f"results_{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    experiment = FederatedIDSExperiment(config, REAL_CSV, SYNTHETIC_CSV, OUTPUT_DIR)
    results    = experiment.run()
    return results


if __name__ == "__main__":
    main()
