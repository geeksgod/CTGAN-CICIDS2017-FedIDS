from component.imp import *
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
    def plot_convergence(self, real_metrics: List[Dict], trts_metrics: List[Dict], tstr_metrics: List[Dict],
                     out_name: str = "02_training_convergence.png"):
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))
        rounds = range(1, len(real_metrics) + 1)
        
        kpi_map = [
            ('accuracy',  'Accuracy',  axes[0, 0]),
            ('f1_score',  'F1-Score',  axes[0, 1]),
            ('auc_roc',   'AUC-ROC',   axes[0, 2]),
            ('precision', 'Precision', axes[1, 0]),
            ('recall',    'Recall',    axes[1, 1]),
        ]
        
        # 1. Update the Line Plots
        for key, title, ax in kpi_map:
            rv = [m[key] for m in real_metrics]
            trv = [m[key] for m in trts_metrics]
            tsv = [m[key] for m in tstr_metrics]
            
            ax.plot(rounds, rv, 'o-', linewidth=2.5, color='#2E86AB', label='Real only')
            ax.plot(rounds, trv, 's--', linewidth=2.5, color='#E63946', label='TRTS')
            ax.plot(rounds, tsv, '^-.', linewidth=2.5, color='#FF9F1C', label='TSTR') # Added orange dash-dot line
            
            ax.set_title(title, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_xlabel('Round')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 2. Update the Final Round Bar Comparison
            ax = axes[1, 2]
            metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
            
            real_final = [real_metrics[-1][m] for m in metrics_names]
            trts_final = [trts_metrics[-1][m] for m in metrics_names]
            tstr_final = [tstr_metrics[-1][m] for m in metrics_names]
            
            x = np.arange(len(metrics_names))
            w = 0.25 # Reduced width slightly so all 3 bars fit cleanly side-by-side
            
            ax.bar(x - w, real_final, w, label='Real only', color='#2E86AB', alpha=0.85, edgecolor='white')
            ax.bar(x, trts_final, w, label='TRTS', color='#E63946', alpha=0.85, edgecolor='white')
            ax.bar(x + w, tstr_final, w, label='TSTR', color='#FF9F1C', alpha=0.85, edgecolor='white')
            
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_names, rotation=30)
            ax.set_ylim(0, 1.1)
            ax.set_title('Final Round Comparison', fontweight='bold')
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add text labels on top of the bars
            for i, (rv, trv, tsv) in enumerate(zip(real_final, trts_final, tstr_final)):
                ax.text(i - w, rv + 0.01, f'{rv:.2f}', ha='center', fontsize=8, fontweight='bold')
                ax.text(i, trv + 0.01, f'{trv:.2f}', ha='center', fontsize=8, fontweight='bold')
                ax.text(i + w, tsv + 0.01, f'{tsv:.2f}', ha='center', fontsize=8, fontweight='bold')

            fig.suptitle('Federated IDS: Training Convergence (Real vs TRTS vs TSTR)',
                        fontsize=15, fontweight='bold')
                        
            path = f"{self.out}/{out_name}"
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            print(f"💾 Saved: {path}")
        
    # ── 3. Confusion matrices ─────────────────────────────────────────────────
    def plot_confusion_matrices(self, real_gm: Dict, trts_gm: Dict, tstr_gm: Dict,
                            out_name: str = "03_confusion_matrices.png"):
            # Expanded figsize from (14, 6) to (20, 6) to hold 3 subplots cleanly
            fig, axes = plt.subplots(1, 3, figsize=(20, 6)) 
            
            zip_data = zip(
                axes,
                [real_gm, trts_gm, tstr_gm],
                ['Global Model – Real Only', 'Global Model – TRTS', 'Global Model – TSTR'],
                ['Blues', 'Reds', 'Oranges'] # Distinct color mapping for easy reading
            )
            
            for ax, gm, title, cmap in zip_data:
                cm = np.array([[gm['true_negatives'],  gm['false_positives']],
                            [gm['false_negatives'], gm['true_positives']]])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                            xticklabels=['Benign','Attack'], yticklabels=['Benign','Attack'],
                            linewidths=1, linecolor='white', cbar_kws={'label':'Count'})
                
                ax.set_title(f'{title}\nAcc={gm["accuracy"]:.4f}  F1={gm["f1_score"]:.4f}  AUC={gm["auc_roc"]:.4f}',
                            fontweight='bold')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                
            fig.suptitle('Confusion Matrix Comparison', fontsize=15, fontweight='bold')
            path = f"{self.out}/{out_name}"
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            print(f"💾 Saved: {path}")

    # ── 4. ROC curves ─────────────────────────────────────────────────────────
    def plot_roc_curves(self, real_gm: Dict, trts_gm: Dict, tstr_gm: Dict,
                org_results_real: List[Dict], org_results_trts: List[Dict], org_results_tstr: List[Dict],
                out_name: str = "04_roc_curves.png"):
        # Expanded width to 24 to comfortably fit 3 panels without overlapping labels
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
        zip_data = zip(
            axes,
            [real_gm, trts_gm, tstr_gm],
            [org_results_real, org_results_trts, org_results_tstr],
            ['Real-Only Federated IDS', 'TRTS Federated IDS', 'TSTR Federated IDS']
        )
        
        for ax, gm, org_res, title in zip_data:
            colors = plt.cm.tab10(np.linspace(0, 1, len(org_res) + 1))
            
            # Plot local organizational curves
            for i, om in enumerate(org_res):
                if 'fpr' in om:
                    ax.plot(om['fpr'], om['tpr'], linewidth=1.5,
                            color=colors[i], alpha=0.7, label=f"{om['name']} (AUC={om['auc']:.3f})")
                    
            # Global ROC from stored probabilities
            if 'all_probs' in gm and 'all_labels' in gm:
                fpr, tpr, _ = roc_curve(gm['all_labels'], gm['all_probs'])
                ax.plot(fpr, tpr, linewidth=3, color=colors[-1],
                        label=f'Global Model (AUC={gm["auc_roc"]:.3f})')
                
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(title, fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
        fig.suptitle('ROC Curve Comparison', fontsize=15, fontweight='bold')
        path = f"{self.out}/{out_name}"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"💾 Saved: {path}")

    # ── 5. Org-level breakdown ─────────────────────────────────────────────────
    def plot_org_breakdown(self, orgs_real: List, orgs_trts: List, orgs_tstr: List,
                       out_name: str = "05_org_breakdown.png"):
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        names = [o.profile['name'].split()[0] for o in orgs_real]
        
        x = np.arange(len(names))
        w = 0.25  # Slimmed down to let 3 bars cluster comfortably side-by-side

        for ax, metric, title in zip(
                axes.flatten(),
                ['accuracy', 'f1_score', 'auc_roc', 'recall'],
                ['Accuracy', 'F1-Score', 'AUC-ROC', 'Recall']):
            
            # Pull performance metrics across all tracks
            rv = [o.evaluate()[metric] for o in orgs_real]
            trv = [o.evaluate()[metric] for o in orgs_trts]
            tsv = [o.evaluate()[metric] for o in orgs_tstr]
            
            # Center the groups around x: left (x-w), middle (x), and right (x+w)
            ax.bar(x - w, rv,  w, label='Real only', color='#2E86AB', alpha=0.85, edgecolor='white')
            ax.bar(x,     trv, w, label='TRTS',      color='#E63946', alpha=0.85, edgecolor='white')
            ax.bar(x + w, tsv, w, label='TSTR',      color='#FF9F1C', alpha=0.85, edgecolor='white')
            
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=30)
            ax.set_ylim(0, 1.1)
            ax.set_title(f'Per-Org {title}', fontweight='bold')
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
            
            # Precision clipped to .2f to avoid text overlapping over dense bar clusters
            for i, (r, tr, ts) in enumerate(zip(rv, trv, tsv)):
                ax.text(i - w, r + 0.01,  f'{r:.2f}',  ha='center', fontsize=8)
                ax.text(i,     tr + 0.01, f'{tr:.2f}', ha='center', fontsize=8)
                ax.text(i + w, ts + 0.01, f'{ts:.2f}', ha='center', fontsize=8)

        fig.suptitle('Per-Organization Performance Breakdown', fontsize=16, fontweight='bold')
        path = f"{self.out}/{out_name}"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"💾 Saved: {path}")

    # ── 6. Comprehensive summary panel ────────────────────────────────────────
    def plot_summary_panel(self, real_gm: Dict, trts_gm: Dict, tstr_gm: Dict,
                       out_name: str = "06_summary_panel.png"):
        fig = plt.figure(figsize=(22, 16))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

        metric_keys   = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'specificity', 'sensitivity']
        metric_labels = ['Acc', 'Prec', 'Recall', 'F1', 'AUC', 'Spec', 'Sens']
        x_pos = np.arange(len(metric_labels))
        w = 0.25  # Adjusted for clean 3-bar positioning

        # 1. Global Model Comparison (All Metrics)
        ax0 = fig.add_subplot(gs[0, :2])
        real_v = [real_gm.get(k, 0) for k in metric_keys]
        trts_v = [trts_gm.get(k, 0) for k in metric_keys]
        tstr_v = [tstr_gm.get(k, 0) for k in metric_keys]
        
        ax0.bar(x_pos - w, real_v, w, color='#2E86AB', label='Real only', alpha=0.85, edgecolor='white')
        ax0.bar(x_pos,     trts_v, w, color='#E63946', label='TRTS',      alpha=0.85, edgecolor='white')
        ax0.bar(x_pos + w, tstr_v, w, color='#FF9F1C', label='TSTR',      alpha=0.85, edgecolor='white')
        
        ax0.set_xticks(x_pos)
        ax0.set_xticklabels(metric_labels)
        ax0.set_ylim(0, 1.15)
        ax0.set_title('All Metrics: Global Model Comparison', fontweight='bold')
        ax0.legend()
        ax0.grid(True, axis='y', alpha=0.3)

        # 2. Performance Deltas (TRTS and TSTR vs Real Baseline)
        ax1 = fig.add_subplot(gs[0, 2:])
        trts_deltas = [trts_gm.get(k, 0) - real_gm.get(k, 0) for k in metric_keys]
        tstr_deltas = [tstr_gm.get(k, 0) - real_gm.get(k, 0) for k in metric_keys]
        
        # Sub-bars within delta axis
        ax1.bar(x_pos - w/2, trts_deltas, w, label='Δ TRTS', color='#E63946', alpha=0.85, edgecolor='white')
        ax1.bar(x_pos + w/2, tstr_deltas, w, label='Δ TSTR', color='#FF9F1C', alpha=0.85, edgecolor='white')
        ax1.axhline(0, color='black', linewidth=1)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(metric_labels)
        ax1.set_ylabel('Δ (Variant − Real Only)')
        ax1.set_title('Performance Delta Over Real Baseline', fontweight='bold')
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Annotate deltas safely without clashing text
        for i, (td, sd) in enumerate(zip(trts_deltas, tstr_deltas)):
            ax1.text(i - w/2, td + (0.005 if td >= 0 else -0.015), f'{td:+.2f}', ha='center', fontsize=8)
            ax1.text(i + w/2, sd + (0.005 if sd >= 0 else -0.015), f'{sd:+.2f}', ha='center', fontsize=8)

        # 3. TP/TN/FP/FN Matrix Count Comparison
        ax2 = fig.add_subplot(gs[1, :2])
        cm_cats = ['TP', 'TN', 'FP', 'FN']
        x2 = np.arange(4)
        
        rv_  = [real_gm['true_positives'],  real_gm['true_negatives'],  real_gm['false_positives'],  real_gm['false_negatives']]
        trv_ = [trts_gm['true_positives'],  trts_gm['true_negatives'],  trts_gm['false_positives'],  trts_gm['false_negatives']]
        tsv_ = [tstr_gm['true_positives'],  tstr_gm['true_negatives'],  tstr_gm['false_positives'],  tstr_gm['false_negatives']]
        
        ax2.bar(x2 - w, rv_,  w, color='#2E86AB', label='Real only', alpha=0.85, edgecolor='white')
        ax2.bar(x2,     trv_, w, color='#E63946', label='TRTS',      alpha=0.85, edgecolor='white')
        ax2.bar(x2 + w, tsv_, w, color='#FF9F1C', label='TSTR',      alpha=0.85, edgecolor='white')
        
        ax2.set_xticks(x2)
        ax2.set_xticklabels(cm_cats)
        ax2.set_title('Confusion Matrix Counts', fontweight='bold')
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)

        # 4. Key Findings Box (Evaluates metrics programmatically)
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Track the highest F1 Score setup
        f1s = {'Real-Only': real_gm['f1_score'], 'TRTS': trts_gm['f1_score'], 'TSTR': tstr_gm['f1_score']}
        best = max(f1s, key=f1s.get)
        
        trts_f1_delta = trts_gm['f1_score'] - real_gm['f1_score']
        tstr_f1_delta = tstr_gm['f1_score'] - real_gm['f1_score']
        
        # Conditional generation interpretation string
        if trts_f1_delta > 0.005 and tstr_f1_delta > 0.005:
            interpretation = "Synthetic variants show solid generalization improvements on cross-testing evaluations."
        elif trts_f1_delta < -0.01 or tstr_f1_delta < -0.01:
            interpretation = "Noticeable degradation present in cross-domain metrics (Evaluate synthetic data fidelity/distribution gap)."
        else:
            interpretation = "Synthetic metrics match baseline performance characteristics closely (Neutral/Stable variance)."

        summary = (
            f"KEY EVALUATION FINDINGS\n"
            f"{'─'*100}\n"
            f"  Top Overall Performer (by F1-Score) : {best}\n"
            f"  TRTS vs Real Baseline F1 Delta       : {trts_f1_delta:+.4f}  |  TSTR vs Real Baseline F1 Delta: {tstr_f1_delta:+.4f}\n"
            f"  Real-Only Reference Global Metrics   : Acc={real_gm['accuracy']:.4f}  F1={real_gm['f1_score']:.4f}  AUC={real_gm['auc_roc']:.4f}\n"
            f"  TRTS Evaluation Track Metrics        : Acc={trts_gm['accuracy']:.4f}  F1={trts_gm['f1_score']:.4f}  AUC={trts_gm['auc_roc']:.4f}\n"
            f"  TSTR Evaluation Track Metrics        : Acc={tstr_gm['accuracy']:.4f}  F1={tstr_gm['f1_score']:.4f}  AUC={tstr_gm['auc_roc']:.4f}\n"
            f"{'─'*100}\n"
            f"  System Interpretation: {interpretation}"
        )
        
        ax4.text(0.01, 0.98, summary, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8, edgecolor='#adb5bd'))

        fig.suptitle('Federated IDS – Comprehensive Evaluation Summary', fontsize=16, fontweight='bold', y=1.01)
        path = f"{self.out}/{out_name}"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"💾 Saved: {path}")

    # ── 7. Privacy budget ────────────────────────────────────────────────────
    def plot_privacy_budget(self, orgs_real: List, orgs_trts: List, orgs_tstr: List,
                        out_name: str = "07_privacy_budget.png"):
        # Expanded from (16, 6) to (24, 6) to keep 3 square-ish subplots crisp
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        
        zip_data = zip(
            axes, 
            [orgs_real, orgs_trts, orgs_tstr],
            ['Real-Only', 'TRTS', 'TSTR']
        )
        
        for ax, orgs, title in zip_data:
            for org in orgs:
                rounds_ = [m['round'] for m in org.metrics_hist]
                privs   = [m['privacy_budget_used'] for m in org.metrics_hist]
                
                # Retained your original per-organization line color splitting style
                ax.plot(rounds_, privs, 'o-', linewidth=2,
                        label=org.profile['name'].split()[0], alpha=0.8)
                        
            ax.set_xlabel('Round')
            ax.set_ylabel('Cumulative ε Used') # Displays epsilon consumption track cleanly
            ax.set_title(f'DP Budget – {title}', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
        fig.suptitle('Differential Privacy Budget Consumption', fontsize=14, fontweight='bold')
        path = f"{self.out}/{out_name}"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"💾 Saved: {path}")