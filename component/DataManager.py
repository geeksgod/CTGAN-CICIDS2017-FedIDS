from component.imp import *
class DataManager:
    def __init__(self, config: FederatedConfig):
        self.config   = config
        self.scaler   = StandardScaler()
        self.selector = SelectKBest(mutual_info_classif, k=config.feature_selection_k)

    def load_data(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        print(f"📂 Loading real data from: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes('object').columns:
            df[col] = df[col].str.strip()
        return self._preprocess(df)

    

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
