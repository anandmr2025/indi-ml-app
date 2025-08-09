from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pandas as pd, numpy as np

def train_rf(df_feat: pd.DataFrame):
    feat_cols = df_feat.columns.drop("Target")
    X, y = df_feat[feat_cols], df_feat["Target"]
    split = int(.8*len(X))
    # Check for empty train/test splits
    if split == 0 or len(X[split:]) == 0:
        raise ValueError("Not enough data to split into train/test sets. Please provide more data.")
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X[:split], y[:split])
    preds = rf.predict(X[split:])
    rmse = np.sqrt(mean_squared_error(y[split:], preds))
    mae  = mean_absolute_error(y[split:], preds)
    r2   = r2_score(y[split:], preds)
    return rf, (rmse, mae, r2), preds, y[split:]
 

if __name__ == "__main__":
    import sys
    import os
    # Ensure the project root is in sys.path for module resolution
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    import pandas as pd
    from indi_ml.ingest import history
    from indi_ml.features import enrich

    # Increase the period to get more data for training/testing
    df = history("TCS", period="12mo")  # Changed from "2mo" to "6mo"
    feat_df = enrich(df)
    feat_df["Target"] = feat_df["Close"].shift(-1)
    feat_df.dropna(inplace=True)
    if len(feat_df) < 10:
        raise ValueError("Not enough rows in feature dataframe after preprocessing. Please use a longer period or check your data source.")
    model, metrics, preds, actual = train_rf(feat_df)
    print("Metrics (RMSE, MAE, R2):", metrics)
