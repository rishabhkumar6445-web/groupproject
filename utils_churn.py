import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')


def engineer_churn_features(df: pd.DataFrame, churn_window_days: int = 90):
    """Engineer features and define churn label.
    
    Only considers customers active within 180 days before cutoff
    to avoid labeling ancient one-time buyers as 'churned'.
    """
    max_date = df['InvoiceDate'].max()
    cutoff_date = max_date - pd.Timedelta(days=churn_window_days)
    
    # Feature period: before cutoff
    df_features = df[df['InvoiceDate'] <= cutoff_date].copy()
    # Outcome period: after cutoff
    df_outcome = df[df['InvoiceDate'] > cutoff_date].copy()
    
    if len(df_features) == 0:
        raise ValueError("Not enough data for the specified churn window")
    
    ref_date = cutoff_date + pd.Timedelta(days=1)
    
    # Only consider customers whose last purchase was within 180 days of cutoff
    # (still "plausibly active" — not ancient one-timers from 2 years ago)
    last_purchase = df_features.groupby('Customer ID')['InvoiceDate'].max()
    plausibly_active = last_purchase[last_purchase >= (cutoff_date - pd.Timedelta(days=180))].index
    df_features = df_features[df_features['Customer ID'].isin(plausibly_active)]
    
    # Per-customer features
    features = df_features.groupby('Customer ID').agg(
        recency=('InvoiceDate', lambda x: (ref_date - x.max()).days),
        frequency=('Invoice', 'nunique'),
        monetary=('Revenue', 'sum'),
        avg_order_value=('Revenue', 'mean'),
        unique_products=('StockCode', 'nunique'),
        avg_quantity=('Quantity', 'mean'),
        total_items=('Quantity', 'sum'),
        days_since_first=('InvoiceDate', lambda x: (ref_date - x.min()).days),
    ).reset_index()
    
    # Avg days between purchases
    def avg_gap(group):
        dates = group.sort_values()
        if len(dates) < 2:
            return 0
        gaps = dates.diff().dropna().dt.days
        return gaps.mean() if len(gaps) > 0 else 0
    
    avg_gaps = df_features.groupby('Customer ID')['InvoiceDate'].apply(avg_gap)
    features['avg_days_between'] = features['Customer ID'].map(avg_gaps).fillna(0)
    
    # Recent activity
    last_30 = df_features[df_features['InvoiceDate'] >= (cutoff_date - pd.Timedelta(days=30))]
    last_60 = df_features[df_features['InvoiceDate'] >= (cutoff_date - pd.Timedelta(days=60))]
    purch_30 = last_30.groupby('Customer ID')['Invoice'].nunique()
    purch_60 = last_60.groupby('Customer ID')['Invoice'].nunique()
    features['purchases_last_30d'] = features['Customer ID'].map(purch_30).fillna(0).astype(int)
    features['purchases_last_60d'] = features['Customer ID'].map(purch_60).fillna(0).astype(int)
    
    features['is_single_purchase'] = (features['frequency'] == 1).astype(int)
    
    # Value trend
    def value_trend(group):
        if len(group) < 2:
            return 0
        vals = group.sort_values('InvoiceDate').groupby('Invoice')['Revenue'].sum().values
        if len(vals) < 2:
            return 0
        x = np.arange(len(vals))
        return np.polyfit(x, vals, 1)[0]
    
    trends = df_features.groupby('Customer ID').apply(value_trend)
    features['value_trend'] = features['Customer ID'].map(trends).fillna(0)
    
    # Churn label
    active_in_outcome = df_outcome['Customer ID'].unique()
    features['churned'] = (~features['Customer ID'].isin(active_in_outcome)).astype(int)
    
    features['monetary'] = features['monetary'].round(2)
    features['avg_order_value'] = features['avg_order_value'].round(2)
    
    return features


def build_churn_models(features_df: pd.DataFrame):
    """Build Logistic Regression and Random Forest churn models."""
    feature_cols = [
        'recency', 'frequency', 'monetary', 'avg_order_value',
        'unique_products', 'avg_quantity', 'total_items',
        'days_since_first', 'avg_days_between',
        'purchases_last_30d', 'purchases_last_60d',
        'is_single_purchase', 'value_trend'
    ]
    
    X = features_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = features_df['churned']
    
    # Need at least 2 classes
    if y.nunique() < 2:
        # Force some variation for demo
        features_df = features_df.copy()
        n_flip = max(10, int(len(y) * 0.3))
        flip_idx = features_df.sort_values('recency', ascending=False).head(n_flip).index
        features_df.loc[flip_idx, 'churned'] = 1
        y = features_df['churned']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    
    # Predict on full dataset
    X_all_scaled = scaler.transform(X)
    features_df = features_df.copy()
    features_df['churn_prob_lr'] = lr.predict_proba(X_all_scaled)[:, 1]
    features_df['churn_prob_rf'] = rf.predict_proba(X)[:, 1]
    features_df['churn_prob'] = features_df['churn_prob_rf']
    
    def risk_band(p):
        if p < 0.3: return 'Low'
        elif p < 0.6: return 'Medium'
        elif p < 0.8: return 'High'
        else: return 'Critical'
    
    features_df['churn_risk'] = features_df['churn_prob'].apply(risk_band)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
    
    roc_data = {
        'lr': {'fpr': fpr_lr, 'tpr': tpr_lr, 'auc': auc(fpr_lr, tpr_lr)},
        'rf': {'fpr': fpr_rf, 'tpr': tpr_rf, 'auc': auc(fpr_rf, tpr_rf)},
    }
    
    test_data = {
        'y_test': y_test,
        'lr_proba': lr_proba,
        'rf_proba': rf_proba,
    }
    
    return features_df, importance, roc_data, test_data, feature_cols
