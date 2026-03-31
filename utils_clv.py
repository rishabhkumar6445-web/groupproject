import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def compute_clv(df: pd.DataFrame):
    """Compute CLV using BG/NBD + Gamma-Gamma via lifetimes library."""
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data

    # Build summary data
    summary = summary_data_from_transaction_data(
        df, 'Customer ID', 'InvoiceDate',
        monetary_value_col='Revenue',
        observation_period_end=df['InvoiceDate'].max()
    )

    # Filter: frequency > 0 for Gamma-Gamma, monetary > 0
    summary = summary[summary['frequency'] > 0]
    summary = summary[summary['monetary_value'] > 0]

    # BG/NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])

    # Predict purchases in next 180 days (6 months)
    summary['predicted_purchases_180d'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        180, summary['frequency'], summary['recency'], summary['T']
    )

    # Gamma-Gamma model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(summary['frequency'], summary['monetary_value'])

    summary['predicted_avg_value'] = ggf.conditional_expected_average_profit(
        summary['frequency'], summary['monetary_value']
    )

    # CLV = predicted purchases * predicted avg value
    summary['predicted_clv'] = (
        summary['predicted_purchases_180d'] * summary['predicted_avg_value']
    ).round(2)

    # Historical CLV
    hist_clv = df.groupby('Customer ID')['Revenue'].sum().reset_index()
    hist_clv.columns = ['Customer ID', 'historical_clv']

    summary = summary.reset_index()
    summary = summary.merge(hist_clv, on='Customer ID', how='left')

    # CLV Tiers
    summary['CLV_Tier'] = pd.qcut(
        summary['predicted_clv'], q=[0, 0.4, 0.7, 0.9, 1.0],
        labels=['Bronze', 'Silver', 'Gold', 'Platinum'],
        duplicates='drop'
    )

    return summary, bgf, ggf


def compute_clv_simple(df: pd.DataFrame):
    """Simplified CLV using historical + projected approach (fallback)."""
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    first_purchase = df.groupby('Customer ID')['InvoiceDate'].min()
    tenure_days = (reference_date - first_purchase).dt.days

    customer_stats = df.groupby('Customer ID').agg(
        total_revenue=('Revenue', 'sum'),
        order_count=('Invoice', 'nunique'),
        avg_order_value=('Revenue', 'mean'),
    ).reset_index()

    customer_stats['tenure_days'] = customer_stats['Customer ID'].map(tenure_days)
    customer_stats['tenure_days'] = customer_stats['tenure_days'].clip(lower=1)

    # Daily revenue rate * 180 days
    customer_stats['daily_rate'] = customer_stats['total_revenue'] / customer_stats['tenure_days']
    customer_stats['predicted_clv'] = (customer_stats['daily_rate'] * 180).round(2)
    customer_stats['historical_clv'] = customer_stats['total_revenue'].round(2)

    customer_stats['CLV_Tier'] = pd.qcut(
        customer_stats['predicted_clv'], q=[0, 0.4, 0.7, 0.9, 1.0],
        labels=['Bronze', 'Silver', 'Gold', 'Platinum'],
        duplicates='drop'
    )

    return customer_stats
