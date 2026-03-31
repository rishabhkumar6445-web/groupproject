import pandas as pd
import numpy as np


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('Revenue', 'sum')
    ).reset_index()

    rfm['Monetary'] = rfm['Monetary'].round(2)

    # Score 1-4 using percentile-based approach (handles duplicates)
    def score_column(col, ascending=True):
        """Score column into 1-4 using percentile ranks."""
        pct = col.rank(pct=True)
        scores = pd.cut(pct, bins=[0, 0.25, 0.5, 0.75, 1.0],
                        labels=[1, 2, 3, 4] if ascending else [4, 3, 2, 1],
                        include_lowest=True)
        return scores.astype(int)

    rfm['Frequency_Score'] = score_column(rfm['Frequency'], ascending=True)
    rfm['Monetary_Score'] = score_column(rfm['Monetary'], ascending=True)
    rfm['Recency_Score'] = score_column(rfm['Recency'], ascending=False)  # lower recency = better

    rfm['RFM_Score'] = (rfm['Recency_Score'].astype(str) +
                         rfm['Frequency_Score'].astype(str) +
                         rfm['Monetary_Score'].astype(str))

    def segment(row):
        r, f, m = row['Recency_Score'], row['Frequency_Score'], row['Monetary_Score']
        if r >= 3 and f >= 3 and m >= 3:
            if r == 4 and f == 4 and m == 4:
                return 'Champions'
            return 'Loyal Customers'
        elif r >= 3 and f <= 2:
            return 'Potential Loyalists'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        elif r <= 2 and f <= 2 and m >= 3:
            return 'About to Sleep'
        elif r == 1 and f == 1 and m == 1:
            return 'Lost'
        else:
            return 'Hibernating'

    rfm['Segment'] = rfm.apply(segment, axis=1)
    return rfm


SEGMENT_COLORS = {
    'Champions': '#2ecc71',
    'Loyal Customers': '#3498db',
    'Potential Loyalists': '#f39c12',
    'At Risk': '#e74c3c',
    'About to Sleep': '#9b59b6',
    'Hibernating': '#95a5a6',
    'Lost': '#34495e',
}

SEGMENT_RECOMMENDATIONS = {
    'Champions': [
        'Early access to new collections & limited editions',
        'VIP referral program with exclusive rewards',
        'Invite to brand ambassador program',
    ],
    'Loyal Customers': [
        'Loyalty tier upgrade nudge (Silver → Gold)',
        'Cross-sell complementary categories',
        'Birthday/anniversary special offers',
    ],
    'Potential Loyalists': [
        'Second purchase incentive: 15% off within 14 days',
        'Personalized product recommendations via email',
        'Welcome series with brand story & styling tips',
    ],
    'At Risk': [
        'URGENT win-back: "We miss you" + 20% off, 7-day validity',
        'Survey: "What can we improve?" with incentive',
        'Retarget with best-sellers from their preferred category',
    ],
    'About to Sleep': [
        'Gentle reminder email with new arrivals',
        'Flash sale notification push',
        'Cart-abandonment style re-engagement',
    ],
    'Hibernating': [
        'Deep discount (25-30% off) re-engagement campaign',
        'Seasonal sale alerts only (reduce email frequency)',
        'Consider excluding from paid campaigns to save budget',
    ],
    'Lost': [
        'Paid social retargeting with fresh creatives',
        'Final win-back attempt or exclude from all campaigns',
        'Reallocate budget to Potential Loyalists instead',
    ],
}
