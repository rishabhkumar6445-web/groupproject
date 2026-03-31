import pandas as pd
import numpy as np


def load_and_clean_data(filepath: str):
    df = pd.read_csv(filepath, parse_dates=['InvoiceDate'])
    raw_records = len(df)

    df = df.dropna(subset=['Customer ID'])
    df['Customer ID'] = df['Customer ID'].astype(int)
    df = df[df['Country'] == 'United Kingdom'].copy()

    df['is_cancel'] = df['Invoice'].astype(str).str.startswith('C')
    df_clean = df[~df['is_cancel']].copy()
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['Price'] > 0)]
    df_clean['Revenue'] = df_clean['Quantity'] * df_clean['Price']
    df_clean = df_clean[(df_clean['Quantity'] <= 1000) & (df_clean['Price'] <= 500)]

    df_clean['Year'] = df_clean['InvoiceDate'].dt.year
    df_clean['Month'] = df_clean['InvoiceDate'].dt.month
    df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.day_name()
    df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M').astype(str)

    stats = {
        'raw_records': raw_records,
        'clean_records': len(df_clean),
        'records_removed': raw_records - len(df_clean),
        'pct_removed': round((raw_records - len(df_clean)) / raw_records * 100, 1),
        'unique_customers': df_clean['Customer ID'].nunique(),
        'unique_invoices': df_clean['Invoice'].nunique(),
        'date_min': df_clean['InvoiceDate'].min(),
        'date_max': df_clean['InvoiceDate'].max(),
        'total_revenue': round(df_clean['Revenue'].sum(), 2),
    }
    return df_clean, stats


def get_cancellation_rates(filepath: str):
    df = pd.read_csv(filepath, parse_dates=['InvoiceDate'])
    df = df.dropna(subset=['Customer ID'])
    df['Customer ID'] = df['Customer ID'].astype(int)
    df = df[df['Country'] == 'United Kingdom']
    total = df.groupby('Customer ID')['Invoice'].nunique()
    cancels = df[df['Invoice'].astype(str).str.startswith('C')].groupby('Customer ID')['Invoice'].nunique()
    cancel_rate = (cancels / total).fillna(0)
    return cancel_rate.reset_index().rename(columns={'Invoice': 'cancel_rate'})
