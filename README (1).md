# 🛍️ StyleKart — Customer Intelligence Platform

### SBR Session 5 | Customer Segmentation for Organisation Growth | 20 Marks

A Streamlit-powered dashboard that solves three interconnected e-commerce problems using data-driven models.

---

## Three Problems Solved

| # | Problem | Method | Key Output |
|---|---------|--------|------------|
| 1 | **Customer Segmentation** | RFM Analysis (Recency, Frequency, Monetary) | 7 named segments with campaign recommendations |
| 2 | **Customer Lifetime Value** | BG/NBD + Gamma-Gamma (lifetimes library) | 6-month CLV prediction + tier classification |
| 3 | **Churn Prediction** | Logistic Regression + Random Forest | Churn probability + risk bands + business impact |

**Key Differentiator:** The Integrated Intelligence tab connects all three models into a single Master Customer Table — enabling unified decision-making rather than three siloed analyses.

---

## Tech Stack

- **Python 3.9+**
- **Streamlit** — Interactive dashboard
- **Pandas / NumPy** — Data processing
- **Plotly** — Interactive visualizations
- **Scikit-learn** — ML models (Logistic Regression, Random Forest)
- **Lifetimes** — BG/NBD + Gamma-Gamma for CLV

---

## Project Structure

```
├── app.py                    # Main Streamlit app (5 tabs)
├── requirements.txt          # Dependencies
├── data/
│   └── online_retail_II.csv  # Dataset
├── utils/
│   ├── data_cleaning.py      # Data cleaning pipeline
│   ├── rfm.py                # RFM segmentation + recommendations
│   ├── clv.py                # CLV prediction models
│   └── churn.py              # Churn feature engineering + modelling
└── .streamlit/
    └── config.toml           # Theme configuration
```

---

## Dataset

**Online Retail II** (proxy for StyleKart transaction data)
- ~175K transactional records
- ~4,300 UK customers
- Period: Jan 2010 – Dec 2011

---

## Limitations

- Dataset is UK general retail (not Indian fashion) — used as analytical proxy
- Currency in GBP (1 GBP ≈ ₹105 for conversion)
- Behavioral segmentation only (no demographic data)
- Data from 2010-2011 — methods are timeless, production use would require current data

---

## References

Based on the 9-part "Data-Driven Growth with Python" blog series on Towards Data Science.
