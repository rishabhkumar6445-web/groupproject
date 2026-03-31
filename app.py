import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_cleaning import load_and_clean_data
from utils.rfm import compute_rfm, SEGMENT_COLORS, SEGMENT_RECOMMENDATIONS
from utils.clv import compute_clv, compute_clv_simple
from utils.churn import engineer_churn_features, build_churn_models

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="StyleKart · Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; max-width: 1200px; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 20px; text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-card h3 { color: #8892b0; font-size: 0.78rem; margin: 0;
        text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h1 { color: #e6f1ff; font-size: 1.7rem; margin: 4px 0 0 0; }
    .segment-card {
        background: #0d1117; border-radius: 10px; padding: 16px;
        border-left: 4px solid; margin-bottom: 8px;
    }
    .segment-card h4 { margin: 0 0 8px 0; color: #e6f1ff; }
    .segment-card p { margin: 2px 0; color: #8892b0; font-size: 0.85rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #161b22; border-radius: 8px; padding: 10px 20px;
        color: #8892b0; border: 1px solid #30363d;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
        color: white !important; border: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═══════════════════════════════════════════════════════════════
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "online_retail_II.csv")


@st.cache_data(show_spinner=False)
def load_data():
    return load_and_clean_data(DATA_PATH)


@st.cache_data(show_spinner=False)
def get_rfm(df_json):
    df = pd.read_json(df_json)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return compute_rfm(df)


@st.cache_data(show_spinner=False)
def get_clv(df_json):
    df = pd.read_json(df_json)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    try:
        result, _, _ = compute_clv(df)
        return result, True
    except Exception:
        return compute_clv_simple(df), False


@st.cache_data(show_spinner=False)
def get_churn(df_json):
    df = pd.read_json(df_json)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    features = engineer_churn_features(df, churn_window_days=90)
    preds, importance, roc, test, cols = build_churn_models(features)
    return preds, importance, roc, test


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><h3>{label}</h3><h1>{value}</h1></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛍️ StyleKart")
    st.markdown("**Customer Intelligence Platform**")
    st.markdown("---")

    with st.spinner("Loading dataset…"):
        df_clean, data_stats = load_data()

    date_min = df_clean["InvoiceDate"].min().date()
    date_max = df_clean["InvoiceDate"].max().date()

    st.markdown("##### 📅 Date Filter")
    date_range = st.date_input("Select range", value=(date_min, date_max),
                               min_value=date_min, max_value=date_max)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        mask = (df_clean["InvoiceDate"].dt.date >= date_range[0]) & \
               (df_clean["InvoiceDate"].dt.date <= date_range[1])
        df_filtered = df_clean[mask].copy()
    else:
        df_filtered = df_clean.copy()

    st.markdown("---")
    st.markdown("##### 📊 Quick Stats")
    st.markdown(f"**Records:** {len(df_filtered):,}")
    st.markdown(f"**Customers:** {df_filtered['Customer ID'].nunique():,}")
    st.markdown(f"**Revenue:** £{df_filtered['Revenue'].sum():,.0f}")
    st.markdown("---")

    with st.expander("ℹ️ About This Project"):
        st.markdown("""
        **SBR Session 5 — Customer Segmentation for Organisation Growth**

        Three interconnected models for StyleKart (fashion e-commerce):
        1. RFM Customer Segmentation
        2. Customer Lifetime Value Prediction
        3. Churn Prediction

        **Dataset:** Online Retail II (proxy for StyleKart)

        **Limitations:** UK general retail 2010-2011, behavioral segmentation only.
        """)

df_json = df_filtered.to_json()

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Overview & EDA",
    "🎯 RFM Segmentation",
    "💰 Customer Lifetime Value",
    "⚠️ Churn Prediction",
    "🧠 Integrated Intelligence",
])


# ───────────────────────────────────────────────────────────────
# TAB 0 — OVERVIEW & EDA
# ───────────────────────────────────────────────────────────────
with tab0:
    st.markdown("# 📋 Overview & Exploratory Analysis")

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Customers", f"{df_filtered['Customer ID'].nunique():,}")
    with c2: metric_card("Total Revenue", f"£{df_filtered['Revenue'].sum():,.0f}")
    with c3: metric_card("Avg Order Value", f"£{df_filtered.groupby('Invoice')['Revenue'].sum().mean():,.2f}")
    with c4: metric_card("Total Orders", f"{df_filtered['Invoice'].nunique():,}")

    st.markdown("####")

    with st.expander("🧹 Data Cleaning Summary", expanded=False):
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Raw Records", f"{data_stats['raw_records']:,}")
        cc2.metric("Clean Records", f"{data_stats['clean_records']:,}")
        cc3.metric("Removed", f"{data_stats['records_removed']:,} ({data_stats['pct_removed']}%)")

    # --- Charts ---
    ca, cb = st.columns(2)
    with ca:
        monthly = df_filtered.groupby("YearMonth")["Revenue"].sum().reset_index().sort_values("YearMonth")
        fig = px.line(monthly, x="YearMonth", y="Revenue", title="Monthly Revenue Trend",
                      template="plotly_dark", labels={"YearMonth": "Month", "Revenue": "Revenue (£)"})
        fig.update_traces(line=dict(color="#388bfd", width=3))
        fig.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        ofreq = df_filtered.groupby("Customer ID")["Invoice"].nunique().clip(upper=20).reset_index()
        ofreq.columns = ["Customer ID", "Orders"]
        fig = px.histogram(ofreq, x="Orders", nbins=20, title="Order Frequency Distribution",
                           template="plotly_dark")
        fig.update_traces(marker_color="#f39c12")
        fig.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    cc, cd = st.columns(2)
    with cc:
        cr = df_filtered.groupby("Customer ID")["Revenue"].sum().sort_values(ascending=False).reset_index()
        cr["cum_pct"] = cr["Revenue"].cumsum() / cr["Revenue"].sum() * 100
        cr["cust_pct"] = np.arange(1, len(cr) + 1) / len(cr) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cr["cust_pct"], y=cr["cum_pct"], mode="lines",
                                 line=dict(color="#2ecc71", width=3), name="Cumulative %"))
        fig.add_hline(y=80, line_dash="dash", line_color="#e74c3c", annotation_text="80% Revenue")
        fig.update_layout(title="Customer Revenue Concentration (Pareto)",
                          xaxis_title="% of Customers", yaxis_title="% of Revenue",
                          template="plotly_dark", height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with cd:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        drev = df_filtered.groupby("DayOfWeek")["Revenue"].sum().reindex(dow_order).reset_index()
        fig = px.bar(drev, x="DayOfWeek", y="Revenue", title="Revenue by Day of Week",
                     template="plotly_dark")
        fig.update_traces(marker_color="#9b59b6")
        fig.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    ce, cf = st.columns(2)
    with ce:
        fp = df_filtered.groupby("Customer ID")["InvoiceDate"].min().reset_index()
        fp.columns = ["Customer ID", "FirstPurchase"]
        fp["FirstMonth"] = fp["FirstPurchase"].dt.to_period("M").astype(str)
        mc = df_filtered.merge(fp[["Customer ID", "FirstMonth"]], on="Customer ID")
        mc["CustMonth"] = mc["InvoiceDate"].dt.to_period("M").astype(str)
        mc["Type"] = np.where(mc["CustMonth"] == mc["FirstMonth"], "New", "Returning")
        nr = mc.groupby(["CustMonth", "Type"])["Customer ID"].nunique().reset_index()
        fig = px.bar(nr, x="CustMonth", y="Customer ID", color="Type",
                     title="New vs Returning Customers", template="plotly_dark", barmode="stack",
                     color_discrete_map={"New": "#e74c3c", "Returning": "#3498db"},
                     labels={"Customer ID": "Customers", "CustMonth": "Month"})
        fig.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with cf:
        aov = df_filtered.groupby("Invoice")["Revenue"].sum().clip(upper=500).reset_index()
        fig = px.histogram(aov, x="Revenue", nbins=50, title="Order Value Distribution",
                           template="plotly_dark", labels={"Revenue": "Order Value (£)"})
        fig.update_traces(marker_color="#1abc9c")
        fig.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Cohort Heatmap
    st.markdown("### 📊 Cohort Retention Heatmap")
    fp2 = df_filtered.groupby("Customer ID")["InvoiceDate"].min().reset_index()
    fp2.columns = ["Customer ID", "FP"]
    fp2["Cohort"] = fp2["FP"].dt.to_period("M")
    dfc = df_filtered.merge(fp2[["Customer ID", "Cohort"]], on="Customer ID")
    dfc["TM"] = dfc["InvoiceDate"].dt.to_period("M")
    dfc["CI"] = (dfc["TM"] - dfc["Cohort"]).apply(lambda x: x.n if hasattr(x, "n") else 0)
    ct = dfc.groupby(["Cohort", "CI"])["Customer ID"].nunique().reset_index()
    cp = ct.pivot(index="Cohort", columns="CI", values="Customer ID")
    cs = cp.iloc[:, 0]
    ret = (cp.divide(cs, axis=0) * 100).iloc[:12, :12]
    ret.index = ret.index.astype(str)
    fig = px.imshow(ret.values, x=[f"M+{i}" for i in range(ret.shape[1])],
                    y=ret.index.tolist(), color_continuous_scale="Blues",
                    title="Cohort Retention Rate (%)", aspect="auto",
                    labels=dict(x="Months After First Purchase", y="Cohort", color="Retention %"))
    fig.update_layout(template="plotly_dark", height=450, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ───────────────────────────────────────────────────────────────
# TAB 1 — RFM SEGMENTATION
# ───────────────────────────────────────────────────────────────
with tab1:
    st.markdown("# 🎯 RFM Customer Segmentation")
    st.markdown("Segment customers by **Recency**, **Frequency**, and **Monetary** value to run targeted campaigns.")

    with st.spinner("Computing RFM scores…"):
        rfm = get_rfm(df_json)

    all_segs = sorted(rfm["Segment"].unique().tolist())
    sel_segs = st.multiselect("Filter by Segment", all_segs, default=all_segs, key="rfm_seg")
    rfm_d = rfm[rfm["Segment"].isin(sel_segs)]

    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Customers", f"{len(rfm_d):,}")
    with k2: metric_card("Avg Recency", f"{rfm_d['Recency'].mean():.0f} days")
    with k3: metric_card("Avg Frequency", f"{rfm_d['Frequency'].mean():.1f}")
    with k4: metric_card("Avg Monetary", f"£{rfm_d['Monetary'].mean():,.0f}")

    st.markdown("####")
    r1, r2 = st.columns(2)
    with r1:
        sc = rfm_d["Segment"].value_counts().reset_index()
        sc.columns = ["Segment", "Count"]
        fig = px.bar(sc, x="Segment", y="Count", title="Customer Distribution by Segment",
                     template="plotly_dark", color="Segment", color_discrete_map=SEGMENT_COLORS)
        fig.update_layout(height=400, showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        sr = rfm_d.groupby("Segment")["Monetary"].sum().reset_index()
        fig = px.pie(sr, values="Monetary", names="Segment", title="Revenue Contribution by Segment",
                     template="plotly_dark", color="Segment", color_discrete_map=SEGMENT_COLORS, hole=0.4)
        fig.update_layout(height=400, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    r3, r4 = st.columns(2)
    with r3:
        chars = rfm_d.groupby("Segment").agg(
            Avg_Recency=("Recency", "mean"), Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean")).reset_index()
        melted = chars.melt(id_vars="Segment", var_name="Metric", value_name="Value")
        fig = px.bar(melted, x="Segment", y="Value", color="Metric", barmode="group",
                     title="Segment Characteristics", template="plotly_dark",
                     color_discrete_sequence=["#e74c3c", "#3498db", "#2ecc71"])
        fig.update_layout(height=400, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with r4:
        fig = px.scatter(rfm_d, x="Frequency", y="Monetary", color="Segment",
                         title="Frequency vs Monetary", template="plotly_dark",
                         color_discrete_map=SEGMENT_COLORS, opacity=0.6)
        fig.update_layout(height=400, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Campaign Recommendations")
    for seg in sel_segs:
        if seg in SEGMENT_RECOMMENDATIONS:
            color = SEGMENT_COLORS.get(seg, "#888")
            recs = SEGMENT_RECOMMENDATIONS[seg]
            recs_html = "".join([f"<p>• {r}</p>" for r in recs])
            cnt = len(rfm_d[rfm_d["Segment"] == seg])
            st.markdown(f"""<div class="segment-card" style="border-color: {color};">
                <h4>{seg} ({cnt:,} customers)</h4>{recs_html}</div>""", unsafe_allow_html=True)

    st.download_button("📥 Download RFM Data", rfm_d.to_csv(index=False),
                       "stylekart_rfm.csv", "text/csv")


# ───────────────────────────────────────────────────────────────
# TAB 2 — CUSTOMER LIFETIME VALUE
# ───────────────────────────────────────────────────────────────
with tab2:
    st.markdown("# 💰 Customer Lifetime Value Prediction")
    st.markdown("Predict 6-month CLV to optimize acquisition & retention spend.")

    with st.spinner("Computing CLV…"):
        clv_data, used_bgf = get_clv(df_json)

    if used_bgf:
        st.success("✅ BG/NBD + Gamma-Gamma model fitted successfully")
    else:
        st.info("ℹ️ Using simplified CLV projection (historical rate × 180 days)")

    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Customers", f"{len(clv_data):,}")
    with k2: metric_card("Avg CLV", f"£{clv_data['predicted_clv'].mean():,.0f}")
    with k3: metric_card("Median CLV", f"£{clv_data['predicted_clv'].median():,.0f}")
    with k4: metric_card("Total Predicted", f"£{clv_data['predicted_clv'].sum():,.0f}")

    st.markdown("####")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(clv_data, x="predicted_clv", nbins=50,
                           title="CLV Distribution (6-Month)", template="plotly_dark",
                           labels={"predicted_clv": "Predicted CLV (£)"})
        fig.update_traces(marker_color="#f39c12")
        for tier, col in [("Platinum", "#e74c3c"), ("Gold", "#f39c12"), ("Silver", "#95a5a6")]:
            td = clv_data[clv_data["CLV_Tier"] == tier]
            if not td.empty:
                fig.add_vline(x=td["predicted_clv"].min(), line_dash="dash", line_color=col,
                              annotation_text=tier)
        fig.update_layout(height=400, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cs = clv_data.sort_values("predicted_clv", ascending=False).copy()
        cs["cum"] = cs["predicted_clv"].cumsum() / cs["predicted_clv"].sum() * 100
        cs["cpct"] = np.arange(1, len(cs) + 1) / len(cs) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cs["cpct"], y=cs["cum"], mode="lines", fill="tozeroy",
                                 line=dict(color="#2ecc71", width=3),
                                 fillcolor="rgba(46,204,113,0.15)"))
        fig.add_hline(y=80, line_dash="dash", line_color="#e74c3c", annotation_text="80% Revenue")
        p80 = cs[cs["cum"] >= 80]["cpct"].iloc[0] if len(cs[cs["cum"] >= 80]) > 0 else 50
        fig.add_vline(x=p80, line_dash="dash", line_color="#e74c3c",
                      annotation_text=f"{p80:.0f}% of customers")
        fig.update_layout(title="Revenue Concentration Curve",
                          xaxis_title="% Customers (by CLV)", yaxis_title="% Predicted Revenue",
                          template="plotly_dark", height=400, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        ts = clv_data.groupby("CLV_Tier").agg(count=("predicted_clv", "count"),
             avg_clv=("predicted_clv", "mean"), total=("predicted_clv", "sum")).reset_index()
        tord = ["Platinum", "Gold", "Silver", "Bronze"]
        ts["CLV_Tier"] = pd.Categorical(ts["CLV_Tier"], categories=tord, ordered=True)
        ts = ts.sort_values("CLV_Tier")
        tcm = {"Platinum": "#e74c3c", "Gold": "#f39c12", "Silver": "#95a5a6", "Bronze": "#6c5b7b"}
        fig = px.bar(ts, x="CLV_Tier", y="avg_clv", title="Average CLV by Tier",
                     template="plotly_dark", color="CLV_Tier", color_discrete_map=tcm,
                     labels={"avg_clv": "Avg CLV (£)"})
        fig.update_layout(height=400, showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.pie(ts, values="count", names="CLV_Tier", title="Customers by Tier",
                     template="plotly_dark", color="CLV_Tier", color_discrete_map=tcm, hole=0.4)
        fig.update_layout(height=400, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧮 Acquisition Budget Calculator")
    cac = st.slider("Customer Acquisition Cost (£)", 1, 50, 5, key="cac")
    bd = ts[["CLV_Tier", "count", "avg_clv"]].copy()
    bd["CAC"] = cac
    bd["CLV:CAC Ratio"] = (bd["avg_clv"] / cac).round(1)
    bd["Profitable?"] = bd["avg_clv"] > cac
    bd["Net Value (£)"] = (bd["avg_clv"] - cac).round(2)
    st.dataframe(bd, use_container_width=True, hide_index=True)

    st.download_button("📥 Download CLV Data", clv_data.to_csv(index=False),
                       "stylekart_clv.csv", "text/csv")


# ───────────────────────────────────────────────────────────────
# TAB 3 — CHURN PREDICTION
# ───────────────────────────────────────────────────────────────
with tab3:
    st.markdown("# ⚠️ Churn Prediction Model")
    st.markdown("Identify customers at risk of leaving — intervene before it's too late.")

    with st.spinner("Training churn models…"):
        churn_data, feat_imp, roc_data, test_data = get_churn(df_json)

    churn_rate = churn_data["churned"].mean() * 100
    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Customers Analyzed", f"{len(churn_data):,}")
    with k2: metric_card("Churn Rate", f"{churn_rate:.1f}%")
    with k3: metric_card("RF Model AUC", f"{roc_data['rf']['auc']:.3f}")
    with k4: metric_card("LR Model AUC", f"{roc_data['lr']['auc']:.3f}")

    st.markdown("####")
    threshold = st.slider("🎚️ Classification Threshold", 0.1, 0.9, 0.5, 0.05,
                           key="thr", help="Adjust to balance precision vs recall")

    h1, h2 = st.columns(2)
    with h1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc_data["rf"]["fpr"], y=roc_data["rf"]["tpr"], mode="lines",
                                 name=f"Random Forest (AUC={roc_data['rf']['auc']:.3f})",
                                 line=dict(color="#2ecc71", width=3)))
        fig.add_trace(go.Scatter(x=roc_data["lr"]["fpr"], y=roc_data["lr"]["tpr"], mode="lines",
                                 name=f"Logistic Reg (AUC={roc_data['lr']['auc']:.3f})",
                                 line=dict(color="#3498db", width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(color="#555", dash="dash")))
        fig.update_layout(title="ROC Curve — Model Comparison",
                          xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          template="plotly_dark", height=420, margin=dict(t=40, b=20),
                          legend=dict(x=0.35, y=0.1))
        st.plotly_chart(fig, use_container_width=True)

    with h2:
        top_f = feat_imp.head(10)
        fig = px.bar(top_f, x="importance", y="feature", orientation="h",
                     title="Top 10 Feature Importance (Random Forest)", template="plotly_dark",
                     color="importance", color_continuous_scale="YlOrRd")
        fig.update_layout(height=420, margin=dict(t=40, b=20), showlegend=False,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    h3, h4 = st.columns(2)
    with h3:
        from sklearn.metrics import confusion_matrix as cm_func
        y_test = test_data["y_test"]
        y_pred = (test_data["rf_proba"] >= threshold).astype(int)
        cm = cm_func(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True,
                        x=["Predicted: Stay", "Predicted: Churn"],
                        y=["Actual: Stay", "Actual: Churn"],
                        title=f"Confusion Matrix (Threshold = {threshold})",
                        template="plotly_dark", color_continuous_scale="Blues", aspect="equal")
        fig.update_layout(height=380, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with h4:
        fig = px.histogram(churn_data, x="churn_prob", nbins=40, color="churned",
                           title="Churn Probability Distribution", template="plotly_dark",
                           color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                           labels={"churn_prob": "Churn Probability", "churned": "Churned"},
                           barmode="overlay", opacity=0.7)
        fig.add_vline(x=threshold, line_dash="dash", line_color="#f39c12",
                      annotation_text=f"Threshold: {threshold}")
        fig.update_layout(height=380, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Business Impact
    st.markdown("### 📊 Business Impact Calculator")
    avg_rev = df_filtered.groupby("Customer ID")["Revenue"].sum().mean()
    cost_int = st.number_input("Cost per Intervention (£)", value=3.0, step=0.5)

    all_pred = (churn_data["churn_prob"] >= threshold).astype(int)
    actual = churn_data["churned"]
    tp = ((all_pred == 1) & (actual == 1)).sum()
    fp = ((all_pred == 1) & (actual == 0)).sum()
    fn = ((all_pred == 0) & (actual == 1)).sum()
    total_int = tp + fp
    cost_total = total_int * cost_int
    rev_saved = tp * avg_rev * 0.3
    net = rev_saved - cost_total

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Churners Caught", f"{tp:,}")
    i2.metric("Churners Missed", f"{fn:,}")
    i3.metric("False Alarms", f"{fp:,}")
    i4.metric("Net ROI", f"£{net:,.0f}",
              delta="Profitable" if net > 0 else "Loss",
              delta_color="normal" if net > 0 else "inverse")

    b1, b2 = st.columns(2)
    b1.metric("Intervention Cost", f"£{cost_total:,.0f}")
    b2.metric("Revenue Saved (est.)", f"£{rev_saved:,.0f}")

    st.markdown("### 🚦 Churn Risk Bands")
    rs = churn_data.groupby("churn_risk").agg(
        Customers=("Customer ID", "count"), Avg_Spend=("monetary", "mean"),
        Avg_Prob=("churn_prob", "mean")).reset_index()
    rord = ["Low", "Medium", "High", "Critical"]
    rs["churn_risk"] = pd.Categorical(rs["churn_risk"], categories=rord, ordered=True)
    rs = rs.sort_values("churn_risk")
    actions = {"Low": "No intervention", "Medium": "Soft nudge + loyalty points",
               "High": "15% discount + re-engagement", "Critical": "25% off + free shipping + call"}
    rs["Action"] = rs["churn_risk"].map(actions)
    st.dataframe(rs.rename(columns={"churn_risk": "Risk Band", "Avg_Spend": "Avg Spend (£)",
                                     "Avg_Prob": "Avg Churn Prob"}),
                 use_container_width=True, hide_index=True)

    st.download_button("📥 Download Churn Data", churn_data.to_csv(index=False),
                       "stylekart_churn.csv", "text/csv")


# ───────────────────────────────────────────────────────────────
# TAB 4 — INTEGRATED INTELLIGENCE
# ───────────────────────────────────────────────────────────────
with tab4:
    st.markdown("# 🧠 Integrated Customer Intelligence")
    st.markdown("Connecting **Segmentation + CLV + Churn** into one decision framework.")

    with st.spinner("Building integrated profiles…"):
        rfm_i = get_rfm(df_json)[["Customer ID", "Segment", "RFM_Score", "Recency", "Frequency", "Monetary"]]
        clv_i, _ = get_clv(df_json)
        clv_i = clv_i[["Customer ID", "predicted_clv", "CLV_Tier"]].copy()
        churn_i, _, _, _ = get_churn(df_json)
        churn_i = churn_i[["Customer ID", "churn_prob", "churn_risk"]].copy()

        master = rfm_i.merge(clv_i, on="Customer ID", how="inner") \
                      .merge(churn_i, on="Customer ID", how="inner")

        med_clv = master["predicted_clv"].median()
        q30_clv = master["predicted_clv"].quantile(0.3)

        def priority(row):
            s, ch, clv = row["Segment"], row["churn_prob"], row["predicted_clv"]
            if s in ["At Risk", "About to Sleep"] and ch > 0.6 and clv > med_clv:
                return "Critical"
            elif ch > 0.5 and clv > q30_clv:
                return "High"
            elif ch > 0.3:
                return "Medium"
            return "Low"

        def recommend(row):
            if row["Priority"] == "Critical":
                return "URGENT: 20% off + free shipping + personal email"
            elif row["Priority"] == "High":
                return "15% discount + re-engagement campaign"
            elif row["Priority"] == "Medium":
                return "Personalized recommendations + loyalty nudge"
            elif row["Segment"] == "Champions":
                return "VIP early access + referral program"
            elif row["Segment"] in ["Lost", "Hibernating"] and row["predicted_clv"] < q30_clv:
                return "Exclude from campaigns — save budget"
            return "Standard experience — monitor"

        master["Priority"] = master.apply(priority, axis=1)
        master["Action"] = master.apply(recommend, axis=1)

    crit = (master["Priority"] == "Critical").sum()
    rev_risk = master[master["Priority"].isin(["Critical", "High"])]["predicted_clv"].sum()
    excludable = master[(master["Segment"].isin(["Lost", "Hibernating"])) &
                        (master["predicted_clv"] < q30_clv)]

    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Total Profiled", f"{len(master):,}")
    with k2: metric_card("Critical Priority", f"{crit:,}")
    with k3: metric_card("Revenue at Risk", f"£{rev_risk:,.0f}")
    with k4: metric_card("Excludable", f"{len(excludable):,}")

    st.markdown("####")

    # Bubble Chart
    st.markdown("### 🔮 CLV Tier × Churn Risk × Segment")
    bd = master.groupby(["CLV_Tier", "churn_risk", "Segment"]).agg(
        count=("Customer ID", "count"), avg_clv=("predicted_clv", "mean")).reset_index()
    fig = px.scatter(bd, x="CLV_Tier", y="churn_risk", size="count", color="Segment",
                     title="Customer Landscape", template="plotly_dark",
                     color_discrete_map=SEGMENT_COLORS, size_max=50,
                     hover_data=["count", "avg_clv"],
                     category_orders={"CLV_Tier": ["Bronze", "Silver", "Gold", "Platinum"],
                                      "churn_risk": ["Low", "Medium", "High", "Critical"]})
    fig.update_layout(height=500, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Priority Table
    st.markdown("### 🚨 Top Priority Customers")
    pf = st.multiselect("Filter by Priority", ["Critical", "High", "Medium", "Low"],
                        default=["Critical", "High"], key="pf")
    pd_show = master[master["Priority"].isin(pf)].sort_values(
        ["Priority", "predicted_clv"], ascending=[True, False]).head(50)
    st.dataframe(pd_show[["Customer ID", "Segment", "RFM_Score", "predicted_clv",
                           "CLV_Tier", "churn_prob", "churn_risk", "Priority", "Action"]].rename(
        columns={"predicted_clv": "CLV (£)", "churn_prob": "Churn Prob",
                 "churn_risk": "Churn Risk"}),
        use_container_width=True, hide_index=True, height=400)

    # Revenue at Risk by Segment
    st.markdown("### 📉 Revenue at Risk by Segment")
    rr = master[master["churn_risk"].isin(["High", "Critical"])].groupby("Segment").agg(
        customers=("Customer ID", "count"), clv_at_risk=("predicted_clv", "sum"),
        avg_churn=("churn_prob", "mean")).reset_index().sort_values("clv_at_risk", ascending=False)
    fig = px.bar(rr, x="Segment", y="clv_at_risk", title="Revenue at Risk (High + Critical Churn)",
                 template="plotly_dark", color="Segment", color_discrete_map=SEGMENT_COLORS,
                 text="customers", labels={"clv_at_risk": "CLV at Risk (£)"})
    fig.update_traces(texttemplate="%{text} customers", textposition="outside")
    fig.update_layout(height=400, showlegend=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Decision Matrix
    st.markdown("### 📋 Decision Matrix")
    sf = st.selectbox("Filter Segment", ["All"] + sorted(master["Segment"].unique().tolist()), key="ds")
    dm = master.groupby(["Segment", "CLV_Tier", "churn_risk"]).agg(
        Customers=("Customer ID", "count"), Avg_CLV=("predicted_clv", "mean")).reset_index()
    dm["Avg_CLV"] = dm["Avg_CLV"].round(2)
    if sf != "All":
        dm = dm[dm["Segment"] == sf]
    st.dataframe(dm, use_container_width=True, hide_index=True)

    st.download_button("📥 Download Master Table", master.to_csv(index=False),
                       "stylekart_master_intelligence.csv", "text/csv")

    # Executive Story
    st.markdown("---")
    st.markdown("### 📖 Executive Summary")
    crit_custs = master[master["Priority"] == "Critical"]
    lost_low = master[(master["Segment"].isin(["Lost", "Hibernating"])) & (master["CLV_Tier"] == "Bronze")]

    st.markdown(f"""
    > We segmented **{len(master):,}** customers into **{master['Segment'].nunique()}** groups,
    > predicted 6-month value, and identified churn risk for each.
    >
    > **{len(crit_custs):,}** high-value customers are at **critical churn risk**,
    > representing **£{crit_custs['predicted_clv'].sum():,.0f}** in predicted revenue.
    > A targeted retention campaign on this group delivers the highest ROI.
    >
    > **{len(lost_low):,}** low-value churned customers should be **excluded from campaigns**,
    > saving budget that can be redirected to nurturing Potential Loyalists.
    >
    > **This is not three models. This is one decision-making framework.**
    """)
