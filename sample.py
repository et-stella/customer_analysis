import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------------
# 1. 샘플 데이터 생성
# -------------------------
st.title("Customer Purchase Flow Analysis")

np.random.seed(42)
customers = [f"CUST_{i}" for i in range(1, 31)]
articles = ["Lipstick", "Foundation", "Cleanser", "Perfume", "Serum", "Toner", "Mascara"]
customer_types = ["Online", "Offline", "Omni"]

records = []
for cust in customers:
    ctype = np.random.choice(customer_types)
    num_purchases = np.random.randint(1, 7)  # 최대 6단계까지 생성
    dates = pd.date_range(start='2024-01-01', periods=90).to_list()
    chosen_dates = sorted(np.random.choice(dates, num_purchases, replace=False))
    for i, d in enumerate(chosen_dates):
        article = np.random.choice(articles)
        records.append([cust, ctype, d, article, i+1])

df = pd.DataFrame(records, columns=['customer_id', 'customer_type', 'order_date', 'article', 'purchase_rank'])
df.sort_values(by=['customer_id', 'order_date'], inplace=True)

# -------------------------
# 2. 고객 유형 필터
# -------------------------
customer_filter = st.selectbox("Select Customer Type", options=["All"] + customer_types)
if customer_filter != "All":
    df = df[df['customer_type'] == customer_filter]

# -------------------------
# 3. 계층 텍스트 출력 (Indent 스타일, 최대 6단계)
# -------------------------
st.subheader("Indented Tree Structure of Purchase Paths")
df_multi = df[df['purchase_rank'] <= 6]
df_multi = df_multi.pivot(index='customer_id', columns='purchase_rank', values='article').dropna()
df_multi.columns = [f'{i}th' for i in range(1, len(df_multi.columns)+1)]

path_counts = df_multi.groupby(list(df_multi.columns)).size().reset_index(name='count')

# 계층 구조 생성
level_dicts = [defaultdict(list) for _ in range(len(df_multi.columns)-1)]

for _, row in path_counts.iterrows():
    keys = row[:-1].tolist()
    c = row['count']
    for i in range(len(keys)-1):
        level_dicts[i][tuple(keys[:i+1])].append((keys[i+1], c))

# 재귀 함수로 출력

def print_path(prefix, key, depth):
    if depth >= len(key):
        return
    indent = "    " * depth
    st.text(f"{indent}└── {key[depth]} ({sum(c for _, c in level_dicts[depth].get(key, []))} customers)")
    if depth < len(level_dicts):
        for child, count in level_dicts[depth].get(key, []):
            print_path(key + (child,), key + (child,), depth+1)

# 출력 루트 노드부터
roots = defaultdict(int)
for k, vlist in level_dicts[0].items():
    roots[k[0]] += sum(c for _, c in vlist)
for root, total in roots.items():
    st.text(f"{root} ({total} customers)")
    for child, count in level_dicts[0][(root,)]:
        print_path((root, child), (root, child), 1)

# -------------------------
# 4.5 고객 유형별 평균 구매 주기 비교
# -------------------------
st.subheader("Average Purchase Interval by Customer Type")

intervals = []
for ctype in ["All"] + customer_types:
    if ctype == "All":
        group_df = df.copy()
    else:
        group_df = df[df['customer_type'] == ctype]
    days = []
    for cust_id, cust_df in group_df.groupby('customer_id'):
        cust_df = cust_df.sort_values('order_date')
        if len(cust_df) > 1:
            diff = cust_df['order_date'].diff().dropna().dt.days
            days.extend(diff.tolist())
    if days:
        avg_days = round(np.mean(days), 1)
    else:
        avg_days = np.nan
    intervals.append({"Customer Type": ctype, "Avg Purchase Interval (days)": avg_days})

interval_df = pd.DataFrame(intervals)
st.dataframe(interval_df)
fig_bar = px.bar(interval_df, x="Customer Type", y="Avg Purchase Interval (days)", text="Avg Purchase Interval (days)", title="Average Days Between Purchases by Customer Type")
st.plotly_chart(fig_bar)

# -------------------------
# 4. Sunburst Chart (구매 플로우 시각화)
# -------------------------
st.subheader("Sunburst Chart: Purchase Flow")
if not path_counts.empty:
    fig_sun = px.sunburst(path_counts, path=list(df_multi.columns), values='count')
    st.plotly_chart(fig_sun)

# -------------------------
# 5. 히트맵 (수치 시각화)
# -------------------------
st.subheader("Heatmaps Between Sequential Purchases")

for i in range(1, 6):
    if f'{i}th' in df_multi.columns and f'{i+1}th' in df_multi.columns:
        heatmap_data = df_multi.groupby([f'{i}th', f'{i+1}th']).size().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="viridis", ax=ax)
        st.subheader(f"{i}th → {i+1}th Purchase Frequency")
        st.pyplot(fig)

# -------------------------
# 5~7 그대로 유지
# ... (이후 코드 생략: Heatmap, Omni 분석, Raw Data)
