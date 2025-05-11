import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

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
    num_purchases = np.random.randint(1, 4)
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
# 3. Sankey 시각화를 위한 데이터 준비 (1st → 2nd → 3rd)
# -------------------------
df_triplet = df[df['purchase_rank'].isin([1, 2, 3])]
df_triplet = df_triplet.pivot(index='customer_id', columns='purchase_rank', values='article').dropna()
df_triplet.columns = ['purchase_rank_1', 'purchase_rank_2', 'purchase_rank_3']

link_counts_1_2 = df_triplet.groupby(['purchase_rank_1', 'purchase_rank_2']).size().reset_index(name='count')
link_counts_2_3 = df_triplet.groupby(['purchase_rank_2', 'purchase_rank_3']).size().reset_index(name='count')

nodes = list(set(link_counts_1_2['purchase_rank_1']).union(set(link_counts_1_2['purchase_rank_2']))
             .union(set(link_counts_2_3['purchase_rank_2'])).union(set(link_counts_2_3['purchase_rank_3'])))
node_map = {name: i for i, name in enumerate(nodes)}

# 링크 구성
sources = [node_map[a] for a in link_counts_1_2['purchase_rank_1']] + [node_map[a] for a in link_counts_2_3['purchase_rank_2']]
targets = [node_map[b] for b in link_counts_1_2['purchase_rank_2']] + [node_map[b] for b in link_counts_2_3['purchase_rank_3']]
values = list(link_counts_1_2['count']) + list(link_counts_2_3['count'])

# -------------------------
# 4. Sankey Chart 생성
# -------------------------
if values:
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=nodes, pad=15, thickness=20),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])
    st.subheader("1st → 2nd → 3rd Purchase Flow")
    st.plotly_chart(fig)
else:
    st.warning("Not enough data to generate Sankey chart.")

# -------------------------
# 5. 히트맵 (수치 시각화)
# -------------------------
st.subheader("Heatmap: 1st vs 2nd Purchase Frequency")
heatmap_1_2 = df_triplet.groupby(['purchase_rank_1', 'purchase_rank_2']).size().unstack().fillna(0)
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.heatmap(heatmap_1_2, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax1)
st.pyplot(fig1)

st.subheader("Heatmap: 2nd vs 3rd Purchase Frequency")
heatmap_2_3 = df_triplet.groupby(['purchase_rank_2', 'purchase_rank_3']).size().unstack().fillna(0)
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(heatmap_2_3, annot=True, fmt=".0f", cmap="YlOrBr", ax=ax2)
st.pyplot(fig2)

# -------------------------
# 6. 원본 데이터 보기
# -------------------------
if st.checkbox("Show Raw Data"):
    st.dataframe(df.reset_index(drop=True))
