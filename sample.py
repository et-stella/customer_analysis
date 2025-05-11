import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from collections import defaultdict
import tempfile
import os

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
    num_purchases = np.random.randint(1, 5)
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
# 3. 마인드맵 스타일 네트워크 시각화 with node size
# -------------------------
st.subheader("Mindmap View of Purchase Paths")
df_quad = df[df['purchase_rank'].isin([1, 2, 3, 4])]
df_quad = df_quad.pivot(index='customer_id', columns='purchase_rank', values='article').dropna()
df_quad.columns = ['1st', '2nd', '3rd', '4th']

path_counts = df_quad.groupby(['1st', '2nd', '3rd', '4th']).size().reset_index(name='count')

G = nx.DiGraph()
node_weights = defaultdict(int)
for _, row in path_counts.iterrows():
    a1, a2, a3, a4, c = row['1st'], row['2nd'], row['3rd'], row['4th'], row['count']
    G.add_edge(a1, a2, weight=c)
    G.add_edge(a2, a3, weight=c)
    G.add_edge(a3, a4, weight=c)
    node_weights[a1] += c
    node_weights[a2] += c
    node_weights[a3] += c
    node_weights[a4] += c

net = Network(height='550px', width='100%', directed=True)
for node in G.nodes():
    net.add_node(node, label=node, size=node_weights[node] * 3)
for source, target, data in G.edges(data=True):
    net.add_edge(source, target, value=data['weight'])

with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
    tmp_path = tmp_file.name
    net.save_graph(tmp_path)
    components.html(open(tmp_path, 'r', encoding='utf-8').read(), height=600)
    os.unlink(tmp_path)

# -------------------------
# 4. Sunburst Chart (구매 플로우 시각화)
# -------------------------
st.subheader("Sunburst Chart: Purchase Flow (1st → 2nd → 3rd → 4th)")
if not path_counts.empty:
    fig_sun = px.sunburst(path_counts, path=['1st', '2nd', '3rd', '4th'], values='count')
    st.plotly_chart(fig_sun)

# -------------------------
# 5. 히트맵 (수치 시각화)
# -------------------------
st.subheader("Heatmap: 1st vs 2nd Purchase Frequency")
heatmap_1_2 = df_quad.groupby(['1st', '2nd']).size().unstack().fillna(0)
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.heatmap(heatmap_1_2, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax1)
st.pyplot(fig1)

st.subheader("Heatmap: 2nd vs 3rd Purchase Frequency")
heatmap_2_3 = df_quad.groupby(['2nd', '3rd']).size().unstack().fillna(0)
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(heatmap_2_3, annot=True, fmt=".0f", cmap="YlOrBr", ax=ax2)
st.pyplot(fig2)

st.subheader("Heatmap: 3rd vs 4th Purchase Frequency")
heatmap_3_4 = df_quad.groupby(['3rd', '4th']).size().unstack().fillna(0)
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(heatmap_3_4, annot=True, fmt=".0f", cmap="BuPu", ax=ax3)
st.pyplot(fig3)

# -------------------------
# 6. 특정 제품 기준 흐름 비중 시각화
# -------------------------
st.subheader("Purchase Transition Ratio from 1st Article (Omni Customers)")
omni_df = df[df['customer_type'] == 'Omni']
omni_pivot = omni_df.pivot(index='customer_id', columns='purchase_rank', values='article').dropna()
if set([1, 2]).issubset(omni_pivot.columns):
    omni_pivot.columns = [f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th" for i in omni_pivot.columns]
    selected_article = st.selectbox("Select 1st Purchase Article", sorted(omni_pivot['1st'].unique()))
    filtered = omni_pivot[omni_pivot['1st'] == selected_article]
    next_counts = filtered['2nd'].value_counts(normalize=True).mul(100).round(1).reset_index()
    next_counts.columns = ['2nd Article', 'Percentage']
    st.dataframe(next_counts)
    fig4 = px.bar(next_counts, x='2nd Article', y='Percentage', text='Percentage', title=f"2nd Purchase Ratio After '{selected_article}'")
    st.plotly_chart(fig4)

# -------------------------
# 7. 원본 데이터 보기
# -------------------------
if st.checkbox("Show Raw Data"):
    st.dataframe(df.reset_index(drop=True))
