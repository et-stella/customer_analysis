import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz

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
# 3. 구매 경로 계층형 트리 시각화 (Graphviz)
# -------------------------
df_triplet = df[df['purchase_rank'].isin([1, 2, 3])]
df_triplet = df_triplet.pivot(index='customer_id', columns='purchase_rank', values='article').dropna()
df_triplet.columns = ['purchase_rank_1', 'purchase_rank_2', 'purchase_rank_3']

path_counts = df_triplet.groupby(['purchase_rank_1', 'purchase_rank_2', 'purchase_rank_3']).size().reset_index(name='count')

st.subheader("Hierarchical Tree View of Purchase Flow")
dot = graphviz.Digraph(format='png')
dot.attr(rankdir='LR')

for _, row in path_counts.iterrows():
    dot.edge(f"1st: {row['purchase_rank_1']}", f"2nd: {row['purchase_rank_2']}", label=str(row['count']))
    dot.edge(f"2nd: {row['purchase_rank_2']}", f"3rd: {row['purchase_rank_3']}", label=str(row['count']))

st.graphviz_chart(dot)

# -------------------------
# 4. 히트맵 (수치 시각화)
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
# 5. 특정 제품 기준 흐름 비중 시각화
# -------------------------
st.subheader("Purchase Transition Ratio from 1st Article (Omni Customers)")
omni_df = df[df['customer_type'] == 'Omni']
omni_pivot = omni_df.pivot(index='customer_id', columns='purchase_rank', values='article').dropna()
omni_pivot.columns = ['1st', '2nd', '3rd']

if not omni_pivot.empty:
    selected_article = st.selectbox("Select 1st Purchase Article", sorted(omni_pivot['1st'].unique()))
    filtered = omni_pivot[omni_pivot['1st'] == selected_article]
    next_counts = filtered['2nd'].value_counts(normalize=True).mul(100).round(1).reset_index()
    next_counts.columns = ['2nd Article', 'Percentage']
    st.dataframe(next_counts)
    fig3 = px.bar(next_counts, x='2nd Article', y='Percentage', text='Percentage', title=f"2nd Purchase Ratio After '{selected_article}'")
    st.plotly_chart(fig3)

# -------------------------
# 6. 원본 데이터 보기
# -------------------------
if st.checkbox("Show Raw Data"):
    st.dataframe(df.reset_index(drop=True))
