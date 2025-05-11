import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
# 3. Sankey 시각화를 위한 데이터 준비
# -------------------------
df_pair = df[df['purchase_rank'].isin([1, 2])]
df_pair = df_pair.pivot(index='customer_id', columns='purchase_rank', values='article').dropna()
link_counts = df_pair.groupby([1, 2]).size().reset_index(name='count')

nodes = list(set(link_counts[1]).union(set(link_counts[2])))
node_map = {name: i for i, name in enumerate(nodes)}

# -------------------------
# 4. Sankey Chart 생성
# -------------------------
if not link_counts.empty:
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=nodes, pad=15, thickness=20),
        link=dict(
            source=[node_map[a] for a in link_counts[1]],
            target=[node_map[b] for b in link_counts[2]],
            value=link_counts['count']
        )
    )])
    st.subheader("1st → 2nd Purchase Flow")
    st.plotly_chart(fig)
else:
    st.warning("Not enough data to generate Sankey chart.")

# -------------------------
# 5. 원본 데이터 보기
# -------------------------
if st.checkbox("Show Raw Data"):
    st.dataframe(df.reset_index(drop=True))
