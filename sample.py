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
# 3. 텍스트 트리 구조 시각화 (구매 경로)
# -------------------------
st.subheader("Indented Tree View of Purchase Paths")
df_quad = df[df['purchase_rank'].isin([1, 2, 3, 4])]
df_quad = df_quad.pivot(index='customer_id', columns='purchase_rank', values='article').dropna()
df_quad.columns = ['1st', '2nd', '3rd', '4th']

path_counts = df_quad.groupby(['1st', '2nd', '3rd', '4th']).size().reset_index(name='count')

first_level = defaultdict(list)
second_level = defaultdict(list)
third_level = defaultdict(list)

for _, row in path_counts.iterrows():
    a1, a2, a3, a4, c = row['1st'], row['2nd'], row['3rd'], row['4th'], row['count']
    first_level[a1].append((a2, c))
    second_level[(a1, a2)].append((a3, c))
    third_level[(a1, a2, a3)].append((a4, c))

for a1 in first_level:
    total_1 = sum([c for _, c in first_level[a1]])
    st.text(f"{a1} ({total_1} customers)")
    for a2, c2 in first_level[a1]:
        st.text(f"  └── {a2} ({c2} customers)")
        for a3, c3 in second_level[(a1, a2)]:
            st.text(f"      └── {a3} ({c3} customers)")
            for a4, c4 in third_level[(a1, a2, a3)]:
                st.text(f"          ── {a4} ({c4} customers)")

# -------------------------
# 4. Sunburst Chart (구매 플로우 시각화)
# -------------------------
st.subheader("Sunburst Chart: Purchase Flow (1st → 2nd → 3rd → 4th)")
if not path_counts.empty:
    fig_sun = px.sunburst(path_counts, path=['1st', '2nd', '3rd', '4th'], values='count')
    st.plotly_chart(fig_sun)

# -------------------------
# 5. 특정 제품 기준 흐름 비중 시각화
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
    fig3 = px.bar(next_counts, x='2nd Article', y='Percentage', text='Percentage', title=f"2nd Purchase Ratio After '{selected_article}'")
    st.plotly_chart(fig3)

# -------------------------
# 6. 원본 데이터 보기
# -------------------------
if st.checkbox("Show Raw Data"):
    st.dataframe(df.reset_index(drop=True))
