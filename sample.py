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
    if depth >= len(key) or depth >= len(level_dicts):
        return
    indent = "    " * depth
    children = level_dicts[depth].get(key, [])
    st.text(f"{indent}└── {key[depth]} ({sum(c for _, c in children)} customers)")
    for child, count in children:
        print_path(key + (child,), key + (child,), depth + 1)

# 출력 루트 노드부터
roots = defaultdict(int)
for k, vlist in level_dicts[0].items():
    roots[k[0]] += sum(c for _, c in vlist)
for root, total in roots.items():
    st.text(f"{root} ({total} customers)")
    for child, count in level_dicts[0][(root,)]:
        print_path((root, child), (root, child), 1)

# -------------------------
# 4.4 고객 유형별 총 구매 횟수, 1인당 평균 구매 횟수, 2회 이상 구매자 비중, 성별/연령/채널 분석
# -------------------------
# -------------------------
st.subheader("Customer Type Summary: Total, Avg Purchases")
sum_stats = []

for ctype in customer_types:
    group_df = df[df['customer_type'] == ctype]
    total_purchase = group_df.shape[0]
    unique_customers = group_df['customer_id'].nunique()
    avg_purchase_per_user = round(total_purchase / unique_customers, 2) if unique_customers > 0 else np.nan

    # 2회 이상 구매한 고객 수
    purchase_counts = group_df.groupby('customer_id').size()
    repeat_buyers = purchase_counts[purchase_counts >= 2].count()
    repeat_ratio = round((repeat_buyers / unique_customers) * 100, 1) if unique_customers > 0 else np.nan

    sum_stats.append({
        "Customer Type": ctype,
        "Total Purchases": total_purchase,
        "Unique Customers": unique_customers,
        "Avg Purchases per Customer": avg_purchase_per_user,
        "Repeat Buyer Ratio (%)": repeat_ratio
    })

summary_df = pd.DataFrame(sum_stats)
st.dataframe(summary_df)
fig = px.bar(summary_df, x="Customer Type", y="Avg Purchases per Customer", text="Avg Purchases per Customer", title="Average Purchases per Customer by Type")
st.plotly_chart(fig)

# -------------------------
# 고객 특성 및 채널 Top 5 분석
# -------------------------

# 성별 및 연령 샘플 추가 생성
gender_map = {cid: np.random.choice(['M', 'F']) for cid in df['customer_id'].unique()}
age_map = {cid: np.random.randint(20, 60) for cid in df['customer_id'].unique()}

df['gender'] = df['customer_id'].map(gender_map)
df['age'] = df['customer_id'].map(age_map)
df['age_group'] = pd.cut(df['age'], bins=[19, 29, 39, 49, 59, 99], labels=['20s', '30s', '40s', '50s', '60+'])

st.subheader("Customer Gender Distribution by Type")
gender_summary = df.groupby(['customer_type', 'gender'])['customer_id'].nunique().reset_index()
gender_summary.columns = ['Customer Type', 'Gender', 'Unique Customers']
fig_gender = px.bar(gender_summary, x='Customer Type', y='Unique Customers', color='Gender', barmode='group', title="Gender Distribution by Customer Type")
st.plotly_chart(fig_gender)

st.subheader("Customer Age Group Distribution by Type")
age_summary = df.groupby(['customer_type', 'age_group'])['customer_id'].nunique().reset_index()
age_summary.columns = ['Customer Type', 'Age Group', 'Unique Customers']
fig_age = px.bar(age_summary, x='Customer Type', y='Unique Customers', color='Age Group', barmode='group', title="Age Group Distribution by Customer Type")
st.plotly_chart(fig_age)

st.subheader("Top 5 Purchased Articles by Customer Type")
top5 = df.groupby(['customer_type', 'article']).size().reset_index(name='count')
top5 = top5.sort_values(['customer_type', 'count'], ascending=[True, False]).groupby('customer_type').head(5)
fig_top5 = px.bar(top5, x='customer_type', y='count', color='article', text='article', barmode='stack', title='Top 5 Purchased Articles by Customer Type')
st.plotly_chart(fig_top5)


# -------------------------
st.subheader("Sunburst Chart: Purchase Flow")
if not path_counts.empty:
    fig_sun = px.sunburst(path_counts, path=list(df_multi.columns), values='count')
    st.plotly_chart(fig_sun)

# -------------------------
# 5. 히트맵 (수치 시각화)
# -------------------------
st.subheader("Purchase Transition Probability Heatmaps")

for i in range(1, 3):  # Only 1st→2nd and 2nd→3rd
    col_from = f"{i}th"
    col_to = f"{i+1}th"
    if col_from in df_multi.columns and col_to in df_multi.columns:
        pair_df = df_multi[[col_from, col_to]].dropna()
        total_from = pair_df.groupby(col_from).size()
        prob_matrix = pair_df.groupby([col_from, col_to]).size().unstack().fillna(0)
        prob_matrix = prob_matrix.div(total_from, axis=0) * 100
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(prob_matrix, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar_kws={'label': 'Transition %'})
        st.subheader(f"{col_from} → {col_to} Transition Probability (%)")
        st.pyplot(fig)

# 5~7 그대로 유지
# ... (이후 코드 생략: Heatmap, Omni 분석, Raw Data)
