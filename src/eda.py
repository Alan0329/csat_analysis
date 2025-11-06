# %%
import pandas as pd
import plotly.express as px 

from pathlib import Path

DATA_DIR = Path("../data")

# %%
csat_df = pd.read_csv(f"{DATA_DIR}/Customer_support_data.csv", encoding="utf-8-sig")

# %%
# 確認有無載入成功
csat_df.head(10)

csat_df.tail(10)

# %%
# 欄位
csat_df.columns

# %%
# 缺失值
csat_df.isnull().sum()

# %%
csat_df.describe()


# %%
# 樣本數分佈
str_dtypes_cols = csat_df.select_dtypes("O").columns
for col in str_dtypes_cols:
    print(f"{col} 樣本數分佈：")
    print(csat_df[col].value_counts(dropna=False))
    print(" ")


# %%
# unique id 是否真的為重複值
len(csat_df['Unique id'].unique())

# %%
# channel_name: 撥入、撥出、email
df = (
    csat_df
    .query("channel_name == 'Outcall'")
    .query("`Customer Remarks`.notnull()")
)
df

# %%
ddf = (
    csat_df
    .query("Order_id == '5be81fde-7fb5-4624-af8a-89892e600acb'")
    [['Customer Remarks' != None]]

)
ddf


# %%
# unique id 是否真的為重複值
len(csat_df['Order_id'].dropna().unique())

# %%
csat_df.dtypes
float_dtypes_cols = csat_df.select_dtypes("float").columns
for col in float_dtypes_cols:
    print(f"{col} 樣本數分佈：")
    print(csat_df[col].value_counts(dropna=False))
    print(" ")

# %%
df = csat_df.query("`connected_handling_time`.notnull()")

# %%
csat_df.dtypes
int_dtypes_cols = csat_df.select_dtypes("int").columns
for col in int_dtypes_cols:
    print(f"{col} 樣本數分佈：")
    print(csat_df[col].value_counts(dropna=False))
    print(" ")


# %%
# === 針對 CSAT Score 進行更深入的 EDA ===
# %%
# 整體 CSAT Score 分佈，確認滿意度的整體輪廓
csat_score_counts = csat_df["CSAT Score"].value_counts().sort_index()
print("CSAT Score 分佈：")
print(csat_score_counts)
px.histogram(
    csat_df,
    x="CSAT Score",
    nbins=5,
    title="CSAT Score 分佈",
    text_auto=True,
)

# %%
# 分析客服人員資歷 (Tenure Bucket) 與對 CSAT 的影響
tenure_csat = (
    csat_df
    .groupby("Tenure Bucket")
    ["CSAT Score"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)
print("不同資歷區間的 CSAT：")
print(tenure_csat)


# %%
# 品類與子品類對 CSAT Score 的影響
category_csat = (
    csat_df
    .groupby("category")
    ["CSAT Score"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)
print("各主品類 CSAT 統計：")
print(category_csat)



# %%
tenure_cate = (
    csat_df
    .groupby(["Tenure Bucket", "category"], as_index=False)
    ['category']
    .agg(["count"])
    .sort_values(by=["Tenure Bucket", "count"], ascending=False)

)
tenure_cate = (
    csat_df
    .groupby(["Tenure Bucket", "category"], as_index=False)
    ['category']
    .agg(["count"])
    .sort_values(by=["Tenure Bucket", "count"], ascending=False)

)
tenure_cate["ratio"] = tenure_cate["count"] / tenure_cate.groupby("Tenure Bucket")["count"].transform("sum") * 100
print("不同資歷處理的類別：")
print(tenure_cate)

# %%
sub_category_csat = (
    csat_df
    .groupby("Sub-category")
    ["CSAT Score"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)
print("各子品類 CSAT 統計：")
print(sub_category_csat.head(20))
px.box(
    csat_df,
    x="category",
    y="CSAT Score",
    color="category",
    title="不同主品類的 CSAT Score 分佈",
)

# %%
# 分析客服人員班別對 CSAT 的影響
shift_csat = (
    csat_df.groupby("Agent Shift")["CSAT Score"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)
print("不同班別的 CSAT：")
print(shift_csat)

px.box(
    csat_df,
    x="Agent Shift",
    y="CSAT Score",
    color="Agent Shift",
    title="班別對 CSAT Score 的影響",
)

# %%
# 各通路的 CSAT 表現，找出通路端的差異
channel_csat = (
    csat_df.groupby("channel_name")["CSAT Score"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)
print("各通路 CSAT 統計：")
print(channel_csat)

px.box(
    csat_df,
    x="channel_name",
    y="CSAT Score",
    color="channel_name",
    title="不同通路的 CSAT Score 分佈",
    points="all",
)


# %%
# 客戶是否留下評語與 CSAT 的關聯
csat_df["has_customer_remarks"] = csat_df["Customer Remarks"].notna()
remarks_csat = (
    csat_df.groupby("has_customer_remarks")["CSAT Score"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)
print("是否留下評語 vs CSAT：")
print(remarks_csat)


# %%
# 將時間欄位轉換為 datetime，計算回覆時長對 CSAT 的影響
csat_df["issue_reported_at_dt"] = pd.to_datetime(
    csat_df["Issue_reported at"], errors="coerce"
)
csat_df["issue_responded_dt"] = pd.to_datetime(
    csat_df["issue_responded"], errors="coerce"
)
csat_df["survey_response_dt"] = pd.to_datetime(
    csat_df["Survey_response_Date"], errors="coerce"
)
csat_df["order_datetime"] = pd.to_datetime(
    csat_df["order_date_time"], errors="coerce"
)
csat_df["respond_hours"] = (
    csat_df["issue_responded_dt"] - csat_df["issue_reported_at_dt"]
).dt.total_seconds() / 3600


csat_df["respond_hours_bin"] = pd.cut(
    csat_df["respond_hours"], bins=[0, 1, 4, 12, 24, 48, 96, 120, 9999]
)

response_time_summary = (
    csat_df.dropna(subset=["respond_hours_bin"])
    .groupby("respond_hours_bin")["CSAT Score"]
    .agg(["count", "mean", "median"])
)

print("回覆時長 (小時) 分段 vs CSAT：")
print(response_time_summary)
px.scatter(
    csat_df,
    x="respond_hours",
    y="CSAT Score",
    title="回覆時長與 CSAT 的關係",
    trendline="ols",
    labels={"respond_hours": "回覆時長(小時)", "CSAT Score": "CSAT Score"},
)

# %%
# 金額與處理時間對 CSAT 的關聯性
csat_df.loc[
    csat_df["Item_price"].notna(), "item_price_decile"
] = pd.qcut(
    csat_df.loc[csat_df["Item_price"].notna(), "Item_price"],
    10,
    duplicates="drop",
)
item_price_csat = (
    csat_df
    .dropna(subset=["item_price_decile"])
    .groupby("item_price_decile")
    ["CSAT Score"]
    .agg(["count", "mean", "median"])
)
print("訂單金額分位 vs CSAT：")
print(item_price_csat)


# %%
# 客服團隊層級 (Agent、Supervisor、Manager) 的 CSAT 表現
agent_csat = (
    csat_df.groupby("Agent_name")["CSAT Score"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)
print("客服人員的 CSAT 表現 (Top 10)：")
print(agent_csat.head(10))
supervisor_csat = (
    csat_df.groupby("Supervisor")["CSAT Score"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)
print("主管層級的 CSAT 表現：")
print(supervisor_csat)
manager_csat = (
    csat_df.groupby("Manager")["CSAT Score"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)
print("經理層級的 CSAT 表現：")
print(manager_csat)
