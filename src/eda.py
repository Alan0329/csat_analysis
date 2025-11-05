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