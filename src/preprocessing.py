# %%
import pandas as pd
import plotly.express as px 
import re

from pathlib import Path

DATA_DIR = Path("../data")

# %%
csat_df = pd.read_csv(f"{DATA_DIR}/Customer_support_data.csv", encoding="utf-8-sig")
import pandas as pd
import plotly.express as px 
import re

from pathlib import Path

DATA_DIR = Path("../data")

# %%
csat_df = pd.read_csv(f"{DATA_DIR}/Customer_support_data.csv", encoding="utf-8-sig")

# %%
GOOD_KEYWORDS = {
    "good", "great", "excellent", "amazing", "thank", "thanks",
    "appreciate", "satisfied", "happy", "delight", "wonderful",
    "nice", "nice help", "so sweet", "sweet", "thanku", "thks",
    "best", "better", "thx", "ty", "best", "better", "awesome", 
    "fantastic", "brilliant", "perfect", "wow", "yay", "helpful", 
    "luv", "love", "super",
    "thank you", "thanks a lot", "many thanks", "so sweet", "nice help",
            "well done", "great job", "love it", "loved it", "so helpful",
            "very helpful", "life saver", "lifesaver", "appreciate it",
            "much appreciated", "excellent service", "good job", "great support",
            "awesome support"
}

OK_KEYWORDS = {
    "ok", "okay", "fine", "average", "satisfactory",
    "decent", "not bad", "soso", "so so", "meh", "alright", "acceptable", "fair",
    "not bad", "so so", "so-so", "it's ok", "it's okay", "its ok",
        "its okay", "just okay", "just ok", "okay then", "could be better",
        "not the best", "ok but", "okay but", "fine but"
}

NEGATIVE_KEYWORDS = {
   "bad", "poor", "terrible", "awful", "horrible", "worst",
            "sucks", "suck", "trash", "garbage", "useless", "unhelpful",
            "rude", "angry", "mad", "frustrated", "annoyed", "cancel", "refund",
            "complaint", "escalate", "bug", "broken", "fail", "failure", "error",
            "slow", "stupid", "hate", "lag", "crash", "crashed", "crashing",
            "not good", "not great", "not okay", "didn't help", "did not help",
            "no help", "not helpful", "issue not solved", "problem not solved",
            "still not working", "still broken", "won't work", "doesn't work",
            "not working", "stopped working", "never again", "very slow",
            "too slow", "unacceptable", "worst experience", "waste of time",
            "waste time", "refund please", "want refund", "request refund",
            "cancel my", "cancelled my", "escalate this", "file a complaint",
            "very rude", "attitude bad",
}

def any_word_in(text: str, words: set[str]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)

def categorize_customer_remarks(remarks) -> str:
    # 1) 處理遺漏值與空白
    if pd.isna(remarks):
        return "no"
    text = str(remarks).strip().lower()
    if text in {"na", "n/a", "nil", "none", "no", "-", "--", ""}:
        return "no"

    # 2) 優先處理多字片語（避免 'not bad' 被誤判為負向）
    if any(phrase in text for phrase in OK_KEYWORDS):
        return "ok"

    # 3) 再處理單字關鍵字
    if any_word_in(text, NEGATIVE_KEYWORDS):
        return "negative"
    if any_word_in(text, GOOD_KEYWORDS):
        return "good"
    if any_word_in(text, OK_KEYWORDS):
        return "ok"

    return "ok"

csat_df['remarks_category'] = csat_df['Customer Remarks'].map(categorize_customer_remarks)

# %%
# === 1) 回覆延遲（分鐘）— 向量化計算 ===
REPORTED_COL  = "Issue_reported at"   # 依你的實際欄位調整
RESPONDED_COL = "issue_responded"  # 依你的實際欄位調整

reported = pd.to_datetime(csat_df[REPORTED_COL],  errors="coerce")
responded = pd.to_datetime(csat_df[RESPONDED_COL], errors="coerce")

csat_df["response_lag_min"] = (
    (responded - reported)
    .dt.total_seconds()
    .div(60)
    .round(2)
)


# %%

CITY_REGION_MAP = {
    # North India
    "DELHI": "North", "NEW DELHI": "North", "GURGAON": "North", 
    "NOIDA": "North", "GHAZIABAD": "North", "JAIPUR": "North", "LUCKNOW": "North", 
    "KANPUR": "North", "VARANASI": "North", "ALLAHABAD": "North", "DEHRADUN": "North",
    "CHANDIGARH": "North", "AGRA": "North",
    # West India
    "MUMBAI": "West", "THANE": "West", "PUNE": "West", "NAGPUR": "West", "AHMEDABAD": "West",
    "SURAT": "West", "VADODARA": "West", "INDORE": "West",
    # South India
    "BANGALORE": "South", "BENGALURU": "South", "HYDERABAD": "South", "CHENNAI": "South",
    "KOCHI": "South", "COIMBATORE": "South", "VISAKHAPATNAM": "South", "MADURAI": "South",
    "MYSORE": "South",
    # East India
    "KOLKATA": "East", "BHUBANESWAR": "East", "GUWAHATI": "East", "RANCHI": "East", 
    "PATNA": "East", "JAMSHEDPUR": "East",  "DHANBAD": "East",
    # Central India
    "RAIPUR": "Central","BHOPAL": "Central",
}

city_norm = (
    csat_df["Customer_City"]
    .astype("string")
    .str.strip()
    .str.upper()
    )

# 建立「最長優先」比對的正則（避免 "DELHI" 先吃掉 "NEW DELHI"）
key_order = sorted(CITY_REGION_MAP.keys(), key=len, reverse=True)
pattern = r"^(" + "|".join(re.escape(k) for k in key_order) + r")(?:\b|$)"

matched_key = city_norm.str.extract(pattern, expand=False)

csat_df["region"] = (
    matched_key.map(CITY_REGION_MAP)
)

# 補齊 Unknown / Other
csat_df.loc[city_norm.isna() | (city_norm == ""), "region"] = "Unknown"
csat_df["region"] = csat_df["region"].fillna("Other")

# %%
csat_df.to_csv(f"{DATA_DIR}/processed_csat_data.csv", encoding="utf-8-sig", index=False)

# %%