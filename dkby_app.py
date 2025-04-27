import akshare as ak
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

# SMA and DKBY functions as in original script
def sma_tdx_fast(x: pd.Series, n: int, m: int) -> pd.Series:
    return x.ewm(alpha=m/n, adjust=False).mean()

def compute_dkby_fast(df: pd.DataFrame) -> pd.DataFrame:
    hhv21 = df["high"].rolling(21).max()
    llv21 = df["low"].rolling(21).min()
    C = df["close"]
    var1 = (hhv21 - C) / (hhv21 - llv21) * 100 - 10
    var2 = (C - llv21) / (hhv21 - llv21) * 100
    var3      = sma_tdx_fast(var2, 13, 8)
    long_side = sma_tdx_fast(var3, 13, 8)
    short_side= sma_tdx_fast(var1, 21, 8)
    return pd.DataFrame({"long": long_side, "short": short_side})

def process_industry(name: str, calc_start: str, ed: str, thresh: float) -> str | None:
    df = ak.stock_board_industry_index_ths(symbol=name, start_date=calc_start, end_date=ed)
    if df.empty:
        return None
    df.rename(columns={"日期":"date","开盘价":"open","最高价":"high","最低价":"low","收盘价":"close"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df_week = df.resample("W-FRI").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    if df_week.empty:
        return None
    dkby = compute_dkby_fast(df_week)
    if dkby.empty:
        return None
    diff = dkby["long"].iat[-1] - dkby["short"].iat[-1]
    return name if abs(diff) < thresh else None

# Streamlit UI
st.title("行业板块 DKBY 筛选工具")
thresh = st.number_input("输入 DKBY long-short 差值阈值", min_value=0.0, value=5.0, step=0.1)
start_date = st.date_input("开始日期（留空=1 年前）")
end_date = st.date_input("结束日期（留空=今天)")

if st.button("开始筛选"):
    # format dates
    today = pd.Timestamp.today().normalize()
    sd = start_date.strftime("%Y-%m-%d") if start_date else (today - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    ed = end_date.strftime("%Y-%m-%d") if end_date else today.strftime("%Y-%m-%d")
    calc_start = (pd.to_datetime(sd) - pd.Timedelta(days=150)).strftime("%Y%m%d")
    ed_fmt = ed.replace("-", "")
    industry_df = ak.stock_board_industry_name_ths()
    names = industry_df["name"].tolist()
    matched = []
    with ThreadPoolExecutor(max_workers=9) as exe:
        futures = {exe.submit(process_industry, name, calc_start, ed_fmt, thresh): name for name in names}
        for fut in futures:
            res = fut.result()
            if res:
                matched.append(res)
    if matched:
        st.success("满足条件的行业板块：")
        for m in matched:
            st.write(m)
    else:
        st.info("未找到满足条件的行业板块。")
