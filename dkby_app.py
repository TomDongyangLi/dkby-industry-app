import akshare as ak
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Streamlit UI with text inputs for dates
st.title("行业板块 DKBY 筛选工具")
thresh = st.number_input("输入 DKBY long-short 差值阈值", min_value=0.0, value=5.0, step=0.1)

sd_str = st.text_input("开始日期（留空=1 年前，格式 YYYY-MM-DD）", value="")
ed_str = st.text_input("结束日期（留空=今天，格式 YYYY-MM-DD）", value="")

if st.button("开始筛选"):
    # 解析或使用默认日期
    today = pd.Timestamp.today().normalize()
    sd = sd_str.strip() or (today - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    ed = ed_str.strip() or today.strftime("%Y-%m-%d")

    calc_start = (pd.to_datetime(sd) - pd.Timedelta(days=150)).strftime("%Y%m%d")
    ed_fmt = ed.replace("-", "")

    names = ak.stock_board_industry_name_ths()["name"].tolist()
    matched = []
    progress = st.progress(0)

    with ThreadPoolExecutor(max_workers=4) as exe:
        futures = {exe.submit(process_industry, n, calc_start, ed_fmt, thresh): n for n in names}
        for i, fut in enumerate(as_completed(futures)):
            res = fut.result()
            if res:
                matched.append(res)
            progress.progress((i+1)/len(names))

    if matched:
        st.success("满足条件的行业板块：")
        for m in matched:
            st.write(m)
    else:
        st.info("未找到满足条件的行业板块。")
