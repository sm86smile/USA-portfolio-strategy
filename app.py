import calendar
import math
import time
from datetime import date
from typing import Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="ETF 자산배분",
    page_icon="📈",
    layout="wide",
)

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
API_CALL_DELAY_SECONDS = 1.25
CACHE_TTL_SECONDS = 12 * 60 * 60

LAA_FIXED = ["IWD", "GLD", "IEF"]
LAA_VARIABLE = ["QQQ", "SHY"]
VAA_ATTACK = ["SPY", "EFA", "EEM", "AGG"]
VAA_SAFE = ["LQD", "IEF", "SHY"]
ODM_ASSETS = ["SPY", "EFA", "BIL", "AGG"]

ALL_TICKERS = sorted(set(LAA_FIXED + LAA_VARIABLE + VAA_ATTACK + VAA_SAFE + ODM_ASSETS))
# LAA는 가격 데이터 없이 수동 조건으로 비중이 결정됩니다.
# Alpha Vantage 호출은 VAA/오리지널 듀얼 모멘텀 계산에 필요한 ETF로만 제한합니다.
DATA_TICKERS = sorted(set(VAA_ATTACK + VAA_SAFE + ODM_ASSETS))

ETF_LABELS = {
    "IWD": "미국 대형 가치주",
    "GLD": "금",
    "IEF": "미국 중기국채",
    "QQQ": "나스닥100",
    "SHY": "미국 단기국채",
    "SPY": "미국 S&P500",
    "EFA": "선진국 주식(미국 제외)",
    "EEM": "신흥국 주식",
    "AGG": "미국 종합채권",
    "LQD": "미국 투자등급 회사채",
    "BIL": "초단기 미국 국채",
}


# =========================================================
# 유틸 함수
# =========================================================
def get_secret_api_key() -> str:
    """Streamlit Secrets에서 Alpha Vantage API Key를 읽습니다. 화면에서는 직접 입력받지 않습니다."""
    try:
        return str(st.secrets["ALPHA_VANTAGE_API_KEY"]).strip()
    except Exception:
        return ""


def format_pct(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x) * 100:.{digits}f}%"


def format_score(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"


def money_krw(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{int(round(float(x))):,}원"


def money_usd(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"${float(x):,.2f}"


def fx_rate_krw(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):,.2f}원/USD"


def usd_price(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"${float(x):,.2f}"


def format_share_count(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{int(float(x)):,}주"


def format_fractional_shares(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):,.2f}주"


def format_input_amount(amount: float, currency: str) -> str:
    if currency == "KRW":
        return money_krw(amount)
    return money_usd(amount)


def is_last_day(d: date) -> bool:
    return d.day == calendar.monthrange(d.year, d.month)[1]


def add_months(d: date, months: int = 1) -> date:
    """월말 리밸런싱이면 다음 월도 월말로 계산합니다."""
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    last_day_target_month = calendar.monthrange(year, month)[1]

    if is_last_day(d):
        day = last_day_target_month
    else:
        day = min(d.day, last_day_target_month)

    return date(year, month, day)


def add_years(d: date, years: int = 1) -> date:
    """2월 29일 같은 예외를 고려해 연 단위 날짜를 계산합니다."""
    try:
        return date(d.year + years, d.month, d.day)
    except ValueError:
        return date(d.year + years, d.month, calendar.monthrange(d.year + years, d.month)[1])


def next_rebalance_date(last_date: date, cycle: str) -> date:
    if cycle == "연 1회":
        return add_years(last_date, 1)
    if cycle == "월 1회":
        return add_months(last_date, 1)
    raise ValueError(f"지원하지 않는 리밸런싱 주기입니다: {cycle}")


def rebalance_status(next_date: date, eval_date: date) -> str:
    return "리밸런싱 필요" if next_date <= eval_date else "대기"


def normalize_strategy_weights(w_laa: float, w_vaa: float, w_odm: float) -> Tuple[float, float, float, float]:
    total = w_laa + w_vaa + w_odm
    if total <= 0:
        return w_laa, w_vaa, w_odm, total
    return w_laa / total, w_vaa / total, w_odm / total, total


def pct_cols(df: pd.DataFrame, cols: List[str], digits: int = 2) -> pd.DataFrame:
    show = df.copy()
    for col in cols:
        if col in show.columns:
            show[col] = show[col].apply(lambda x: format_pct(x, digits=digits))
    return show


def money_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    show = df.copy()
    for col in cols:
        if col in show.columns:
            show[col] = show[col].apply(money_krw)
    return show


def usd_money_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    show = df.copy()
    for col in cols:
        if col in show.columns:
            show[col] = show[col].apply(money_usd)
    return show


def date_cols_to_string(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    show = df.copy()
    for col in cols:
        if col in show.columns:
            show[col] = show[col].apply(lambda x: x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else str(x))
    return show


# =========================================================
# Alpha Vantage 데이터 수집
# =========================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_monthly_adjusted(symbol: str, api_key: str) -> pd.DataFrame:
    """Alpha Vantage TIME_SERIES_MONTHLY_ADJUSTED 데이터를 불러옵니다."""
    params = {
        "function": "TIME_SERIES_MONTHLY_ADJUSTED",
        "symbol": symbol,
        "apikey": api_key,
    }
    response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "Error Message" in data:
        raise ValueError(f"{symbol}: Alpha Vantage 오류 - {data['Error Message']}")
    if "Note" in data:
        raise ValueError(
            f"{symbol}: Alpha Vantage 호출 제한 메시지 - {data['Note']}\n"
            "해결 방법: 잠시 후 다시 실행하거나, 오늘 여러 번 실행했다면 다음 날 다시 시도하세요."
        )
    if "Information" in data:
        raise ValueError(
            f"{symbol}: Alpha Vantage 호출 제한 또는 안내 메시지 - {data['Information']}\n"
            "해결 방법: 잠시 후 다시 실행하거나, 오늘 여러 번 실행했다면 다음 날 다시 시도하세요. "
            "무료 API는 호출 제한이 있으므로 캐시 초기화/반복 실행을 줄이는 것이 좋습니다."
        )

    key = "Monthly Adjusted Time Series"
    if key not in data:
        raise ValueError(f"{symbol}: 월봉 데이터를 찾지 못했습니다. 응답 키: {list(data.keys())}")

    df = pd.DataFrame.from_dict(data[key], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df = df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adjusted_close",
            "6. volume": "volume",
            "7. dividend amount": "dividend",
        }
    )

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "adjusted_close" not in df.columns:
        raise ValueError(f"{symbol}: adjusted_close 컬럼이 없습니다.")

    df["symbol"] = symbol
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_usdkrw_daily(api_key: str) -> pd.DataFrame:
    """Alpha Vantage FX_DAILY로 USD/KRW 일별 환율을 불러옵니다."""
    params = {
        "function": "FX_DAILY",
        "from_symbol": "USD",
        "to_symbol": "KRW",
        "outputsize": "compact",
        "apikey": api_key,
    }
    response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "Error Message" in data:
        raise ValueError(f"USD/KRW: Alpha Vantage 오류 - {data['Error Message']}")
    if "Note" in data:
        raise ValueError(
            f"USD/KRW: Alpha Vantage 호출 제한 메시지 - {data['Note']}\n"
            "해결 방법: 잠시 후 다시 실행하거나, 오늘 여러 번 실행했다면 다음 날 다시 시도하세요."
        )
    if "Information" in data:
        raise ValueError(
            f"USD/KRW: Alpha Vantage 호출 제한 또는 안내 메시지 - {data['Information']}\n"
            "해결 방법: 잠시 후 다시 실행하거나, 오늘 여러 번 실행했다면 다음 날 다시 시도하세요. "
            "무료 API는 호출 제한이 있으므로 캐시 초기화/반복 실행을 줄이는 것이 좋습니다."
        )

    key = "Time Series FX (Daily)"
    if key not in data:
        raise ValueError(f"USD/KRW: 일별 환율 데이터를 찾지 못했습니다. 응답 키: {list(data.keys())}")

    df = pd.DataFrame.from_dict(data[key], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
        }
    )

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "close" not in df.columns:
        raise ValueError("USD/KRW: close 컬럼이 없습니다.")

    return df


def select_fx_rate(fx_df: pd.DataFrame, eval_date: date) -> Tuple[float, pd.Timestamp]:
    """평가 기준일 이하의 가장 최근 USD/KRW 종가를 선택합니다."""
    if fx_df.empty:
        raise ValueError("USD/KRW 환율 데이터가 비어 있습니다.")

    eval_ts = pd.Timestamp(eval_date)
    usable = fx_df.loc[fx_df.index <= eval_ts].dropna(subset=["close"])

    if usable.empty:
        usable = fx_df.dropna(subset=["close"])

    if usable.empty:
        raise ValueError("USD/KRW 환율 close 데이터를 찾지 못했습니다.")

    rate_date = usable.index.max()
    rate = float(usable.loc[rate_date, "close"])

    if rate <= 0:
        raise ValueError("USD/KRW 환율이 0 이하입니다. 데이터를 확인하세요.")

    return rate, rate_date


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_global_quote(symbol: str, api_key: str) -> Dict[str, object]:
    """Alpha Vantage GLOBAL_QUOTE로 ETF의 최신 1주 가격을 불러옵니다."""
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key,
    }
    response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "Error Message" in data:
        raise ValueError(f"{symbol}: Alpha Vantage 오류 - {data['Error Message']}")
    if "Note" in data:
        raise ValueError(
            f"{symbol}: Alpha Vantage 호출 제한 메시지 - {data['Note']}\n"
            "해결 방법: 잠시 후 다시 실행하거나, 오늘 여러 번 실행했다면 다음 날 다시 시도하세요."
        )
    if "Information" in data:
        raise ValueError(
            f"{symbol}: Alpha Vantage 호출 제한 또는 안내 메시지 - {data['Information']}\n"
            "해결 방법: 잠시 후 다시 실행하거나, 오늘 여러 번 실행했다면 다음 날 다시 시도하세요. "
            "무료 API는 호출 제한이 있으므로 캐시 초기화/반복 실행을 줄이는 것이 좋습니다."
        )

    quote = data.get("Global Quote", {})
    if not quote:
        raise ValueError(f"{symbol}: 최신 가격 데이터를 찾지 못했습니다. 응답 키: {list(data.keys())}")

    price = pd.to_numeric(quote.get("05. price"), errors="coerce")
    latest_trading_day = quote.get("07. latest trading day")

    if pd.isna(price) or float(price) <= 0:
        raise ValueError(f"{symbol}: 유효한 최신 가격을 찾지 못했습니다.")

    return {
        "ETF": symbol,
        "1주 가격(USD)": float(price),
        "가격 기준일": latest_trading_day or "-",
    }


def load_latest_quotes(tickers: List[str], api_key: str) -> pd.DataFrame:
    """최종 선택된 ETF만 최신 가격을 추가 조회합니다."""
    rows: List[Dict[str, object]] = []
    errors: List[str] = []

    if not tickers:
        return pd.DataFrame(columns=["ETF", "1주 가격(USD)", "가격 기준일"])

    progress = st.progress(0, text="최종 선택 ETF의 최신 1주 가격을 불러오는 중입니다.")
    for i, ticker in enumerate(tickers, start=1):
        try:
            rows.append(fetch_global_quote(ticker, api_key))
        except Exception as e:
            errors.append(str(e))
            rows.append({"ETF": ticker, "1주 가격(USD)": pd.NA, "가격 기준일": "-"})

        progress.progress(i / len(tickers), text=f"최신 가격 로딩: {ticker} ({i}/{len(tickers)})")

        if i < len(tickers):
            time.sleep(API_CALL_DELAY_SECONDS)

    progress.empty()

    if errors:
        with st.expander("최신 가격 로딩 오류 보기", expanded=True):
            for err in errors:
                st.error(err)

    return pd.DataFrame(rows)


def add_share_plan(df: pd.DataFrame, quote_df: pd.DataFrame, usdkrw_rate: float) -> pd.DataFrame:
    """목표 투자금과 1주 가격을 이용해 실제 매수 가능한 정수 주식 수를 계산합니다."""
    result = df.copy()

    if quote_df.empty:
        result["1주 가격(USD)"] = pd.NA
        result["가격 기준일"] = "-"
    else:
        result = result.merge(quote_df, on="ETF", how="left")

    theoretical_shares = []
    buy_shares = []
    actual_usd = []
    actual_krw = []
    remain_usd = []
    remain_krw = []

    for _, row in result.iterrows():
        target_usd = pd.to_numeric(row.get("목표 투자금(USD)"), errors="coerce")
        price = pd.to_numeric(row.get("1주 가격(USD)"), errors="coerce")

        if pd.isna(target_usd) or pd.isna(price) or float(price) <= 0:
            theoretical_shares.append(pd.NA)
            buy_shares.append(pd.NA)
            actual_usd.append(pd.NA)
            actual_krw.append(pd.NA)
            remain_usd.append(pd.NA)
            remain_krw.append(pd.NA)
            continue

        raw_shares = float(target_usd) / float(price)
        shares = math.floor(raw_shares)
        invested_usd = shares * float(price)
        cash_left_usd = float(target_usd) - invested_usd

        theoretical_shares.append(raw_shares)
        buy_shares.append(shares)
        actual_usd.append(invested_usd)
        actual_krw.append(invested_usd * usdkrw_rate)
        remain_usd.append(cash_left_usd)
        remain_krw.append(cash_left_usd * usdkrw_rate)

    result["이론 주수"] = theoretical_shares
    result["매수 주수"] = buy_shares
    result["실제 매수금액(USD)"] = actual_usd
    result["실제 매수금액(KRW)"] = actual_krw
    result["잔여금(USD)"] = remain_usd
    result["잔여금(KRW)"] = remain_krw

    return result


def load_all_monthly_prices(tickers: List[str], api_key: str) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    errors: List[str] = []

    progress = st.progress(0, text="Alpha Vantage에서 ETF 월봉 데이터를 불러오는 중입니다.")
    for i, ticker in enumerate(tickers, start=1):
        try:
            result[ticker] = fetch_monthly_adjusted(ticker, api_key)
        except Exception as e:
            errors.append(str(e))

        progress.progress(i / len(tickers), text=f"ETF 데이터 로딩: {ticker} ({i}/{len(tickers)})")

        if i < len(tickers):
            time.sleep(API_CALL_DELAY_SECONDS)

    progress.empty()

    if errors:
        with st.expander("데이터 로딩 오류 보기", expanded=True):
            for err in errors:
                st.error(err)

    return result


def build_price_matrix(
    data: Dict[str, pd.DataFrame],
    tickers: List[str],
    eval_date: date,
    lookback_months: int,
    exclude_current_month: bool = True,
) -> pd.DataFrame:
    closes = []
    for ticker in tickers:
        if ticker not in data:
            continue
        closes.append(data[ticker]["adjusted_close"].rename(ticker))

    if not closes:
        return pd.DataFrame()

    prices = pd.concat(closes, axis=1).sort_index().dropna(how="all")
    eval_ts = pd.Timestamp(eval_date)
    prices = prices.loc[prices.index <= eval_ts]

    if exclude_current_month and not prices.empty:
        latest = prices.index.max()
        if latest.year == eval_ts.year and latest.month == eval_ts.month:
            prices = prices.loc[prices.index < latest]

    if lookback_months > 0:
        prices = prices.tail(lookback_months)

    return prices


# =========================================================
# 전략 계산 함수
# =========================================================
def calculate_returns(prices: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """1, 3, 6, 12개월 수익률을 계산합니다."""
    rows = []
    for ticker in tickers:
        if ticker not in prices.columns:
            rows.append({"ETF": ticker, "자산군": ETF_LABELS.get(ticker, ""), "1개월 수익률": pd.NA, "3개월 수익률": pd.NA, "6개월 수익률": pd.NA, "12개월 수익률": pd.NA})
            continue

        s = prices[ticker].dropna()
        if len(s) < 13:
            rows.append(
                {
                    "ETF": ticker,
                    "자산군": ETF_LABELS.get(ticker, ""),
                    "기준월": s.index[-1].strftime("%Y-%m-%d") if len(s) else "-",
                    "현재 조정종가": s.iloc[-1] if len(s) else pd.NA,
                    "1개월 수익률": pd.NA,
                    "3개월 수익률": pd.NA,
                    "6개월 수익률": pd.NA,
                    "12개월 수익률": pd.NA,
                }
            )
            continue

        rows.append(
            {
                "ETF": ticker,
                "자산군": ETF_LABELS.get(ticker, ""),
                "기준월": s.index[-1].strftime("%Y-%m-%d"),
                "현재 조정종가": s.iloc[-1],
                "1개월 수익률": s.iloc[-1] / s.iloc[-2] - 1,
                "3개월 수익률": s.iloc[-1] / s.iloc[-4] - 1,
                "6개월 수익률": s.iloc[-1] / s.iloc[-7] - 1,
                "12개월 수익률": s.iloc[-1] / s.iloc[-13] - 1,
            }
        )

    return pd.DataFrame(rows)


def calculate_vaa(prices: pd.DataFrame, zero_is_defensive: bool) -> Tuple[str, pd.DataFrame, str]:
    tickers = VAA_ATTACK + VAA_SAFE
    returns = calculate_returns(prices, tickers)

    need_cols = ["1개월 수익률", "3개월 수익률", "6개월 수익률", "12개월 수익률"]
    if returns.empty or returns[need_cols].isna().any().any():
        raise ValueError("VAA 계산에 필요한 13개월 이상의 월봉 데이터가 부족합니다.")

    returns["모멘텀 스코어"] = (
        12 * returns["1개월 수익률"]
        + 4 * returns["3개월 수익률"]
        + 2 * returns["6개월 수익률"]
        + returns["12개월 수익률"]
    )
    returns["구분"] = returns["ETF"].apply(lambda x: "공격형" if x in VAA_ATTACK else "안전자산")

    attack = returns[returns["ETF"].isin(VAA_ATTACK)].copy()
    safe = returns[returns["ETF"].isin(VAA_SAFE)].copy()

    if zero_is_defensive:
        attack_ok = bool((attack["모멘텀 스코어"] > 0).all())
        threshold_text = "공격형 4개 ETF의 모멘텀 스코어가 모두 0 초과"
    else:
        attack_ok = bool((attack["모멘텀 스코어"] >= 0).all())
        threshold_text = "공격형 4개 ETF의 모멘텀 스코어가 모두 0 이상"

    if attack_ok:
        pool = attack
        reason = f"{threshold_text} → 공격형 중 최고 점수 ETF 선택"
    else:
        pool = safe
        reason = f"{threshold_text} 조건 미충족 → 안전자산 중 최고 점수 ETF 선택"

    selected = pool.sort_values("모멘텀 스코어", ascending=False).iloc[0]["ETF"]
    scores = returns.sort_values(["구분", "모멘텀 스코어"], ascending=[True, False])
    return selected, scores, reason


def calculate_dual_momentum(prices: pd.DataFrame) -> Tuple[str, pd.DataFrame, str]:
    returns = calculate_returns(prices, ODM_ASSETS)
    if returns.empty or returns["12개월 수익률"].isna().any():
        raise ValueError("오리지널 듀얼 모멘텀 계산에 필요한 13개월 이상의 월봉 데이터가 부족합니다.")

    r = returns.set_index("ETF")["12개월 수익률"]
    spy_r = float(r["SPY"])
    efa_r = float(r["EFA"])
    bil_r = float(r["BIL"])

    if spy_r > bil_r:
        selected = "SPY" if spy_r >= efa_r else "EFA"
        reason = "SPY 12개월 수익률이 BIL보다 높아 SPY/EFA 중 더 강한 ETF 선택"
    else:
        selected = "AGG"
        reason = "SPY 12개월 수익률이 BIL보다 낮거나 같아 AGG 선택"

    returns = returns.sort_values("12개월 수익률", ascending=False)
    return selected, returns, reason


def allocation_rows(
    strategy: str,
    strategy_weight: float,
    inner_weights: Dict[str, float],
    rebalance_info: Dict[str, Dict[str, object]],
    eval_date: date,
    total_investment_krw: float,
    total_investment_usd: float,
    input_currency: str,
    reason: str = "",
) -> List[Dict[str, object]]:
    rows = []
    for ticker, inner_weight in inner_weights.items():
        info = rebalance_info[ticker]
        cycle = str(info["cycle"])
        last_date = info["last_date"]
        next_date = next_rebalance_date(last_date, cycle)
        total_weight = strategy_weight * inner_weight
        rows.append(
            {
                "하위전략": strategy,
                "ETF": ticker,
                "자산군": ETF_LABELS.get(ticker, ""),
                "하위전략 내 비중": inner_weight,
                "강환국 전략 전체 비중": total_weight,
                "목표 투자금(KRW)": total_investment_krw * total_weight,
                "목표 투자금(USD)": total_investment_usd * total_weight,
                "배분 기준 화폐": input_currency,
                "리밸런싱 주기": cycle,
                "최근 리밸런싱일": last_date,
                "다음 리밸런싱일": next_date,
                "리밸런싱 상태": rebalance_status(next_date, eval_date),
                "선정 사유": reason,
            }
        )
    return rows


def appendix_buy_strategy_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "전략": "LAA",
                "대상 ETF": "IWD, GLD, IEF, QQQ, SHY",
                "매수 전략": "IWD/GLD/IEF 각 25% 고정. 나머지 25%는 S&P500 200일선 하회와 미국 실업률 12개월 평균 상회가 동시에 O이면 SHY, 아니면 QQQ.",
                "데이터/입력": "S&P500/실업률 조건은 사용자가 직접 확인 후 O/X 입력",
                "리밸런싱": "IWD/GLD/IEF 연 1회, QQQ/SHY 월 1회",
            },
            {
                "전략": "VAA 공격형",
                "대상 ETF": "공격형: SPY, EFA, EEM, AGG / 안전자산: LQD, IEF, SHY",
                "매수 전략": "공격형 4개 ETF의 모멘텀 스코어가 모두 양호하면 공격형 1위 100%, 하나라도 방어 신호면 안전자산 1위 100%.",
                "데이터/입력": "Alpha Vantage 월별 조정종가로 자동 계산",
                "리밸런싱": "월 1회",
            },
            {
                "전략": "오리지널 듀얼 모멘텀",
                "대상 ETF": "SPY, EFA, BIL, AGG",
                "매수 전략": "SPY 12개월 수익률이 BIL보다 높으면 SPY/EFA 중 높은 ETF 100%, 낮거나 같으면 AGG 100%.",
                "데이터/입력": "Alpha Vantage 월별 조정종가로 자동 계산",
                "리밸런싱": "월 1회",
            },
        ]
    )


def appendix_rebalance_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "구분": "월 1회 리밸런싱",
                "계산 방식": "최근 리밸런싱일 + 1개월",
                "월말 처리": "최근 리밸런싱일이 월말이면 다음 리밸런싱일도 다음 달 월말로 계산",
                "상태 판정": "다음 리밸런싱일이 평가 기준일 이하이면 리밸런싱 필요",
            },
            {
                "구분": "연 1회 리밸런싱",
                "계산 방식": "최근 리밸런싱일 + 1년",
                "월말 처리": "2월 29일 같은 예외는 다음 해 가능한 마지막 날짜로 보정",
                "상태 판정": "다음 리밸런싱일이 평가 기준일 이하이면 리밸런싱 필요",
            },
            {
                "구분": "VAA 모멘텀 스코어",
                "계산 방식": "12×1개월 수익률 + 4×3개월 수익률 + 2×6개월 수익률 + 1×12개월 수익률",
                "월말 처리": "진행 중인 월 데이터 제외 옵션 사용 시 완성된 최근 월봉까지만 사용",
                "상태 판정": "공격형 4개 ETF가 모두 기준 이상인지 확인",
            },
            {
                "구분": "오리지널 듀얼 모멘텀",
                "계산 방식": "SPY, EFA, BIL의 12개월 수익률 비교",
                "월말 처리": "완성된 최근 월봉 기준으로 12개월 수익률 계산",
                "상태 판정": "SPY > BIL이면 SPY/EFA 비교, 아니면 AGG 선택",
            },
            {
                "구분": "투자금 입력 화폐",
                "계산 방식": "KRW 또는 USD 중 선택한 화폐를 기준으로 전체 배분금액 계산",
                "월말 처리": "USD/KRW 환율은 평가 기준일 이하의 가장 최근 일별 종가 사용",
                "상태 판정": "KRW 입력 시 USD=KRW/환율, USD 입력 시 KRW=USD×환율로 환산",
            },
            {
                "구분": "정수 주식 수 계산",
                "계산 방식": "목표 투자금(USD) ÷ ETF 1주 가격(USD) 후 소수점 이하는 버림",
                "월말 처리": "최종 선택 ETF만 최신 1주 가격을 추가 조회",
                "상태 판정": "실제 매수금액과 남는 현금을 KRW/USD로 함께 표시",
            },
            {
                "구분": "USD/KRW 환산",
                "계산 방식": "원화 투자금과 달러 투자금을 모두 결과표에 표시",
                "월말 처리": "평가 기준일 이하의 가장 최근 일별 USD/KRW 종가 사용",
                "상태 판정": "환율 기준일과 적용 환율을 결과표에 함께 표시",
            },
        ]
    )


# =========================================================
# 화면
# =========================================================
st.title("ETF 자산배분")
st.caption(
    "VAA 공격형 · LAA · 오리지널 듀얼 모멘텀 전략의 최종 매수 비중, 목표 투자금, 리밸런싱 일정을 계산합니다."
)

api_key = get_secret_api_key()
today = date.today()
default_monthly_last = today - relativedelta(months=1)
default_annual_last = today - relativedelta(years=1)

with st.sidebar:
    st.header("1) 조회 기준")

    if api_key:
        st.success("Alpha Vantage API Key를 Secrets에서 불러왔습니다.")
    else:
        st.error("Secrets에 ALPHA_VANTAGE_API_KEY가 없습니다.")

    eval_date = st.date_input("평가 기준일", value=today)
    lookback_months = st.slider(
        "데이터 조회기간",
        min_value=13,
        max_value=60,
        value=15,
        help="12개월 수익률 계산을 위해 최소 13개월 이상의 월봉 데이터가 필요합니다.",
    )
    investment_currency = st.radio(
        "배분 기준 화폐",
        options=["KRW", "USD"],
        format_func=lambda x: "원화(KRW)로 입력" if x == "KRW" else "달러(USD)로 입력",
        horizontal=True,
        help="선택한 화폐로 총 투자금을 입력하고, 결과표에는 KRW와 USD 목표 투자금을 모두 표시합니다.",
    )

    if investment_currency == "KRW":
        total_investment_input = st.number_input(
            "총 투자금(KRW)",
            min_value=0,
            value=10_000_000,
            step=1,
            format="%d",
            help="원 단위까지 입력할 수 있습니다. USD 목표금액은 USD/KRW 환율로 자동 환산합니다.",
        )
        total_investment_input = int(total_investment_input) if total_investment_input else 0
    else:
        total_investment_input = st.number_input(
            "총 투자금(USD)",
            min_value=0.0,
            value=10_000.0,
            step=1.0,
            format="%.2f",
            help="달러 단위로 입력할 수 있습니다. KRW 목표금액은 USD/KRW 환율로 자동 환산합니다.",
        )
        total_investment_input = float(total_investment_input) if total_investment_input else 0.0

    st.header("2) LAA 수동 조건")
    laa_defensive = st.radio(
        "S&P500 200일선 하회 + 미국 실업률 12개월 평균 상회 여부",
        options=[False, True],
        format_func=lambda x: "X: 조건 미충족 → QQQ" if not x else "O: 조건 충족 → SHY",
        index=0,
        help="두 조건이 모두 O일 때만 LAA 변동 25%를 SHY로 선택합니다. 그 외에는 QQQ입니다.",
    )

    st.header("3) 최근 리밸런싱일")
    laa_annual_last = st.date_input(
        "LAA 고정자산 최근 리밸런싱일",
        value=default_annual_last,
        help="IWD, GLD, IEF의 최근 연간 리밸런싱일",
    )
    laa_monthly_last = st.date_input(
        "LAA 변동자산 최근 리밸런싱일",
        value=default_monthly_last,
        help="QQQ 또는 SHY의 최근 월간 리밸런싱일",
    )
    vaa_monthly_last = st.date_input(
        "VAA 공격형 최근 리밸런싱일",
        value=default_monthly_last,
        help="VAA 공격형 전략의 최근 월간 리밸런싱일",
    )
    odm_monthly_last = st.date_input(
        "오리지널 듀얼 모멘텀 최근 리밸런싱일",
        value=default_monthly_last,
        help="오리지널 듀얼 모멘텀 전략의 최근 월간 리밸런싱일",
    )

    st.header("4) 하위전략 비중")
    w_laa_input = st.number_input(
        "LAA 비중",
        min_value=0.0,
        max_value=1.0,
        value=1 / 3,
        step=0.01,
        format="%.4f",
    )
    w_vaa_input = st.number_input(
        "VAA 공격형 비중",
        min_value=0.0,
        max_value=1.0,
        value=1 / 3,
        step=0.01,
        format="%.4f",
    )
    w_odm_input = st.number_input(
        "오리지널 듀얼 모멘텀 비중",
        min_value=0.0,
        max_value=1.0,
        value=1 / 3,
        step=0.01,
        format="%.4f",
    )

    st.header("5) VAA 판정 기준")
    zero_is_defensive = st.checkbox(
        "모멘텀 스코어 0점은 방어 신호로 처리",
        value=True,
        help="체크하면 공격형 ETF 4개가 모두 0점 초과일 때만 공격형으로 판단합니다. 해제하면 0점 이상도 공격형으로 봅니다.",
    )

    st.header("6) 데이터 옵션")
    exclude_current_month = st.checkbox(
        "진행 중인 월 데이터 제외",
        value=True,
        help="월말 리밸런싱 전략이므로 기본값은 현재 진행 중인 월 데이터를 제외합니다.",
    )
    st.caption("KRW↔USD 환산은 Alpha Vantage USD/KRW 일별 환율의 평가 기준일 이하 최근 종가를 사용합니다.")
    if st.button("캐시 초기화"):
        st.cache_data.clear()
        st.success("캐시를 초기화했습니다. 다시 계산 버튼을 눌러주세요.")

w_laa, w_vaa, w_odm, weight_sum = normalize_strategy_weights(w_laa_input, w_vaa_input, w_odm_input)

if weight_sum <= 0:
    st.error("하위전략 비중 합계가 0입니다. 비중을 입력하세요.")
    st.stop()

if abs(weight_sum - 1.0) > 1e-8:
    st.warning(f"하위전략 비중 합계가 {weight_sum:.4f}입니다. 계산 시 100% 기준으로 자동 정규화합니다.")

with st.expander("미국 ETF 구성 확인", expanded=False):
    etf_rows = []
    for ticker in ALL_TICKERS:
        uses = []
        if ticker in LAA_FIXED:
            uses.append("LAA 고정")
        if ticker in LAA_VARIABLE:
            uses.append("LAA 변동")
        if ticker in VAA_ATTACK:
            uses.append("VAA 공격형")
        if ticker in VAA_SAFE:
            uses.append("VAA 안전자산")
        if ticker in ODM_ASSETS:
            uses.append("듀얼모멘텀")
        etf_rows.append(
            {
                "ETF": ticker,
                "자산군": ETF_LABELS.get(ticker, ""),
                "사용 전략": ", ".join(uses),
                "Alpha Vantage 호출 여부": "O" if ticker in DATA_TICKERS else "X - LAA 수동 계산용",
            }
        )
    st.dataframe(pd.DataFrame(etf_rows), use_container_width=True, hide_index=True)

st.subheader("입력된 리밸런싱 일정")
schedule_preview = pd.DataFrame(
    [
        {
            "구분": "LAA 고정자산",
            "대상": "IWD, GLD, IEF",
            "주기": "연 1회",
            "최근 리밸런싱일": laa_annual_last,
            "다음 리밸런싱일": next_rebalance_date(laa_annual_last, "연 1회"),
            "상태": rebalance_status(next_rebalance_date(laa_annual_last, "연 1회"), eval_date),
        },
        {
            "구분": "LAA 변동자산",
            "대상": "QQQ 또는 SHY",
            "주기": "월 1회",
            "최근 리밸런싱일": laa_monthly_last,
            "다음 리밸런싱일": next_rebalance_date(laa_monthly_last, "월 1회"),
            "상태": rebalance_status(next_rebalance_date(laa_monthly_last, "월 1회"), eval_date),
        },
        {
            "구분": "VAA 공격형",
            "대상": "선택 ETF 1개",
            "주기": "월 1회",
            "최근 리밸런싱일": vaa_monthly_last,
            "다음 리밸런싱일": next_rebalance_date(vaa_monthly_last, "월 1회"),
            "상태": rebalance_status(next_rebalance_date(vaa_monthly_last, "월 1회"), eval_date),
        },
        {
            "구분": "오리지널 듀얼 모멘텀",
            "대상": "선택 ETF 1개",
            "주기": "월 1회",
            "최근 리밸런싱일": odm_monthly_last,
            "다음 리밸런싱일": next_rebalance_date(odm_monthly_last, "월 1회"),
            "상태": rebalance_status(next_rebalance_date(odm_monthly_last, "월 1회"), eval_date),
        },
    ]
)
st.dataframe(date_cols_to_string(schedule_preview, ["최근 리밸런싱일", "다음 리밸런싱일"]), use_container_width=True, hide_index=True)

with st.expander("Appendix. 매수 전략 및 리밸런싱 일정 계산 방식", expanded=False):
    tab1, tab2 = st.tabs(["매수 전략 요약", "리밸런싱 일정 계산 방식"])
    with tab1:
        st.dataframe(appendix_buy_strategy_table(), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(appendix_rebalance_table(), use_container_width=True, hide_index=True)

c_total, c_weight, c_api = st.columns(3)
c_total.metric("현재 총 투자금", format_input_amount(total_investment_input, investment_currency))
c_weight.metric("하위전략 비중", f"LAA {w_laa:.2%} / VAA {w_vaa:.2%} / ODM {w_odm:.2%}")
c_api.metric("API 호출 대상", f"{len(DATA_TICKERS)}개 ETF + 환율 + 선택 ETF 가격")

if not api_key:
    st.warning("Streamlit Secrets에 ALPHA_VANTAGE_API_KEY를 저장한 뒤 실행하세요.")
    st.stop()

st.info(
    "LAA의 S&P500 200일선/실업률 조건은 수동 O/X 입력값을 사용합니다. "
    "VAA와 오리지널 듀얼 모멘텀은 Alpha Vantage 월별 조정종가를 사용합니다. "
    f"총 투자금은 {investment_currency} 기준으로 배분하고, 결과표에는 KRW와 USD를 함께 표시합니다. "
    f"이번 실행의 기본 API 호출 대상은 {', '.join(DATA_TICKERS)} 및 USD/KRW 일별 환율입니다. "
    "계산 후 최종 선택 ETF의 최신 1주 가격을 추가 조회해 정수 주식 수를 계산합니다."
)

run = st.button("전략 비중 계산", type="primary")

if run:
    data = load_all_monthly_prices(DATA_TICKERS, api_key)
    prices = build_price_matrix(
        data,
        DATA_TICKERS,
        eval_date=eval_date,
        lookback_months=lookback_months,
        exclude_current_month=exclude_current_month,
    )

    if prices.empty:
        st.error("ETF 가격 데이터를 가져오지 못했습니다. 기준일, API Key, 호출 제한 상태를 확인하세요.")
        st.stop()

    # USD 환산용 일별 USD/KRW 환율 조회
    try:
        time.sleep(API_CALL_DELAY_SECONDS)
        fx_df = fetch_usdkrw_daily(api_key)
        usdkrw_rate, usdkrw_rate_date = select_fx_rate(fx_df, eval_date)
    except Exception as e:
        st.error(f"USD/KRW 환율 데이터를 가져오지 못했습니다: {e}")
        st.stop()

    if investment_currency == "KRW":
        total_investment_krw = float(total_investment_input)
        total_investment_usd = total_investment_krw / usdkrw_rate
    else:
        total_investment_usd = float(total_investment_input)
        total_investment_krw = total_investment_usd * usdkrw_rate

    actual_eval_dt = prices.index.max()
    st.success(f"계산 기준월: {actual_eval_dt.strftime('%Y-%m-%d')}")

    fx_c1, fx_c2, fx_c3, fx_c4 = st.columns(4)
    fx_c1.metric("배분 기준 화폐", investment_currency)
    fx_c2.metric("총 투자금(KRW)", money_krw(total_investment_krw))
    fx_c3.metric("총 투자금(USD)", money_usd(total_investment_usd))
    fx_c4.metric("적용 USD/KRW 환율", fx_rate_krw(usdkrw_rate))
    st.caption(f"환율 기준일: {usdkrw_rate_date.strftime('%Y-%m-%d')}")

    with st.expander("계산 기준월 ETF 가격 확인", expanded=False):
        latest_rows = []
        for ticker in DATA_TICKERS:
            if ticker in prices.columns and pd.notna(prices[ticker].dropna().iloc[-1]):
                latest_rows.append(
                    {
                        "ETF": ticker,
                        "자산군": ETF_LABELS.get(ticker, ""),
                        "기준월": actual_eval_dt.strftime("%Y-%m-%d"),
                        "조정종가": prices[ticker].dropna().iloc[-1],
                    }
                )
        latest_df = pd.DataFrame(latest_rows)
        if not latest_df.empty:
            latest_df["조정종가"] = latest_df["조정종가"].apply(lambda x: f"{x:,.2f}")
        st.dataframe(latest_df, use_container_width=True, hide_index=True)

    rows: List[Dict[str, object]] = []

    # LAA
    laa_variable = "SHY" if laa_defensive else "QQQ"
    laa_reason = (
        "S&P500 200일선 하회와 미국 실업률 12개월 평균 상회 조건이 모두 충족되어 SHY 선택"
        if laa_defensive
        else "두 조건이 동시에 충족되지 않아 QQQ 선택"
    )
    laa_inner = {"IWD": 0.25, "GLD": 0.25, "IEF": 0.25, laa_variable: 0.25}
    laa_rebalance = {
        "IWD": {"cycle": "연 1회", "last_date": laa_annual_last},
        "GLD": {"cycle": "연 1회", "last_date": laa_annual_last},
        "IEF": {"cycle": "연 1회", "last_date": laa_annual_last},
        laa_variable: {"cycle": "월 1회", "last_date": laa_monthly_last},
    }
    rows += allocation_rows(
        "LAA",
        w_laa,
        laa_inner,
        laa_rebalance,
        eval_date,
        total_investment_krw,
        total_investment_usd,
        investment_currency,
        laa_reason,
    )

    # VAA
    try:
        vaa_selected, vaa_scores, vaa_reason = calculate_vaa(prices, zero_is_defensive)
    except Exception as e:
        st.error(str(e))
        st.stop()

    rows += allocation_rows(
        "VAA 공격형",
        w_vaa,
        {vaa_selected: 1.0},
        {vaa_selected: {"cycle": "월 1회", "last_date": vaa_monthly_last}},
        eval_date,
        total_investment_krw,
        total_investment_usd,
        investment_currency,
        vaa_reason,
    )

    # ODM
    try:
        odm_selected, odm_returns, odm_reason = calculate_dual_momentum(prices)
    except Exception as e:
        st.error(str(e))
        st.stop()

    rows += allocation_rows(
        "오리지널 듀얼 모멘텀",
        w_odm,
        {odm_selected: 1.0},
        {odm_selected: {"cycle": "월 1회", "last_date": odm_monthly_last}},
        eval_date,
        total_investment_krw,
        total_investment_usd,
        investment_currency,
        odm_reason,
    )

    st.subheader("하위전략별 선택 결과")
    c1, c2, c3 = st.columns(3)
    c1.metric("LAA 변동 25%", f"{laa_variable} · {ETF_LABELS[laa_variable]}")
    c1.caption(f"다음 월간 리밸런싱일: {next_rebalance_date(laa_monthly_last, '월 1회')}")

    c2.metric("VAA 공격형", f"{vaa_selected} · {ETF_LABELS[vaa_selected]}")
    c2.caption(f"{vaa_reason} / 다음 리밸런싱일: {next_rebalance_date(vaa_monthly_last, '월 1회')}")

    c3.metric("오리지널 듀얼 모멘텀", f"{odm_selected} · {ETF_LABELS[odm_selected]}")
    c3.caption(f"{odm_reason} / 다음 리밸런싱일: {next_rebalance_date(odm_monthly_last, '월 1회')}")

    st.subheader("VAA 모멘텀 스코어")
    vaa_display = pct_cols(
        vaa_scores,
        ["1개월 수익률", "3개월 수익률", "6개월 수익률", "12개월 수익률"],
    )
    vaa_display["모멘텀 스코어"] = vaa_display["모멘텀 스코어"].apply(format_score)
    if "현재 조정종가" in vaa_display.columns:
        vaa_display["현재 조정종가"] = vaa_display["현재 조정종가"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
    st.dataframe(vaa_display, use_container_width=True, hide_index=True)

    st.subheader("오리지널 듀얼 모멘텀 12개월 수익률")
    odm_display = pct_cols(odm_returns, ["1개월 수익률", "3개월 수익률", "6개월 수익률", "12개월 수익률"])
    if "현재 조정종가" in odm_display.columns:
        odm_display["현재 조정종가"] = odm_display["현재 조정종가"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
    st.dataframe(odm_display, use_container_width=True, hide_index=True)

    detail = pd.DataFrame(rows)
    detail["적용 환율"] = usdkrw_rate
    detail["환율 기준일"] = usdkrw_rate_date.date()

    # 최종 선택된 ETF의 최신 1주 가격을 조회해 실제 매수 가능한 정수 주식 수를 계산합니다.
    selected_tickers = sorted(detail["ETF"].dropna().unique().tolist())
    time.sleep(API_CALL_DELAY_SECONDS)
    quote_df = load_latest_quotes(selected_tickers, api_key)

    with st.expander("최종 선택 ETF 최신 1주 가격 확인", expanded=False):
        quote_display = quote_df.copy()
        if not quote_display.empty and "1주 가격(USD)" in quote_display.columns:
            quote_display["1주 가격(USD)"] = quote_display["1주 가격(USD)"].apply(usd_price)
        st.dataframe(quote_display, use_container_width=True, hide_index=True)

    detail = add_share_plan(detail, quote_df, usdkrw_rate)

    st.subheader("하위전략별 투입 비중 및 리밸런싱 일정")
    detail_display = pct_cols(detail, ["하위전략 내 비중", "강환국 전략 전체 비중"])
    detail_display = money_cols(detail_display, ["목표 투자금(KRW)", "실제 매수금액(KRW)", "잔여금(KRW)"])
    detail_display = usd_money_cols(detail_display, ["목표 투자금(USD)", "실제 매수금액(USD)", "잔여금(USD)"])
    if "1주 가격(USD)" in detail_display.columns:
        detail_display["1주 가격(USD)"] = detail_display["1주 가격(USD)"].apply(usd_price)
    if "이론 주수" in detail_display.columns:
        detail_display["이론 주수"] = detail_display["이론 주수"].apply(format_fractional_shares)
    if "매수 주수" in detail_display.columns:
        detail_display["매수 주수"] = detail_display["매수 주수"].apply(format_share_count)
    detail_display["적용 환율"] = detail_display["적용 환율"].apply(fx_rate_krw)
    detail_display = date_cols_to_string(detail_display, ["최근 리밸런싱일", "다음 리밸런싱일", "환율 기준일"])
    st.dataframe(detail_display, use_container_width=True, hide_index=True)

    final = (
        detail.groupby(["ETF", "자산군"], as_index=False)
        .agg(
            {
                "강환국 전략 전체 비중": "sum",
                "목표 투자금(KRW)": "sum",
                "목표 투자금(USD)": "sum",
                "다음 리밸런싱일": lambda x: min(v for v in x if v is not None),
                "리밸런싱 상태": lambda x: "리밸런싱 필요" if "리밸런싱 필요" in list(x) else "대기",
            }
        )
        .sort_values("강환국 전략 전체 비중", ascending=False)
    )

    final["적용 환율"] = usdkrw_rate
    final["환율 기준일"] = usdkrw_rate_date.date()
    final = add_share_plan(final, quote_df, usdkrw_rate)

    st.subheader("최종 매수 비중 및 대략 매수 주수")
    final_display = pct_cols(final, ["강환국 전략 전체 비중"])
    final_display = money_cols(final_display, ["목표 투자금(KRW)", "실제 매수금액(KRW)", "잔여금(KRW)"])
    final_display = usd_money_cols(final_display, ["목표 투자금(USD)", "실제 매수금액(USD)", "잔여금(USD)"])
    if "1주 가격(USD)" in final_display.columns:
        final_display["1주 가격(USD)"] = final_display["1주 가격(USD)"].apply(usd_price)
    if "이론 주수" in final_display.columns:
        final_display["이론 주수"] = final_display["이론 주수"].apply(format_fractional_shares)
    if "매수 주수" in final_display.columns:
        final_display["매수 주수"] = final_display["매수 주수"].apply(format_share_count)
    final_display["적용 환율"] = final_display["적용 환율"].apply(fx_rate_krw)
    final_display = date_cols_to_string(final_display, ["다음 리밸런싱일", "환율 기준일"])
    st.dataframe(final_display, use_container_width=True, hide_index=True)

    share_summary = final.dropna(subset=["매수 주수"]).copy()
    if not share_summary.empty:
        invested_usd_total = share_summary["실제 매수금액(USD)"].sum()
        remaining_usd_total = total_investment_usd - invested_usd_total
        s1, s2, s3 = st.columns(3)
        s1.metric("실제 매수금액(USD)", money_usd(invested_usd_total))
        s2.metric("남는 현금(USD)", money_usd(remaining_usd_total))
        s3.metric("남는 현금(KRW)", money_krw(remaining_usd_total * usdkrw_rate))

    st.bar_chart(final.set_index("ETF")["강환국 전략 전체 비중"])

    csv = final.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "최종 비중 CSV 다운로드",
        data=csv,
        file_name=f"kang_us_etf_allocation_{actual_eval_dt.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

else:
    st.info("배분 기준 화폐와 총 투자금, 조회 기준, LAA 조건, 최근 리밸런싱일, 하위전략 비중을 선택한 뒤 '전략 비중 계산'을 누르세요. 계산 후 최종 선택 ETF별 대략 매수 주수도 함께 표시됩니다.")

    st.subheader("기본 전략별 ETF")
    st.dataframe(
        pd.DataFrame(
            [
                {"전략": "LAA", "ETF": ", ".join(LAA_FIXED + LAA_VARIABLE)},
                {"전략": "VAA 공격형", "ETF": ", ".join(VAA_ATTACK + VAA_SAFE)},
                {"전략": "오리지널 듀얼 모멘텀", "ETF": ", ".join(ODM_ASSETS)},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
