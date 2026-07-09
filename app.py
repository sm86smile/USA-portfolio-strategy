import calendar
import time
from datetime import date
from typing import Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="강환국 ETF 전략 대시보드",
    page_icon="📈",
    layout="wide",
)

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
# Alpha Vantage 무료 API는 호출 제한이 있으므로 요청 사이에 여유를 둡니다.
API_CALL_DELAY_SECONDS = 1.25

LAA_CORE = ["IWD", "GLD", "IEF"]
LAA_TACTICAL = ["QQQ", "SHY"]
VAA_OFFENSIVE = ["SPY", "EFA", "EEM", "AGG"]
VAA_DEFENSIVE = ["LQD", "IEF", "SHY"]
ODM_ASSETS = ["SPY", "EFA", "BIL", "AGG"]
ALL_TICKERS = sorted(set(LAA_CORE + LAA_TACTICAL + VAA_OFFENSIVE + VAA_DEFENSIVE + ODM_ASSETS))
# LAA는 가격 데이터 없이 수동 조건으로 비중이 결정됩니다.
# 따라서 API 호출은 VAA/듀얼모멘텀 계산에 필요한 ETF로만 제한합니다.
DATA_TICKERS = sorted(set(VAA_OFFENSIVE + VAA_DEFENSIVE + ODM_ASSETS))

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
    """
    Streamlit Secrets에서 Alpha Vantage API Key를 읽는다.
    앱 화면에서는 API Key를 직접 입력받지 않는다.
    """
    try:
        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
    except Exception:
        return ""

    return str(api_key).strip()


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x * 100:.2f}%"


def format_score(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:.4f}"


def format_krw(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{int(round(x)):,.0f}원"


def is_last_day(d: date) -> bool:
    return d.day == calendar.monthrange(d.year, d.month)[1]


def add_months(d: date, months: int = 1) -> date:
    """월말에 리밸런싱했다면 다음 달도 월말로 계산한다."""
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    last_day_target_month = calendar.monthrange(year, month)[1]

    if is_last_day(d):
        day = last_day_target_month
    else:
        day = min(d.day, last_day_target_month)

    return date(year, month, day)


def add_years(d: date, years: int = 1) -> date:
    """2월 29일 같은 예외를 고려해 연 단위 날짜를 계산한다."""
    try:
        return date(d.year + years, d.month, d.day)
    except ValueError:
        # 예: 2024-02-29 + 1년 => 2025-02-28
        return date(d.year + years, d.month, calendar.monthrange(d.year + years, d.month)[1])


def normalize_amount_input(value: int) -> int:
    return int(value) if value else 0


# =========================================================
# Alpha Vantage 데이터 수집
# =========================================================
@st.cache_data(ttl="12h", show_spinner=False)
def fetch_monthly_adjusted(symbol: str, api_key: str) -> pd.DataFrame:
    """
    Alpha Vantage TIME_SERIES_MONTHLY_ADJUSTED 사용.
    월말 리밸런싱 전략이므로 일봉보다 월봉 조정종가가 적합하다.
    """
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
        raise ValueError(f"{symbol}: Alpha Vantage 호출 제한 메시지 - {data['Note']}")
    if "Information" in data:
        info = data["Information"]
        raise ValueError(
            f"{symbol}: Alpha Vantage 호출 제한 또는 안내 메시지 - {info} "
            "\n해결 방법: 잠시 후 다시 실행하거나, 오늘 이미 여러 번 실행했다면 다음 날 다시 시도하세요. "
            "무료 API는 일일 호출 수 제한이 있어 캐시 초기화/반복 실행을 줄이는 것이 좋습니다."
        )

    key = "Monthly Adjusted Time Series"
    if key not in data:
        raise ValueError(f"{symbol}: 월봉 데이터를 찾지 못했습니다. 응답: {list(data.keys())}")

    df = pd.DataFrame.from_dict(data[key], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    rename_map = {
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. adjusted close": "adjusted_close",
        "6. volume": "volume",
        "7. dividend amount": "dividend",
    }
    df = df.rename(columns=rename_map)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "adjusted_close" not in df.columns:
        raise ValueError(f"{symbol}: adjusted_close 컬럼이 없습니다.")

    df["symbol"] = symbol
    return df


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

        # 무료 API의 초당 호출 제한을 피하기 위해 실제 호출 사이에 간격을 둡니다.
        # 캐시에 이미 저장된 데이터도 이 루프를 지나지만, 안정성을 우선합니다.
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
    exclude_current_month: bool = True,
) -> pd.DataFrame:
    closes = []
    for ticker in tickers:
        if ticker not in data:
            continue
        s = data[ticker]["adjusted_close"].rename(ticker)
        closes.append(s)

    if not closes:
        return pd.DataFrame()

    prices = pd.concat(closes, axis=1).sort_index()
    prices = prices.dropna(how="all")

    if exclude_current_month and len(prices) > 0:
        today = pd.Timestamp.today()
        latest = prices.index.max()
        if latest.year == today.year and latest.month == today.month:
            prices = prices.loc[prices.index < latest]

    return prices


# =========================================================
# 전략 계산 함수
# =========================================================
def calculate_returns(prices: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """1, 3, 6, 12개월 수익률 계산."""
    rows = []
    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        s = prices[ticker].dropna()
        if len(s) < 13:
            rows.append({
                "ETF": ticker,
                "자산군": ETF_LABELS.get(ticker, ""),
                "1M": pd.NA,
                "3M": pd.NA,
                "6M": pd.NA,
                "12M": pd.NA,
            })
            continue

        rows.append({
            "ETF": ticker,
            "자산군": ETF_LABELS.get(ticker, ""),
            "기준월": s.index[-1].strftime("%Y-%m-%d"),
            "현재 조정종가": s.iloc[-1],
            "1M": s.iloc[-1] / s.iloc[-2] - 1,
            "3M": s.iloc[-1] / s.iloc[-4] - 1,
            "6M": s.iloc[-1] / s.iloc[-7] - 1,
            "12M": s.iloc[-1] / s.iloc[-13] - 1,
        })

    return pd.DataFrame(rows)


def calculate_vaa(prices: pd.DataFrame, total_investment: int, vaa_zero_rule: str) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    tickers = VAA_OFFENSIVE + VAA_DEFENSIVE
    returns = calculate_returns(prices, tickers)

    if returns.empty or returns[["1M", "3M", "6M", "12M"]].isna().any().any():
        raise ValueError("VAA 계산에 필요한 13개월 이상의 월봉 데이터가 부족합니다.")

    returns["모멘텀 스코어"] = (
        12 * returns["1M"] +
        4 * returns["3M"] +
        2 * returns["6M"] +
        1 * returns["12M"]
    )
    returns["그룹"] = returns["ETF"].apply(lambda x: "공격형" if x in VAA_OFFENSIVE else "안전자산")

    offensive_scores = returns[returns["ETF"].isin(VAA_OFFENSIVE)].set_index("ETF")["모멘텀 스코어"]

    if vaa_zero_rule == "공격형 4개 모두 0 이상이면 공격형":
        use_offensive = bool((offensive_scores >= 0).all())
        rule_text = "공격형 4개 ETF의 모멘텀 스코어가 모두 0 이상"
    else:
        use_offensive = bool((offensive_scores > 0).all())
        rule_text = "공격형 4개 ETF의 모멘텀 스코어가 모두 0 초과"

    if use_offensive:
        candidate = returns[returns["ETF"].isin(VAA_OFFENSIVE)].copy()
        selected_group = "공격형"
    else:
        candidate = returns[returns["ETF"].isin(VAA_DEFENSIVE)].copy()
        selected_group = "안전자산"

    selected_etf = candidate.sort_values("모멘텀 스코어", ascending=False).iloc[0]["ETF"]

    allocation_rows = []
    for ticker in tickers:
        weight = 1.0 if ticker == selected_etf else 0.0
        allocation_rows.append({
            "전략": "VAA 공격형",
            "ETF": ticker,
            "자산군": ETF_LABELS.get(ticker, ""),
            "비중": weight,
            "투자금": total_investment * weight,
            "선정 여부": "선정" if weight > 0 else "-",
            "선정 사유": f"{selected_group} 중 모멘텀 스코어 1위" if weight > 0 else "-",
        })

    allocation = pd.DataFrame(allocation_rows)
    returns = returns.sort_values(["그룹", "모멘텀 스코어"], ascending=[True, False])
    decision = f"{rule_text} 여부: {'O' if use_offensive else 'X'} → {selected_group} 선택 → {selected_etf} 100%"

    return allocation, returns, selected_etf, decision


def calculate_dual_momentum(prices: pd.DataFrame, total_investment: int) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    returns = calculate_returns(prices, ODM_ASSETS)

    if returns.empty or returns["12M"].isna().any():
        raise ValueError("오리지널 듀얼 모멘텀 계산에 필요한 13개월 이상의 월봉 데이터가 부족합니다.")

    r = returns.set_index("ETF")["12M"]
    spy_r = r["SPY"]
    efa_r = r["EFA"]
    bil_r = r["BIL"]

    if spy_r > bil_r:
        selected_etf = "SPY" if spy_r >= efa_r else "EFA"
        decision = f"SPY 12개월 수익률이 BIL보다 높음 → SPY/EFA 중 높은 ETF 선택 → {selected_etf}"
    else:
        selected_etf = "AGG"
        decision = "SPY 12개월 수익률이 BIL보다 낮거나 같음 → AGG 선택"

    allocation_rows = []
    for ticker in ODM_ASSETS:
        weight = 1.0 if ticker == selected_etf else 0.0
        allocation_rows.append({
            "전략": "오리지널 듀얼 모멘텀",
            "ETF": ticker,
            "자산군": ETF_LABELS.get(ticker, ""),
            "비중": weight,
            "투자금": total_investment * weight,
            "선정 여부": "선정" if weight > 0 else "-",
            "선정 사유": decision if weight > 0 else "-",
        })

    allocation = pd.DataFrame(allocation_rows)
    returns = returns.sort_values("12M", ascending=False)
    return allocation, returns, decision


def calculate_laa(
    total_investment: int,
    sp500_below_200ma: bool,
    unemployment_above_12ma: bool,
) -> Tuple[pd.DataFrame, str]:
    tactical_etf = "SHY" if sp500_below_200ma and unemployment_above_12ma else "QQQ"
    decision = (
        "S&P500 < 200일선 = O, 실업률 > 12개월 평균 = O → SHY 선택"
        if tactical_etf == "SHY"
        else "두 조건이 동시에 충족되지 않음 → QQQ 선택"
    )

    rows = []
    for ticker in LAA_CORE:
        rows.append({
            "전략": "LAA",
            "ETF": ticker,
            "자산군": ETF_LABELS.get(ticker, ""),
            "비중": 0.25,
            "투자금": total_investment * 0.25,
            "선정 여부": "고정 보유",
            "선정 사유": "IWD/GLD/IEF 각 25% 고정",
            "리밸런싱 주기": "연 1회",
        })

    for ticker in LAA_TACTICAL:
        weight = 0.25 if ticker == tactical_etf else 0.0
        rows.append({
            "전략": "LAA",
            "ETF": ticker,
            "자산군": ETF_LABELS.get(ticker, ""),
            "비중": weight,
            "투자금": total_investment * weight,
            "선정 여부": "선정" if weight > 0 else "-",
            "선정 사유": decision if weight > 0 else "-",
            "리밸런싱 주기": "월 1회",
        })

    return pd.DataFrame(rows), decision


def add_rebalance_columns(
    allocation: pd.DataFrame,
    strategy_name: str,
    annual_last: date = None,
    monthly_last: date = None,
) -> pd.DataFrame:
    df = allocation.copy()

    if strategy_name == "LAA":
        annual_next = add_years(annual_last, 1)
        monthly_next = add_months(monthly_last, 1)

        df["최근 리밸런싱"] = df["리밸런싱 주기"].apply(
            lambda x: annual_last.strftime("%Y-%m-%d") if x == "연 1회" else monthly_last.strftime("%Y-%m-%d")
        )
        df["다음 리밸런싱"] = df["리밸런싱 주기"].apply(
            lambda x: annual_next.strftime("%Y-%m-%d") if x == "연 1회" else monthly_next.strftime("%Y-%m-%d")
        )
    else:
        monthly_next = add_months(monthly_last, 1)
        df["리밸런싱 주기"] = "월 1회"
        df["최근 리밸런싱"] = monthly_last.strftime("%Y-%m-%d")
        df["다음 리밸런싱"] = monthly_next.strftime("%Y-%m-%d")

    return df


def display_allocation_table(df: pd.DataFrame):
    show = df.copy()
    show["비중"] = show["비중"].apply(lambda x: f"{x * 100:.1f}%")
    show["투자금"] = show["투자금"].apply(format_krw)
    st.dataframe(show, use_container_width=True, hide_index=True)


def display_return_table(df: pd.DataFrame, include_score: bool = False):
    show = df.copy()
    for col in ["1M", "3M", "6M", "12M"]:
        if col in show.columns:
            show[col] = show[col].apply(format_pct)
    if "모멘텀 스코어" in show.columns:
        show["모멘텀 스코어"] = show["모멘텀 스코어"].apply(format_score)
    if "현재 조정종가" in show.columns:
        show["현재 조정종가"] = show["현재 조정종가"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
    st.dataframe(show, use_container_width=True, hide_index=True)


# =========================================================
# 사이드바 입력
# =========================================================
st.title("📈 강환국 ETF 전략 대시보드")
st.caption("LAA, VAA 공격형, 오리지널 듀얼 모멘텀의 ETF 비중과 투자금, 다음 리밸런싱 일정을 계산합니다.")

with st.sidebar:
    st.header("기본 입력")

    api_key = get_secret_api_key()

    if api_key:
        st.success("Alpha Vantage API Key를 Streamlit Secrets에서 불러왔습니다.")
    else:
        st.error("Streamlit Secrets에 ALPHA_VANTAGE_API_KEY가 없습니다.")

    total_investment = st.number_input(
        "총 투자금(원)",
        min_value=0,
        value=10_000_000,
        step=1,
        format="%d",
        help="원 단위까지 입력 가능합니다.",
    )
    total_investment = normalize_amount_input(total_investment)

    st.divider()
    st.header("LAA 수동 조건")
    sp500_below_200ma = st.radio(
        "S&P500 지수 가격이 200일 이동평균보다 낮습니까?",
        options=[False, True],
        format_func=lambda x: "O" if x else "X",
        horizontal=True,
    )
    unemployment_above_12ma = st.radio(
        "미국 실업률이 12개월 이동평균보다 높습니까?",
        options=[False, True],
        format_func=lambda x: "O" if x else "X",
        horizontal=True,
    )

    st.divider()
    st.header("리밸런싱 날짜")
    laa_annual_last = st.date_input("LAA IWD/GLD/IEF 최근 연간 리밸런싱일", value=date.today())
    laa_monthly_last = st.date_input("LAA QQQ/SHY 최근 월간 리밸런싱일", value=date.today())
    vaa_monthly_last = st.date_input("VAA 최근 월간 리밸런싱일", value=date.today())
    odm_monthly_last = st.date_input("오리지널 듀얼 모멘텀 최근 월간 리밸런싱일", value=date.today())

    st.divider()
    st.header("계산 옵션")
    exclude_current_month = st.checkbox(
        "진행 중인 월 데이터 제외",
        value=True,
        help="월말 리밸런싱 전략이므로 기본값은 현재 진행 중인 월 데이터를 제외합니다.",
    )
    vaa_zero_rule = st.selectbox(
        "VAA 공격형 판단 기준",
        options=[
            "공격형 4개 모두 0 이상이면 공격형",
            "공격형 4개 모두 0 초과이면 공격형",
        ],
        index=0,
    )

    if st.button("캐시 초기화"):
        st.cache_data.clear()
        st.success("캐시를 초기화했습니다. 다시 계산 버튼을 눌러주세요.")


# =========================================================
# 메인 화면
# =========================================================
if not api_key:
    st.warning(
        "Alpha Vantage API Key가 설정되어 있지 않습니다. "
        "Streamlit Secrets에 ALPHA_VANTAGE_API_KEY를 저장해주세요."
    )
    st.stop()

st.info(
    "VAA와 오리지널 듀얼 모멘텀은 Alpha Vantage의 월별 조정종가를 사용합니다. "
    "LAA의 S&P500 200일선/실업률 조건은 사용자가 직접 확인한 후 O/X로 입력하는 구조입니다. "
    f"이번 실행의 API 호출 대상은 {len(DATA_TICKERS)}개 ETF입니다: {', '.join(DATA_TICKERS)}"
)

run = st.button("전략 계산하기", type="primary")

if run:
    data = load_all_monthly_prices(DATA_TICKERS, api_key)
    prices = build_price_matrix(data, DATA_TICKERS, exclude_current_month=exclude_current_month)

    if prices.empty:
        st.error("가격 데이터를 불러오지 못했습니다. API Key, 호출 제한, 티커명을 확인해주세요.")
        st.stop()

    st.subheader("데이터 기준")
    c1, c2, c3 = st.columns(3)
    c1.metric("가격 데이터 ETF 수", f"{len([t for t in DATA_TICKERS if t in prices.columns])}개")
    c2.metric("가장 이른 월봉", prices.index.min().strftime("%Y-%m-%d"))
    c3.metric("전략 계산 기준월", prices.index.max().strftime("%Y-%m-%d"))

    tab_laa, tab_vaa, tab_odm, tab_total, tab_appendix = st.tabs(
        ["LAA", "VAA 공격형", "오리지널 듀얼 모멘텀", "통합 요약", "Appendix"]
    )

    with tab_laa:
        st.header("LAA 계산 결과")
        laa_allocation, laa_decision = calculate_laa(
            total_investment,
            sp500_below_200ma,
            unemployment_above_12ma,
        )
        laa_allocation = add_rebalance_columns(
            laa_allocation,
            "LAA",
            annual_last=laa_annual_last,
            monthly_last=laa_monthly_last,
        )
        st.success(laa_decision)
        display_allocation_table(laa_allocation)

    with tab_vaa:
        st.header("VAA 공격형 계산 결과")
        try:
            vaa_allocation, vaa_scores, vaa_selected, vaa_decision = calculate_vaa(
                prices,
                total_investment,
                vaa_zero_rule,
            )
            vaa_allocation = add_rebalance_columns(
                vaa_allocation,
                "VAA 공격형",
                monthly_last=vaa_monthly_last,
            )
            st.success(vaa_decision)
            display_allocation_table(vaa_allocation)

            st.subheader("VAA 모멘텀 스코어")
            display_return_table(vaa_scores, include_score=True)
        except Exception as e:
            st.error(str(e))

    with tab_odm:
        st.header("오리지널 듀얼 모멘텀 계산 결과")
        try:
            odm_allocation, odm_returns, odm_decision = calculate_dual_momentum(prices, total_investment)
            odm_allocation = add_rebalance_columns(
                odm_allocation,
                "오리지널 듀얼 모멘텀",
                monthly_last=odm_monthly_last,
            )
            st.success(odm_decision)
            display_allocation_table(odm_allocation)

            st.subheader("12개월 수익률")
            display_return_table(odm_returns)
        except Exception as e:
            st.error(str(e))

    with tab_total:
        st.header("통합 요약")
        summary_tables = []

        try:
            summary_tables.append(laa_allocation)
        except NameError:
            pass

        try:
            summary_tables.append(vaa_allocation)
        except NameError:
            pass

        try:
            summary_tables.append(odm_allocation)
        except NameError:
            pass

        if summary_tables:
            total_summary = pd.concat(summary_tables, ignore_index=True)
            selected_only = total_summary[total_summary["비중"] > 0].copy()

            st.subheader("선정 ETF만 보기")
            display_allocation_table(selected_only)

            csv = total_summary.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "전체 결과 CSV 다운로드",
                data=csv,
                file_name="etf_strategy_result.csv",
                mime="text/csv",
            )

    with tab_appendix:
        st.header("Appendix: 매수 전략 및 계산 방식")

        strategy_table = pd.DataFrame([
            {
                "전략": "LAA",
                "대상 ETF": "IWD, GLD, IEF, QQQ, SHY",
                "매수 규칙": "IWD/GLD/IEF 각 25% 고정. 나머지 25%는 S&P500 200일선 조건과 실업률 12개월 평균 조건이 모두 악화이면 SHY, 아니면 QQQ.",
                "리밸런싱": "IWD/GLD/IEF 연 1회, QQQ/SHY 월 1회",
                "앱 입력 방식": "S&P500 조건과 실업률 조건은 수동 O/X 입력",
            },
            {
                "전략": "VAA 공격형",
                "대상 ETF": "공격형: SPY, EFA, EEM, AGG / 안전자산: LQD, IEF, SHY",
                "매수 규칙": "공격형 4개 ETF의 모멘텀이 모두 양호하면 공격형 중 1위 100%, 하나라도 불량이면 안전자산 중 1위 100%.",
                "리밸런싱": "월 1회",
                "앱 입력 방식": "Alpha Vantage 월별 조정종가로 자동 계산",
            },
            {
                "전략": "오리지널 듀얼 모멘텀",
                "대상 ETF": "SPY, EFA, BIL, AGG",
                "매수 규칙": "SPY 12개월 수익률이 BIL보다 높으면 SPY/EFA 중 12개월 수익률이 높은 ETF 100%, 아니면 AGG 100%.",
                "리밸런싱": "월 1회",
                "앱 입력 방식": "Alpha Vantage 월별 조정종가로 자동 계산",
            },
        ])
        st.dataframe(strategy_table, use_container_width=True, hide_index=True)

        calc_table = pd.DataFrame([
            {
                "항목": "VAA 모멘텀 스코어",
                "계산식": "12 × 1개월 수익률 + 4 × 3개월 수익률 + 2 × 6개월 수익률 + 1 × 12개월 수익률",
            },
            {
                "항목": "1개월 수익률",
                "계산식": "기준월 조정종가 / 1개월 전 조정종가 - 1",
            },
            {
                "항목": "3개월 수익률",
                "계산식": "기준월 조정종가 / 3개월 전 조정종가 - 1",
            },
            {
                "항목": "6개월 수익률",
                "계산식": "기준월 조정종가 / 6개월 전 조정종가 - 1",
            },
            {
                "항목": "12개월 수익률",
                "계산식": "기준월 조정종가 / 12개월 전 조정종가 - 1",
            },
            {
                "항목": "월간 리밸런싱 다음 일정",
                "계산식": "최근 리밸런싱일 + 1개월. 최근일이 월말이면 다음 일정도 월말로 계산.",
            },
            {
                "항목": "연간 리밸런싱 다음 일정",
                "계산식": "최근 리밸런싱일 + 1년",
            },
        ])
        st.subheader("계산식")
        st.dataframe(calc_table, use_container_width=True, hide_index=True)

        st.subheader("주의사항")
        st.markdown(
            """
            - 이 앱은 투자 참고용 계산 도구이며 매수·매도 추천이 아닙니다.
            - VAA와 듀얼 모멘텀은 월말 리밸런싱 전략이므로 월별 조정종가를 기준으로 계산합니다.
            - Alpha Vantage 무료 API는 호출 제한이 있으므로 캐시를 사용합니다.
            - LAA의 S&P500 200일선 및 미국 실업률 조건은 자동 수집하지 않고 사용자가 직접 O/X로 입력합니다.
            """
        )

else:
    st.markdown(
        """
        ### 사용 방법
        1. Streamlit Secrets에 Alpha Vantage API Key를 저장합니다.
        2. 총 투자금을 원 단위로 입력합니다.
        3. LAA의 S&P500 200일선 조건과 미국 실업률 조건을 O/X로 선택합니다.
        4. 각 전략의 최근 리밸런싱 날짜를 입력합니다.
        5. **전략 계산하기** 버튼을 누릅니다.
        """
    )

    st.subheader("전략별 ETF")
    etf_map = pd.DataFrame([
        {"전략": "LAA", "ETF": ", ".join(LAA_CORE + LAA_TACTICAL)},
        {"전략": "VAA 공격형", "ETF": ", ".join(VAA_OFFENSIVE + VAA_DEFENSIVE)},
        {"전략": "오리지널 듀얼 모멘텀", "ETF": ", ".join(ODM_ASSETS)},
    ])
    st.dataframe(etf_map, use_container_width=True, hide_index=True)