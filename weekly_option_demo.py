import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ─── 페이지 설정 ───────────────────────────────────────────────
st.set_page_config(
    page_title="위클리 옵션 변동성 예측 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 16px 20px; margin: 4px 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #58a6ff; }
    .metric-label { font-size: 0.85rem; color: #8b949e; margin-bottom: 4px; }
    .metric-delta-pos { color: #3fb950; font-size: 0.85rem; }
    .section-header {
        border-left: 3px solid #58a6ff; padding-left: 12px;
        margin: 24px 0 16px 0; font-size: 1.1rem;
        font-weight: 600;
    }
    .data-badge {
        display: inline-block; padding: 2px 8px;
        border-radius: 10px; font-size: 0.75rem; font-weight: 600;
        margin-left: 8px;
    }
    .real { background: #1a3c2b; color: #3fb950; }
    .dummy { background: #3c2a1a; color: #f0b429; }
</style>
""", unsafe_allow_html=True)
LAYOUT = dict(
    plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
    font=dict(color="#e6edf3"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    hovermode="x unified", margin=dict(t=20, b=40)
)

def badge(label, ok, real_text="실제", dummy_text="더미"):
    cls = "real" if ok else "dummy"
    icon = "✅" if ok else "⚠️"
    txt = real_text if ok else dummy_text
    return f'<span class="data-badge {cls}">{icon} {label} {txt}</span>'

DATE_START, DATE_END = "2023-01-01", "2025-12-31"
WEEKLY_DATES = pd.date_range(DATE_START, DATE_END, freq="W-THU")

# ─── KOSPI200 일별 실데이터 (VKOSPI 프록시·만기일효과·백테스팅의 공통 기반) ─────
@st.cache_data(ttl=3600)
def load_kospi200_daily():
    """
    yfinance ^KS200(코스피200) 일별 종가를 로드합니다.
    이 시계열이 ① VKOSPI 실현변동성 프록시, ② 만기일 효과 t검정, ③ 백테스팅
    세 곳의 공통 원자료입니다. 실패 시 세 곳 모두 동일하게 더미로 폴백합니다.
    """
    try:
        import yfinance as yf
        raw = yf.download("^KS200", start="2022-11-01", end=DATE_END, progress=False)
        if raw is None or raw.empty:
            raise ValueError("빈 데이터프레임")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) < 100:
            raise ValueError(f"데이터 부족 ({len(close)}행)")
        daily = pd.DataFrame({"close": close})
        return daily, True
    except Exception as e:
        print(f"[KOSPI200 일별 데이터 로드 실패] {e}")
        return pd.DataFrame(columns=["close"]), False

def compute_vkospi_proxy(daily, daily_ok):
    """
    일별 종가 → 로그수익률 → 20일 롤링 표준편차 연율화(×√252×100) → 주간(W-THU) 리샘플.
    ⚠️ 옵션 내재변동성 지수(VKOSPI) 자체가 아니라 '실현변동성 기반 프록시'입니다.
    """
    if daily_ok and len(daily) > 25:
        log_ret = np.log(daily["close"] / daily["close"].shift(1))
        realized = log_ret.rolling(20).std() * np.sqrt(252) * 100
        weekly = realized.resample("W-THU").last().reindex(WEEKLY_DATES, method="ffill").bfill()
        if weekly.notna().sum() > 20:
            return weekly.values, True
    np.random.seed(42)
    n = len(WEEKLY_DATES)
    dummy = np.clip(20 + np.cumsum(np.random.randn(n) * 0.8) + np.sin(np.arange(n) * 0.3) * 5, 10, 50)
    return dummy, False

def compute_weekly_returns(daily, daily_ok):
    """KOSPI200 실제 주간 종가·수익률 (백테스팅의 기초자산 수익률)."""
    if daily_ok and len(daily) > 25:
        weekly_close = daily["close"].resample("W-THU").last().reindex(WEEKLY_DATES, method="ffill")
        weekly_ret = weekly_close.pct_change(fill_method=None)
        if weekly_close.notna().sum() > 20:
            return weekly_close, weekly_ret, True
    np.random.seed(88)
    n = len(WEEKLY_DATES)
    ret = np.random.randn(n) * 0.02
    price = pd.Series(100 * np.cumprod(1 + ret), index=WEEKLY_DATES)
    return price, price.pct_change(fill_method=None), False

# ─── 시장 지수(VKOSPI 프록시·VIX·VHSI 등) 로드 ──────────────────
@st.cache_data(ttl=3600)
def load_market_indices():
    dates = WEEKLY_DATES
    n = len(dates)
    daily, daily_ok = load_kospi200_daily()
    vkospi, vkospi_real = compute_vkospi_proxy(daily, daily_ok)

    yf_series, yf_ok = {}, {"vhsi": False, "vix": False}
    try:
        import yfinance as yf
        for key, ticker in {"vhsi": "^VHSI", "vix": "^VIX"}.items():
            df_raw = yf.download(ticker, start="2023-01-01", end=DATE_END, progress=False)
            if df_raw is None or df_raw.empty:
                continue
            close = df_raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            reind = close.resample("W-THU").last().reindex(dates, method="ffill")
            if reind.notna().sum() > 10:
                yf_series[key] = reind.values
                yf_ok[key] = True
    except Exception as e:
        print(f"[VHSI/VIX 로드 실패] {e}")

    df_out = pd.DataFrame({"date": dates, "vkospi": vkospi})

    if "vhsi" in yf_series:
        df_out["vhsi"] = yf_series["vhsi"]
    else:
        np.random.seed(10)
        df_out["vhsi"] = np.clip(22 + np.cumsum(np.random.randn(n) * 0.85) + np.sin(np.arange(n) * 0.25) * 5, 8, 55)

    if "vix" in yf_series:
        df_out["vix"] = yf_series["vix"]
    else:
        np.random.seed(50)
        df_out["vix"] = np.clip(20 + np.cumsum(np.random.randn(n) * 0.85) + np.sin(np.arange(n) * 0.25) * 5, 8, 55)

    # VJPX·VTWN: 일본·대만 옵션 변동성지수는 무료 공개 API가 없어 시연용 값으로 유지합니다.
    # (이번 작업 범위 밖 — 의도적 시뮬레이션이며 버그가 아닙니다)
    for col, seed, base in [("vjpx", 20, 18), ("vtwn", 30, 21)]:
        np.random.seed(seed)
        df_out[col] = np.clip(base + np.cumsum(np.random.randn(n) * 0.8) + np.sin(np.arange(n) * 0.3) * 5, 8, 50)

    # PCR·미결제약정 변화율: 국내 개별주식 위클리옵션이 아직 상장되지 않아 원천적으로 실거래 데이터가
    # 존재하지 않습니다 (이 프로젝트가 다루는 문제의식 그 자체). 상장 전까지는 시뮬레이션 값으로 유지합니다.
    np.random.seed(70)
    df_out["pcr"] = np.clip(0.8 + np.random.randn(n) * 0.15, 0.4, 1.5)
    np.random.seed(71)
    df_out["oi_change"] = np.random.randn(n) * 0.05

    df_out["high_vol"] = (df_out["vkospi"] > 28).astype(int)

    flags = {
        "vkospi": vkospi_real, "vix": yf_ok["vix"], "vhsi": yf_ok["vhsi"],
        "vjpx": False, "vtwn": False, "pcr": False, "oi_change": False,
    }
    return df_out, flags, daily, daily_ok

# ─── XGBoost / 기준모델(Logistic) 실제 학습 (TimeSeriesSplit OOF) ─────
def build_features(df):
    feat = pd.DataFrame({"date": df["date"]})
    feat["vkospi_t"] = df["vkospi"]
    feat["vkospi_lag1"] = df["vkospi"].shift(1)
    feat["vkospi_ma4"] = df["vkospi"].rolling(4).mean()
    feat["vix_t"] = df["vix"]
    feat["vhsi_t"] = df["vhsi"]
    feat["pcr_t"] = df["pcr"]
    feat["oi_change_t"] = df["oi_change"]
    feat["target"] = df["high_vol"].shift(-1)  # 다음 주 고변동성 여부를 예측
    return feat.dropna().reset_index(drop=True)

def _fit_oof(X, y, model_type, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full(len(X), np.nan)
    for train_idx, test_idx in tscv.split(X):
        y_train = y.iloc[train_idx]
        if y_train.nunique() < 2:
            continue  # 해당 fold의 학습구간에 클래스가 하나뿐이면 스킵 (조기 구간 흔함)
        if model_type == "xgboost" and XGBOOST_AVAILABLE:
            model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.08,
                                   eval_metric="logloss", random_state=42)
        else:
            model = LogisticRegression(max_iter=1000)
        model.fit(X.iloc[train_idx], y_train)
        oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return oof

@st.cache_data
def train_predict_models(df):
    """
    TimeSeriesSplit(5-fold) 기반 Out-of-Fold 예측 — 각 fold의 테스트 구간 예측만 모아
    성능을 계산하므로(원본 기획서의 Stacking 설계와 동일한 원칙) 데이터 누수가 없습니다.
    XGBoost(실전 모델)와 Logistic Regression(기준모델)을 동일한 방식으로 실제 학습합니다.
    TabNet은 의존성이 무거워 이번 접수 버전에서는 학습하지 않습니다(향후 과제).
    """
    feat = build_features(df)
    if len(feat) < 30:
        return None, {}, f"학습 가능한 표본이 부족합니다 (n={len(feat)})"
    X = feat.drop(columns=["date", "target"])
    y = feat["target"].astype(int)
    if y.nunique() < 2:
        return None, {}, "target(다음주 고변동성)이 단일 클래스라 학습할 수 없습니다"

    results = {}
    for name, mtype in [("xgb", "xgboost"), ("logit", "logistic")]:
        proba = _fit_oof(X, y, mtype)
        pred = (proba >= 0.5).astype(float)
        valid = ~np.isnan(proba)
        if valid.sum() < 10 or y[valid].nunique() < 2:
            results[name] = None
            continue
        results[name] = {
            "proba": proba, "pred": pred,
            "auc": roc_auc_score(y[valid], proba[valid]),
            "precision": precision_score(y[valid], pred[valid].astype(int), zero_division=0),
            "recall": recall_score(y[valid], pred[valid].astype(int), zero_division=0),
            "f1": f1_score(y[valid], pred[valid].astype(int), zero_division=0),
            "n_valid": int(valid.sum()),
        }

    feat_out = feat[["date"]].copy()
    if results.get("xgb"):
        feat_out["pred_xgb"] = results["xgb"]["pred"]
        feat_out["pred_xgb_proba"] = results["xgb"]["proba"]
    df_merged = df.merge(feat_out, on="date", how="left")
    error = None if results.get("xgb") else "XGBoost OOF 예측 표본이 부족합니다"
    return df_merged, results, error

# ─── 만기일 효과 t검정 (일별 원자료 기준) ────────────────────────
@st.cache_data
def compute_expiry_effect(daily, daily_ok):
    """
    ⚠️ 설계 유의사항: 대시보드의 주간(W-THU) 리샘플 시계열은 '전부' 목요일이므로
    그 위에서는 요일 비교가 원천적으로 불가능합니다. 따라서 이 검정만은 일별 원자료를
    별도로 사용해 목요일 vs 그 외 요일의 일중 변동성 프록시(|일수익률|×√252×100)를
    Welch's t-test(scipy.stats.ttest_ind, equal_var=False)로 비교합니다.
    """
    if not daily_ok or len(daily) < 60:
        return None, None, "일별 실데이터 로드 실패로 검정을 생략합니다 (더미 폴백)"
    d = daily.copy()
    d["ret"] = d["close"].pct_change(fill_method=None)
    d["vol_proxy"] = d["ret"].abs() * np.sqrt(252) * 100
    d["weekday"] = d.index.day_name()
    d = d.dropna(subset=["vol_proxy"])
    thu = d.loc[d["weekday"] == "Thursday", "vol_proxy"]
    other = d.loc[d["weekday"] != "Thursday", "vol_proxy"]
    if len(thu) < 10 or len(other) < 10:
        return None, None, "표본 수가 부족해 검정을 생략합니다"
    tstat, pval = stats.ttest_ind(thu, other, equal_var=False)
    ed = pd.DataFrame({
        "구분": ["만기일(목요일)", "일반 요일(월·화·수·금)"],
        "평균 일중 변동성 프록시": [thu.mean(), other.mean()],
        "표준편차": [thu.std(), other.std()],
    })
    ed_stats = {"tstat": tstat, "pvalue": pval, "n_thu": len(thu), "n_other": len(other),
                "diff": thu.mean() - other.mean()}
    return ed, ed_stats, None

# ─── 규칙 기반 백테스팅 (실제 KOSPI200 주간수익률) ────────────────
@st.cache_data
def run_backtest(weekly_close, weekly_ret, weekly_ok, pred_df):
    """
    모델 전략 = XGBoost가 '다음 주 고변동성'을 p≥0.5로 예측하면 노출 30%로 축소, 아니면 100%.
    이동평균 전략 = 지난주 종가가 8주 이동평균 위였을 때만 보유(룩어헤드 방지를 위해 신호는 1주 지연).
    Buy&Hold = 상시 100% 노출. 세 전략 모두 KOSPI200 실제 주간수익률을 기초자산으로 사용하는
    단순화된 규칙 기반 시뮬레이션이며, 실제 옵션 프리미엄·거래비용은 반영하지 않습니다.
    """
    dates = WEEKLY_DATES
    if not weekly_ok:
        np.random.seed(99)
        n = len(dates)
        dummy = pd.DataFrame({
            "date": dates,
            "model": np.cumprod(1 + np.random.randn(n) * 0.012 + 0.003),
            "buy_hold": np.cumprod(1 + np.random.randn(n) * 0.015 + 0.001),
            "moving_avg": np.cumprod(1 + np.random.randn(n) * 0.013 + 0.0015),
        })
        return dummy, None, False

    ret = weekly_ret.reindex(dates).fillna(0.0)
    price = weekly_close.reindex(dates).ffill()

    buy_hold_curve = (1 + ret).cumprod()

    ma = price.rolling(8).mean()
    ma_signal = (price.shift(1) > ma.shift(1)).astype(float).fillna(0.0)
    ma_ret = ret * ma_signal
    moving_avg_curve = (1 + ma_ret).cumprod()

    if pred_df is not None and "pred_xgb_proba" in pred_df.columns:
        proba_aligned = pred_df.set_index("date")["pred_xgb_proba"].reindex(dates)
        # proba_aligned[t] = "t시점까지의 정보로 예측한 t+1주 고변동성 확률" → 포지션에는 한 주 밀어서 적용
        exposure_series = proba_aligned.shift(1)
        exposure = np.where(exposure_series >= 0.5, 0.3, 1.0)
        exposure = pd.Series(exposure, index=dates)
        exposure[exposure_series.isna()] = 1.0
    else:
        exposure = pd.Series(1.0, index=dates)
    model_ret = ret * exposure
    model_curve = (1 + model_ret).cumprod()

    bt = pd.DataFrame({"date": dates, "model": model_curve.values,
                        "buy_hold": buy_hold_curve.values, "moving_avg": moving_avg_curve.values})

    def perf_stats(r):
        r = r.dropna()
        if len(r) < 2 or r.std() == 0:
            return dict(cum=np.nan, ann=np.nan, sharpe=np.nan, mdd=np.nan, winrate=np.nan)
        cum = (1 + r).prod() - 1
        ann = (1 + cum) ** (52 / len(r)) - 1
        sharpe = (r.mean() / r.std()) * np.sqrt(52)
        curve = (1 + r).cumprod()
        mdd = (curve / curve.cummax() - 1).min()
        winrate = (r > 0).mean()
        return dict(cum=cum, ann=ann, sharpe=sharpe, mdd=mdd, winrate=winrate)

    bt_stats = {"model": perf_stats(model_ret), "buy_hold": perf_stats(ret), "moving_avg": perf_stats(ma_ret)}
    return bt, bt_stats, True

# ─── 커버드콜 헤지효과 데이터 ──────────────────────────────────
@st.cache_data(ttl=3600)
def load_covered_call_data():
    """커버드콜 ETF 실제 데이터 로드 — KR 티커 우선, 실패시 US 티커, 그마저 실패시 더미"""
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="W-THU")
    try:
        import yfinance as yf
        candidates = [
            ("279530.KS", "TIGER 200커버드콜ATM (KR)"),
            ("QYLD", "Global X NASDAQ 100 Covered Call (US)"),
        ]
        for ticker, label in candidates:
            d = yf.download(ticker, start="2023-01-01", end="2025-12-31", progress=False)
            if not d.empty:
                s = d["Close"].resample("W-THU").last().reindex(dates, method="ffill")
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                df_out = pd.DataFrame({"date": dates, "covered_call": s.values})
                if df_out["covered_call"].notna().sum() > 10:
                    return df_out, True, label
    except Exception:
        pass
    np.random.seed(77)
    n = len(dates)
    cc = 100 + np.cumsum(np.random.randn(n)*0.5) - np.sin(np.arange(n)*0.3)*3
    return pd.DataFrame({"date": dates, "covered_call": cc}), False, "시연용 더미 데이터"
# ─── 공포탐욕지수 데이터 ──────────────────────────────────────
@st.cache_data(ttl=3600)
def load_fear_greed_data():
    """CNN Fear & Greed Index 공개 JSON 엔드포인트에서 로드 — 실패시 더미"""
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="W-THU")
    try:
        import requests
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = r.json()
        hist = data["fear_and_greed_historical"]["data"]
        fg = pd.DataFrame(hist)
        fg["date"] = pd.to_datetime(fg["x"], unit="ms")
        fg = fg.rename(columns={"y": "score"})[["date", "score"]]
        weekly = fg.set_index("date")["score"].resample("W-THU").last().reindex(dates, method="ffill")
        if weekly.notna().sum() > 10:
            return pd.DataFrame({"date": dates, "fear_greed": weekly.values}), True
    except Exception:
        pass
    np.random.seed(55)
    n = len(dates)
    fgv = np.clip(50 + np.cumsum(np.random.randn(n)*3) + np.sin(np.arange(n)*0.2)*15, 0, 100)
    return pd.DataFrame({"date": dates, "fear_greed": fgv}), False
# ─── 보조 거시지표 (환율·미국채 10년물) ──────────────────────────
@st.cache_data(ttl=3600)
def load_macro_data():
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="W-THU")
    try:
        import yfinance as yf
        fx = yf.download("USDKRW=X", start="2023-01-01", end="2025-12-31", progress=False)
        rate = yf.download("^TNX", start="2023-01-01", end="2025-12-31", progress=False)
        if not fx.empty and not rate.empty:
            fx_s = fx["Close"].resample("W-THU").last().reindex(dates, method="ffill")
            rate_s = rate["Close"].resample("W-THU").last().reindex(dates, method="ffill")
            if isinstance(fx_s, pd.DataFrame):
                fx_s = fx_s.iloc[:, 0]
            if isinstance(rate_s, pd.DataFrame):
                rate_s = rate_s.iloc[:, 0]
            if fx_s.notna().sum() > 10 and rate_s.notna().sum() > 10:
                return pd.DataFrame({"date": dates, "usdkrw": fx_s.values, "us10y": rate_s.values}), True
    except Exception:
        pass
    np.random.seed(66)
    n = len(dates)
    fx = 1300 + np.cumsum(np.random.randn(n)*5)
    rate = 4.0 + np.cumsum(np.random.randn(n)*0.05)
    return pd.DataFrame({"date": dates, "usdkrw": fx, "us10y": rate}), False

# ─── 데이터·모델 파이프라인 실행 ─────────────────────────────────
df, flags, daily, daily_ok = load_market_indices()
weekly_close, weekly_ret, weekly_ok = compute_weekly_returns(daily, daily_ok)
df_pred, ml_results, ml_error = train_predict_models(df)
bt, bt_stats, bt_real = run_backtest(weekly_close, weekly_ret, weekly_ok, df_pred)
ed, ed_stats, ed_error = compute_expiry_effect(daily, daily_ok)
cc_df, cc_real, cc_label = load_covered_call_data()
fg_df, fg_real = load_fear_greed_data()
macro_df, macro_real = load_macro_data()

is_real = flags["vkospi"]  # 헤더 대표 배지 — VKOSPI(실현변동성 프록시) 실계산 여부가 기준축
data_label = (
    badge("VKOSPI(실현변동성)", flags["vkospi"])
    + badge("VIX", flags["vix"])
    + badge("VHSI", flags["vhsi"])
)
# ─── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 분석 설정")
    st.markdown("---")
    ticker = st.selectbox("📌 종목 선택", [
        "삼성전자 (005930)", "SK하이닉스 (000660)",
        "현대차 (005380)", "LG에너지솔루션 (373220)"
    ])
    horizon = st.radio("📅 시간 지평", ["단기 (주간)", "중장기 (월간)"])
    model_choice = st.multiselect("🤖 모델 선택", ["XGBoost", "TabNet"],
                                   default=["XGBoost", "TabNet"])
    countries = st.multiselect(
        "🌏 비교 국가 선택",
        ["🇭🇰 홍콩 (VHSI)", "🇯🇵 일본 (VJPX)", "🇹🇼 대만 (VTWN)"],
        default=["🇭🇰 홍콩 (VHSI)", "🇯🇵 일본 (VJPX)", "🇹🇼 대만 (VTWN)"]
    )
    st.markdown("---")
    st.markdown("**데이터 출처**")
    st.caption("Yahoo Finance (yfinance) — VIX·VHSI·KOSPI200(^KS200)")
    st.caption("CNN Fear & Greed Index")
    st.caption("커버드콜 ETF (KR/US)")
    st.caption("⚠️ PCR·미결제약정·VJPX·VTWN·개별종목 옵션 데이터는 국내 미상장으로 시뮬레이션 값")
    st.markdown("**참고문헌**")
    st.caption("Kang(2022) 한국증권학회지 — 위클리옵션 변동성지수 개선")
    st.caption("노성호(2026) 자본시장연구원 — 자본시장 심리지수(CMSI)")
# ─── 헤더 ─────────────────────────────────────────────────────
st.markdown(f"""
<h1 style='font-size:1.8rem; margin-bottom:4px;'>
📊 위클리 옵션 변동성 예측 대시보드
</h1>
<div style='margin-bottom:12px;'>{data_label}</div>
<p style='color:#8b949e; margin-bottom:24px;'>
아시아권 옵션 시장 크로스마켓 전이학습 · 다중 시간 지평 모델 · 백테스팅 전략 검증 · 헤지효과·심리국면 분석
</p>
""", unsafe_allow_html=True)
# ─── KPI ──────────────────────────────────────────────────────
xgb_r = ml_results.get("xgb") if ml_results else None
logit_r = ml_results.get("logit") if ml_results else None
model_stats = bt_stats["model"] if bt_stats else None
bh_stats = bt_stats["buy_hold"] if bt_stats else None

if xgb_r:
    xgb_auc_str = f"{xgb_r['auc']:.2f}"
    if logit_r:
        _d = xgb_r['auc'] - logit_r['auc']
        xgb_delta_str = f"{'▲' if _d >= 0 else '▼'} {_d:+.2f} vs 기준모델(Logistic)"
    else:
        xgb_delta_str = f"OOF n={xgb_r['n_valid']}"
else:
    xgb_auc_str, xgb_delta_str = "N/A", f"⚠️ {ml_error or '학습 실패'}"

tabnet_auc_str, tabnet_delta_str = "미구현", "향후 과제 (XGBoost 우선 검증)"

if model_stats and pd.notna(model_stats["sharpe"]):
    sharpe_str = f"{model_stats['sharpe']:.2f}"
    sharpe_delta_str = f"{model_stats['sharpe']-bh_stats['sharpe']:+.2f} vs Buy&Hold" if bh_stats else ""
else:
    sharpe_str, sharpe_delta_str = "N/A", "백테스팅 실패"

if model_stats and pd.notna(model_stats["mdd"]):
    mdd_str = f"{model_stats['mdd']*100:.1f}%"
    mdd_delta_str = f"{(model_stats['mdd']-bh_stats['mdd'])*100:+.1f}%p vs Buy&Hold" if bh_stats else ""
else:
    mdd_str, mdd_delta_str = "N/A", "백테스팅 실패"

c1,c2,c3,c4 = st.columns(4)
for col,label,val,delta in zip(
    [c1,c2,c3,c4],
    ["XGBoost AUC (실측)","TabNet AUC","모델 전략 샤프비율 (실측)","최대 낙폭 MDD (실측)"],
    [xgb_auc_str, tabnet_auc_str, sharpe_str, mdd_str],
    [xgb_delta_str, tabnet_delta_str, sharpe_delta_str, mdd_delta_str]
):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{val}</div>
        <div class='metric-delta-pos'>{delta}</div>
    </div>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
# ─── 탭 ───────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "🌏 아시아권 변동성 비교","🤖 모델 예측 결과",
    "⏱ 시간 지평 비교","💰 백테스팅 결과",
    "🛡️ 커버드콜 헤지효과","😨 공포탐욕지수 국면"
])
country_map = {
    "🇭🇰 홍콩 (VHSI)": ("vhsi","#f0b429","VHSI"),
    "🇯🇵 일본 (VJPX)": ("vjpx","#3fb950","VJPX"),
    "🇹🇼 대만 (VTWN)": ("vtwn","#bc8cff","VTWN"),
}
# ── Tab 1 ───────────────────────────────────────────────────
with tab1:
    st.markdown(f"<div class='section-header'>국가별 변동성 지수 비교 — VKOSPI (한국) 기준 {badge('VKOSPI', flags['vkospi'])}</div>",
                unsafe_allow_html=True)
    st.caption("📚 학술 근거: Kang(2022, 한국증권학회지)에 따르면 위클리옵션은 차근월물 대비 거래량 3.7배·체결빈도 6~10배로 "
               "유동성이 높아 변동성지수(V-KOSPI200)의 급등락·과소변동 왜곡을 줄이는 데 기여합니다 — "
               "본 대시보드가 위클리옵션 기반 신호를 핵심 데이터로 삼는 근거입니다.")
    st.caption("⚠️ 'VKOSPI' 라벨은 편의상 표기이며, 실제로는 KOSPI200 실제 종가로 계산한 20일 롤링 실현변동성(연율화) "
               "프록시입니다. 홍콩(VHSI)·미국(VIX)은 실제 내재변동성 지수, 일본(VJPX)·대만(VTWN)은 공개 무료 API가 없어 "
               "시연용 값입니다.")
    # VIX 보조 차트
    if "vix" in df.columns:
        col_main, col_vix = st.columns([3,1])
    else:
        col_main = st.container()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["date"],y=df["vkospi"],name="🇰🇷 VKOSPI (한국)",
        line=dict(color="#58a6ff",width=2.5),fill="tozeroy",fillcolor="rgba(88,166,255,0.06)"))
    for c in countries:
        k,color,lbl = country_map[c]
        fig1.add_trace(go.Scatter(x=df["date"],y=df[k],
            name=f"{c.split(' ')[0]} {lbl}",line=dict(color=color,width=1.8,dash="dot")))
    fig1.add_hline(y=28,line_dash="dash",line_color="#f85149",opacity=0.5,
                   annotation_text="고변동성 임계값 (28)",annotation_font_color="#f85149")
    fig1.update_layout(**{**LAYOUT,"height":360,"yaxis":{**LAYOUT["yaxis"],"title":"변동성 지수"}})
    st.plotly_chart(fig1,use_container_width=True)
    # VIX 별도 표시
    if "vix" in df.columns:
        st.markdown(f"<div class='section-header'>🇺🇸 미국 VIX (글로벌 공포지수) {badge('VIX', flags['vix'])}</div>",
                    unsafe_allow_html=True)
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(x=df["date"],y=df["vix"],name="VIX",
            line=dict(color="#ff7b7b",width=2),fill="tozeroy",fillcolor="rgba(255,123,123,0.06)"))
        fig_vix.add_hline(y=30,line_dash="dash",line_color="#f85149",opacity=0.5,
                           annotation_text="공포 구간 (30)")
        fig_vix.add_hline(y=20,line_dash="dash",line_color="#f0b429",opacity=0.4,
                           annotation_text="주의 구간 (20)")
        fig_vix.update_layout(**{**LAYOUT,"height":260,
                                  "yaxis":{**LAYOUT["yaxis"],"title":"VIX"}})
        st.plotly_chart(fig_vix,use_container_width=True)
        st.caption("💡 VIX는 VKOSPI의 선행 시그널로 활용 — 미국 공포지수 상승 → 아시아 변동성 연쇄 상승 패턴 확인")
    # 국가별 서브탭
    st.markdown("<div class='section-header'>국가별 상세 분석</div>",unsafe_allow_html=True)
    st.markdown(f"{badge('VHSI', flags['vhsi'])} {badge('VJPX', flags['vjpx'])} {badge('VTWN', flags['vtwn'])}", unsafe_allow_html=True)
    sub_hk,sub_jp,sub_tw = st.tabs(["🇭🇰 홍콩","🇯🇵 일본","🇹🇼 대만"])
    def country_detail(subtab,col_key,label,color,threshold):
        with subtab:
            cl,cr = st.columns([2,1])
            with cl:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["date"],y=df["vkospi"],
                    name="VKOSPI (한국)",line=dict(color="#58a6ff",width=1.5)))
                fig.add_trace(go.Scatter(x=df["date"],y=df[col_key],
                    name=label,line=dict(color=color,width=2)))
                fig.add_hline(y=threshold,line_dash="dash",line_color="#f85149",opacity=0.5,
                              annotation_text=f"고변동성 임계값 ({threshold})")
                fig.update_layout(**{**LAYOUT,"height":300,
                                     "yaxis":{**LAYOUT["yaxis"],"title":"변동성 지수"}})
                st.plotly_chart(fig,use_container_width=True)
            with cr:
                corr = df["vkospi"].corr(df[col_key])  # pandas .corr()는 NaN을 쌍별로 제외해 np.corrcoef보다 안전
                st.markdown("<br>",unsafe_allow_html=True)
                st.metric(f"{label} 평균",f"{df[col_key].mean():.1f}")
                st.metric("VKOSPI 상관계수",f"{corr:.2f}","높을수록 학습 전이 유효")
                st.metric("고변동성 빈도",f"{(df[col_key]>threshold).mean()*100:.1f}%")
    country_detail(sub_hk,"vhsi","VHSI (홍콩)","#f0b429",28)
    country_detail(sub_jp,"vjpx","VJPX (일본)","#3fb950",25)
    country_detail(sub_tw,"vtwn","VTWN (대만)","#bc8cff",27)
    # 상관관계 히트맵
    st.markdown("<div class='section-header'>국가간 변동성 상관관계 히트맵</div>",unsafe_allow_html=True)
    cols_for_corr = ["vkospi","vhsi","vjpx","vtwn"]
    if "vix" in df.columns:
        cols_for_corr.append("vix")
    rename_map = {"vkospi":"🇰🇷 한국","vhsi":"🇭🇰 홍콩",
                  "vjpx":"🇯🇵 일본","vtwn":"🇹🇼 대만","vix":"🇺🇸 VIX"}
    corr_df = df[cols_for_corr].rename(columns=rename_map).corr()
    fig_c = px.imshow(corr_df,text_auto=".2f",color_continuous_scale="Blues",zmin=0,zmax=1)
    fig_c.update_layout(plot_bgcolor="#161b22",paper_bgcolor="#0d1117",
                        font=dict(color="#e6edf3"),height=340,margin=dict(t=20,b=20))
    st.plotly_chart(fig_c,use_container_width=True)
    ca,cb = st.columns(2)
    with ca:
        st.markdown(f"<div class='section-header'>PCR (풋/콜 비율) {badge('PCR', False, dummy_text='시뮬레이션')}</div>",unsafe_allow_html=True)
        fp = go.Figure()
        fp.add_trace(go.Bar(x=df["date"],y=df["pcr"],
            marker_color=np.where(df["pcr"]>1.0,"#f85149","#3fb950")))
        fp.add_hline(y=1.0,line_dash="dash",line_color="#8b949e",opacity=0.7,annotation_text="PCR=1.0")
        fp.update_layout(**{**LAYOUT,"height":260,"showlegend":False})
        st.plotly_chart(fp,use_container_width=True)
    with cb:
        st.markdown(f"<div class='section-header'>미결제약정 주간 변화율 {badge('OI', False, dummy_text='시뮬레이션')}</div>",unsafe_allow_html=True)
        fo = go.Figure()
        fo.add_trace(go.Bar(x=df["date"],y=df["oi_change"]*100,
            marker_color=np.where(df["oi_change"]>0,"#58a6ff","#f85149")))
        fo.update_layout(**{**LAYOUT,"height":260,"showlegend":False,
                             "yaxis":{**LAYOUT["yaxis"],"title":"변화율 (%)"}})
        st.plotly_chart(fo,use_container_width=True)
    st.caption("⚠️ PCR·미결제약정은 국내 개별주식 위클리옵션이 아직 상장되지 않아 실거래 데이터가 없습니다 — "
               "상장 전까지는 시뮬레이션 값이며, 상장 이후 실제 값으로 교체할 예정입니다.")
# ── Tab 2 ───────────────────────────────────────────────────
with tab2:
    st.markdown(f"<div class='section-header'>고변동성 예측 — {ticker.split(' ')[0]} · {horizon}</div>",
                unsafe_allow_html=True)
    working_df = df_pred if df_pred is not None else df
    recent = working_df.tail(40).copy()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=recent["date"],y=recent["vkospi"],
        name="실제 VKOSPI(실현변동성 프록시)",line=dict(color="#8b949e",width=1.5)))
    if "XGBoost" in model_choice:
        if "pred_xgb" in recent.columns:
            m = recent["pred_xgb"] == 1
            fig2.add_trace(go.Scatter(x=recent[m]["date"],y=recent[m]["vkospi"],
                mode="markers",name="XGBoost 고변동성 예측(OOF)",
                marker=dict(symbol="triangle-up",size=10,color="#58a6ff")))
        else:
            st.warning(f"⚠️ XGBoost 예측을 표시할 수 없습니다: {ml_error}")
    if "TabNet" in model_choice:
        st.info("ℹ️ TabNet은 이번 접수 버전에서 실제로 학습하지 않았습니다 — 의존성이 무거워 XGBoost로 "
                "우선 검증하고, TabNet 비교 학습은 향후 과제로 남겨둡니다.")
    fig2.add_hline(y=28,line_dash="dash",line_color="#f85149",opacity=0.5,annotation_text="고변동성 임계값")
    fig2.update_layout(**{**LAYOUT,"height":360,"yaxis":{**LAYOUT["yaxis"],"title":"VKOSPI"}})
    st.plotly_chart(fig2,use_container_width=True)

    rows = []
    for name, key in [("XGBoost","xgb"), ("기준모델 (Logistic)","logit")]:
        r = ml_results.get(key) if ml_results else None
        if r:
            rows.append({"모델":name,"정밀도":r["precision"],"재현율":r["recall"],"F1-Score":r["f1"],"AUC":r["auc"]})
        else:
            rows.append({"모델":name,"정밀도":np.nan,"재현율":np.nan,"F1-Score":np.nan,"AUC":np.nan})
    rows.append({"모델":"TabNet (미구현·향후 과제)","정밀도":np.nan,"재현율":np.nan,"F1-Score":np.nan,"AUC":np.nan})
    perf_df = pd.DataFrame(rows)
    st.markdown("<div class='section-header'>모델 성능 비교</div>",unsafe_allow_html=True)
    st.dataframe(
        perf_df.style
        .background_gradient(subset=["AUC"],cmap="Blues")
        .format({"정밀도":"{:.2f}","재현율":"{:.2f}","F1-Score":"{:.2f}","AUC":"{:.2f}"}, na_rep="—"),
        use_container_width=True,hide_index=True)
    if ml_results and ml_results.get("xgb"):
        st.caption(f"✅ TimeSeriesSplit(5-fold) Out-of-Fold 예측 기준 실측값 (검증표본 n={ml_results['xgb']['n_valid']}/전체 "
                   f"{len(build_features(df))}). 각 fold의 테스트 구간 예측만 모아 계산해 데이터 누수를 방지했습니다.")
    if ml_error:
        st.warning(f"⚠️ {ml_error}")
# ── Tab 3 ───────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'>단기(주간) vs 중장기(월간) 예측력 비교</div>",
                unsafe_allow_html=True)
    st.caption("⚠️ 아래 시간지평 비교는 이번 버전에서 별도 재학습을 하지 않은 예시(참고용) 수치입니다 — "
               "위 KPI·모델 성능 비교의 XGBoost AUC(실측)와는 별개이며, 시간지평별 모델 재학습은 향후 과제입니다.")
    hd = pd.DataFrame({
        "모델":["XGBoost","XGBoost","TabNet","TabNet"],
        "시간지평":["주간 (단기)","월간 (중장기)","주간 (단기)","월간 (중장기)"],
        "AUC":[0.72,0.81,0.69,0.77],
    })
    fig3 = px.bar(hd,x="모델",y="AUC",color="시간지평",barmode="group",
                  color_discrete_map={"주간 (단기)":"#58a6ff","월간 (중장기)":"#3fb950"},text="AUC")
    fig3.update_traces(texttemplate="%{text:.2f}",textposition="outside")
    fig3.update_layout(**{**LAYOUT,"height":360,"yaxis":{**LAYOUT["yaxis"],"range":[0.5,0.95]}})
    st.plotly_chart(fig3,use_container_width=True)
    st.info("💡 (참고용 예시치) 중장기(월간) 모델 AUC가 단기(주간) 대비 0.08~0.09 높음.")

    st.markdown(f"<div class='section-header'>만기일 효과 분석 {badge('일별 KOSPI200 t검정', ed is not None)}</div>",
                unsafe_allow_html=True)
    st.caption("📌 주간(목요일 기준) 리샘플 시계열은 전부 목요일이라 요일 비교가 불가능합니다 — 이 검정만은 "
               "일별 KOSPI200 원자료를 별도로 사용해 목요일 vs 그 외 요일의 일중 변동성 프록시를 "
               "Welch's t-test로 비교합니다.")
    if ed is not None:
        ce1,ce2 = st.columns(2)
        with ce1:
            fe = px.bar(ed,x="구분",y="평균 일중 변동성 프록시",color="구분",
                        color_discrete_map={"만기일(목요일)":"#f85149","일반 요일(월·화·수·금)":"#58a6ff"},error_y="표준편차")
            fe.update_layout(**{**LAYOUT,"height":280,"showlegend":False})
            st.plotly_chart(fe,use_container_width=True)
        with ce2:
            st.markdown("<br>",unsafe_allow_html=True)
            st.metric("만기일(목) 평균 변동성 프록시", f"{ed_stats['diff']:+.2f} vs 그 외 요일")
            st.metric("t-검정 p-value", f"{ed_stats['pvalue']:.3f}",
                       "✅ 유의수준 0.05 이하" if ed_stats["pvalue"] < 0.05 else "❌ 유의수준 0.05 초과")
            if ed_stats["pvalue"] < 0.05 and ed_stats["diff"] > 0:
                st.success("목요일 변동성이 통계적으로 유의하게 높음 — 만기일 효과 확인")
            elif ed_stats["pvalue"] < 0.05 and ed_stats["diff"] < 0:
                st.warning("목요일 변동성이 통계적으로 유의하게 '낮게' 나타남 — 원 가설(H5)과 반대 방향입니다. "
                           "표본 기간·프록시 정의를 재검토할 필요가 있습니다.")
            else:
                st.info("이번 표본에서는 목요일과 다른 요일 간 통계적으로 유의한 차이가 확인되지 않았습니다 "
                        "(p ≥ 0.05) — 개별주식 위클리옵션 만기 효과와는 다른 결과일 수 있음에 유의하세요.")
        st.caption(f"📊 표본: 목요일 {ed_stats['n_thu']}일 · 그 외 {ed_stats['n_other']}일 "
                   f"(2022.11~2025.12 KOSPI200 일별 데이터, t={ed_stats['tstat']:.2f}, p={ed_stats['pvalue']:.3f})")
    else:
        st.warning(f"⚠️ 만기일 효과 검정을 실행하지 못했습니다: {ed_error}")
# ── Tab 4 ───────────────────────────────────────────────────
with tab4:
    st.markdown(f"<div class='section-header'>누적 수익률 비교 (2023.01 ~ 2025.12) {badge('KOSPI200 실수익률 규칙전략', bt_real)}</div>",
                unsafe_allow_html=True)
    if bt_real:
        st.caption("📌 규칙: 모델전략 = XGBoost가 다음 주 고변동성(p≥0.5)을 예측하면 노출 30%로 축소, 아니면 100% 노출 · "
                   "이동평균 전략 = 지난주 종가가 8주 이동평균 위일 때만 보유 · Buy&Hold = 상시 100% 노출. "
                   "기초자산은 KOSPI200 실제 주간수익률이며, 실제 옵션 프리미엄·거래비용·슬리피지는 반영하지 않은 "
                   "단순화된 규칙 기반 시뮬레이션입니다.")
    else:
        st.warning("⚠️ KOSPI200 실데이터 로드에 실패해 더미 곡선으로 대체되었습니다.")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=bt["date"],y=(bt["model"]-1)*100,name="모델 기반 전략",
        line=dict(color="#3fb950",width=2.5),fill="tozeroy",fillcolor="rgba(63,185,80,0.06)"))
    fig4.add_trace(go.Scatter(x=bt["date"],y=(bt["buy_hold"]-1)*100,name="Buy & Hold",
        line=dict(color="#8b949e",width=1.5,dash="dot")))
    fig4.add_trace(go.Scatter(x=bt["date"],y=(bt["moving_avg"]-1)*100,name="이동평균 전략",
        line=dict(color="#f0b429",width=1.5,dash="dash")))
    fig4.add_hline(y=0,line_color="#30363d")
    fig4.update_layout(**{**LAYOUT,"height":380,"yaxis":{**LAYOUT["yaxis"],"title":"누적 수익률 (%)"}})
    st.plotly_chart(fig4,use_container_width=True)
    st.markdown("<div class='section-header'>전략별 리스크/수익 지표</div>",unsafe_allow_html=True)
    if bt_stats:
        def fmt_row(name, s):
            if pd.isna(s["sharpe"]):
                return {"전략": name, "누적 수익률":"N/A","연환산 수익률":"N/A","샤프비율":"N/A","최대 낙폭(MDD)":"N/A","승률":"N/A"}
            return {"전략": name, "누적 수익률": f"{s['cum']*100:.1f}%", "연환산 수익률": f"{s['ann']*100:.1f}%",
                    "샤프비율": f"{s['sharpe']:.2f}", "최대 낙폭(MDD)": f"{s['mdd']*100:.1f}%",
                    "승률": f"{s['winrate']*100:.1f}%"}
        rd = pd.DataFrame([
            fmt_row("🟢 모델 기반 전략", bt_stats["model"]),
            fmt_row("⚪ Buy & Hold", bt_stats["buy_hold"]),
            fmt_row("🟡 이동평균 전략", bt_stats["moving_avg"]),
        ])
        st.dataframe(rd,use_container_width=True,hide_index=True)
        m, b = bt_stats["model"], bt_stats["buy_hold"]
        if pd.notna(m["sharpe"]) and pd.notna(b["sharpe"]):
            diff_cum = (m["cum"]-b["cum"])*100
            diff_sharpe = m["sharpe"]-b["sharpe"]
            diff_mdd = (m["mdd"]-b["mdd"])*100
            msg = f"모델 전략이 Buy&Hold 대비 누적 수익률 {diff_cum:+.1f}%p, 샤프비율 {diff_sharpe:+.2f}, MDD {diff_mdd:+.1f}%p."
            if diff_cum > 0 and diff_sharpe > 0:
                st.success(f"✅ {msg}")
            else:
                st.info(f"ℹ️ {msg} 이 단순 규칙에서는 모델 전략이 Buy&Hold를 반드시 능가하지 않습니다 — "
                        "실제 옵션 프리미엄·정교한 포지션 사이징이 빠진 단순화된 백테스팅의 한계입니다.")
    else:
        st.warning("실제 수익률 데이터를 불러오지 못해 지표를 계산할 수 없습니다.")
# ── Tab 5 : 커버드콜 헤지효과 ─────────────────────────────────
with tab5:
    cc_badge = '<span class="data-badge real">✅ 실제 데이터</span>' if cc_real \
               else '<span class="data-badge dummy">⚠️ 시연용 더미 데이터</span>'
    st.markdown(f"<div class='section-header'>커버드콜(옵션 매도) 포지션의 변동성 완화 효과 검증 {cc_badge}</div>",
                unsafe_allow_html=True)
    st.caption(f"📌 참조 데이터: {cc_label if cc_real else '커버드콜 ETF (시연용 더미)'} · "
               "가설: 커버드콜 등 매도 포지션 수요가 클수록 옵션 매도가 변동성을 완화하는 방향으로 작용한다")
    merged_h3 = df[["date","vkospi"]].merge(cc_df[["date","covered_call"]], on="date", how="inner")
    merged_h3["cc_return"] = merged_h3["covered_call"].pct_change(fill_method=None)
    merged_h3["cc_vol"] = merged_h3["cc_return"].rolling(4).std() * 100
    valid_h3 = merged_h3.dropna(subset=["cc_vol"])
    corr_h3 = valid_h3["cc_vol"].corr(valid_h3["vkospi"]) if len(valid_h3) > 5 else 0.0
    cl3, cr3 = st.columns([3,1])
    with cl3:
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=merged_h3["date"], y=merged_h3["vkospi"], name="VKOSPI",
            line=dict(color="#58a6ff", width=2), yaxis="y1"))
        fig5.add_trace(go.Scatter(x=merged_h3["date"], y=merged_h3["covered_call"], name="커버드콜 ETF 가격",
            line=dict(color="#3fb950", width=1.8, dash="dot"), yaxis="y2"))
        fig5.update_layout(**{**LAYOUT, "height": 320,
            "yaxis": {**LAYOUT["yaxis"], "title": "VKOSPI"},
            "yaxis2": dict(overlaying="y", side="right", title="커버드콜 ETF", gridcolor="#21262d")})
        st.plotly_chart(fig5, use_container_width=True)
        fig5b = go.Figure()
        fig5b.add_trace(go.Bar(x=valid_h3["date"], y=valid_h3["cc_vol"],
            name="커버드콜 4주 실현변동성(%)", marker_color="#bc8cff"))
        fig5b.update_layout(**{**LAYOUT, "height": 240, "showlegend": False,
            "yaxis": {**LAYOUT["yaxis"], "title": "실현변동성 (%)"}})
        st.plotly_chart(fig5b, use_container_width=True)
    with cr3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("커버드콜 변동성-VKOSPI 상관계수", f"{corr_h3:.2f}",
                   "음수일수록 헤지효과 강함")
        st.metric("커버드콜 평균 실현변동성", f"{valid_h3['cc_vol'].mean():.2f}%")
        st.metric("VKOSPI 고변동성(>28) 구간 빈도", f"{(merged_h3['vkospi']>28).mean()*100:.1f}%")
        if corr_h3 < -0.1:
            st.success("커버드콜 변동성이 시장 변동성과 역행하는 경향 → 헤지효과 확인")
        elif corr_h3 > 0.1:
            st.warning("커버드콜 변동성이 시장 변동성과 동행 → 헤지효과 뚜렷하지 않음")
        else:
            st.info("뚜렷한 상관관계 확인 안됨 — 표본 확대 필요")
    st.caption("⚠️ 실제 매도 포지션 잔고·델타 헤지 비율 데이터는 공개되지 않아, 커버드콜 ETF 가격 변동성을 대리지표로 사용한 간접 검증입니다.")
    # ── 상승·하락 국면 비대칭 분석 (옵션 정리 노트 반영) ──
    st.markdown("<div class='section-header'>상승·하락 국면별 비대칭 효과</div>", unsafe_allow_html=True)
    st.caption("💡 커버드콜은 완전한 헤지가 아니라 '상승 이익은 제한, 하락 손실은 프리미엄만큼만 완충'하는 "
               "비대칭 구조입니다. 이에 따라 상승 주간과 하락·보합 주간을 나누어 헤지효과가 실제로 "
               "비대칭적으로 나타나는지 확인합니다.")
    valid_dir = merged_h3.dropna(subset=["cc_vol", "cc_return"]).copy()
    valid_dir["regime_dir"] = np.where(valid_dir["cc_return"] > 0, "상승 주간", "하락·보합 주간")
    dir_rows = []
    for r in ["상승 주간", "하락·보합 주간"]:
        g = valid_dir[valid_dir["regime_dir"] == r]
        if len(g) > 0:
            c = g["cc_vol"].corr(g["vkospi"]) if len(g) > 5 else np.nan
            dir_rows.append({"국면": r, "평균 VKOSPI": g["vkospi"].mean(),
                              "커버드콜-VKOSPI 상관계수": c, "표본수": len(g)})
    dir_stats = pd.DataFrame(dir_rows).set_index("국면")
    cu1, cu2 = st.columns([2,1])
    with cu1:
        st.dataframe(
            dir_stats.style.format({"평균 VKOSPI":"{:.1f}","커버드콜-VKOSPI 상관계수":"{:.2f}","표본수":"{:.0f}"}),
            use_container_width=True)
    with cu2:
        st.markdown("<br>", unsafe_allow_html=True)
        up_c = dir_stats.loc["상승 주간","커버드콜-VKOSPI 상관계수"] if "상승 주간" in dir_stats.index else np.nan
        down_c = dir_stats.loc["하락·보합 주간","커버드콜-VKOSPI 상관계수"] if "하락·보합 주간" in dir_stats.index else np.nan
        if pd.notna(up_c) and pd.notna(down_c) and down_c < up_c:
            st.success("하락 주간에서 상관이 더 뚜렷하게 역행 → 노트에서 설명한 '하락 손실 완충' 구조와 부합")
        else:
            st.info("상승·하락 주간 간 뚜렷한 비대칭이 확인되지 않음 — 표본 확대 후 재검증 필요")
# ── Tab 6 : 공포탐욕지수 국면 ─────────────────────────────────
with tab6:
    fg_badge = '<span class="data-badge real">✅ 실제 데이터 (CNN F&G)</span>' if fg_real \
               else '<span class="data-badge dummy">⚠️ 시연용 더미 데이터</span>'
    st.markdown(f"<div class='section-header'>공포·탐욕 국면별 변동성 반응 차이 검증 {fg_badge}</div>",
                unsafe_allow_html=True)
    st.caption("가설: 시장 공포·탐욕 국면에 따라 위클리옵션 도입 효과(변동성 반응)의 방향성이 달라진다")
    merged_h4 = df[["date","vkospi"]].merge(fg_df, on="date", how="inner")
    def fg_regime(v):
        if v < 25: return "극단적 공포"
        if v < 45: return "공포"
        if v < 55: return "중립"
        if v < 75: return "탐욕"
        return "극단적 탐욕"
    merged_h4["regime"] = merged_h4["fear_greed"].apply(fg_regime)
    regime_order = ["극단적 공포","공포","중립","탐욕","극단적 탐욕"]
    regime_colors = {"극단적 공포":"#f85149","공포":"#f0b429","중립":"#8b949e",
                      "탐욕":"#3fb950","극단적 탐욕":"#58a6ff"}
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=merged_h4["date"], y=merged_h4["vkospi"], name="VKOSPI",
        line=dict(color="#58a6ff", width=2), yaxis="y1"))
    fig6.add_trace(go.Scatter(x=merged_h4["date"], y=merged_h4["fear_greed"], name="공포탐욕지수",
        line=dict(color="#f0b429", width=1.8, dash="dot"), yaxis="y2"))
    fig6.update_layout(**{**LAYOUT, "height": 320,
        "yaxis": {**LAYOUT["yaxis"], "title": "VKOSPI"},
        "yaxis2": dict(overlaying="y", side="right", title="공포탐욕지수 (0~100)", range=[0,100], gridcolor="#21262d")})
    st.plotly_chart(fig6, use_container_width=True)
    cg1, cg2 = st.columns([2,1])
    with cg1:
        regime_stats = merged_h4.groupby("regime")["vkospi"].agg(["mean","std","count"]).reindex(regime_order).dropna(how="all")
        fig7 = go.Figure()
        fig7.add_trace(go.Bar(
            x=regime_stats.index, y=regime_stats["mean"],
            error_y=dict(type="data", array=regime_stats["std"].fillna(0)),
            marker_color=[regime_colors.get(r,"#8b949e") for r in regime_stats.index]))
        fig7.update_layout(**{**LAYOUT, "height": 300, "showlegend": False,
            "yaxis": {**LAYOUT["yaxis"], "title": "평균 VKOSPI"}})
        st.plotly_chart(fig7, use_container_width=True)
    with cg2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(regime_stats.rename(columns={"mean":"평균 VKOSPI","std":"표준편차","count":"표본수"})
                     .style.format({"평균 VKOSPI":"{:.1f}","표준편차":"{:.1f}","표본수":"{:.0f}"}),
                     use_container_width=True)
        spread = regime_stats["mean"].max() - regime_stats["mean"].min()
        if spread > 3:
            st.success(f"국면별 평균 VKOSPI 격차 {spread:.1f}p → 국면에 따라 변동성 반응이 유의하게 다름")
        else:
            st.info(f"국면별 평균 VKOSPI 격차 {spread:.1f}p로 크지 않음")
    st.markdown("<div class='section-header'>보조 거시지표 — 환율 · 미국채 10년물</div>", unsafe_allow_html=True)
    macro_badge = '<span class="data-badge real">✅ 실제 데이터</span>' if macro_real \
                  else '<span class="data-badge dummy">⚠️ 시연용 더미 데이터</span>'
    st.markdown(macro_badge, unsafe_allow_html=True)
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=macro_df["date"], y=macro_df["usdkrw"], name="USD/KRW",
        line=dict(color="#58a6ff", width=1.8), yaxis="y1"))
    fig8.add_trace(go.Scatter(x=macro_df["date"], y=macro_df["us10y"], name="미국채 10년물(%)",
        line=dict(color="#f0b429", width=1.8, dash="dot"), yaxis="y2"))
    fig8.update_layout(**{**LAYOUT, "height": 260,
        "yaxis": {**LAYOUT["yaxis"], "title": "USD/KRW"},
        "yaxis2": dict(overlaying="y", side="right", title="10Y (%)", gridcolor="#21262d")})
    st.plotly_chart(fig8, use_container_width=True)
    st.caption("💡 환율·금리는 국면 분석의 보조 거시 변수로, 공포탐욕 국면 전환 시점과의 동행 여부를 함께 참고합니다. "
               "국내 CPI·CP·채권금리는 한국은행 ECOS API 키 발급 후 연동 예정입니다.")
    st.caption("📚 참고: CNN Fear & Greed Index는 미국 시장 기준 지표로, 국내 시장에는 자본시장연구원의 "
               "자본시장 심리지수(CMSI, 노성호 2026 — 국내 증권뉴스를 LLM으로 학습해 구축)가 방법론적으로 "
               "더 적합합니다. CMSI가 공개 데이터로 제공되면 본 지표를 대체할 예정입니다.")
st.markdown("---")
st.caption("⚠️ 본 대시보드는 공모전 시연용 프로토타입입니다.")
st.caption("📁 데이터 출처: KOSPI200(^KS200)·VIX·VHSI (Yahoo Finance) · CNN Fear & Greed Index | 모델: XGBoost(실측) · Logistic(기준모델, 실측) · TabNet(미구현)")
st.caption("📚 참고문헌: 강태훈(2022) 「변동성지수의 개선을 위한 위클리옵션의 활용에 관한 연구」 한국증권학회지 51(6) · "
           "노성호(2026) 「자본시장 심리지수의 구축과 활용」 자본시장연구원 이슈보고서 26-01")
