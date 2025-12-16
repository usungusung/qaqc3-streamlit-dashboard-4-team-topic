import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Theme / Constants
# =========================
FAIL_COLOR   = "#E74C3C"   # 상태: 불량
OK_COLOR     = "#6B7280"   # 상태: 정상(중립)

SIGMA_BAND_COLORS = {
    3: "#f1f5f9",
    2: "#e0e7ff",
    1: "#c7d2fe",
}
ALARM_RULE_COLOR = "#7c3aed"
N_BINS_FIXED = 50

# ✅ 단 하나의 메타 (절대 재정의 금지)
SENSOR_META = {
    "ampere":      {"y_title": "Current (A)",      "unit": "A"},
    "volt":        {"y_title": "Voltage (V)",      "unit": "V"},
    "temperature": {"y_title": "Temperature (°C)", "unit": "°C"},
}

# =========================================================
# 0) Page Config + Sidebar UI CSS
# =========================================================
st.set_page_config(page_title="[QAQC 3기] streamlit 대시보드_4팀(방산-양극산화피막 공정의 불량률 예측)", layout="wide")

st.markdown(
    """
<style>
section[data-testid="stSidebar"] div[role="radiogroup"] {
    display:flex; flex-direction:column; gap:0.5rem;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background:#fff; padding:12px 16px; border-radius:10px;
    border:1px solid #d0d4dd; cursor:pointer; transition:all .15s ease;
    box-shadow:0 1px 2px rgba(0,0,0,.08);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background:#f5f7ff; border-color:#a5b4fc; box-shadow:0 2px 6px rgba(0,0,0,.12);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-selected="true"] {
    background:#eef2ff; border:2px solid #6366f1; box-shadow:0 2px 6px rgba(99,102,241,.25);
}
section[data-testid="stSidebar"] div[role="radiogroup"] span {
    font-size:16px !important; font-weight:600 !important;
}
</style>
""",
    unsafe_allow_html=True
)

st.caption("대시보드 프로젝트")
st.title("양극 산화 피막 데이터 기반 불량 예측 분석")

# =========================================================
# 1) IO
# =========================================================
@st.cache_data
def load_raw_data(csv_path: str = "방산통합데이터셋.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["pk_datetime"] = pd.to_datetime(df["pk_datetime"], errors="coerce")
    df = df.dropna(subset=["pk_datetime"]).copy()
    return df

@st.cache_resource
def load_model_and_meta(pkl_path="best_rf.pkl", meta_path="rf_metrics.json"):
    model = joblib.load(pkl_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

# =========================================================
# 2) Common Utilities (중복 제거 핵심)
# =========================================================
def sequence_failure_label(df: pd.DataFrame) -> pd.DataFrame:
    """sequence_index별 failure를 '정상/불량'으로 집계"""
    out = (
        df.groupby("sequence_index")["failure"]
        .agg(lambda s: "불량" if (s == -1).any() else "정상")
        .reset_index()
        .rename(columns={"failure": "failure_label"})
    )
    return out

def add_norm_time(df: pd.DataFrame) -> pd.DataFrame:
    """sequence별 pk_datetime을 0~1로 정규화한 norm_time 생성"""
    out = df.sort_values(["sequence_index", "pk_datetime"]).copy()
    t_min = out.groupby("sequence_index")["pk_datetime"].transform("min")
    t_max = out.groupby("sequence_index")["pk_datetime"].transform("max")
    dt = (out["pk_datetime"] - t_min).dt.total_seconds()
    total = (t_max - t_min).dt.total_seconds().replace(0, 1)
    out["norm_time"] = dt / total
    return out

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """lag/rolling/diff feature (row-level)"""
    df = df.sort_values(["sequence_index", "pk_datetime"]).copy()

    # lag
    for col in ["ampere", "volt", "temperature"]:
        df[f"{col}_lag1"] = df.groupby("sequence_index")[col].shift(1)

    # rolling mean/std (window=3, shift=1)
    def _roll(g, col, fn):
        return g[col].rolling(window=3).agg(fn).shift(1)

    for col, ko in [("ampere","전류"), ("volt","전압"), ("temperature","온도")]:
        df[f"{ko}이동평균"] = (
            df.groupby("sequence_index", group_keys=False)
              .apply(lambda g: _roll(g, col, "mean"))
        )
        df[f"{ko}이동표준편차"] = (
            df.groupby("sequence_index", group_keys=False)
              .apply(lambda g: _roll(g, col, "std"))
        )

    # diff
    df["△전류"] = df.groupby("sequence_index")["ampere"].diff()
    df["△전압"] = df.groupby("sequence_index")["volt"].diff()
    df["△온도"] = df.groupby("sequence_index")["temperature"].diff()

    return df

def classification_report_to_df(report_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame(report_dict).T.round(3)
    if "support" in df.columns:
        df["support"] = df["support"].fillna(0).astype(int)
    preferred = ["0", "1", "accuracy", "macro avg", "weighted avg"]
    keep = [i for i in preferred if i in df.index]
    return df.loc[keep]

def compute_sigma_band(
    df_raw: pd.DataFrame,
    sensor: str,
    n_bins: int = 50,
    sigma_k: float = 2.0,
    rec_num: int | None = None
) -> pd.DataFrame:
    """
    정상 데이터 기준, norm_time bin별 μ±kσ band 생성
    rec_num이 주어지면 해당 rec의 정상만으로 band 생성
    """
    df = df_raw.copy()
    if rec_num is not None and "rec_num" in df.columns:
        df = df[df["rec_num"] == rec_num].copy()

    df = df[df["failure"] != -1].copy()  # 정상만
    df = add_norm_time(df)

    df["time_bin"] = pd.cut(df["norm_time"], bins=n_bins, labels=False)
    band = (
        df.groupby("time_bin")
        .agg(
            t_mean=("norm_time", "mean"),
            mu=(sensor, "mean"),
            sigma=(sensor, "std")
        )
        .dropna()
        .reset_index()
    )
    band["upper"] = band["mu"] + sigma_k * band["sigma"]
    band["lower"] = band["mu"] - sigma_k * band["sigma"]
    return band

def compute_sequence_oob_ratio(
    df_raw: pd.DataFrame,
    band: pd.DataFrame,
    sensor: str,
    n_bins: int = 50
) -> pd.DataFrame:
    """band 대비 sequence별 이탈 비율 계산(%)"""
    df = add_norm_time(df_raw.copy())
    df["time_bin"] = pd.cut(df["norm_time"], bins=n_bins, labels=False)
    df = df.dropna(subset=["time_bin", sensor, "sequence_index"]).copy()
    df["time_bin"] = df["time_bin"].astype(int)

    band2 = band[["time_bin", "upper", "lower"]].copy()
    merged = df.merge(band2, on="time_bin", how="left").dropna(subset=["upper", "lower"])

    merged["is_pos"] = (merged[sensor] > merged["upper"]).astype(int)
    merged["is_neg"] = (merged[sensor] < merged["lower"]).astype(int)
    merged["is_oob"] = ((merged["is_pos"] == 1) | (merged["is_neg"] == 1)).astype(int)

    out = (
        merged.groupby("sequence_index")
        .agg(
            pos_ratio=("is_pos", "mean"),
            neg_ratio=("is_neg", "mean"),
            oob_ratio=("is_oob", "mean"),
            n_points=("is_oob", "size")
        )
        .reset_index()
    )
    for c in ["pos_ratio", "neg_ratio", "oob_ratio"]:
        out[c] *= 100
    return out

# =========================================================
# [추가] 2σ 패턴 차트 축을 단조롭게 만드는 유틸
# (유틸은 유지하되, SENSOR_META 재정의는 절대 금지)
# =========================================================
def minimal_x_axis():
    return alt.Axis(
        title=None,
        tickCount=6,
        labelColor="#6B7280",
        domainColor="#9CA3AF",
        tickColor="#9CA3AF",
        grid=True,
        gridColor="#E5E7EB"
    )

def minimal_y_axis(title: str):
    return alt.Axis(
        title=title,
        labelColor="#6B7280",
        titleColor="#374151",
        domainColor="#9CA3AF",
        tickColor="#9CA3AF",
        grid=True,
        gridColor="#E5E7EB"
    )

def configure_minimal(chart: alt.Chart) -> alt.Chart:
    return chart.configure_view(stroke=None).configure_axis(
        labelFontSize=11,
        titleFontSize=12
    )

# =========================================================
# 3) KPI 계산
# =========================================================
def compute_quality_metrics(mil: pd.DataFrame, k: float = 3.0):
    df = mil.copy()
    df["is_defect"] = (df["failure"] == -1).astype(int)

    defect_rate = df["is_defect"].mean()
    segment_defect_rate = df.groupby("sequence_index")["is_defect"].mean()

    df_time = df.set_index("pk_datetime")
    hourly_defect_rate = df_time["is_defect"].resample("1H").mean()

    mask_def = df["is_defect"] == 1
    mask_ok = df["is_defect"] == 0

    volt_diff = df.loc[mask_def, "volt"].mean() - df.loc[mask_ok, "volt"].mean()
    amp_diff  = df.loc[mask_def, "ampere"].mean() - df.loc[mask_ok, "ampere"].mean()
    temp_diff = df.loc[mask_def, "temperature"].mean() - df.loc[mask_ok, "temperature"].mean()

    volt_std_def = df.loc[mask_def, "volt"].std()
    volt_std_ok  = df.loc[mask_ok, "volt"].std()
    ISI_volt = np.nan if (np.isnan(volt_std_ok) or volt_std_ok == 0) else (volt_std_def / volt_std_ok)

    DRI_current = df.loc[mask_def, "△전류"].abs().mean()
    MSK_temp = df.loc[mask_def, "온도이동표준편차"].mean()

    def _calc_ooc_and_drift(data: pd.DataFrame, value_col: str, ma_col: str, std_col: str, time_col: str, k: float):
        if not all(c in data.columns for c in [value_col, ma_col, std_col, time_col]):
            return np.nan, np.nan

        s = data[[time_col, value_col, ma_col, std_col]].dropna().sort_values(time_col)
        if len(s) == 0:
            return np.nan, np.nan

        dev = (s[value_col] - s[ma_col]).abs()
        limit = k * s[std_col]
        ooc_ratio = (dev > limit).mean()

        drift = np.nan
        if len(s) > 1:
            x = (s[time_col] - s[time_col].min()).dt.total_seconds()
            y = s[ma_col]
            drift = np.polyfit(x, y, 1)[0]

        return ooc_ratio, drift

    OOC_volt, drift_volt = _calc_ooc_and_drift(df, "volt", "전압이동평균", "전압이동표준편차", "pk_datetime", k)
    OOC_amp,  drift_amp  = _calc_ooc_and_drift(df, "ampere", "전류이동평균", "전류이동표준편차", "pk_datetime", k)
    OOC_temp, drift_temp = _calc_ooc_and_drift(df, "temperature", "온도이동평균", "온도이동표준편차", "pk_datetime", k)

    summary = {
        "defect_rate": defect_rate,
        "volt_diff": volt_diff,
        "amp_diff": amp_diff,
        "temp_diff": temp_diff,
        "ISI_volt": ISI_volt,
        "DRI_current": DRI_current,
        "MSK_temp": MSK_temp,
        "OOC_volt": OOC_volt,
        "drift_volt": drift_volt,
        "OOC_amp": OOC_amp,
        "drift_amp": drift_amp,
        "OOC_temp": OOC_temp,
        "drift_temp": drift_temp,
    }
    return summary, segment_defect_rate, hourly_defect_rate

# =========================================================
# 4) ML Dataset Build (training-basis)
# =========================================================
@st.cache_data
def make_ml_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    span = (
        df.groupby("sequence_index")["pk_datetime"]
          .agg(t_min="min", t_max="max")
          .reset_index()
    )
    span["시간변화량(초)"] = (span["t_max"] - span["t_min"]).dt.total_seconds()
    df = df.merge(span[["sequence_index", "시간변화량(초)"]], on="sequence_index", how="left")

    df["두께변화량"] = df["ampere"] * df["시간변화량(초)"]
    df["최종두께"] = df.groupby("sequence_index")["두께변화량"].transform("sum")

    df = add_time_features(df)

    def split_into_tertiles(g: pd.DataFrame) -> pd.DataFrame:
        n = len(g)
        g = g.sort_values("pk_datetime")
        g["tertile"] = pd.qcut(np.arange(n), 3, labels=[0, 1, 2])
        return g

    df = df.groupby("sequence_index", group_keys=False).apply(split_into_tertiles)
    df["tertile"] = df["tertile"].astype(int)

    agg = (
        df.groupby(["sequence_index", "tertile"], as_index=False)
          .mean(numeric_only=True)
    )

    features_to_use = [
        "volt", "ampere", "temperature",
        "ampere_lag1", "volt_lag1", "temperature_lag1",
        "전류이동평균", "전압이동평균", "온도이동평균",
        "전류이동표준편차", "전압이동표준편차", "온도이동표준편차",
        "△전류", "△전압", "△온도",
        "tertile", "시간변화량(초)", "rec_num",
        "두께변화량", "최종두께",
        "failure", "sequence_index",
    ]
    missing = sorted(set(features_to_use) - set(agg.columns))
    if missing:
        raise KeyError(f"make_ml_data()에서 누락된 컬럼: {missing}")

    agg = agg[features_to_use].dropna().copy()
    return agg

# =========================================================
# 5) Model bootstrapping
# =========================================================
mil_raw = load_raw_data()
mil_ml  = make_ml_data(mil_raw)

rf_model, rf_meta = load_model_and_meta()
RF_THRESHOLD = float(rf_meta.get("threshold", 0.5))

feature_names = rf_meta.get("feature_importance", {}).get("features")
if not feature_names:
    feature_names = [c for c in mil_ml.columns if c not in ["failure", "sequence_index"]]

missing_for_X = sorted(set(feature_names) - set(mil_ml.columns))
if missing_for_X:
    raise KeyError(f"X 만들 때 누락된 컬럼: {missing_for_X}")

X_all = mil_ml[feature_names].copy()
y_all = (mil_ml["failure"] == -1.0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

feature_means = rf_meta.get("feature_means")
if not feature_means:
    feature_means = X_train.mean(numeric_only=True).to_dict()
for c in feature_names:
    feature_means.setdefault(c, 0.0)

# =========================================================
# 6) Point Predict Utilities
# =========================================================
def make_rf_input_row(
    ampere: float,
    volt: float,
    temperature: float,
    rec_num: int,
    tertile: int,
    feature_cols: list[str],
    base_means: dict
) -> pd.DataFrame:
    values = {c: float(base_means.get(c, 0.0)) for c in feature_cols}

    if "ampere" in values: values["ampere"] = float(ampere)
    if "volt" in values: values["volt"] = float(volt)
    if "temperature" in values: values["temperature"] = float(temperature)
    if "rec_num" in values: values["rec_num"] = int(rec_num)
    if "tertile" in values: values["tertile"] = int(tertile)

    if "ampere_lag1" in values: values["ampere_lag1"] = float(ampere)
    if "volt_lag1" in values: values["volt_lag1"] = float(volt)
    if "temperature_lag1" in values: values["temperature_lag1"] = float(temperature)

    if "전류이동평균" in values: values["전류이동평균"] = float(ampere)
    if "전압이동평균" in values: values["전압이동평균"] = float(volt)
    if "온도이동평균" in values: values["온도이동평균"] = float(temperature)

    for c in ["전류이동표준편차", "전압이동표준편차", "온도이동표준편차", "△전류", "△전압", "△온도"]:
        if c in values: values[c] = 0.0

    return pd.DataFrame([values])[feature_cols]

def get_training_ranges(mil_ml_: pd.DataFrame, rec_num: int, tertile: int) -> dict:
    df = mil_ml_.copy()
    if "rec_num" in df.columns:
        df = df[df["rec_num"] == rec_num]
    if "tertile" in df.columns:
        df = df[df["tertile"] == tertile]
    if len(df) == 0:
        df = mil_ml_.copy()

    ranges = {}
    for col in ["ampere", "volt", "temperature"]:
        if col in df.columns:
            ranges[col] = (float(df[col].min()), float(df[col].max()))
    return ranges

def render_range_caption_under_input(value: float, mn: float, mx: float, unit: str = "") -> bool:
    if np.isnan(mn) or np.isnan(mx):
        st.caption("학습 범위: 계산 불가")
        return False
    out = (value < mn) or (value > mx)
    unit_str = f" {unit}" if unit else ""
    tag = " (범위 밖)" if out else ""
    st.caption(f"학습 범위: {mn:.2f} ~ {mx:.2f}{unit_str}{tag}")
    return out

def get_ok_sigma_stats(mil_ml_: pd.DataFrame, rec_num: int, tertile: int) -> dict:
    df = mil_ml_.copy()
    if "rec_num" in df.columns:
        df = df[df["rec_num"] == rec_num]
    if "tertile" in df.columns:
        df = df[df["tertile"] == tertile]
    df = df[df["failure"] != -1.0] if "failure" in df.columns else df

    if len(df) == 0:
        df = mil_ml_.copy()
        df = df[df["failure"] != -1.0] if "failure" in df.columns else df

    stats = {}
    for col in ["ampere", "volt", "temperature"]:
        if col in df.columns and df[col].notna().any():
            mu = float(df[col].mean())
            sigma = float(df[col].std(ddof=0))
            n = int(df[col].dropna().shape[0])
            stats[col] = {"mu": mu, "sigma": sigma, "n": n}
        else:
            stats[col] = {"mu": np.nan, "sigma": np.nan, "n": 0}
    return stats

def sigma_zone(value: float, mu: float, sigma: float) -> tuple[float, int]:
    if np.isnan(mu) or np.isnan(sigma) or sigma == 0:
        return np.nan, 99
    z = abs((value - mu) / sigma)
    if z <= 1: return z, 0
    if z <= 2: return z, 1
    if z <= 3: return z, 2
    return z, 3

def sigma_band_chart(value: float, mu: float, sigma: float, title: str):
    if np.isnan(mu) or np.isnan(sigma) or sigma == 0:
        st.warning(f"{title}: σ 계산 불가(데이터 부족 또는 분산 0)")
        return

    x_min = mu - 3.5 * sigma
    x_max = mu + 3.5 * sigma

    bands = []
    for k in [3, 2, 1]:
        bands.append({"k": f"±{k}σ", "x1": mu - k * sigma, "x2": mu + k * sigma, "level": k})
    band_df = pd.DataFrame(bands)
    point_df = pd.DataFrame([{"x": value, "label": "입력값"}])

    base = alt.Chart(band_df).transform_calculate(dummy='" "')

    band = base.mark_bar(orient="horizontal").encode(
        x=alt.X("x1:Q", title=None, scale=alt.Scale(domain=[x_min, x_max])),
        x2="x2:Q",
        y=alt.Y("dummy:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
        color=alt.Color(
            "level:O",
            legend=None,
            scale=alt.Scale(domain=[1, 2, 3], range=[SIGMA_BAND_COLORS[1], SIGMA_BAND_COLORS[2], SIGMA_BAND_COLORS[3]])
        ),
        tooltip=[
            alt.Tooltip("k:N", title="범위"),
            alt.Tooltip("x1:Q", title="하한", format=".2f"),
            alt.Tooltip("x2:Q", title="상한", format=".2f"),
        ],
    )

    mu_line = alt.Chart(pd.DataFrame([{"mu": mu}])).mark_rule(color="#111827", strokeDash=[4, 4]).encode(
        x="mu:Q",
        tooltip=[alt.Tooltip("mu:Q", title="평균", format=".2f")]
    )

    pt = alt.Chart(point_df).mark_point(size=120, filled=True, color=FAIL_COLOR).encode(
        x=alt.X("x:Q"),
        y=alt.value(0),
        tooltip=[alt.Tooltip("x:Q", title="입력값", format=".2f")]
    )

    st.markdown(f"**{title}**  ")
    st.altair_chart((band + mu_line + pt).properties(height=70), use_container_width=True)

# =========================================================
# 7) Pages
# =========================================================
def page_kpi():
    st.markdown("#### 📊 공정 KPI 지표")
    st.info(
        "전체 공정의 품질 상태를 한눈에 확인하는 페이지입니다. \n\n"
        "현황을 점검하고 이상이 존재하는 구간을 빠르게 파악할 수 있습니다."
    )

    rec_options = sorted(mil_raw["rec_num"].dropna().unique())
    rec_selected = st.selectbox("rec_num 선택", rec_options)

    mil = mil_raw[mil_raw["rec_num"] == rec_selected].copy()
    mil = add_time_features(mil)

    quality_summary, seg_defect_rate, hourly_defect_rate = compute_quality_metrics(mil, k=3.0)

    col_left, col_mid, col_right = st.columns([1, 1, 1])
    with col_left:
        st.markdown("##### 🧪 품질 지표")
        st.metric("전체 불량률", f"{quality_summary['defect_rate'] * 100:.1f} %")
        st.metric("Volt 평균 차이 (불량-정상)", f"{quality_summary['volt_diff']:.2f}")
        st.metric("Ampere 평균 차이 (불량-정상)", f"{quality_summary['amp_diff']:.2f}")
        st.metric("온도 평균 차이 (불량-정상)", f"{quality_summary['temp_diff']:.2f}")

    with col_mid:
        st.markdown("##### 🔧 센서 기반 품질 지표")
        ISI = quality_summary["ISI_volt"]
        st.metric("ISI_volt (전압 변동성 민감도)", f"{ISI:.2f}" if not np.isnan(ISI) else "N/A")
        st.metric("DRI_current (변화량 기반 위험)", f"{quality_summary['DRI_current']:.3f}")
        st.metric("MSK_temp (온도 변동 민감도)", f"{quality_summary['MSK_temp']:.3f}")

    with col_right:
        st.markdown("##### 🏭 공정 상태 KPI")
        st.metric("OOC_volt", f"{quality_summary['OOC_volt'] * 100:.1f} %")
        st.metric("OOC_amp",  f"{quality_summary['OOC_amp'] * 100:.1f} %")
        st.metric("OOC_temp", f"{quality_summary['OOC_temp'] * 100:.1f} %")

    st.markdown("---")
    st.markdown("#### 🔥 불량 발생 sequence/날짜")

    seg_df = seg_defect_rate.reset_index()
    seg_df.columns = ["sequence_index", "defect_rate"]
    if not seg_df.empty:
        seg_chart = alt.Chart(seg_df).mark_bar(color=FAIL_COLOR).encode(
            x=alt.X("sequence_index:O", title="Sequence"),
            y=alt.Y("defect_rate:Q", title="불량률"),
        ).properties(height=120)
        st.altair_chart(seg_chart, use_container_width=True)

    hour_df = hourly_defect_rate.reset_index()
    hour_df.columns = ["pk_datetime", "defect_rate"]
    if not hour_df.empty:
        line = alt.Chart(hour_df).mark_line(color=FAIL_COLOR).encode(
            x=alt.X("pk_datetime:T", title="일시", axis=alt.Axis(format="%m-%d %H:%M", labelAngle=-45)),
            y=alt.Y("defect_rate:Q", title="불량률"),
            tooltip=[alt.Tooltip("pk_datetime:T", title="일시"),
                     alt.Tooltip("defect_rate:Q", title="불량률", format=".3f")]
        )
        points = alt.Chart(hour_df).mark_point(color=FAIL_COLOR, filled=True, size=40).encode(
            x="pk_datetime:T", y="defect_rate:Q",
            tooltip=[alt.Tooltip("pk_datetime:T", title="일시"),
                     alt.Tooltip("defect_rate:Q", title="불량률", format=".3f")]
        )
        st.altair_chart((line + points).properties(height=200), use_container_width=True)

def page_sequence_patterns():
    st.subheader("📅 시퀀스 패턴 + 2σ 기준 이탈 비율")
    st.info(
        "각 Sequence의 공정 패턴을 확인할 수 있습니다.\n\n"
        "공정 전반에서 정상 범위를 벗어난 비율을 통해 이상 비율이 높은 sequence를 식별합니다."
    )

    tab1, tab2 = st.tabs(["시퀀스 패턴", "2σ 기준 이탈 비율"])

    # =============================
    # TAB 1: 시퀀스 패턴 + 2σ band
    # =============================
    with tab1:
        rec_options = sorted(mil_raw["rec_num"].dropna().unique())
        rec_selected = st.selectbox("rec_num 선택", rec_options, key="rec_select_tab1")

        mil = mil_raw.loc[mil_raw["rec_num"] == rec_selected].copy()
        if mil.empty:
            st.warning("해당 rec_num 데이터가 없습니다.")
            st.stop()

        seq_status = sequence_failure_label(mil)
        seq_status["status_icon"] = np.where(seq_status["failure_label"] == "불량", "⚠", "✅")
        seq_status["option_label"] = seq_status.apply(
            lambda r: f"{int(r.sequence_index)} - {r.status_icon}", axis=1
        )
        label_to_seq = dict(zip(seq_status["option_label"], seq_status["sequence_index"]))

        options = seq_status["option_label"].tolist()
        default_vals = options[:3] if len(options) >= 3 else options

        selected_labels = st.multiselect(
            "Sequence 선택 (✅양품, ⚠불량)",
            options=options,
            default=default_vals,
            key="seq_multi_tab1",
        )
        if not selected_labels:
            st.info("최소 1개 이상의 시퀀스를 선택하세요.")
            st.stop()

        selected_seqs = [label_to_seq[l] for l in selected_labels]

        mil_sel = mil.loc[mil["sequence_index"].isin(selected_seqs)].copy()
        mil_sel = add_norm_time(mil_sel)

        charts = []
        sensors = ["ampere", "volt", "temperature"]

        for idx, sensor in enumerate(sensors):
            meta = SENSOR_META.get(sensor, {"y_title": sensor, "unit": ""})
            y_title = meta["y_title"]  # ✅ Current (A) / Voltage (V) / Temperature (°C)
            unit = meta["unit"]

            band_df = compute_sigma_band(
                df_raw=mil,
                sensor=sensor,
                n_bins=N_BINS_FIXED,
                sigma_k=2.0,
                rec_num=rec_selected
            ).copy()

            if band_df.empty:
                st.warning(f"{sensor}: 정상 데이터 부족으로 2σ band를 만들 수 없습니다.")
                continue

            # ✅ 축은 base에서 한 번만 잡고, Y축 title은 y_title로 고정
            base = alt.Chart().encode(
                x=alt.X(
                    "norm_time:Q",
                    title=None,
                    axis=alt.Axis(
                        tickCount=6,
                        grid=True,
                        labelColor="#6B7280",
                        tickColor="#9CA3AF",
                        domainColor="#9CA3AF",
                        labels=(idx == len(sensors) - 1),
                    ),
                ),
                y=alt.Y(
                    f"{sensor}:Q",
                    title=y_title,  # ✅ 여기!
                    axis=alt.Axis(
                        title=y_title,  # ✅ 여기!
                        grid=True,
                        labelColor="#6B7280",
                        tickColor="#9CA3AF",
                        domainColor="#9CA3AF",
                        titleColor="#374151",
                    ),
                ),
            )

            band_df = band_df.rename(columns={"t_mean": "norm_time"})

            # ✅ line 먼저 (축 기준 유지)
            line_layer = base.mark_line().encode(
                y=alt.Y(f"{sensor}:Q"),
                color=alt.Color(
                    "sequence_index:N",
                    legend=alt.Legend(title="Sequence"),
                    scale=alt.Scale(scheme="tableau10"),
                ),
                tooltip=[
                    alt.Tooltip("sequence_index:N", title="Sequence"),
                    alt.Tooltip("norm_time:Q", title="t(norm)", format=".3f"),
                    alt.Tooltip(f"{sensor}:Q", title=y_title, format=".2f"),
                ],
            ).properties(data=mil_sel)

            # ✅ band가 y(lower)로 축을 덮어쓰지 않게 title/axis를 동일하게 명시
            band_layer = base.mark_area(
                opacity=0.35,
                color=SIGMA_BAND_COLORS[1]
            ).encode(
                y=alt.Y(
                    "lower:Q",
                    title=y_title,  # ✅ 여기!
                    axis=alt.Axis(
                        title=y_title,  # ✅ 여기!
                        grid=True,
                        labelColor="#6B7280",
                        tickColor="#9CA3AF",
                        domainColor="#9CA3AF",
                        titleColor="#374151",
                    ),
                ),
                y2=alt.Y2("upper:Q"),
                tooltip=[
                    alt.Tooltip("norm_time:Q", title="t(norm)", format=".3f"),
                    alt.Tooltip("lower:Q", title="2σ lower", format=".2f"),
                    alt.Tooltip("upper:Q", title="2σ upper", format=".2f"),
                ],
            ).properties(data=band_df)

            layer = alt.layer(line_layer, band_layer).properties(height=180)
            charts.append(layer)

        if not charts:
            st.warning("표시할 차트가 없습니다(정상 band 생성 실패 또는 데이터 없음).")
            st.stop()

        v = alt.vconcat(*charts).resolve_axis(y="independent")
        v = v.configure_view(stroke=None).configure_axis(labelFontSize=11, titleFontSize=12)
        st.altair_chart(v, use_container_width=True)

    # =============================
    # TAB 2: 2σ 기준 이탈 비율
    # =============================
    with tab2:
        st.caption("전체 정상(전 rec_num 통합) 기준 band를 만들고, 시퀀스별 이탈 비율(%)을 계산합니다.")

        sensor = st.selectbox("센서 선택", ["ampere", "volt", "temperature"], index=0, key="tab2_sensor")
        alarm_th = st.slider("경고 기준(이탈 %)", 0, 60, 20, 1, key="tab2_alarm")

        band = compute_sigma_band(mil_raw, sensor=sensor, n_bins=N_BINS_FIXED, sigma_k=2.0, rec_num=None)
        ratio_df = compute_sequence_oob_ratio(mil_raw, band=band, sensor=sensor, n_bins=N_BINS_FIXED)
        if ratio_df.empty:
            st.info("계산 가능한 데이터가 없습니다.")
            st.stop()

        plot_df = ratio_df.copy()
        plot_df["sequence_index"] = plot_df["sequence_index"].astype(int)

        seq_lab = sequence_failure_label(mil_raw)
        plot_df = plot_df.merge(seq_lab, on="sequence_index", how="left")
        plot_df["failure_label"] = plot_df["failure_label"].fillna("정상")

        plot_df["abs_oob_ratio"] = plot_df["oob_ratio"].abs()
        plot_df["is_alarm"] = plot_df["abs_oob_ratio"] >= float(alarm_th)
        alarm_cnt = int(plot_df["is_alarm"].sum())

        st.caption(f"경고 기준: abs_oob_ratio ≥ {alarm_th:.0f}% | 경고 시퀀스 수: {alarm_cnt}개")

        bars = alt.Chart(plot_df).mark_bar().encode(
            x=alt.X("sequence_index:O", title="sequence_index"),
            y=alt.Y("abs_oob_ratio:Q", title="2σ 영역 이탈 비율(%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "failure_label:N",
                scale=alt.Scale(domain=["정상", "불량"], range=[OK_COLOR, FAIL_COLOR]),
                legend=alt.Legend(title="failure"),
            ),
            tooltip=[
                alt.Tooltip("sequence_index:O", title="sequence_index"),
                alt.Tooltip("failure_label:N", title="failure"),
                alt.Tooltip("abs_oob_ratio:Q", title="이탈 비율(|%|)", format=".2f"),
                alt.Tooltip("pos_ratio:Q", title="상한 초과(%)", format=".2f"),
                alt.Tooltip("neg_ratio:Q", title="하한 미만(%)", format=".2f"),
                alt.Tooltip("n_points:Q", title="포인트 수"),
            ],
        )

        rule = alt.Chart(pd.DataFrame({"y": [float(alarm_th)]})).mark_rule(
            strokeDash=[6, 4], color=ALARM_RULE_COLOR
        ).encode(y="y:Q")

        st.altair_chart((bars + rule).properties(height=380), use_container_width=True)

        if alarm_cnt > 0:
            top_alarm = (
                plot_df[plot_df["is_alarm"]]
                .sort_values("abs_oob_ratio", ascending=False)
                .head(10)
            )
            seq_list_str = ", ".join(top_alarm["sequence_index"].astype(str).tolist())

            st.warning(
                f"⚠ 경고 시퀀스가 있습니다.\n\n"
                f"- 이탈 비율 상위 시퀀스: **{seq_list_str}**\n\n"
            )
        else:
            st.success("현재 기준선에서는 경고 시퀀스가 없습니다.")

def page_ml_results():
    st.subheader("💻 ML 예측 결과")
    st.info(
        "Sequence 패턴을 기반으로 학습한 rf 모델의 성능을 확인할 수 있습니다. \n\n"
        "Feature Importance 지표를 통해 공정 개선의 우선순위를 결정에 활용할 수 있습니다. "
    )

    y_proba = rf_model.predict_proba(X_test[feature_names])[:, 1]
    y_proba_s = pd.Series(y_proba, index=y_test.index)

    col_left, col_gap, col_right = st.columns([1, 0.2, 1])

    with col_right:
        st.markdown("#### 🧮 임계값 & 핵심 성능 지표")

        default_th = float(st.session_state.get("user_th", RF_THRESHOLD))
        user_th = st.slider("Threshold (불량으로 예측할 최소 확률)", 0.0, 1.0, value=default_th, step=0.01, key="th_slider_ml")
        st.session_state["user_th"] = float(user_th)

        y_pred_user = (y_proba_s >= user_th).astype(int)

        report_dict = classification_report(y_test, y_pred_user, output_dict=True, zero_division=0)
        acc = report_dict["accuracy"]
        f1_defect = report_dict.get("1", {}).get("f1-score", 0.0)
        recall_defect = report_dict.get("1", {}).get("recall", 0.0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc*100:.1f} %")
        m2.metric("F1 (불량)", f"{f1_defect:.3f}")
        m3.metric("Recall (불량)", f"{recall_defect:.3f}")

        st.caption("📄 전체 분류 리포트")
        st.dataframe(classification_report_to_df(report_dict), use_container_width=True, hide_index=False)

    with col_left:
        st.markdown("#### 🪟 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_user)

        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).reset_index().rename(columns={"index": "actual"})
        cm_long = cm_df.melt(id_vars="actual", var_name="predicted", value_name="count")

        heatmap = alt.Chart(cm_long).mark_rect().encode(
            x=alt.X("predicted:N", title="Predicted"),
            y=alt.Y("actual:N", title="Actual"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="reds"), legend=alt.Legend(title="Count")),
            tooltip=[alt.Tooltip("actual:N"), alt.Tooltip("predicted:N"), alt.Tooltip("count:Q")]
        ).properties(height=500)

        text = alt.Chart(cm_long).mark_text(fontSize=14, fontWeight="bold", color="black").encode(
            x="predicted:N", y="actual:N", text="count:Q"
        )

        st.altair_chart(heatmap + text, use_container_width=True)

    st.markdown("#### 📊 Feature Importance")
    fi = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    fi_df = pd.DataFrame({"feature": fi.index, "importance": fi.values})

    fi_chart = alt.Chart(fi_df).mark_bar(color=FAIL_COLOR).encode(
        x=alt.X("feature:N", sort="-y", axis=alt.Axis(labelAngle=-45, title="Feature")),
        y=alt.Y("importance:Q", title="Importance"),
        tooltip=[alt.Tooltip("feature:N", title="Feature"),
                 alt.Tooltip("importance:Q", title="Importance", format=".4f")]
    ).properties(height=350)

    st.altair_chart(fi_chart, use_container_width=True)

def page_fault_sequences():
    st.subheader("🧯 불량 시퀀스 한눈에 보기")
    st.info(
        "모델이 불량으로 판단한 sequence를 한눈에 확인할 수 있습니다. \n\n"
        "실제 불량과 오진 사례를 구분하여 추가 점검이 필요한 sequence 선별에 활용할 수 있습니다."
    )

    th_default = float(st.session_state.get("user_th", RF_THRESHOLD))
    user_th = st.slider("Threshold (불량으로 예측할 최소 확률)", 0.0, 1.0, value=th_default, step=0.01, key="th_slider_fault")
    st.session_state["user_th"] = float(user_th)

    y_proba_test = rf_model.predict_proba(X_test[feature_names])[:, 1]
    y_proba_s = pd.Series(y_proba_test, index=y_test.index)
    y_pred_user = (y_proba_s >= user_th).astype(int)

    proba_all = rf_model.predict_proba(X_all[feature_names])[:, 1]
    mil_all = mil_ml.copy()
    mil_all["proba_fail"] = proba_all

    seq_prob_all = (
        mil_all.groupby("sequence_index", as_index=False)
        .agg(
            mean_proba=("proba_fail", "mean"),
            failure_seq=("failure", lambda s: -1.0 if (s == -1.0).any() else 1.0),
        )
    )
    seq_prob_all["pred_seq"] = (seq_prob_all["mean_proba"] >= user_th).astype(int)
    bad_seq_df = seq_prob_all[seq_prob_all["pred_seq"] == 1].sort_values("mean_proba", ascending=False)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 🔍 시퀀스별 불량 확률")
        seq_list = sorted(mil_ml["sequence_index"].unique())
        seq_choice = st.selectbox("시퀀스를 선택하세요", seq_list, key="seq_choice_fault")

        seq_df = mil_ml[mil_ml["sequence_index"] == seq_choice]
        if len(seq_df) > 0:
            proba_seq = rf_model.predict_proba(seq_df[feature_names])[:, 1]
            mean_proba = float(np.mean(proba_seq))
            pred_seq = int(mean_proba >= user_th)

            c1, c2, c3 = st.columns(3)
            c1.metric("평균 불량 확률", f"{mean_proba:.3f}")
            c2.metric("임계값", f"{user_th:.3f}")
            c3.metric("예측 결과", "⚠" if pred_seq == 1 else "✅")

            with st.expander("선택 시퀀스 (세그먼트 기반) 상세 보기", expanded=False):
                seq_view = seq_df.copy()
                seq_view["불량확률(모델)"] = proba_seq
                st.dataframe(seq_view, use_container_width=True)
        else:
            st.info("해당 시퀀스 데이터 없음")

    with col_right:
        st.markdown("#### ❌ 오진(예측 틀린) 케이스")
        wrong_mask = (y_test != y_pred_user)
        wrong_idx = y_test.index[wrong_mask]

        if len(wrong_idx) == 0:
            st.success("현재 오진 케이스 없음")
        else:
            st.write(f"총 **{len(wrong_idx)}건**의 오진 케이스")
            with st.expander("오진 케이스 상세 보기", expanded=False):
                wrong_cases = mil_ml.loc[wrong_idx].copy()
                wrong_cases["실제값(y_true)"] = y_test.loc[wrong_idx]
                wrong_cases["예측값(y_pred)"] = y_pred_user.loc[wrong_idx]
                wrong_cases["불량확률(모델)"] = y_proba_s.loc[wrong_idx]
                st.dataframe(wrong_cases, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📊 불량으로 예측된 시퀀스 전체 보기")
    st.write(f"현재 임계값 기준으로 불량으로 예측된 시퀀스는 총 **{len(bad_seq_df)}개** 입니다.")

    if len(bad_seq_df) == 0:
        st.info("이 임계값에서는 불량으로 예측된 시퀀스가 없습니다.")
        return

    chart_df = bad_seq_df.copy()
    chart_df["sequence_index"] = chart_df["sequence_index"].astype(str)
    chart_df["실제라벨"] = np.where(chart_df["failure_seq"] == -1.0, "실제 불량", "실제 양품")

    bad_chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("sequence_index:N", sort="-y", title="Sequence Index"),
        y=alt.Y("mean_proba:Q", title="평균 불량 예측 확률"),
        color=alt.Color(
            "실제라벨:N",
            scale=alt.Scale(domain=["실제 불량", "실제 양품"], range=[FAIL_COLOR, OK_COLOR]),
            legend=alt.Legend(title="실제 라벨")
        ),
        tooltip=[
            alt.Tooltip("sequence_index:N", title="Sequence"),
            alt.Tooltip("mean_proba:Q", title="평균 불량 확률", format=".3f"),
            alt.Tooltip("실제라벨:N", title="실제 라벨"),
        ],
    ).properties(height=300)

    st.altair_chart(bad_chart, use_container_width=True)

def page_point_predict():
    st.subheader("🪄 센서값 기반 합부 판정")
    st.info(
        "개별 공정 조건을 기준으로 합/부 판정을 판단할 수 있습니다. \n\n"
        "입력값이 정상 분포 내 어니 위치에 해당하는지 시각적으로 확인하여 위험도를 직관적으로 판단할 수 있습니다."
    )

    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.markdown("#### 입력 조건")

        rec_label = st.selectbox("정류기(rec_num)", options=["rec1", "rec2"])
        rec_num_input = 1 if rec_label == "rec1" else 2

        tertile_label = st.selectbox("공정 내 위치 (tertile)", options=["Ramp-up(0)", "Plateau(1)", "Ramp-down(2)"])
        tertile_input = 0 if "(0)" in tertile_label else (1 if "(1)" in tertile_label else 2)

        ranges = get_training_ranges(mil_ml, rec_num=rec_num_input, tertile=tertile_input)
        (a_min, a_max) = ranges.get("ampere", (np.nan, np.nan))
        (v_min, v_max) = ranges.get("volt", (np.nan, np.nan))
        (t_min, t_max) = ranges.get("temperature", (np.nan, np.nan))

        ampere_input = st.number_input("전류 (ampere)", value=551.5, step=0.1, format="%.2f")
        ood_a = render_range_caption_under_input(ampere_input, a_min, a_max)

        volt_input = st.number_input("전압 (volt)", value=23.2, step=0.1, format="%.2f")
        ood_v = render_range_caption_under_input(volt_input, v_min, v_max)

        temp_input = st.number_input("온도 (℃)", value=12.4, step=0.1, format="%.2f")
        ood_t = render_range_caption_under_input(temp_input, t_min, t_max, unit="℃")

        is_ood = bool(ood_a or ood_v or ood_t)

        st.markdown("")
        run_button = st.button("이 조건으로 예측하기", type="primary")

    with col_right:
        if not run_button:
            st.info("좌측에서 조건을 입력한 후 **[이 조건으로 예측하기]** 버튼을 눌러주세요.")
            return

        if is_ood:
            st.error("입력값이 학습 데이터 범위를 벗어났습니다. (OOD) 예측 신뢰도가 낮아 실행을 중단했습니다.")
            st.stop()

        X_input = make_rf_input_row(
            ampere=ampere_input,
            volt=volt_input,
            temperature=temp_input,
            rec_num=rec_num_input,
            tertile=tertile_input,
            feature_cols=feature_names,
            base_means=feature_means
        )

        proba_bad = float(rf_model.predict_proba(X_input)[0, 1])
        pred = int(proba_bad >= RF_THRESHOLD)
        label_text = "불량" if pred == 1 else "정상"

        st.markdown("#### 예측 결과")
        m1, m2 = st.columns(2)
        m1.metric("판정 라벨", label_text)
        m2.metric("불량 확률", f"{proba_bad * 100:.1f} %")

        ok_stats = get_ok_sigma_stats(mil_ml, rec_num=rec_num_input, tertile=tertile_input)
        a_mu, a_sig = ok_stats["ampere"]["mu"], ok_stats["ampere"]["sigma"]
        v_mu, v_sig = ok_stats["volt"]["mu"], ok_stats["volt"]["sigma"]
        t_mu, t_sig = ok_stats["temperature"]["mu"], ok_stats["temperature"]["sigma"]

        st.markdown("---")
        st.markdown("#### 정상 기준 σ 위치(1σ/2σ/3σ)")

        sigma_band_chart(ampere_input, a_mu, a_sig, "Current (A)")
        sigma_band_chart(volt_input,   v_mu, v_sig, "Voltage (V)")
        sigma_band_chart(temp_input,   t_mu, t_sig, "Temperature (°C)")

        z_a, zone_a = sigma_zone(ampere_input, a_mu, a_sig)
        z_v, zone_v = sigma_zone(volt_input,   v_mu, v_sig)
        z_t, zone_t = sigma_zone(temp_input,   t_mu, t_sig)

        over_items = []
        if zone_a in [2, 3]: over_items.append("Current (A)")
        if zone_v in [2, 3]: over_items.append("Voltage (V)")
        if zone_t in [2, 3]: over_items.append("Temperature (°C)")

        if over_items:
            st.warning(
                "⚠ 정상 분포 기준 **±2σ** 를 초과한 입력값이 있습니다.\n\n"
                f"- 초과 항목: {', '.join(over_items)}\n\n"
                "해당 조건은 **공정 이상(알람 영역)** 가능성이 있으므로 확인이 필요합니다."
            )
        else:
            st.success("정상 분포 기준 **±2σ** 범위 내입니다. (알람 영역 아님)")

# =========================================================
# 8) Sidebar Navigation + Router
# =========================================================
page = st.sidebar.radio(
    "페이지 선택",
    ("📊 공정 KPI", "📅 시퀀스 패턴 + 2σ 기준 이탈 비율", "💻 ML 예측 결과", "🧯 불량 시퀀스 한눈에", "🪄 센서값 기반 합부 판정"),
)

if page == "📊 공정 KPI":
    page_kpi()
elif page == "📅 시퀀스 패턴 + 2σ 기준 이탈 비율":
    page_sequence_patterns()
elif page == "💻 ML 예측 결과":
    page_ml_results()
elif page == "🧯 불량 시퀀스 한눈에":
    page_fault_sequences()
elif page == "🪄 센서값 기반 합부 판정":
    page_point_predict()
