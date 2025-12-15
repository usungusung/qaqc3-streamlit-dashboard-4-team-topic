import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Color System (Theme)
# =========================
DEFECT_RED  = "#E74C3C"
OK_GRAY     = "#9CA3AF"
NEUTRAL_GRAY = "#6B7280"  # 그래프/정보용 중립색


# =========================================================
# 0) Page Config + Sidebar UI CSS
# =========================================================
st.set_page_config(page_title="밀스펙 2.0", layout="wide")

st.markdown(
    """
<style>
/* 🔹 라디오 그룹 전체 간격 */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

/* 🔹 각 라디오 항목을 카드처럼 보이도록 변형 */
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background-color: #ffffff;
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid #d0d4dd;
    cursor: pointer;
    transition: all 0.15s ease;
    box-shadow: 0px 1px 2px rgba(0,0,0,0.08);
}

/* 🔹 마우스 올리면 입체감 */
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: #f5f7ff;
    border-color: #a5b4fc;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.12);
}

/* 🔹 선택된 항목 강조 */
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-selected="true"] {
    background-color: #eef2ff;
    border: 2px solid #6366f1;
    box-shadow: 0px 2px 6px rgba(99, 102, 241, 0.25);
}

/* 🔹 텍스트 조금 키운다 */
section[data-testid="stSidebar"] div[role="radiogroup"] span {
    font-size: 16px !important;
    font-weight: 600 !important;
}
</style>
""",
    unsafe_allow_html=True
)

st.caption("대시보드 프로젝트")
st.title("양극 산화 피막 데이터 기반 불량 예측 분석")


# =========================================================
# 1) Data Load
# =========================================================
@st.cache_data
def load_raw_data(csv_path: str = "방산통합데이터셋.csv") -> pd.DataFrame:
    mil = pd.read_csv(csv_path)
    mil["pk_datetime"] = pd.to_datetime(mil["pk_datetime"], errors="coerce")
    mil.dropna(subset=["pk_datetime"], inplace=True)
    return mil

mil_raw = load_raw_data()


# =========================================================
# 2) Common Feature Engineering
# =========================================================
def add_time_features(mil: pd.DataFrame) -> pd.DataFrame:
    mil = mil.sort_values(["sequence_index", "pk_datetime"]).copy()

    # lag
    mil["ampere_lag1"] = mil.groupby("sequence_index")["ampere"].shift(1)
    mil["volt_lag1"] = mil.groupby("sequence_index")["volt"].shift(1)
    mil["temperature_lag1"] = mil.groupby("sequence_index")["temperature"].shift(1)

    # rolling mean/std (window=3, shift=1)
    mil["전류이동평균"] = (
        mil.groupby("sequence_index")["ampere"]
        .rolling(window=3).mean().shift(1)
        .reset_index(level=0, drop=True)
    )
    mil["전압이동평균"] = (
        mil.groupby("sequence_index")["volt"]
        .rolling(window=3).mean().shift(1)
        .reset_index(level=0, drop=True)
    )
    mil["온도이동평균"] = (
        mil.groupby("sequence_index")["temperature"]
        .rolling(window=3).mean().shift(1)
        .reset_index(level=0, drop=True)
    )

    mil["전류이동표준편차"] = (
        mil.groupby("sequence_index")["ampere"]
        .rolling(window=3).std().shift(1)
        .reset_index(level=0, drop=True)
    )
    mil["전압이동표준편차"] = (
        mil.groupby("sequence_index")["volt"]
        .rolling(window=3).std().shift(1)
        .reset_index(level=0, drop=True)
    )
    mil["온도이동표준편차"] = (
        mil.groupby("sequence_index")["temperature"]
        .rolling(window=3).std().shift(1)
        .reset_index(level=0, drop=True)
    )

    # diff by sequence
    mil["△전류"] = mil.groupby("sequence_index")["ampere"].diff()
    mil["△전압"] = mil.groupby("sequence_index")["volt"].diff()
    mil["△온도"] = mil.groupby("sequence_index")["temperature"].diff()

    return mil


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
    amp_diff = df.loc[mask_def, "ampere"].mean() - df.loc[mask_ok, "ampere"].mean()
    temp_diff = df.loc[mask_def, "temperature"].mean() - df.loc[mask_ok, "temperature"].mean()

    volt_std_def = df.loc[mask_def, "volt"].std()
    volt_std_ok = df.loc[mask_ok, "volt"].std()
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
    OOC_amp, drift_amp = _calc_ooc_and_drift(df, "ampere", "전류이동평균", "전류이동표준편차", "pk_datetime", k)
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


def classification_report_to_df(report_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame(report_dict).T.round(3)
    if "support" in df.columns:
        df["support"] = df["support"].fillna(0).astype(int)

    preferred = ["0", "1", "accuracy", "macro avg", "weighted avg"]
    keep = [i for i in preferred if i in df.index]
    return df.loc[keep]


# =========================================================
# 3) ML Data for Dashboard (same as training pipeline basis)
# =========================================================
@st.cache_data
def make_ml_data(raw: pd.DataFrame) -> pd.DataFrame:
    mil = raw.copy()

    # 생성시간 / 두께 관련
    time_diff = mil.groupby("sequence_index").agg(
        생성시간=("pk_datetime", lambda x: x.max() - x.min())
    ).reset_index()

    mil = pd.merge(mil, time_diff, on="sequence_index", how="left")
    mil["시간변화량(초)"] = mil["생성시간"].dt.total_seconds()
    mil["두께변화량"] = mil["ampere"] * mil["시간변화량(초)"]
    mil["최종두께"] = mil.groupby("sequence_index")["두께변화량"].transform("sum")

    # 시계열 엔지니어링
    mil = add_time_features(mil)

    # tertile 부여
    def split_into_tertiles(group: pd.DataFrame) -> pd.DataFrame:
        n = len(group)
        group = group.sort_values("pk_datetime")
        group["tertile"] = pd.qcut(np.arange(n), 3, labels=[0, 1, 2])
        return group

    mil_tertile = mil.groupby("sequence_index").apply(split_into_tertiles).reset_index(drop=True)

    # 구간별 평균 집계
    mil_tertile = (
        mil_tertile
        .groupby(["sequence_index", "tertile"])
        .mean(numeric_only=True)
        .reset_index()
    )

    features_to_use = [
        "volt", "ampere", "temperature",
        "ampere_lag1", "volt_lag1", "temperature_lag1",
        "전류이동평균", "전압이동평균", "온도이동평균",
        "전류이동표준편차", "전압이동표준편차", "온도이동표준편차",
        "△전류", "△전압", "△온도",
        "failure", "tertile",
        "시간변화량(초)", "rec_num",
        "두께변화량", "최종두께",
        "sequence_index"
    ]

    missing = sorted(set(features_to_use) - set(mil_tertile.columns))
    if missing:
        raise KeyError(f"make_ml_data()에서 누락된 컬럼: {missing}")

    mil_tertile = mil_tertile[features_to_use].dropna()
    mil_tertile["tertile"] = mil_tertile["tertile"].astype(int)

    return mil_tertile


# =========================================================
# 4) Model + Meta
# =========================================================
@st.cache_resource
def load_model_and_meta(pkl_path="best_rf.pkl", meta_path="rf_metrics.json"):
    model = joblib.load(pkl_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


mil_ml = make_ml_data(mil_raw)
rf_model, rf_meta = load_model_and_meta()

RF_THRESHOLD = float(rf_meta.get("threshold", 0.5))

feature_names = rf_meta.get("feature_importance", {}).get("features", None)
if feature_names is None:
    feature_names = [c for c in mil_ml.columns if c != "failure"]

missing_for_X = sorted(set(feature_names) - set(mil_ml.columns))
if missing_for_X:
    raise KeyError(f"X 만들 때 누락된 컬럼: {missing_for_X}")

X_all = mil_ml[feature_names].copy()
y_all = (mil_ml["failure"] == -1.0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

feature_means = rf_meta.get("feature_means", None)
if feature_means is None:
    feature_means = X_train.mean(numeric_only=True).to_dict()

for col in feature_names:
    if col not in feature_means:
        feature_means[col] = 0.0


# =========================================================
# 5) 1-point Input → RF Input Row
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

    if "전류이동표준편차" in values: values["전류이동표준편차"] = 0.0
    if "전압이동표준편차" in values: values["전압이동표준편차"] = 0.0
    if "온도이동표준편차" in values: values["온도이동표준편차"] = 0.0

    if "△전류" in values: values["△전류"] = 0.0
    if "△전압" in values: values["△전압"] = 0.0
    if "△온도" in values: values["△온도"] = 0.0

    return pd.DataFrame([values])[feature_cols]


# =========================================================
# 6) Range Helpers (per rec_num/tertile)
# =========================================================
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


# =========================================================
# 7) Sidebar Navigation
# =========================================================
page = st.sidebar.radio(
    "페이지 선택",
    (
        "📊 공정 KPI",
        "📅 Sequence 패턴 한눈에",
        "💻 ML 예측",
        "🧯 불량 시퀀스 한눈에",
        "🪄 이상값 알려드림",
    ),
)


# =========================================================
# 8) Pages
# =========================================================
def page_kpi():
    st.markdown("#### 📊 공정 KPI 지표")

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
        st.metric("ISI_volt (전압 변동성 불량 민감도)", f"{ISI:.2f}" if not np.isnan(ISI) else "N/A")
        st.metric("DRI_current (변화량 기반 품질 위험지수)", f"{quality_summary['DRI_current']:.3f}")
        st.metric("MSK_temp (이동표준편차 기반 온도 민감도)", f"{quality_summary['MSK_temp']:.3f}")

    with col_right:
        st.markdown("##### 🏭 공정 상태 KPI")
        st.metric("OOC_volt (전압 정상 영역 일탈 비율)", f"{quality_summary['OOC_volt'] * 100:.1f} %")
        st.metric("OOC_amp (전류 정상 영역 일탈 비율)", f"{quality_summary['OOC_amp'] * 100:.1f} %")
        st.metric("OOC_temp (온도 정상 영역 일탈 비율)", f"{quality_summary['OOC_temp'] * 100:.1f} %")

    st.markdown("---")
    st.markdown("#### 🔥 불량 발생 sequence/날짜")

    seg_df = seg_defect_rate.reset_index()
    seg_df.columns = ["sequence_index", "defect_rate"]
    if not seg_df.empty:
        seg_chart = (
            alt.Chart(seg_df)
            .mark_bar(color=DEFECT_RED)
            .encode(
                x=alt.X("sequence_index:O", title="Sequence"),
                y=alt.Y("defect_rate:Q", title="불량률"),
            )
            .properties(height=120)
        )
        st.altair_chart(seg_chart, use_container_width=True)

    hour_df = hourly_defect_rate.reset_index()
    hour_df.columns = ["pk_datetime", "defect_rate"]
    if not hour_df.empty:

        line = (
            alt.Chart(hour_df)
            .mark_line(color=DEFECT_RED)
            .encode(
                x=alt.X("pk_datetime:T", title="일시",
                        axis=alt.Axis(format="%m-%d %H:%M", labelAngle=-45)),
                y=alt.Y("defect_rate:Q", title="불량률"),
                tooltip=[
                    alt.Tooltip("pk_datetime:T", title="일시"),
                    alt.Tooltip("defect_rate:Q", title="불량률", format=".3f"),
                ],
            )
        )

        points = (
            alt.Chart(hour_df)
            .mark_point(color=DEFECT_RED, filled=True, size=40)
            .encode(
                x="pk_datetime:T",
                y="defect_rate:Q",
                tooltip=[
                    alt.Tooltip("pk_datetime:T", title="일시"),
                    alt.Tooltip("defect_rate:Q", title="불량률", format=".3f"),
                ],
            )
        )

        st.altair_chart((line + points).properties(height=200), use_container_width=True)


def page_sequence_patterns():
    st.subheader("📅 Sequence별 패턴 한눈에 보기")

    rec_options = sorted(mil_raw["rec_num"].dropna().unique())
    rec_selected = st.selectbox("rec_num 선택", rec_options)

    mil = mil_raw[mil_raw["rec_num"] == rec_selected].copy()
    mil = add_time_features(mil)

    seq_status = (
        mil.groupby("sequence_index")["failure"]
        .agg(lambda s: -1 if (s == -1).any() else 1)
        .reset_index(name="seq_failure")
    )
    seq_status["status_label"] = seq_status["seq_failure"].map({-1: "⚠", 1: "✅"})
    seq_status["option_label"] = seq_status.apply(lambda r: f"{int(r.sequence_index)} - {r.status_label}", axis=1)

    label_to_seq = dict(zip(seq_status["option_label"], seq_status["sequence_index"]))

    options = seq_status["option_label"].tolist()
    default_vals = options[:3] if len(options) >= 3 else options

    selected_labels = st.multiselect("Sequence 선택(✅양품, ⚠불량)", options=options, default=default_vals)
    if not selected_labels:
        st.info("최소 1개 이상의 시퀀스를 선택하세요.")
        st.stop()

    selected_seqs = [label_to_seq[l] for l in selected_labels]
    mil_sel = mil[mil["sequence_index"].isin(selected_seqs)].copy()
    mil_sel = mil_sel.sort_values(["sequence_index", "pk_datetime"])

    mil_sel["t_min"] = mil_sel.groupby("sequence_index")["pk_datetime"].transform("min")
    mil_sel["t_max"] = mil_sel.groupby("sequence_index")["pk_datetime"].transform("max")

    dt = (mil_sel["pk_datetime"] - mil_sel["t_min"]).dt.total_seconds()
    total = (mil_sel["t_max"] - mil_sel["t_min"]).dt.total_seconds().replace(0, 1)
    mil_sel["norm_time"] = dt / total

    st.caption("※ x축은 각 시퀀스의 시작-끝을 0-1 범위로 정규화한 상대 시간입니다.")

    charts = []
    for sensor in ["ampere", "volt", "temperature"]:
        df_s = mil_sel[["sequence_index", "norm_time", sensor]].copy()
        chart = (
            alt.Chart(df_s)
            .mark_line()
            .encode(
                x=alt.X("norm_time:Q", title=""),
                y=alt.Y(f"{sensor}:Q", title=sensor),
                color=alt.Color(
                    "sequence_index:N",
                    title="Sequence",
                    scale=alt.Scale(scheme="tableau10")  # ✅ 기본 파랑 단일색 제거
                ),
                tooltip=[
                    alt.Tooltip("sequence_index:N", title="Sequence"),
                    alt.Tooltip("norm_time:Q", title="시간(0~1)", format=".2f"),
                    alt.Tooltip(f"{sensor}:Q", title=sensor, format=".2f"),
                ],
            )
            .properties(height=220)
        )
        charts.append(chart)

    combined = alt.vconcat(*charts).resolve_scale(y="independent")
    st.altair_chart(combined, use_container_width=True)


def page_ml_results():
    st.subheader("💻 ML 예측")

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

    with col_gap:
        st.subheader("")

    with col_left:
        st.markdown("#### 🪟 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_user)

        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).reset_index().rename(columns={"index": "actual"})
        cm_long = cm_df.melt(id_vars="actual", var_name="predicted", value_name="count")

        heatmap = (
            alt.Chart(cm_long)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", title="Predicted"),
                y=alt.Y("actual:N", title="Actual"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="reds"), legend=alt.Legend(title="Count")),
                tooltip=[
                    alt.Tooltip("actual:N", title="Actual"),
                    alt.Tooltip("predicted:N", title="Predicted"),
                    alt.Tooltip("count:Q", title="Count"),
                ],
            )
            .properties(height=500)
        )

        text = (
            alt.Chart(cm_long)
            .mark_text(fontSize=14, fontWeight="bold", color="black")
            .encode(x="predicted:N", y="actual:N", text="count:Q")
        )

        st.altair_chart(heatmap + text, use_container_width=True)

    st.markdown("#### 📊 Feature Importance")
    fi = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)

    fi_df = pd.DataFrame({"feature": fi.index, "importance": fi.values})
    fi_chart = (
        alt.Chart(fi_df)
        .mark_bar(color="#C0392B")  
        .encode(
            x=alt.X("feature:N", sort="-y", axis=alt.Axis(labelAngle=-45, title="Feature")),
            y=alt.Y("importance:Q", title="Importance"),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".4f"),
            ],
        )
        .properties(height=350))
    st.altair_chart(fi_chart, use_container_width=True)


def page_fault_sequences():
    st.subheader("🧯 불량 시퀀스 한눈에 보기")

    th_default = float(st.session_state.get("user_th", RF_THRESHOLD))
    user_th = st.slider("Threshold (불량으로 예측할 최소 확률)", 0.0, 1.0, value=th_default, step=0.01, key="th_slider_fault")
    st.session_state["user_th"] = float(user_th)

    y_proba_test = rf_model.predict_proba(X_test[feature_names])[:, 1]
    y_proba_s = pd.Series(y_proba_test, index=y_test.index)
    y_pred_user = (y_proba_s >= user_th).astype(int)

    # all data -> seq average proba
    proba_all = rf_model.predict_proba(X_all[feature_names])[:, 1]
    mil_all = mil_ml.copy()
    mil_all["proba_fail"] = proba_all

    seq_prob_all = (
        mil_all.groupby("sequence_index")
        .agg(
            mean_proba=("proba_fail", "mean"),
            failure_seq=("failure", lambda s: -1.0 if (s == -1.0).any() else 1.0),
        )
        .reset_index()
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

    bad_chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("sequence_index:N", sort="-y", title="Sequence Index"),
            y=alt.Y("mean_proba:Q", title="평균 불량 예측 확률"),
            color=alt.Color(
                "실제라벨:N",
                scale=alt.Scale(
                    domain=["실제 불량", "실제 양품"],
                    range=[DEFECT_RED, OK_GRAY]
                ),
                legend=alt.Legend(title="실제 라벨")
            ),
            tooltip=[
                alt.Tooltip("sequence_index:N", title="Sequence"),
                alt.Tooltip("mean_proba:Q", title="평균 불량 확률", format=".3f"),
                alt.Tooltip("실제라벨:N", title="실제 라벨"),
            ],
        )
        .properties(height=300)
    )

    # ✅ 그래프/테이블 출력 (여기가 빠지면 화면에 안 나옴)
   
    st.altair_chart(bad_chart, use_container_width=True)


def page_point_predict():
    st.subheader("🪄 이상값 알려드림")
    st.caption(
        "정류기(rec_num), 공정 구간(tertile), 온도·전류·전압 1포인트를 입력하면 "
        "기존 RandomForest 모델로 이 조건이 양품/불량 분포 중 어디에 가까운지 판정합니다."
    )

    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.markdown("#### 입력 조건")

        rec_label = st.selectbox("정류기(rec_num)", options=["rec1", "rec2"])
        rec_num_input = 1 if rec_label == "rec1" else 2

        tertile_label = st.selectbox(
            "공정 내 위치 (tertile)",
            options=["Ramp-up(0)", "Plateau(1)", "Ramp-down(2)"]
        )
        if "0" in tertile_label:
            tertile_input = 0
        elif "1" in tertile_label:
            tertile_input = 1
        else:
            tertile_input = 2

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

        # ✅ st.bar_chart(기본 파랑) 대신 Altair로 색 고정
        prob_plot_df = pd.DataFrame({
            "label": ["정상", "불량"],
            "prob": [1 - proba_bad, proba_bad]
        })
        prob_chart = (
            alt.Chart(prob_plot_df)
            .mark_bar()
            .encode(
                x=alt.X("label:N", title=""),
                y=alt.Y("prob:Q", title="확률", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "label:N",
                    scale=alt.Scale(domain=["정상", "불량"], range=[OK_GRAY, DEFECT_RED]),
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="라벨"),
                    alt.Tooltip("prob:Q", title="확률", format=".3f"),
                ]
            )
            .properties(height=240)
        )
        st.altair_chart(prob_chart, use_container_width=True)

        st.markdown("---")
        if pred == 1:
            st.warning(
                f"이 조건은 불량 확률이 **{proba_bad:.2f}**로 "
                f"임계값({RF_THRESHOLD:.2f})을 초과하여 **불량 분포**에 더 가깝습니다."
            )
        else:
            st.success(
                f"이 조건은 불량 확률이 **{proba_bad:.2f}**로 "
                f"임계값({RF_THRESHOLD:.2f})보다 낮아 **정상 분포**에 더 가깝습니다."
            )


# =========================================================
# 9) Router
# =========================================================
if page == "📊 공정 KPI":
    page_kpi()
elif page == "📅 Sequence 패턴 한눈에":
    page_sequence_patterns()
elif page == "💻 ML 예측":
    page_ml_results()
elif page == "🧯 불량 시퀀스 한눈에":
    page_fault_sequences()
elif page == "🪄 이상값 알려드림":
    page_point_predict()
