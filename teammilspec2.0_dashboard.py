import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#==========================
# 0. 페이지 명 설정 및 사이드바 디자인
#==========================
st.set_page_config(page_title="밀스펙 2.0", layout='wide')

st.markdown("""
<style>

/* 🔹 라디오 그룹 전체 간격 */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;   /* 카드 사이 간격 */
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
""", unsafe_allow_html=True)

st.caption("대시보드 프로젝트")
st.title("양극 산화 피막 데이터 기반 불량 예측 분석")

# ===================================
# 1. 데이터 로드
# ===================================
@st.cache_data
def load_data():
    mil = pd.read_csv('방산통합데이터셋.csv')
    mil['pk_datetime'] = pd.to_datetime(mil['pk_datetime'])
    return mil

mil_raw = load_data()   # 원본

# ===================================
# 2. 공통 함수들
# ===================================
def add_time_features(mil: pd.DataFrame) -> pd.DataFrame:
    mil = mil.sort_values(["sequence_index", "pk_datetime"]).copy()

    # lag
    mil['ampere_lag1']      = mil.groupby('sequence_index')['ampere'].shift(1)
    mil['volt_lag1']        = mil.groupby('sequence_index')['volt'].shift(1)
    mil['temperature_lag1'] = mil.groupby('sequence_index')['temperature'].shift(1)

    # rolling mean / std (window=3, shift 1)
    mil['전류이동평균'] = (
        mil.groupby('sequence_index')['ampere']
           .rolling(window=3).mean().shift(1)
           .reset_index(level=0, drop=True)
    )
    mil['전압이동평균'] = (
        mil.groupby('sequence_index')['volt']
           .rolling(window=3).mean().shift(1)
           .reset_index(level=0, drop=True)
    )
    mil['온도이동평균'] = (
        mil.groupby('sequence_index')['temperature']
           .rolling(window=3).mean().shift(1)
           .reset_index(level=0, drop=True)
    )

    mil['전류이동표준편차'] = (
        mil.groupby('sequence_index')['ampere']
           .rolling(window=3).std().shift(1)
           .reset_index(level=0, drop=True)
    )
    mil['전압이동표준편차'] = (
        mil.groupby('sequence_index')['volt']
           .rolling(window=3).std().shift(1)
           .reset_index(level=0, drop=True)
    )
    mil['온도이동표준편차'] = (
        mil.groupby('sequence_index')['temperature']
           .rolling(window=3).std().shift(1)
           .reset_index(level=0, drop=True)
    )

    # diff (sequence 별)
    mil['△전류'] = mil.groupby('sequence_index')['ampere'].diff()
    mil['△전압'] = mil.groupby('sequence_index')['volt'].diff()
    mil['△온도'] = mil.groupby('sequence_index')['temperature'].diff()

    return mil


def compute_quality_metrics(mil: pd.DataFrame, k: float = 3.0):
    df = mil.copy()

    # failure: 1=정상, -1=불량
    df["is_defect"] = (df["failure"] == -1).astype(int)

    # 1) 전체 불량률
    defect_rate = df["is_defect"].mean()

    # 2) sequence별 불량률
    segment_defect_rate = df.groupby("sequence_index")["is_defect"].mean()

    # 3) 시간대별 불량률 (1H)
    df_time = df.set_index("pk_datetime")
    hourly_defect_rate = df_time["is_defect"].resample("1H").mean()

    # 4) 센서 평균 차이 (불량 - 정상)
    mask_def = df["is_defect"] == 1
    mask_ok  = df["is_defect"] == 0

    volt_diff = df.loc[mask_def, "volt"].mean()        - df.loc[mask_ok, "volt"].mean()
    amp_diff  = df.loc[mask_def, "ampere"].mean()      - df.loc[mask_ok, "ampere"].mean()
    temp_diff = df.loc[mask_def, "temperature"].mean() - df.loc[mask_ok, "temperature"].mean()

    # 5) ISI_volt : 불량/정상 표준편차 비
    volt_std_def = df.loc[mask_def, "volt"].std()
    volt_std_ok  = df.loc[mask_ok, "volt"].std()
    ISI_volt = np.nan
    if not np.isnan(volt_std_ok) and volt_std_ok != 0:
        ISI_volt = volt_std_def / volt_std_ok

    # 6) DRI_current : 불량 시 전류 변화량 절대값 평균
    DRI_current = df.loc[mask_def, "△전류"].abs().mean()

    # 7) MSK_temp : 불량 시 온도 이동표준편차 평균
    MSK_temp = df.loc[mask_def, "온도이동표준편차"].mean()

    # ------------------------------------------------------------------
    # 8) 공통 함수: OOC 비율과 Drift(기울기)를 한 번에 계산
    # ------------------------------------------------------------------
    def _calc_ooc_and_drift(
        data: pd.DataFrame,
        value_col: str,      # 원시 센서 값 (volt / ampere / temperature)
        ma_col: str,         # 이동평균 컬럼명
        std_col: str,        # 이동표준편차 컬럼명
        time_col: str = "pk_datetime",
        k: float = 3.0,
    ):
        """
        관리한계 k*σ 기준 OOC 비율과 이동평균에 대한 시간-기울기(drift)를 계산.
        관련 컬럼이 없거나 데이터가 부족하면 np.nan 반환.
        """
        if not all(c in data.columns for c in [value_col, ma_col, std_col, time_col]):
            return np.nan, np.nan

        s = data[[time_col, value_col, ma_col, std_col]].dropna().sort_values(time_col)
        if len(s) == 0:
            return np.nan, np.nan

        # OOC 비율
        dev   = (s[value_col] - s[ma_col]).abs()
        limit = k * s[std_col]
        ooc_ratio = (dev > limit).mean()

        # Drift(시간 대비 이동평균의 기울기)
        drift = np.nan
        if len(s) > 1:
            x = (s[time_col] - s[time_col].min()).dt.total_seconds()
            y = s[ma_col]
            drift = np.polyfit(x, y, 1)[0]

        return ooc_ratio, drift

    # ------------------------------------------------------------------
    # 9) 센서별 공정 상태 KPI
    #    - volt  : 전압이동평균 / 전압이동표준편차
    #    - ampere: 전류이동평균 / 전류이동표준편차   (컬럼명 다르면 여기만 수정)
    #    - temp  : 온도이동평균 / 온도이동표준편차   (컬럼명 다르면 여기만 수정)
    # ------------------------------------------------------------------
    OOC_volt, drift_volt = _calc_ooc_and_drift(
        df, "volt", "전압이동평균", "전압이동표준편차", "pk_datetime", k
    )

    OOC_amp, drift_amp = _calc_ooc_and_drift(
        df, "ampere", "전류이동평균", "전류이동표준편차", "pk_datetime", k
    )

    OOC_temp, drift_temp = _calc_ooc_and_drift(
        df, "temperature", "온도이동평균", "온도이동표준편차", "pk_datetime", k
    )

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

def classification_report_to_df(report_dict):
    """
    sklearn classification_report(output_dict=True)을
    DataFrame 형태로 변환하여 표 형태 시각화에 적합하게 만든다.
    """
    import pandas as pd

    df = pd.DataFrame(report_dict).transpose()
    df = df.round(3)

    # support 값이 float로 나올 수 있으므로 int로 정리
    if "support" in df.columns:
        df["support"] = df["support"].astype(int)

    # 행 순서를 사람이 보기 좋게 재배치
    preferred_index = ["0", "1", "accuracy", "macro avg", "weighted avg"]
    df = df.loc[preferred_index]

    return df


# ---------------- ML 용 데이터 & 모델 ----------------
@st.cache_data
def make_ml_data():
    # 1) 원본 로드 + pk_datetime 처리
    mil = pd.read_csv("방산통합데이터셋.csv")
    mil["pk_datetime"] = pd.to_datetime(mil["pk_datetime"], errors="coerce")
    mil.dropna(subset=["pk_datetime"], inplace=True)

    # 2) 시퀀스별 생성시간 / 두께 관련 파생변수
    time_diff = mil.groupby('sequence_index').agg(
        생성시간=('pk_datetime', lambda x: x.max() - x.min())
    ).reset_index()

    mil = pd.merge(mil, time_diff, on='sequence_index', how='left')
    mil['시간변화량(초)'] = mil['생성시간'].dt.total_seconds()
    mil['두께변화량'] = mil['ampere'] * mil['시간변화량(초)']
    mil['최종두께'] = mil.groupby('sequence_index')['두께변화량'].transform('sum')

    # 3) 시계열 엔지니어링
    mil = add_time_features(mil)

    # 4) 3구간 tertile 분할
    sequence_area = mil.groupby('sequence_index')

    def split_into_tertiles(group):
        n = len(group)
        group = group.sort_index()  # 시간 순 정렬
        group['tertile'] = pd.qcut(np.arange(n), 3, labels=[0, 1, 2])
        return group

    mil_tertile = sequence_area.apply(split_into_tertiles).reset_index(drop=True)

    # 5) 구간별 집계 (평균)
    mil_tertile = (
        mil_tertile
        .groupby(['sequence_index', 'tertile'])
        .mean()
        .reset_index()
    )

    features_to_use = [
        'volt','ampere','temperature','ampere_lag1',
        'volt_lag1','temperature_lag1','전류이동평균','전압이동평균','온도이동평균',
        '전류이동표준편차','전압이동표준편차','온도이동표준편차',
        '△전류','△전압','△온도',
        'failure','tertile',
        '시간변화량(초)', 'rec_num',
        '두께변화량', '최종두께',
        'sequence_index'
    ]

    missing = set(features_to_use) - set(mil_tertile.columns)
    if missing:
        raise KeyError(f"make_ml_data()에서 누락된 컬럼: {sorted(missing)}")

    mil_tertile = mil_tertile[features_to_use].dropna()

    return mil_tertile


@st.cache_resource
def load_rf_model():
    model = joblib.load("best_rf.pkl")
    with open("rf_metrics.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

mil_ml = make_ml_data()
rf_model, rf_meta = load_rf_model()

threshold = float(rf_meta["threshold"])

if "feature_importance" in rf_meta and "features" in rf_meta["feature_importance"]:
    feature_names = rf_meta["feature_importance"]["features"]
else:
    feature_names = list(mil_ml.drop(columns=["failure"]).columns)

missing_for_X = set(feature_names) - set(mil_ml.columns)
if missing_for_X:
    raise KeyError(f"X 만들 때 누락된 컬럼: {sorted(missing_for_X)}")

X = mil_ml[feature_names]
y = (mil_ml["failure"] == -1.0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===================================
# 3. 페이지 선택 (사이드바)
# ===================================
page = st.sidebar.radio(
    "페이지 선택",
    (
        "📊 공정 KPI",
        "📅 Sequence 패턴 한눈에",
        "💻 ML 예측 결과",
        "🧯 불량 시퀀스 한눈에",
    )
)

# ===================================
# 4. 페이지별 내용
# ===================================

# ===================================
# 4-1. 공정 KPI
# ===================================
if page == "📊 공정 KPI":
    st.markdown("#### 📊 공정 KPI 지표")

    # rec_num 필터
    rec_options = sorted(mil_raw["rec_num"].unique())
    rec_selected = st.selectbox("rec_num 선택", rec_options)

    mil = mil_raw[mil_raw["rec_num"] == rec_selected].copy()

    mil = add_time_features(mil)
    quality_summary, seg_defect_rate, hourly_defect_rate = \
        compute_quality_metrics(mil, k=3.0)

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
        st.metric(
            "ISI_volt (전압 변동성 불량 민감도)",
            f"{ISI:.2f}" if not np.isnan(ISI) else "N/A"
        )
        st.metric("DRI_current (변화량 기반 품질 위험지수)",
                  f"{quality_summary['DRI_current']:.3f}")
        st.metric("MSK_temp (이동표준편차 기반 온도 민감도)",
                  f"{quality_summary['MSK_temp']:.3f}")

    with col_right:
        st.markdown("##### 🏭 공정 상태 KPI")

        # 1) volt KPI
        st.metric(
            "OOC_volt (전압 정상 영역 일탈 비율)",
            f"{quality_summary['OOC_volt'] * 100:.1f} %"
        )
        

        # 2) ampere KPI
        if "OOC_amp" in quality_summary:
            st.metric(
                "OOC_amp (전류 정상 영역 일탈 비율)",
                f"{quality_summary['OOC_amp'] * 100:.1f} %"
            )
            

        # 3) temperature KPI
        if "OOC_temp" in quality_summary:
            st.metric(
                "OOC_temp (온도 정상 영역 일탈 비율)",
                f"{quality_summary['OOC_temp'] * 100:.1f} %"
            )



    st.markdown("---")
    st.markdown("#### 🔥불량 발생 sequence/날짜")

    seg_df = seg_defect_rate.reset_index()
    seg_df.columns = ["sequence_index", "defect_rate"]
    if not seg_df.empty:
        seg_chart = (
            alt.Chart(seg_df)
            .mark_bar()
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
        hour_chart = (
            alt.Chart(hour_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("pk_datetime:T",
                        title="일시",
                        axis=alt.Axis(format="%m-%d %H:%M", labelAngle=-45)),
                y=alt.Y("defect_rate:Q", title="불량률"),
            )
            .properties(height=200)
        )
        st.altair_chart(hour_chart, use_container_width=True)


# ===================================
# 4-2. Sequence 패턴 한눈에
# ===================================
elif page == "📅 Sequence 패턴 한눈에":
    st.subheader("📅 Sequence별 패턴 한눈에 보기")

    # 1) rec_num 필터
    rec_options = sorted(mil_raw["rec_num"].unique())
    rec_selected = st.selectbox("rec_num 선택", rec_options)
    mil = mil_raw[mil_raw["rec_num"] == rec_selected].copy()
    mil = add_time_features(mil)

    # 2) 시퀀스별 양품/불량 라벨
    seq_status = (
        mil.groupby("sequence_index")["failure"]
           .agg(lambda s: -1 if (s == -1).any() else 1)
           .reset_index(name="seq_failure")
    )

    seq_status["status_label"] = seq_status["seq_failure"].map({
        -1: "⚠ ",
         1: "✅ ",
    })

    seq_status["option_label"] = seq_status.apply(
        lambda r: f"{int(r.sequence_index)} - {r.status_label}", axis=1
    )

    label_to_seq = dict(
        zip(seq_status["option_label"], seq_status["sequence_index"])
    )

    # 3) 여러 시퀀스 선택
    options = seq_status["option_label"].tolist()
    default_vals = options[:3] if len(options) >= 3 else options

    selected_labels = st.multiselect(
        "Sequence 선택(✅양품, ⚠불량)",
        options=options,
        default=default_vals
    )

    if not selected_labels:
        st.info("최소 1개 이상의 시퀀스를 선택하세요.")
        st.stop()

    selected_seqs = [label_to_seq[l] for l in selected_labels]

    # 4) 선택된 시퀀스 데이터만 사용
    mil_sel = mil[mil["sequence_index"].isin(selected_seqs)].copy()
    mil_sel = mil_sel.sort_values(["sequence_index", "pk_datetime"])

    # 5) 각 시퀀스별 정규화 시간(norm_time = 0~1)
    mil_sel["t_min"] = mil_sel.groupby("sequence_index")["pk_datetime"].transform("min")
    mil_sel["t_max"] = mil_sel.groupby("sequence_index")["pk_datetime"].transform("max")

    dt = (mil_sel["pk_datetime"] - mil_sel["t_min"]).dt.total_seconds()
    total = (mil_sel["t_max"] - mil_sel["t_min"]).dt.total_seconds().replace(0, 1)
    mil_sel["norm_time"] = dt / total

    st.caption(
        "※ x축은 각 시퀀스의 시작-끝을 0-1 범위로 정규화한 상대 시간입니다. ")

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
                    legend=alt.Legend(orient="right")
                ),
                tooltip=[
                    alt.Tooltip("sequence_index:N", title="Sequence"),
                    alt.Tooltip("norm_time:Q", title="시간(0~1)", format=".2f"),
                    alt.Tooltip(f"{sensor}:Q", title=sensor, format=".1f"),
                ],
            )
            .properties(height=220)
        )
        charts.append(chart)

    combined = alt.vconcat(*charts).resolve_scale(y="independent")
    st.altair_chart(combined, use_container_width=True)


# ===================================
# 4-3. 머신러닝 예측 결과
# ===================================
elif page == "💻 ML 예측 결과":
    st.subheader("💻 ML 예측 결과")

    # 2) 임계값 + KPI / Confusion Matrix
    

    # ---------------------------------
    # 0) 공통: Test 확률 예측 (제일 먼저!)
    # ---------------------------------
    X_test_rf = X_test[feature_names]
    y_proba = rf_model.predict_proba(X_test_rf)[:, 1]
    y_proba_s = pd.Series(y_proba, index=y_test.index)

    # ---------------------------------
    # 2) 임계값 + KPI / Confusion Matrix
    # ---------------------------------
    

    col_left, col_gap, col_right = st.columns([1, 0.2, 1])

    with col_right:
        st.markdown("#### 🧮 임계값 & 핵심 성능 지표")

        # threshold slider
        if "user_th" in st.session_state:
            default_th = float(st.session_state["user_th"])
        else:
            default_th = float(threshold)

        user_th = st.slider(
            "Threshold (불량으로 예측할 최소 확률)",
            0.0, 1.0,
            value=default_th,
            step=0.01,
            key="th_slider_ml"
        )
        st.session_state["user_th"] = float(user_th)

        # ★ 여기서는 이미 y_proba_s가 위에서 계산되어 있음
        y_pred_user = (y_proba_s >= user_th).astype(int)

        report_dict = classification_report(
            y_test, y_pred_user, output_dict=True, zero_division=0
        )
        acc = report_dict["accuracy"]
        f1_defect = report_dict["1"]["f1-score"]
        recall_defect = report_dict["1"]["recall"]

        # KPI metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc*100:.1f} %")
        m2.metric("F1 (불량)", f"{f1_defect:.3f}")
        m3.metric("Recall (불량)", f"{recall_defect:.3f}")

        st.caption("📄 전체 분류 리포트")
        report_df = classification_report_to_df(report_dict)
        st.dataframe(
            report_df, use_container_width=True, hide_index=False
        )

    with col_gap:
        st.subheader("")

    with col_left:
        st.markdown("#### 🪟 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred_user)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"]
        ).reset_index().rename(columns={"index": "actual"})

        cm_long = cm_df.melt(
            id_vars="actual",
            var_name="predicted",
            value_name="count"
        )

        heatmap = (
            alt.Chart(cm_long)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", title="Predicted"),
                y=alt.Y("actual:N", title="Actual"),
                color=alt.Color(
                    "count:Q",
                    scale=alt.Scale(scheme="blues"),
                    legend=alt.Legend(title="Count")
                ),
                tooltip=[
                    alt.Tooltip("actual:N", title="Actual"),
                    alt.Tooltip("predicted:N", title="Predicted"),
                    alt.Tooltip("count:Q", title="Count")
                ],
            )
            .properties(height=500)
        )

        text = (
            alt.Chart(cm_long)
            .mark_text(fontSize=14, fontWeight="bold", color="black")
            .encode(
                x="predicted:N",
                y="actual:N",
                text="count:Q",
            )
        )

        st.altair_chart(heatmap + text, use_container_width=True)
    
    

    # ---------------------------------
    #  Feature Importance
    # ---------------------------------
    st.markdown("#### 📊 Feature Importance")

    importances = rf_model.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    fi_df = pd.DataFrame({
        "feature": fi.index,
        "importance": fi.values
    })

    fi_chart = (
        alt.Chart(fi_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "feature:N",
                sort='-y',
                axis=alt.Axis(labelAngle=-45, title="Feature")
            ),
            y=alt.Y("importance:Q", title="Importance"),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".4f"),
            ],
        )
        .properties(height=350)
    )

    st.altair_chart(fi_chart, use_container_width=True)




# ---------- 불량 원인 분석 ----------
elif page == "🧯 불량 시퀀스 한눈에":
    st.subheader("🧯 불량 시퀀스 한눈에 보기")

    # 0) ML 페이지와 연동되는 Threshold 슬라이더
    if "user_th" in st.session_state:
        th_default = float(st.session_state["user_th"])
    else:
        th_default = float(threshold)

    user_th = st.slider(
        "Threshold (불량으로 예측할 최소 확률)",
        0.0, 1.0,
        value=th_default,
        step=0.01,
        key="th_slider_fault"
    )
    st.session_state["user_th"] = float(user_th)

    # 이 페이지에서도 test 예측 다시 계산 (오진 케이스용)
    X_test_rf = X_test[feature_names]
    y_proba_test = rf_model.predict_proba(X_test_rf)[:, 1]
    y_proba_s = pd.Series(y_proba_test, index=y_test.index)
    y_pred_user = (y_proba_s >= user_th).astype(int)

    # 🔹 NEW 0-1) 전체 데이터 기준 시퀀스별 평균 불량 확률 계산
    #      (이걸 가지고 '불량으로 예측된 시퀀스 전체' 섹션을 그린다)
    X_all = mil_ml[feature_names]
    proba_all = rf_model.predict_proba(X_all)[:, 1]

    mil_all = mil_ml.copy()
    mil_all["proba_fail"] = proba_all

    seq_prob_all = (
        mil_all
        .groupby("sequence_index")
        .agg(
            mean_proba=("proba_fail", "mean"),                 # 시퀀스 평균 불량 확률
            failure_seq=("failure",                           # 시퀀스 실제 라벨(양품/불량)
                        lambda s: -1.0 if (s == -1.0).any() else 1.0)
        )
        .reset_index()
    )

    # 임계값 기준 시퀀스 단위 예측 라벨
    seq_prob_all["pred_seq"] = (seq_prob_all["mean_proba"] >= user_th).astype(int)

    # 불량으로 예측된 시퀀스만 추출
    bad_seq_df = (
        seq_prob_all[seq_prob_all["pred_seq"] == 1]
        .sort_values("mean_proba", ascending=False)
    )

    # 1) 시퀀스별 불량 확률 + 오진 케이스 (좌/우 배치)
    col_left, col_right = st.columns([1, 1])

    # 1-1) 시퀀스별 불량 확률 (LEFT) - 기존 코드 유지
    with col_left:
        st.markdown("#### 🔍 시퀀스별 불량 확률")

        seq_list = sorted(mil_ml["sequence_index"].unique())
        seq_choice = st.selectbox("시퀀스를 선택하세요", seq_list, key="seq_choice_fault")
        seq_df = mil_ml[mil_ml["sequence_index"] == seq_choice]

        if len(seq_df) > 0:
            X_seq_seg = seq_df[feature_names]
            proba_seq = rf_model.predict_proba(X_seq_seg)[:, 1]
            proba_seq_s = pd.Series(proba_seq, index=seq_df.index)

            mean_proba = proba_seq_s.mean()
            pred_seq = int(mean_proba >= user_th)

            c1, c2, c3 = st.columns(3)
            c1.metric("평균 불량 확률", f"{mean_proba:.3f}")
            c2.metric("임계값", f"{user_th:.3f}")
            c3.metric("예측 결과", "⚠" if pred_seq == 1 else "✅")

            with st.expander("선택 시퀀스 (세그먼트 기반) 상세 보기", expanded=False):
                seq_view = seq_df.copy()
                seq_view["불량확률(모델)"] = proba_seq_s
                st.dataframe(seq_view)
        else:
            st.info("해당 시퀀스 데이터 없음")

    # 1-2) 오진 케이스 (RIGHT) - 기존 코드 유지
    with col_right:
        st.markdown("#### ❌ 오진(예측 틀린) 케이스")

        wrong_mask = (y_test != y_pred_user)
        wrong_idx = y_test.index[wrong_mask]

        if len(wrong_idx) == 0:
            st.success("현재 오진 케이스 없음 🎉")
        else:
            st.write(f"총 **{len(wrong_idx)}건**의 오진 케이스")
            with st.expander("오진 케이스 상세 보기", expanded=False):
                wrong_cases = mil_ml.loc[wrong_idx].copy()
                wrong_cases["실제값(y_true)"] = y_test.loc[wrong_idx]
                wrong_cases["예측값(y_pred)"] = y_pred_user.loc[wrong_idx]
                wrong_cases["불량확률(모델)"] = y_proba_s.loc[wrong_idx]
                st.dataframe(wrong_cases)

    # ----------------------------------------------------
    # 2) 불량으로 예측된 시퀀스 전체 보기  (NEW 섹션)
    # ----------------------------------------------------
    st.markdown("---")
    st.markdown("#### 📊 불량으로 예측된 시퀀스 전체 보기")

    n_bad = len(bad_seq_df)
    st.write(f"현재 임계값 기준으로 불량으로 예측된 시퀀스는 총 **{n_bad}개** 입니다.")

    if n_bad == 0:
        st.info("이 임계값에서는 불량으로 예측된 시퀀스가 없습니다.")
    else:
        # 2-1) 리스트 표로 보여주기
        show_df = bad_seq_df.copy()
        show_df["실제라벨"] = np.where(
            show_df["failure_seq"] == -1.0, "실제 불량", "실제 양품"
        )

  

        # 2-2) 막대 그래프로 시각화
        st.markdown("")

        chart_df = show_df.copy()
        chart_df["sequence_index"] = chart_df["sequence_index"].astype(str)
        chart_df["실제라벨"] = np.where(
            chart_df["failure_seq"] == -1.0, "실제 불량", "실제 양품"
        )

        bad_chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("sequence_index:N",
                        sort="-y",
                        title="Sequence Index"),
                y=alt.Y("mean_proba:Q", title="평균 불량 예측 확률"),
                color=alt.Color("실제라벨:N", title="실제 라벨"),
                tooltip=[
                    alt.Tooltip("sequence_index:N", title="Sequence"),
                    alt.Tooltip("mean_proba:Q", title="평균 불량 확률", format=".3f"),
                    alt.Tooltip("실제라벨:N", title="실제 라벨"),
                ],
            )
            .properties(height=300)
        )

        st.altair_chart(bad_chart, use_container_width=True)
