"""RoadLens Anyang - Streamlit 시제품.

실행: streamlit run app.py
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import config, repository
from src.data_loader import fetch_precipitation, load_damage_csv, load_school_zone_csv
from src.image_inference import MODEL_MISSING_MESSAGE, detect_damage
from src.preprocessing import (
    clean_damage_frame,
    clean_school_zone_frame,
    summarize_base_date,
)
from src.privacy import anonymize_image
from src.risk_score import compute_risk_score, photo_severity_from_detections
from src.route_optimizer import (
    DISTANCE_MODE_HAVERSINE,
    format_distance,
    optimize_route,
)
from src.schemas import (
    COL_BASE_DATE,
    COL_COLLECTED_AT,
    COL_HAZARD_TYPE,
    COL_LAT,
    COL_LON,
    COL_NODE_LINK,
    COL_ROAD_NAME,
    InspectionStatus,
)
from src.spatial_analysis import (
    cluster_damage_points,
    cluster_summary,
    distances_from_point_m,
    history_density_count,
    history_density_score,
    nearest_school_zone_distance_m,
    rainfall_score,
    road_importance_score,
    road_importance_table,
    school_zone_score,
)

st.set_page_config(
    page_title="RoadLens Anyang",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "1. 서비스 소개",
    "2. 과거 파손 이력 지도",
    "3. 사진 분석",
    "4. 점검 우선순위",
    "5. 점검목록 및 순찰경로",
    "6. 처리결과 기록",
]

AI_DISCLAIMER = (
    "AI 분석 결과는 참고 정보이며 보수 여부를 확정하지 않습니다. "
    "최종 판단은 담당자 확인 결과로 기록됩니다."
)


def bullet(text: str) -> None:
    """강조 상자 대신 '- 내용' 한 줄로 안내 문구를 표시한다."""
    st.markdown("- " + str(text))


# ---------------------------------------------------------------------------
# 데이터 로딩 (캐시)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="파손 이력 데이터를 불러오는 중입니다...")
def load_damage_data():
    frame, load_report = load_damage_csv()
    clean, clean_report = clean_damage_frame(frame)
    return clean, load_report, clean_report


@st.cache_data(show_spinner="어린이보호구역 데이터를 불러오는 중입니다...")
def load_school_zone_data():
    frame, load_report = load_school_zone_csv()
    clean, clean_report = clean_school_zone_frame(frame)
    return clean, load_report, clean_report


@st.cache_data(show_spinner="밀집구간을 계산하는 중입니다...")
def compute_clusters(frame: pd.DataFrame, eps_m: float, min_samples: int):
    return cluster_damage_points(frame, eps_m=eps_m, min_samples=min_samples)


def get_config() -> config.AppConfig:
    if "config" not in st.session_state:
        config.ensure_directories()
        st.session_state["config"] = config.load_config()
    return st.session_state["config"]


def data_origin_badge(is_sample: bool) -> None:
    """실제 데이터 / 테스트용 샘플을 화면에서 구분해 표시한다."""
    if is_sample:
        bullet("테스트용 샘플 데이터를 표시하고 있습니다. 실제 안양시 데이터가 아닙니다.")
    else:
        bullet("안양시 공공데이터(공공데이터포털)를 사용하고 있습니다.")
        bullet(
            "파손 이력은 2021년 9~11월 수집분(데이터기준일자 2021-12-17)이며, "
            "2026년 9월 기준 안양시가 개방한 가장 최신 자료입니다."
        )


# ---------------------------------------------------------------------------
# 1. 서비스 소개
# ---------------------------------------------------------------------------
def page_intro(cfg: config.AppConfig) -> None:
    st.title("RoadLens Anyang")
    st.caption("안양시 도로 점검 우선순위 지원 시제품 · 2026년 안양시 공공데이터와 AI 활용 대학생 경진대회 출품용")

    st.subheader("서비스 목적")
    st.markdown(
        """
    - 안양시 과거 도로부속물 파손 이력을 지도에서 확인합니다.
    - 현장에서 찍은 도로 사진을 업로드하면 포트홀·균열을 탐지합니다.
    - 과거 이력 밀도, 사진상 파손 정도, 어린이보호구역 인접도, 도로구간 중요도,
      강수 영향을 결합해 점검 우선순위를 계산합니다.
    - 선택한 점검 대상의 순찰 순서를 만들고 처리상태를 기록합니다.
        """
    )

    damage, load_report, clean_report = load_damage_data()
    school_zones, sz_load, _ = load_school_zone_data()

    st.subheader("데이터 기준일 및 출처")
    base_date = summarize_base_date(damage)
    col1, col2, col3 = st.columns(3)
    col1.metric("파손 이력 건수(정제 후)", f"{len(damage):,}")
    col2.metric("데이터기준일자", base_date or "확인 불가")
    col3.metric("어린이보호구역 건수", f"{len(school_zones):,}")

    if load_report.is_sample or sz_load.is_sample:
        data_origin_badge(True)
    else:
        data_origin_badge(False)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "데이터": "경기도 안양시_도로위험물 현황 (포트홀·균열)",
                    "출처": "https://www.data.go.kr/data/15096553/fileData.do",
                    "적재 방식": "data/raw/anyang_road_hazard.csv",
                    "현재 사용 중": ("테스트 샘플" if load_report.is_sample else "실제 파일"),
                },
                {
                    "데이터": "경기도 안양시_도로부속물 파손 현황",
                    "출처": "https://www.data.go.kr/data/15096549/fileData.do",
                    "적재 방식": "data/raw/anyang_road_damage.csv",
                    "현재 사용 중": ("테스트 샘플" if load_report.is_sample else "실제 파일"),
                },
                {
                    "데이터": "전국어린이보호구역표준데이터",
                    "출처": "https://www.data.go.kr/data/15012891/standard.do",
                    "적재 방식": "data/raw/school_zones.csv (안양시만 필터)",
                    "현재 사용 중": ("테스트 샘플" if sz_load.is_sample else "실제 파일"),
                },
                {
                    "데이터": "기상청_단기예보 조회서비스",
                    "출처": "https://www.data.go.kr/data/15084084/openapi.do",
                    "적재 방식": "KMA_SERVICE_KEY 환경변수가 있을 때만 호출",
                    "현재 사용 중": ("사용 가능" if cfg.kma_enabled else "미사용(키 없음)"),
                },
                {
                    "데이터": "RDD2022 도로 파손 이미지",
                    "출처": "https://figshare.com/articles/dataset/21431547",
                    "적재 방식": "Google Colab 에서 학습, best.onnx 를 models/ 에 배치",
                    "현재 사용 중": ("모델 있음" if cfg.model_available else "모델 없음"),
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("AI 의 역할과 한계")
    bullet("AI 는 업로드된 사진에서 파손으로 보이는 영역을 표시하고 신뢰도를 제시합니다.")

    st.subheader("현재 실행 환경 상태")
    status_rows = [
        {
            "기능": "과거 파손 이력 지도",
            "상태": "사용 가능" if len(damage) else "데이터 없음",
            "비고": f"정제 후 {len(damage):,}건",
        },
        {
            "기능": "사진 분석(AI 탐지)",
            "상태": "사용 가능" if cfg.model_available else "비활성",
            "비고": (
                f"{cfg.model_path}" if cfg.model_available
                else "models/best.onnx 또는 models/best.pt 필요"
            ),
        },
        {
            "기능": "강수 정보 자동 조회",
            "상태": "사용 가능" if cfg.kma_enabled else "비활성",
            "비고": "KMA_SERVICE_KEY 미설정 시 강수량 직접 입력" if not cfg.kma_enabled else "기상청 단기예보",
        },
        {
            "기능": "실제 도로거리 경로",
            "상태": "사용 가능" if cfg.routing_api_enabled else "비활성",
            "비고": "도로망 API 미연동 - Haversine 근사 경로 사용",
        },
    ]
    st.dataframe(pd.DataFrame(status_rows), width="stretch", hide_index=True)

    if cfg.notes:
        with st.expander("비활성화된 기능의 사유", expanded=True):
            for note in cfg.notes:
                st.write("- " + note)

    st.subheader("개인정보 처리 안내")
    bullet(config.PHOTO_RETENTION_NOTICE)


# ---------------------------------------------------------------------------
# 2. 과거 파손 이력 지도
# ---------------------------------------------------------------------------
def page_history_map(cfg: config.AppConfig) -> None:
    st.title("과거 파손 이력 지도")
    st.caption("표시되는 지점은 모두 과거에 수집된 파손 이력이며, 현재 파손 상태가 아닙니다.")

    damage, load_report, clean_report = load_damage_data()
    data_origin_badge(load_report.is_sample)

    if damage.empty:
        bullet(
            "표시할 데이터가 없습니다. `data/raw/anyang_road_damage.csv` 를 배치한 뒤 다시 실행하세요.",
        )
        with st.expander("불러오기 기록"):
            for message in load_report.messages:
                st.write("- " + message)
        return

    bullet(f"데이터 정제 완료 (사용 {len(damage):,}건)")

    # --- 필터 ---
    st.subheader("필터")
    fcol1, fcol2, fcol3 = st.columns([2, 2, 3])

    hazard_options = sorted(
        [h for h in damage[COL_HAZARD_TYPE].dropna().unique().tolist() if str(h).strip()]
    ) if COL_HAZARD_TYPE in damage.columns else []
    selected_hazards = fcol1.multiselect(
        "위험물 유형", hazard_options, default=hazard_options
    )

    filtered = damage.copy()
    if selected_hazards and COL_HAZARD_TYPE in filtered.columns:
        filtered = filtered[filtered[COL_HAZARD_TYPE].isin(selected_hazards)]

    if COL_COLLECTED_AT in damage.columns and damage[COL_COLLECTED_AT].notna().any():
        min_date = pd.to_datetime(damage[COL_COLLECTED_AT]).min().date()
        max_date = pd.to_datetime(damage[COL_COLLECTED_AT]).max().date()
        date_range = fcol2.date_input(
            "수집날짜 범위",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            dates = pd.to_datetime(filtered[COL_COLLECTED_AT])
            filtered = filtered[
                (dates.dt.date >= start) & (dates.dt.date <= end) | dates.isna()
            ]
    else:
        fcol2.info("수집날짜 항목이 없어 날짜 필터를 사용할 수 없습니다.")

    road_query = fcol3.text_input("도로명 검색", placeholder="예: 안양로")
    if road_query and COL_ROAD_NAME in filtered.columns:
        filtered = filtered[
            filtered[COL_ROAD_NAME].astype(str).str.contains(road_query, na=False)
        ]

    st.write(f"필터 결과: {len(filtered):,}건 (전체 {len(damage):,}건)")

    # --- 표시 방식 ---
    st.subheader("지도")
    mcol1, mcol2, mcol3 = st.columns([2, 2, 2])
    layer_mode = mcol1.radio(
        "표시 레이어",
        ["마커 클러스터", "히트맵", "밀집구간(군집)"],
        horizontal=False,
        help="개별 지점을 모두 그리지 않고 집계 레이어로 표시해 대용량에서도 멈추지 않습니다.",
    )
    cluster_eps = mcol2.slider("밀집 판정 반경(m)", 50, 500, 150, step=50)
    cluster_min = mcol3.slider("밀집 최소 건수", 2, 30, 5)

    if filtered.empty:
        bullet("필터 조건에 해당하는 데이터가 없습니다.")
        return

    try:
        import folium
        from folium.plugins import HeatMap, MarkerCluster
        from streamlit_folium import st_folium
    except ImportError as exc:
        bullet(f"지도 라이브러리를 불러오지 못했습니다: {exc}")
        st.dataframe(filtered.head(200), width="stretch")
        return

    center = [float(filtered[COL_LAT].mean()), float(filtered[COL_LON].mean())]
    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    if layer_mode == "마커 클러스터":
        marker_cluster = MarkerCluster(name="과거 파손 이력").add_to(fmap)
        # 매우 큰 데이터에서도 브라우저가 버티도록 상한을 둔다.
        max_markers = 5000
        subset = filtered.head(max_markers)
        if len(filtered) > max_markers:
            bullet(
                f"마커는 상위 {max_markers:,}건만 그립니다. "
                "전체 분포는 히트맵 또는 밀집구간 레이어를 사용하세요."
            )
        for row in subset.itertuples(index=False):
            lat = getattr(row, COL_LAT)
            lon = getattr(row, COL_LON)
            hazard = getattr(row, COL_HAZARD_TYPE, "")
            road = getattr(row, COL_ROAD_NAME, "")
            collected = getattr(row, COL_COLLECTED_AT, "")
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="#d95f02",
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>과거 파손 이력</b><br>유형: {hazard}<br>도로: {road}<br>수집: {collected}",
                    max_width=260,
                ),
            ).add_to(marker_cluster)

    elif layer_mode == "히트맵":
        HeatMap(
            filtered[[COL_LAT, COL_LON]].astype(float).values.tolist(),
            radius=12,
            blur=18,
            name="과거 파손 이력 밀도",
        ).add_to(fmap)

    else:
        clustered, info = compute_clusters(filtered, float(cluster_eps), int(cluster_min))
        summary = cluster_summary(clustered)
        if summary.empty:
            bullet(
                info.get("reason", "밀집구간이 탐지되지 않았습니다. 반경 또는 최소 건수를 조정하세요.")
            )
        else:
            st.caption(
                f"군집 알고리즘: {info.get('algorithm')} · "
                f"밀집구간 {info.get('n_clusters')}개 · 잡음 {info.get('n_noise')}건"
            )
            max_count = float(summary["count"].max())
            for row in summary.itertuples(index=False):
                folium.CircleMarker(
                    location=[row.lat, row.lon],
                    radius=8 + 22 * (row.count / max_count),
                    color="#7570b3",
                    fill=True,
                    fill_opacity=0.5,
                    popup=folium.Popup(
                        f"<b>밀집구간 #{row.cluster}</b><br>과거 이력 {row.count}건<br>"
                        f"대표 도로: {row.road_address}<br>유형: {row.hazard_types}",
                        max_width=300,
                    ),
                ).add_to(fmap)
            st.dataframe(
                summary.rename(
                    columns={
                        "cluster": "군집",
                        "count": "과거 이력 건수",
                        "lat": "중심 위도",
                        "lon": "중심 경도",
                        "road_address": "대표 도로명",
                        "hazard_types": "포함 유형",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    folium.LayerControl().add_to(fmap)
    st_folium(fmap, height=560, width="stretch", returned_objects=[])

    st.session_state["filtered_damage"] = filtered


# ---------------------------------------------------------------------------
# 3. 사진 분석
# ---------------------------------------------------------------------------
def page_photo_analysis(cfg: config.AppConfig) -> None:
    st.title("사진 분석")
    st.caption("업로드한 사진은 비식별 처리 후 AI 탐지를 수행합니다.")

    bullet(config.PHOTO_RETENTION_NOTICE)

    with st.expander("개인정보 처리 설정", expanded=False):
        st.write(
            f"- 얼굴 비식별 처리: {'사용' if cfg.face_blur_enabled else '미사용'} "
            "(.env 의 `FACE_BLUR_ENABLED`)"
        )
        st.write(
            f"- 원본 사진 저장: {'저장함' if cfg.store_original_image else '저장하지 않음(기본값)'} "
            "(.env 의 `STORE_ORIGINAL_IMAGE`)"
        )
        experimental_plate = st.checkbox(
            "번호판 후보 영역 실험적 모자이크 적용",
            value=False,
            help=(
                "한국 번호판 전용 탐지모델이 없어 러시아 번호판용 보조 탐지기를 사용합니다. "
                "정확도가 검증되지 않았으므로 '번호판 제거 완료'로 간주하지 마십시오."
            ),
        )

    if not cfg.model_available:
        bullet(MODEL_MISSING_MESSAGE)
        st.caption(
            "모델이 없으므로 탐지 결과를 만들어 표시하지 않습니다. "
            "비식별 처리 결과만 확인할 수 있습니다."
        )

    uploaded = st.file_uploader(
        "도로 사진 업로드 (JPG / PNG)", type=["jpg", "jpeg", "png"]
    )
    if uploaded is None:
        return

    import cv2

    file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        bullet("이미지를 읽지 못했습니다. 다른 파일을 시도하세요.")
        return

    # 1) 비식별 처리
    st.subheader("1단계 · 비식별 처리")
    anon = anonymize_image(
        image_bgr,
        blur_faces=cfg.face_blur_enabled,
        experimental_plate_blur=experimental_plate,
    )
    pcol1, pcol2 = st.columns(2)
    pcol1.metric("얼굴 처리", f"{anon.face_count}건", anon.face_status)
    pcol2.metric("번호판 처리", f"{anon.plate_count}건", anon.plate_status)
    for message in anon.messages:
        st.write("- " + message)

    working = anon.image if anon.image is not None else image_bgr

    # 2) AI 탐지
    st.subheader("2단계 · AI 분석 결과")
    if not cfg.model_available:
        bullet(
            "모델 파일이 없어 파손 탐지를 수행하지 않았습니다. "
            "아래 이미지는 비식별 처리만 적용된 결과입니다.",
        )
        st.image(cv2.cvtColor(working, cv2.COLOR_BGR2RGB), caption="비식별 처리 결과",
                 width="stretch")
        st.session_state["photo_result"] = {
            "available": False,
            "detections": None,
            "damage_area_ratio": None,
        }
        return

    conf = st.slider(
        "신뢰도 임계값", 0.05, 0.9, float(cfg.detection_conf_threshold), step=0.05
    )
    result = detect_damage(
        working,
        model_path=cfg.model_path,
        model_kind=cfg.model_kind,
        conf_threshold=conf,
    )

    if not result.available:
        bullet(result.reason)
        st.image(cv2.cvtColor(working, cv2.COLOR_BGR2RGB), caption="비식별 처리 결과",
                 width="stretch")
        st.session_state["photo_result"] = result.to_dict()
        return

    icol1, icol2 = st.columns(2)
    icol1.image(
        cv2.cvtColor(working, cv2.COLOR_BGR2RGB),
        caption="비식별 처리 결과",
        width="stretch",
    )
    icol2.image(
        cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB),
        caption="AI 탐지 결과",
        width="stretch",
    )

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("탐지 개수", f"{result.count}건")
    top_conf = (
        max(d["confidence"] for d in result.detections) if result.detections else 0.0
    )
    mcol2.metric("최고 신뢰도", f"{top_conf:.2f}")
    mcol3.metric("파손영역 비율", f"{(result.damage_area_ratio or 0.0) * 100:.2f}%")

    if result.detections:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "유형(코드)": d["class_name"],
                        "유형": d["label_ko"],
                        "신뢰도": round(d["confidence"], 3),
                        "x1": round(d["bbox"][0]),
                        "y1": round(d["bbox"][1]),
                        "x2": round(d["bbox"][2]),
                        "y2": round(d["bbox"][3]),
                    }
                    for d in result.detections
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        bullet("탐지된 파손이 없습니다. (분석은 수행되었습니다)")

    bullet(AI_DISCLAIMER)

    severity = photo_severity_from_detections(
        result.detections, result.damage_area_ratio
    )
    st.metric("사진상 파손 정도 점수 (0~100)", f"{severity:.1f}" if severity is not None else "-")

    # 다음 화면에서 사용할 수 있도록 저장
    payload = result.to_dict()
    payload["photo_severity"] = severity
    st.session_state["photo_result"] = payload
    st.session_state["photo_privacy"] = anon.to_dict()

    # 이미지 저장 (기본: 비식별 이미지만)
    if st.button("이 사진과 분석 결과를 점검 건으로 접수"):
        config.ensure_directories()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        anon_path = config.UPLOAD_IMAGE_DIR / f"anon_{stamp}.jpg"
        cv2.imwrite(str(anon_path), working)
        original_path = None
        if cfg.store_original_image:
            original_path = config.UPLOAD_IMAGE_DIR / f"orig_{stamp}.jpg"
            cv2.imwrite(str(original_path), image_bgr)

        inspection_id = repository.create_inspection(
            anonymized_image_path=str(anon_path),
            original_image_path=str(original_path) if original_path else None,
            ai_result=payload,
            privacy_result=anon.to_dict(),
            status=InspectionStatus.RECEIVED.value,
        )
        bullet(
            f"점검 건 #{inspection_id} 로 접수했습니다. "
            "'6. 처리결과 기록' 화면에서 담당자 확인 결과를 입력하세요."
        )


# ---------------------------------------------------------------------------
# 4. 점검 우선순위
# ---------------------------------------------------------------------------
def page_priority(cfg: config.AppConfig) -> None:
    st.title("점검 우선순위")
    st.caption("지점별 점검 우선순위 점수(0~100)를 계산합니다.")

    damage, load_report, _ = load_damage_data()
    school_zones, sz_report, _ = load_school_zone_data()
    data_origin_badge(load_report.is_sample)

    if damage.empty:
        bullet("파손 이력 데이터가 없어 우선순위를 계산할 수 없습니다.")
        return

    st.subheader("대상 지점 선택")
    mode = st.radio(
        "계산 대상",
        ["밀집구간 상위 지점", "직접 좌표 입력"],
        horizontal=True,
    )

    targets: list[dict] = []
    if mode == "밀집구간 상위 지점":
        col1, col2, col3 = st.columns(3)
        eps = col1.slider("밀집 판정 반경(m)", 50, 500, 150, step=50, key="prio_eps")
        min_samples = col2.slider("밀집 최소 건수", 2, 30, 5, key="prio_min")
        top_n = col3.slider("상위 지점 수", 1, 30, 10)
        clustered, info = compute_clusters(damage, float(eps), int(min_samples))
        summary = cluster_summary(clustered)
        if summary.empty:
            bullet(info.get("reason", "밀집구간이 없습니다. 조건을 조정하세요."))
            return
        for row in summary.head(top_n).itertuples(index=False):
            targets.append(
                {
                    "name": f"밀집구간 #{row.cluster}",
                    "lat": float(row.lat),
                    "lon": float(row.lon),
                    "road_address": row.road_address,
                    "cluster_count": int(row.count),
                }
            )
    else:
        col1, col2 = st.columns(2)
        lat = col1.number_input(
            "위도", value=float(config.ANYANG_CENTER[0]), format="%.6f"
        )
        lon = col2.number_input(
            "경도", value=float(config.ANYANG_CENTER[1]), format="%.6f"
        )
        targets.append(
            {"name": "직접 입력 지점", "lat": lat, "lon": lon, "road_address": "", "cluster_count": None}
        )

    st.subheader("입력 항목 설정")
    icol1, icol2, icol3 = st.columns(3)
    radius = icol1.slider("이력 밀도 계산 반경(m)", 100, 1000, 300, step=50)
    reference = icol2.slider("밀도 100점 기준 건수", 5, 100, 20, step=5)
    sz_limit = icol3.slider(
        "보호구역 인접도 0점 거리(m)", 100, 1500,
        int(config.SCHOOL_ZONE_MAX_DISTANCE_M), step=50,
    )

    # --- 강수 정보 ---
    st.subheader("강수 정보")
    rain_mm: float | None = None
    if cfg.kma_enabled:
        if st.button("기상청 단기예보 조회"):
            weather = fetch_precipitation(cfg.kma_service_key, cfg.kma_nx, cfg.kma_ny)
            st.session_state["weather"] = weather
        weather = st.session_state.get("weather")
        if weather and weather.get("available"):
            bullet(
                f"기상청 단기예보 (발표 {weather['base']}, 예보 {weather['fcst']}): "
                f"강수량 {weather['rain_mm']}mm, 강수확률 {weather.get('pop')}%"
            )
            rain_mm = float(weather["rain_mm"])
        elif weather:
            bullet(weather.get("reason", "기상청 조회에 실패했습니다."))
            st.caption("강수량을 직접 입력하세요.")
    else:
        bullet(
            "KMA_SERVICE_KEY 가 설정되지 않아 기상청 단기예보를 조회할 수 없습니다. "
            "강수량을 직접 입력하세요.",
        )

    use_manual = st.checkbox(
        "강수량 직접 입력", value=(rain_mm is None), key="manual_rain_check"
    )
    if use_manual:
        manual = st.number_input("강수량 (mm)", min_value=0.0, max_value=200.0,
                                 value=0.0, step=0.5)
        rain_mm = float(manual)
    elif rain_mm is None:
        bullet("강수 정보를 사용하지 않습니다.")

    # --- 사진 점수 ---
    photo_result = st.session_state.get("photo_result")
    photo_severity = (photo_result or {}).get("photo_severity")
    if photo_severity is None:
        bullet("'3. 사진 분석' 화면에서 사진을 분석하면 사진상 파손 정도가 반영됩니다.")

    # --- 계산 ---
    importance_table = road_importance_table(damage)
    school_available = not school_zones.empty

    rows = []
    details: dict[str, dict] = {}
    for target in targets:
        count = history_density_count(damage, target["lat"], target["lon"], radius)
        density = history_density_score(count, reference)

        distance = (
            nearest_school_zone_distance_m(school_zones, target["lat"], target["lon"])
            if school_available else None
        )
        sz_score = school_zone_score(distance, sz_limit)

        # 도로구간 중요도: 반경 내에서 가장 많이 등장한 노드링크/도로명을 대표값으로 사용
        key_value = _representative_road_key(damage, target["lat"], target["lon"], radius)
        importance = road_importance_score(importance_table, key_value)

        risk = compute_risk_score(
            history_density=density,
            photo_severity=photo_severity,
            school_zone=sz_score,
            road_importance=importance,
            rainfall=rainfall_score(rain_mm),
        )
        details[target["name"]] = {"risk": risk, "target": target,
                                   "count": count, "distance": distance,
                                   "road_key": key_value}
        rows.append(
            {
                "지점": target["name"],
                "총점": risk.total_score,
                "위도": round(target["lat"], 6),
                "경도": round(target["lon"], 6),
                "도로명": target.get("road_address", ""),
                f"반경 {radius}m 내 과거 이력": count,
                "보호구역 최근접(m)": round(distance) if distance is not None else None,
            }
        )

    table = pd.DataFrame(rows).sort_values("총점", ascending=False).reset_index(drop=True)
    st.subheader("우선순위 결과")
    st.dataframe(table, width="stretch", hide_index=True)

    st.session_state["priority_table"] = table
    st.session_state["priority_details"] = {
        name: {
            "lat": d["target"]["lat"],
            "lon": d["target"]["lon"],
            "score": d["risk"].total_score,
            "road_address": d["target"].get("road_address", ""),
        }
        for name, d in details.items()
    }



def _representative_road_key(
    damage: pd.DataFrame, lat: float, lon: float, radius_m: float
) -> str | None:
    """반경 내에서 가장 자주 등장하는 노드링크(없으면 도로명)를 반환."""
    if damage.empty:
        return None
    key = None
    if COL_NODE_LINK in damage.columns and damage[COL_NODE_LINK].notna().any():
        key = COL_NODE_LINK
    elif COL_ROAD_NAME in damage.columns and damage[COL_ROAD_NAME].notna().any():
        key = COL_ROAD_NAME
    if key is None:
        return None

    distances = distances_from_point_m(damage, lat, lon)
    nearby = damage.loc[distances <= radius_m, key].dropna().astype(str)
    if nearby.empty:
        return None
    return str(nearby.mode().iloc[0])


# ---------------------------------------------------------------------------
# 5. 점검목록 및 순찰경로
# ---------------------------------------------------------------------------
def page_route(cfg: config.AppConfig) -> None:
    st.title("점검목록 및 순찰경로")

    details = st.session_state.get("priority_details")
    if not details:
        bullet(
            "'4. 점검 우선순위' 화면에서 먼저 우선순위를 계산하세요. "
            "계산된 지점 목록을 여기서 선택할 수 있습니다.",
        )
        return

    if cfg.routing_api_enabled:
        bullet("외부 도로망 API 가 활성화되어 실제 도로거리 계산을 사용합니다.")
        distance_mode = "road_network_api"
    else:
        bullet(
            "도로망 데이터/API 가 연동되지 않아 직선(Haversine) 거리 기반 근사 경로를 계산합니다. "
            "실제 주행거리와 다를 수 있습니다.",
        )
        distance_mode = DISTANCE_MODE_HAVERSINE

    names = list(details.keys())
    selected = st.multiselect("점검 대상 선택", names, default=names[: min(5, len(names))])

    col1, col2 = st.columns(2)
    return_to_start = col1.checkbox("출발지로 복귀(순환 경로)", value=False)
    time_limit = col2.slider("최적화 시간 제한(초)", 1, 20, 3)

    points = [(details[n]["lat"], details[n]["lon"]) for n in selected]

    if len(points) == 0:
        bullet("점검 대상을 선택하지 않았습니다. 경로를 계산하지 않습니다.")
        return

    result = optimize_route(
        points,
        return_to_start=return_to_start,
        time_limit_s=time_limit,
        distance_mode=distance_mode,
    )

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("방문 지점 수", f"{len(selected)}곳")
    mcol2.metric("총 이동거리", format_distance(result.total_distance_m))
    mcol3.metric("계산 방식", result.solver or "-")

    for message in result.messages:
        bullet(message)

    order_rows = []
    for step, idx in enumerate(result.order, start=1):
        name = selected[idx]
        leg = result.leg_distances_m[step - 1] if step - 1 < len(result.leg_distances_m) else None
        order_rows.append(
            {
                "순번": step,
                "지점": name,
                "점검 우선순위 점수": details[name]["score"],
                "도로명": details[name].get("road_address", ""),
                "다음 지점까지": format_distance(leg) if leg is not None else "-",
            }
        )
    st.subheader("추천 순찰 순서")
    st.dataframe(pd.DataFrame(order_rows), width="stretch", hide_index=True)

    if len(points) < 1:
        return

    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError as exc:
        bullet(f"지도 라이브러리를 불러오지 못했습니다: {exc}")
        return

    ordered_points = [points[i] for i in result.order]
    center = [
        float(np.mean([p[0] for p in ordered_points])),
        float(np.mean([p[1] for p in ordered_points])),
    ]
    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
    for step, idx in enumerate(result.order, start=1):
        name = selected[idx]
        folium.Marker(
            location=list(points[idx]),
            tooltip=f"{step}. {name} (점수 {details[name]['score']})",
            icon=folium.DivIcon(
                html=(
                    "<div style='background:#1b6ca8;color:#fff;border-radius:50%;"
                    "width:26px;height:26px;line-height:26px;text-align:center;"
                    f"font-weight:700'>{step}</div>"
                )
            ),
        ).add_to(fmap)

    line = ordered_points + ([ordered_points[0]] if return_to_start and len(ordered_points) > 1 else [])
    if len(line) > 1:
        folium.PolyLine(line, color="#1b6ca8", weight=3, opacity=0.8,
                        dash_array="8,6").add_to(fmap)
    st_folium(fmap, height=520, width="stretch", returned_objects=[])

    st.subheader("점검목록 등록")
    if st.button("선택한 지점을 점검 건으로 접수"):
        created = []
        for idx in result.order:
            name = selected[idx]
            info = details[name]
            inspection_id = repository.create_inspection(
                lat=info["lat"],
                lon=info["lon"],
                road_address=info.get("road_address"),
                risk_result={"total_score": info["score"]},
                status=InspectionStatus.SCHEDULED.value,
                is_sample_data=bool(st.session_state.get("is_sample_data", False)),
            )
            created.append(inspection_id)
        bullet(f"{len(created)}건을 '점검예정' 상태로 등록했습니다. (#{', #'.join(map(str, created))})")


# ---------------------------------------------------------------------------
# 6. 처리결과 기록
# ---------------------------------------------------------------------------
def page_records(cfg: config.AppConfig) -> None:
    st.title("처리결과 기록")
    st.caption("AI 분석 결과와 담당자 확인 결과를 서로 다른 필드에 저장합니다.")

    repository.init_db()
    counts = repository.status_counts()
    cols = st.columns(len(counts))
    for col, (status, n) in zip(cols, counts.items()):
        col.metric(status, f"{n}건")

    st.subheader("점검 건 목록")
    status_filter = st.selectbox(
        "상태 필터", ["전체"] + InspectionStatus.values()
    )
    rows = repository.list_inspections(
        status=None if status_filter == "전체" else status_filter
    )
    if not rows:
        bullet(
            "등록된 점검 건이 없습니다. "
            "'3. 사진 분석' 또는 '5. 점검목록 및 순찰경로' 화면에서 접수할 수 있습니다."
        )
        return

    table = pd.DataFrame(rows)
    display_cols = {
        "id": "번호",
        "created_at": "접수시각",
        "status": "처리상태",
        "lat": "위도",
        "lon": "경도",
        "road_address": "도로명",
        "ai_available": "AI 분석 수행",
        "ai_top_class": "AI 최고 유형",
        "ai_top_confidence": "AI 신뢰도",
        "ai_risk_score": "우선순위 점수",
        "reviewer_judgement": "담당자 확인 결과",
        "is_sample_data": "샘플데이터",
    }
    view = table[[c for c in display_cols if c in table.columns]].rename(columns=display_cols)
    if "AI 분석 수행" in view.columns:
        view["AI 분석 수행"] = view["AI 분석 수행"].map({1: "예", 0: "아니오"})
    if "샘플데이터" in view.columns:
        view["샘플데이터"] = view["샘플데이터"].map({1: "샘플", 0: "실데이터"})
    st.dataframe(view, width="stretch", hide_index=True)

    st.subheader("상세 · 상태 변경 · 담당자 확인 결과 입력")
    selected_id = st.selectbox("점검 건 번호", table["id"].tolist())
    record = repository.get_inspection(int(selected_id))
    if record is None:
        bullet("선택한 점검 건을 찾을 수 없습니다.")
        return

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.markdown("#### AI 분석 결과 (자동)")
        if record["ai_available"]:
            st.write(f"- 모델: `{record['ai_model_path']}`")
            st.write(f"- 탐지 개수: {record['ai_detection_count']}")
            st.write(f"- 최고 유형: {record['ai_top_class']}")
            st.write(
                f"- 최고 신뢰도: "
                f"{record['ai_top_confidence']:.3f}" if record["ai_top_confidence"] is not None else "- 최고 신뢰도: -"
            )
            ratio = record["ai_damage_area_ratio"]
            st.write(f"- 파손영역 비율: {ratio * 100:.2f}%" if ratio is not None else "- 파손영역 비율: -")
        else:
            st.write("- AI 분석을 수행하지 않았습니다(모델 없음 또는 사진 미첨부).")
        if record["ai_risk_score"] is not None:
            st.write(f"- 점검 우선순위 점수: {record['ai_risk_score']}")
        st.caption(AI_DISCLAIMER)

    with dcol2:
        st.markdown("#### 담당자 확인 결과 (수동)")
        st.write(f"- 현재 기록: {record['reviewer_judgement'] or '미입력'}")
        st.write(f"- 확인 시각: {record['reviewed_at'] or '-'}")
        with st.form(f"reviewer_form_{selected_id}"):
            judgement = st.selectbox(
                "담당자 판단",
                ["보수 필요", "보수 불필요", "재확인 필요", "타 부서 이관"],
            )
            reviewer_name = st.text_input(
                "담당 부서/직위 (개인 식별 정보는 입력하지 마세요)", value=""
            )
            note = st.text_area("의견", value="")
            submitted = st.form_submit_button("담당자 확인 결과 저장")
        if submitted:
            repository.set_reviewer_judgement(
                int(selected_id), judgement, note=note, reviewer_name=reviewer_name
            )
            bullet("담당자 확인 결과를 저장했습니다. AI 분석 결과는 변경되지 않습니다.")
            st.rerun()

    st.markdown("#### 처리상태 변경")
    scol1, scol2, scol3 = st.columns([2, 3, 1])
    new_status = scol1.selectbox(
        "변경할 상태",
        InspectionStatus.values(),
        index=InspectionStatus.values().index(record["status"]),
    )
    status_note = scol2.text_input("변경 사유", value="")
    if scol3.button("상태 변경"):
        repository.update_status(int(selected_id), new_status, note=status_note)
        bullet(f"#{selected_id} 상태를 '{new_status}' 로 변경했습니다.")
        st.rerun()

    history = repository.get_status_history(int(selected_id))
    if history:
        st.markdown("#### 상태 변경 이력")
        st.dataframe(
            pd.DataFrame(history)[["changed_at", "from_status", "to_status", "note"]]
            .rename(
                columns={
                    "changed_at": "변경 시각",
                    "from_status": "이전 상태",
                    "to_status": "변경 상태",
                    "note": "사유",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    if record["anonymized_image_path"] and Path(record["anonymized_image_path"]).exists():
        st.markdown("#### 첨부 사진 (비식별 처리본)")
        st.image(record["anonymized_image_path"], width=480)

    st.divider()
    st.caption(config.PHOTO_RETENTION_NOTICE)
    if st.button("이 점검 건 삭제 (보유기간 경과 처리)"):
        repository.delete_inspection(int(selected_id))
        bullet(f"#{selected_id} 을(를) 삭제했습니다.")
        st.rerun()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = get_config()

    with st.sidebar:
        st.title("RoadLens Anyang")
        st.caption("안양시 도로 점검 우선순위 지원 시제품")
        page = st.radio("화면 이동", PAGES, label_visibility="collapsed")
        st.divider()
        st.markdown("실행 상태")
        st.write("모델: " + ("사용 가능" if cfg.model_available else "없음"))
        st.write("기상청 API: " + ("사용 가능" if cfg.kma_enabled else "키 없음"))
        st.write("원본 사진 저장: " + ("함" if cfg.store_original_image else "안 함"))
        st.divider()
        st.caption(AI_DISCLAIMER)

    if page == PAGES[0]:
        page_intro(cfg)
    elif page == PAGES[1]:
        page_history_map(cfg)
    elif page == PAGES[2]:
        page_photo_analysis(cfg)
    elif page == PAGES[3]:
        page_priority(cfg)
    elif page == PAGES[4]:
        page_route(cfg)
    else:
        page_records(cfg)


if __name__ == "__main__":
    main()
