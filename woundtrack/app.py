"""Streamlit dashboard for patient intake, visits, progress, and reports."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from woundtrack.config import (
    DATABASE_PATH,
    MEDICAL_DISCLAIMER,
    REPORTS_DIR,
    UPLOADS_DIR,
)
from woundtrack.src import db
from woundtrack.src.compare import align_later_to_earlier, compare_areas
from woundtrack.src.measure import (
    AreaMeasurement,
    CalibrationResult,
    detect_reference_scale,
    manual_calibration,
    measure_mask,
)
from woundtrack.src.report import generate_pdf_report
from woundtrack.src.segment import load_model, predict_mask
from woundtrack.src.tissue import TissueComposition, analyze_tissue

st.set_page_config(page_title="WoundTrack", page_icon="🩹", layout="wide")


@st.cache_resource(show_spinner="Loading WoundTrack model on CPU…")
def _cached_model():
    """Load the model once per Streamlit server process."""
    return load_model(device="cpu")


def _selected_patient() -> dict[str, Any] | None:
    """Resolve the session's selected patient from the database."""
    patient_id = st.session_state.get("patient_id")
    if patient_id is None:
        return None
    return db.get_patient(int(patient_id))


def _image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode user-uploaded bytes to a three-channel RGB uint8 array."""
    with Image.open(BytesIO(image_bytes)) as pil_image:
        return np.asarray(pil_image.convert("RGB"), dtype=np.uint8)


def _make_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a translucent red wound mask over an RGB image."""
    overlay = image_rgb.copy()
    inside = np.asarray(mask) > 0
    if inside.any():
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        overlay[inside] = (0.55 * overlay[inside] + 0.45 * red[inside]).astype(np.uint8)
    return overlay


def _patient_page() -> None:
    """Add/select patient and show the local patient list."""
    st.header("Patients")
    with st.form("add_patient_form", clear_on_submit=True):
        name = st.text_input("Patient name", max_chars=120)
        age = st.number_input("Age (optional; enter 0 if unknown)", min_value=0, max_value=130, value=0)
        notes = st.text_area("Notes (optional)", height=90)
        submitted = st.form_submit_button("Add patient", type="primary")
    if submitted:
        try:
            patient_id = db.add_patient(name, int(age) if age > 0 else None, notes)
            st.session_state["patient_id"] = patient_id
            st.success("Patient added and selected.")
            st.rerun()
        except ValueError as error:
            st.error(str(error))

    patients = db.list_patients()
    if not patients:
        st.info("Add a patient to begin recording wound visits.")
        return
    st.subheader("Select a patient")
    labels = {
        int(patient["id"]): f"{patient['name']}  ·  ID {patient['id']}"
        for patient in patients
    }
    patient_ids = list(labels)
    selected_id = st.session_state.get("patient_id")
    selected_index = patient_ids.index(selected_id) if selected_id in patient_ids else 0
    choice = st.selectbox(
        "Patient record",
        patient_ids,
        index=selected_index,
        format_func=lambda patient_id: labels[int(patient_id)],
    )
    if st.button("Select patient", type="primary"):
        st.session_state["patient_id"] = int(choice)
        st.success(f"Selected {labels[int(choice)]}.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "ID": patient["id"],
                    "Name": patient["name"],
                    "Age": patient["age"] if patient["age"] is not None else "—",
                    "Notes": patient["notes"],
                }
                for patient in patients
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    current = _selected_patient()
    if current:
        st.caption(f"Selected patient: **{current['name']}** (ID {current['id']})")


def _new_visit_page(patient: dict[str, Any] | None) -> None:
    """Upload/capture, segment, measure, analyze, and save one visit."""
    st.header("New Visit")
    if patient is None:
        st.info("Add or select a patient on the Patients page first.")
        return
    st.caption(f"Patient: {patient['name']} · ID {patient['id']}")
    input_method = st.radio("Image source", ["Upload photo", "Camera"], horizontal=True)
    uploaded = (
        st.file_uploader("Choose a wound photo", type=["jpg", "jpeg", "png"])
        if input_method == "Upload photo"
        else st.camera_input("Take a wound photo")
    )
    if uploaded is None:
        st.info("Upload or take a well-lit photo that shows the full wound and, if possible, a 1-rupee coin or 2 cm square marker.")
        return

    image_bytes = uploaded.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    try:
        image_rgb = _image_from_bytes(image_bytes)
    except Exception as error:
        st.error(f"Could not read this image: {error}")
        return
    visit_date = st.date_input("Visit date", value=date.today())
    auto_calibration = detect_reference_scale(image_rgb)
    if auto_calibration is not None:
        st.success(
            f"Reference detected: {auto_calibration.source}, "
            f"about {auto_calibration.pixels_per_cm:.1f} pixels/cm."
        )
        scale_options = [
            "Use detected marker",
            "Manual pixels/cm estimate",
            "Pixels only (uncalibrated)",
        ]
        default_scale_index = 0
    else:
        st.caption("No reference marker was detected. You can enter a manual pixels/cm estimate or save a relative pixel area.")
        scale_options = ["Manual pixels/cm estimate", "Pixels only (uncalibrated)"]
        default_scale_index = 0
    scale_mode = st.radio("Area scale", scale_options, index=default_scale_index, horizontal=True)
    manual_ppcm: float | None = None
    if scale_mode == "Manual pixels/cm estimate":
        manual_ppcm = st.number_input(
            "Estimated pixels per centimeter",
            min_value=0.1,
            max_value=10000.0,
            value=40.0,
            step=1.0,
            help="Manual scale estimates are always labeled uncalibrated / relative.",
        )
    analysis_key = (
        f"{patient['id']}:{image_hash}:{visit_date.isoformat()}:{scale_mode}:{manual_ppcm}"
    )

    if st.button("Run segmentation and measurement", type="primary"):
        try:
            with st.spinner("Segmenting the image on CPU…"):
                model = _cached_model()
                mask = predict_mask(image_rgb, model=model, device="cpu")
                calibration: CalibrationResult | None
                if scale_mode == "Use detected marker":
                    calibration = auto_calibration
                elif scale_mode == "Manual pixels/cm estimate":
                    calibration = manual_calibration(float(manual_ppcm or 0.0))
                else:
                    calibration = None
                measurement = measure_mask(mask, calibration)
                tissue = analyze_tissue(image_rgb, mask)
            st.session_state["visit_analysis"] = {
                "analysis_key": analysis_key,
                "image_hash": image_hash,
                "image_bytes": image_bytes,
                "image_rgb": image_rgb,
                "mask": mask,
                "measurement": measurement,
                "tissue": tissue,
            }
        except FileNotFoundError as error:
            st.error(f"{error} Train the model before using New Visit.")
        except Exception as error:
            st.error(f"Analysis failed: {error}")

    analysis = st.session_state.get("visit_analysis")
    if not analysis or analysis.get("analysis_key") != analysis_key:
        return

    mask = analysis["mask"]
    measurement: AreaMeasurement = analysis["measurement"]
    tissue: TissueComposition = analysis["tissue"]
    columns = st.columns(3)
    columns[0].image(image_rgb, caption="Original photo", use_container_width=True)
    columns[1].image(mask * 255, caption="Wound mask", clamp=True, use_container_width=True)
    columns[2].image(_make_overlay(image_rgb, mask), caption="Mask overlay", use_container_width=True)

    st.subheader("Measurement")
    measure_columns = st.columns(4)
    measure_columns[0].metric("Wound pixels", f"{measurement.area_pixels:,} px²")
    measure_columns[1].metric(
        "Area",
        f"{measurement.area_cm2:.2f} cm²" if measurement.area_cm2 is not None else "—",
    )
    measure_columns[2].metric(
        "Perimeter",
        f"{measurement.perimeter_cm:.2f} cm"
        if measurement.perimeter_cm is not None
        else f"{measurement.perimeter_pixels:.1f} px",
    )
    measure_columns[3].metric("Scale", measurement.display_label)
    if not measurement.calibrated:
        st.warning("Uncalibrated / relative result. Pixel area is not a real-world cm² measurement.")
    st.caption(f"Scale source: {measurement.calibration_method}")

    st.subheader("Heuristic tissue color analysis")
    tissue_columns = st.columns(4)
    tissue_columns[0].metric("Granulation", f"{tissue.granulation_pct:.1f}%")
    tissue_columns[1].metric("Slough", f"{tissue.slough_pct:.1f}%")
    tissue_columns[2].metric("Necrotic", f"{tissue.necrotic_pct:.1f}%")
    tissue_columns[3].metric("Other", f"{tissue.other_pct:.1f}%")
    st.caption("HSV thresholds are heuristic and have not been clinically validated.")

    if st.button("Save visit", type="primary"):
        try:
            existing_visits = db.list_visits(int(patient["id"]))
            comparable = [visit for visit in existing_visits if visit.get("area_cm2") is not None]
            trend = None
            if measurement.area_cm2 is not None and comparable:
                baseline_visit = comparable[0]
                previous_visit = comparable[-1]
                days = (
                    date.fromisoformat(visit_date.isoformat())
                    - date.fromisoformat(str(previous_visit["date"]))
                ).days
                trend = compare_areas(
                    current_area_cm2=measurement.area_cm2,
                    baseline_area_cm2=float(baseline_visit["area_cm2"]),
                    previous_area_cm2=float(previous_visit["area_cm2"]),
                    days_since_previous=float(days) if days > 0 else None,
                )
            elif measurement.area_cm2 is not None:
                trend = compare_areas(measurement.area_cm2, measurement.area_cm2)

            patient_upload_dir = UPLOADS_DIR / f"patient_{patient['id']}"
            patient_upload_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%H%M%S%f")
            image_path = patient_upload_dir / f"{visit_date.isoformat()}_{timestamp}.png"
            mask_path = patient_upload_dir / f"{visit_date.isoformat()}_{timestamp}_mask.png"
            Image.fromarray(analysis["image_rgb"]).save(image_path)
            Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)

            perimeter_value = (
                measurement.perimeter_cm
                if measurement.perimeter_cm is not None
                else measurement.perimeter_pixels
            )
            status_label = (
                trend.status_label
                if trend is not None
                else "Uncalibrated / relative"
            )
            db.add_visit(
                int(patient["id"]),
                visit_date=visit_date.isoformat(),
                image_path=str(image_path),
                mask_path=str(mask_path),
                area_cm2=measurement.area_cm2,
                area_pixels=measurement.area_pixels,
                perimeter=perimeter_value,
                perimeter_pixels=measurement.perimeter_pixels,
                perimeter_unit="cm" if measurement.perimeter_cm is not None else "px",
                calibrated=measurement.calibrated,
                calibration_method=measurement.calibration_method,
                change_from_baseline_pct=(trend.change_from_baseline_pct if trend else None),
                change_from_previous_pct=(trend.change_from_previous_pct if trend else None),
                reduction_from_baseline_pct=(trend.reduction_from_baseline_pct if trend else None),
                healing_rate_cm2_per_day=(trend.healing_rate_cm2_per_day if trend else None),
                status_label=status_label,
                granulation_pct=tissue.granulation_pct,
                slough_pct=tissue.slough_pct,
                necrotic_pct=tissue.necrotic_pct,
                other_tissue_pct=tissue.other_pct,
            )
            st.success("Visit saved.")
            st.session_state.pop("visit_analysis", None)
            st.rerun()
        except Exception as error:
            st.error(f"Could not save visit: {error}")


def _visit_overlay(visit: dict[str, Any]) -> np.ndarray | None:
    """Load one stored visit photo and blend its binary mask over the image."""
    image_path = Path(str(visit["image_path"]))
    mask_path = Path(str(visit["mask_path"]))
    if not image_path.is_file() or not mask_path.is_file():
        return None
    with Image.open(image_path) as pil_image:
        image_rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
    with Image.open(mask_path) as pil_mask:
        mask = np.asarray(pil_mask.convert("L")) > 0
    if image_rgb.shape[:2] != mask.shape:
        return None
    return _make_overlay(image_rgb, mask.astype(np.uint8))


def _progress_page(patient: dict[str, Any] | None) -> None:
    """Display longitudinal area, healing, tissue, photo, and status views."""
    st.header("Progress")
    if patient is None:
        st.info("Add or select a patient on the Patients page first.")
        return
    visits = db.list_visits(int(patient["id"]))
    if not visits:
        st.info("No visits saved for this patient yet.")
        return
    st.caption(f"Progress for {patient['name']} · {len(visits)} saved visit(s)")
    latest = visits[-1]
    status = str(latest.get("status_label", "Unclassified"))
    status_column, rate_column, area_column = st.columns(3)
    status_column.metric("Current status", status)
    rate = latest.get("healing_rate_cm2_per_day")
    rate_column.metric("Latest healing rate", f"{float(rate):.3f} cm²/day" if rate is not None else "—")
    area = latest.get("area_cm2")
    area_column.metric(
        "Latest wound area",
        f"{float(area):.2f} cm²" if area is not None else f"{latest['area_pixels']:,} px²",
        help="Calibrated" if latest.get("calibrated") else "Uncalibrated / relative estimate",
    )

    visits_with_cm2 = [visit for visit in visits if visit.get("area_cm2") is not None]
    if visits_with_cm2:
        area_frame = pd.DataFrame(
            [{"date": visit["date"], "area_cm2": float(visit["area_cm2"])} for visit in visits_with_cm2]
        )
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=area_frame["date"],
                y=area_frame["area_cm2"],
                mode="lines+markers",
                name="Area (cm²)",
                line={"color": "#2463eb", "width": 3},
            )
        )
        figure.update_layout(title="Wound area vs date", xaxis_title="Visit date", yaxis_title="Area (cm²)", height=360)
        st.plotly_chart(figure, use_container_width=True)
        if len(visits_with_cm2) != len(visits):
            st.caption("Pixel-only visits are omitted from the cm² trend chart.")
    else:
        area_frame = pd.DataFrame(
            [{"date": visit["date"], "area_pixels": int(visit["area_pixels"])} for visit in visits]
        )
        figure = go.Figure(
            go.Scatter(x=area_frame["date"], y=area_frame["area_pixels"], mode="lines+markers", name="Pixel area")
        )
        figure.update_layout(title="Relative wound pixel area vs date (uncalibrated)", xaxis_title="Visit date", yaxis_title="Area (pixels²)", height=360)
        st.plotly_chart(figure, use_container_width=True)

    if any(not visit.get("calibrated", False) for visit in visits):
        st.caption(
            "At least one visit is uncalibrated / relative; manual-scale estimates and "
            "pixel-only areas are not calibrated real-world measurements."
        )

    tissue_figure = go.Figure()
    for key, label, color in (
        ("granulation_pct", "Granulation", "#de5b75"),
        ("slough_pct", "Slough", "#e8c547"),
        ("necrotic_pct", "Necrotic", "#414141"),
        ("other_tissue_pct", "Other / unclassified", "#b9c4d0"),
    ):
        tissue_figure.add_bar(
            x=[visit["date"] for visit in visits],
            y=[float(visit.get(key, 0.0) or 0.0) for visit in visits],
            name=label,
            marker_color=color,
        )
    tissue_figure.update_layout(
        barmode="stack",
        title="Heuristic tissue composition over time",
        xaxis_title="Visit date",
        yaxis_title="Share inside wound mask (%)",
        yaxis={"range": [0, 100]},
        height=360,
    )
    st.plotly_chart(tissue_figure, use_container_width=True)
    st.caption("Tissue color thresholds are heuristic, not a validated diagnostic model.")

    if len(visits) > 1:
        st.subheader("Baseline vs latest photo")
        baseline_visit = visits[0]
        latest_visit = visits[-1]
        baseline_image = None
        latest_image = None
        try:
            with Image.open(baseline_visit["image_path"]) as img:
                baseline_image = np.asarray(img.convert("RGB"), dtype=np.uint8)
            with Image.open(latest_visit["image_path"]) as img:
                latest_image = np.asarray(img.convert("RGB"), dtype=np.uint8)
        except (OSError, KeyError):
            pass
        aligned_later = (
            align_later_to_earlier(baseline_image, latest_image)
            if baseline_image is not None and latest_image is not None
            else None
        )
        columns = st.columns(2)
        if baseline_image is not None:
            columns[0].image(baseline_image, caption=f"Baseline — {baseline_visit['date']}", use_container_width=True)
        if aligned_later is not None:
            columns[1].image(aligned_later, caption=f"Latest, aligned for display — {latest_visit['date']}", use_container_width=True)
        elif latest_image is not None:
            columns[1].image(latest_image, caption=f"Latest — {latest_visit['date']}", use_container_width=True)
        st.caption("Optional image alignment, when available, is for visual comparison only; it does not alter measured areas.")

    st.subheader("Visit timeline")
    for start in range(0, len(visits), 3):
        columns = st.columns(3)
        for column, visit in zip(columns, visits[start : start + 3]):
            overlay = _visit_overlay(visit)
            if overlay is not None:
                column.image(overlay, caption=f"{visit['date']} · {visit.get('status_label', '—')}", use_container_width=True)
            else:
                column.warning(f"Stored photo unavailable for {visit['date']}.")
            if visit.get("area_cm2") is not None:
                column.caption(f"{float(visit['area_cm2']):.2f} cm² · {'calibrated' if visit['calibrated'] else 'uncalibrated / relative'}")
            else:
                column.caption(f"{int(visit['area_pixels']):,} px² · uncalibrated / relative")


def _report_page(patient: dict[str, Any] | None) -> None:
    """Generate and offer a downloadable PDF progress report."""
    st.header("Report")
    if patient is None:
        st.info("Add or select a patient on the Patients page first.")
        return
    visits = db.list_visits(int(patient["id"]))
    if not visits:
        st.info("Save at least one visit before generating a report.")
        return
    st.caption(f"PDF report for {patient['name']} · {len(visits)} visit(s)")
    if st.button("Generate PDF report", type="primary"):
        try:
            output_path = REPORTS_DIR / f"patient_{patient['id']}_progress.pdf"
            with st.spinner("Building PDF report…"):
                generate_pdf_report(patient, visits, output_path)
            st.session_state["latest_report_path"] = str(output_path)
            st.session_state["latest_report_patient_id"] = int(patient["id"])
            st.success("PDF report generated.")
        except Exception as error:
            st.error(f"Report generation failed: {error}")
    report_path = st.session_state.get("latest_report_path")
    report_patient_id = st.session_state.get("latest_report_patient_id")
    if report_patient_id == int(patient["id"]) and report_path and Path(report_path).is_file():
        report_bytes = Path(report_path).read_bytes()
        st.download_button(
            "Download PDF",
            data=report_bytes,
            file_name=f"woundtrack_patient_{patient['id']}_report.pdf",
            mime="application/pdf",
            type="primary",
        )


def main() -> None:
    """Initialize local storage and render the four WoundTrack pages."""
    db.initialize_db(DATABASE_PATH)
    st.title("🩹 WoundTrack")
    st.caption(MEDICAL_DISCLAIMER)
    st.sidebar.title("WoundTrack")
    selected_page = st.sidebar.radio("Page", ["Patients", "New Visit", "Progress", "Report"])
    patient = _selected_patient()
    if patient is not None:
        st.sidebar.caption(f"Selected: {patient['name']}")
    if selected_page == "Patients":
        _patient_page()
    elif selected_page == "New Visit":
        _new_visit_page(patient)
    elif selected_page == "Progress":
        _progress_page(patient)
    else:
        _report_page(patient)


if __name__ == "__main__":
    main()
