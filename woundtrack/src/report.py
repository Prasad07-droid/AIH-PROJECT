"""Generate a patient summary PDF with progress charts and photo comparison."""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as ReportImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from woundtrack.config import MEDICAL_DISCLAIMER
from woundtrack.src.compare import align_later_to_earlier


def _make_area_chart(visits: list[dict[str, Any]]) -> BytesIO | None:
    """Build a PNG area-vs-date chart for the PDF."""
    measured = [visit for visit in visits if visit.get("area_cm2") is not None]
    if not measured:
        return None
    dates = [str(visit["date"]) for visit in measured]
    areas = [float(visit["area_cm2"]) for visit in measured]
    figure, axis = plt.subplots(figsize=(7.0, 3.1))
    axis.plot(dates, areas, marker="o", color="#2463eb", linewidth=2)
    axis.set_ylabel("Area (cm²)")
    axis.set_title("Wound area over time")
    axis.grid(True, alpha=0.25)
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    output.seek(0)
    return output


def _make_tissue_chart(visits: list[dict[str, Any]]) -> BytesIO | None:
    """Build a stacked tissue-percentage chart for the PDF."""
    if not visits:
        return None
    dates = [str(visit["date"]) for visit in visits]
    figure, axis = plt.subplots(figsize=(7.0, 3.1))
    bottom = np.zeros(len(visits), dtype=np.float32)
    for key, label, color in (
        ("granulation_pct", "Granulation", "#de5b75"),
        ("slough_pct", "Slough", "#e8c547"),
        ("necrotic_pct", "Necrotic", "#414141"),
        ("other_tissue_pct", "Other / unclassified", "#b9c4d0"),
    ):
        values = np.asarray([float(visit.get(key, 0.0) or 0.0) for visit in visits])
        axis.bar(dates, values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set_ylim(0, 100)
    axis.set_ylabel("Share of pixels inside mask (%)")
    axis.set_title("Heuristic tissue color estimates")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    output.seek(0)
    return output


def _make_photo_comparison(visits: list[dict[str, Any]]) -> BytesIO | None:
    """Build an earliest/latest comparison; alignment is only for display."""
    if not visits:
        return None
    first = visits[0]
    last = visits[-1]
    first_bgr = cv2.imread(str(first["image_path"]), cv2.IMREAD_COLOR)
    last_bgr = cv2.imread(str(last["image_path"]), cv2.IMREAD_COLOR)
    if first_bgr is None or last_bgr is None:
        return None
    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    last_rgb = cv2.cvtColor(last_bgr, cv2.COLOR_BGR2RGB)
    aligned_last = align_later_to_earlier(first_rgb, last_rgb)
    display_last = aligned_last if aligned_last is not None else last_rgb

    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(first_rgb)
    axes[0].set_title(f"Baseline — {first['date']}")
    axes[1].imshow(display_last)
    axes[1].set_title(f"Latest — {last['date']}")
    for axis in axes:
        axis.axis("off")
    figure.suptitle("Photo comparison (alignment, if successful, is visual only)")
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    output.seek(0)
    return output


def _footer(canvas: Any, document: Any) -> None:
    """Draw the required disclaimer and page number on every PDF page."""
    canvas.saveState()
    width, _ = letter
    canvas.setStrokeColor(colors.HexColor("#d9dee8"))
    canvas.line(document.leftMargin, 0.58 * inch, width - document.rightMargin, 0.58 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#3f4652"))
    canvas.drawString(document.leftMargin, 0.42 * inch, MEDICAL_DISCLAIMER)
    canvas.drawRightString(width - document.rightMargin, 0.42 * inch, f"Page {document.page}")
    canvas.restoreState()


def generate_pdf_report(
    patient: dict[str, Any],
    visits: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Create and return a PDF with patient details, summary, graphs, and photos."""
    if not visits:
        raise ValueError("at least one visit is required to generate a report")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.alignment = TA_CENTER
    story: list[Any] = [
        Paragraph("WoundTrack Progress Report", title_style),
        Paragraph(MEDICAL_DISCLAIMER, styles["Heading3"]),
        Spacer(1, 0.15 * inch),
    ]

    patient_rows = [
        ["Patient", str(patient.get("name", "—"))],
        ["Age", str(patient.get("age") if patient.get("age") is not None else "Not provided")],
        ["Report date", date.today().isoformat()],
        ["Visits", str(len(visits))],
    ]
    patient_table = Table(patient_rows, colWidths=[1.5 * inch, 5.5 * inch])
    patient_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f8")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#202633")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9dee8")),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([patient_table, Spacer(1, 0.18 * inch)])
    story.append(Paragraph("Summary", styles["Heading2"]))

    first = visits[0]
    latest = visits[-1]
    summary_lines = [
        f"Baseline visit: {first['date']}",
        f"Latest visit: {latest['date']}",
        f"Latest status: {latest.get('status_label', 'Unclassified')}",
    ]
    if latest.get("area_cm2") is not None:
        summary_lines.append(
            f"Latest area: {float(latest['area_cm2']):.2f} cm² "
            f"({'calibrated' if latest.get('calibrated') else 'uncalibrated / relative'})"
        )
    else:
        summary_lines.append(
            f"Latest area: {int(latest.get('area_pixels', 0))} pixels "
            "(uncalibrated / relative)"
        )
    reduction = latest.get("reduction_from_baseline_pct")
    if reduction is not None:
        summary_lines.append(f"Area reduction from baseline: {float(reduction):.1f}%")
    healing_rate = latest.get("healing_rate_cm2_per_day")
    if healing_rate is not None:
        summary_lines.append(f"Latest healing rate: {float(healing_rate):.3f} cm²/day")
    for line in summary_lines:
        story.append(Paragraph(line, styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Progress graphs", styles["Heading2"]))
    area_chart = _make_area_chart(visits)
    if area_chart is not None:
        story.append(ReportImage(area_chart, width=6.8 * inch, height=3.1 * inch))
    else:
        story.append(Paragraph("No pixel-to-cm² scale has been recorded for these visits.", styles["BodyText"]))
    tissue_chart = _make_tissue_chart(visits)
    if tissue_chart is not None:
        story.append(ReportImage(tissue_chart, width=6.8 * inch, height=3.1 * inch))
    story.append(
        Paragraph(
            "Tissue percentages are heuristic HSV color estimates, not a validated tissue classifier.",
            styles["Italic"],
        )
    )

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Image comparison", styles["Heading2"]))
    comparison = _make_photo_comparison(visits)
    if comparison is not None:
        story.append(ReportImage(comparison, width=6.8 * inch, height=3.4 * inch))
    else:
        story.append(Paragraph("A photo comparison could not be generated from the stored image files.", styles["BodyText"]))

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Visit details", styles["Heading2"]))
    visit_rows = [["Date", "Area", "Scale", "Perimeter", "Status"]]
    for visit in visits:
        area = (
            f"{float(visit['area_cm2']):.2f} cm²"
            if visit.get("area_cm2") is not None
            else f"{int(visit.get('area_pixels', 0))} px²"
        )
        perimeter = visit.get("perimeter")
        perimeter_text = (
            f"{float(perimeter):.1f} {visit.get('perimeter_unit', 'px')}"
            if perimeter is not None
            else "—"
        )
        visit_rows.append(
            [
                str(visit["date"]),
                area,
                "Calibrated" if visit.get("calibrated") else "Relative",
                perimeter_text,
                str(visit.get("status_label", "—")),
            ]
        )
    visits_table = Table(visit_rows, repeatRows=1, colWidths=[1.0, 1.15, 1.1, 1.1, 2.5])
    visits_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2463eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d9dee8")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([visits_table, Spacer(1, 0.1 * inch)])
    notes = str(patient.get("notes", "")).strip()
    if notes:
        story.append(Paragraph("Patient notes", styles["Heading2"]))
        story.append(Paragraph(escape(notes).replace("\n", "<br/>"), styles["BodyText"]))

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.78 * inch,
        title="WoundTrack Progress Report",
        author="WoundTrack",
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output_path
