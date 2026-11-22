"""
PDF Report generation engine (100% Python via xhtml2pdf).
Compiles an executive audit report of data preprocessing, model performance, and leaderboard.
"""

from __future__ import annotations

import datetime
from io import BytesIO
from typing import Any, Dict
from django.http import HttpResponse
from django.template.loader import get_template
from django.utils.text import slugify
from xhtml2pdf import pisa

from apps.datasets.models import Dataset
from apps.preprocessing.services import load_prepared
from apps.ml_engine.services import load_latest_result, load_leaderboard
from apps.export_inference.exporter import ExportError
from apps.model_viz.report_charts import report_images


def build_report_context(dataset: Dataset) -> Dict[str, Any]:
    """Collect all dataset, preprocessing, training, and leaderboard details for the audit report."""
    prepared = load_prepared(dataset)
    latest_result = load_latest_result(dataset)
    leaderboard = load_leaderboard(dataset)

    # Determine champion model from leaderboard
    champion = None
    if leaderboard:
        champion = next((b for b in leaderboard if b.get("is_best")), leaderboard[0])

    prep_config = prepared.config if prepared else {}

    return {
        "dataset": dataset,
        "prepared": prepared,
        "prep_config": prep_config,
        "latest_result": latest_result,
        "leaderboard": leaderboard,
        "champion": champion,
        # Static diagnostic charts (confusion matrix, ROC, residuals, clusters…) from the diagnostics module.
        "diagnostic_images": report_images(dataset),
        "generated_at": datetime.datetime.now().strftime("%d/%m/%Y à %H:%M"),
    }


def generate_pdf_report_bytes(dataset: Dataset) -> bytes:
    """Render the HTML report template and compile it into raw PDF bytes."""
    context = build_report_context(dataset)
    template = get_template("export_inference/pdf_report_template.html")
    html_content = template.render(context)

    pdf_buffer = BytesIO()
    pisa_status = pisa.pisaDocument(
        src=BytesIO(html_content.encode("utf-8")),
        dest=pdf_buffer,
        encoding="utf-8",
    )

    if pisa_status.err:
        raise ExportError(f"Erreur lors de la compilation du PDF ({pisa_status.err})")

    return pdf_buffer.getvalue()


def generate_pdf_report_response(dataset: Dataset) -> HttpResponse:
    """Generate the HTTP FileResponse to download the compiled PDF."""
    pdf_bytes = generate_pdf_report_bytes(dataset)
    slug = slugify(dataset.name) or "rapport"
    filename = f"rapport_ml_studio_{slug}.pdf"

    response = HttpResponse(pdf_bytes, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response
