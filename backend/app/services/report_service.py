"""Geração do relatório PDF de uma classificação."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    CondPageBreak,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus import (
    Image as RLImage,
)

from backend.app.db.models import Analysis
from backend.app.services.output_files import resolve_output_file

_PAGE_MARGIN = 2 * cm
_CONTENT_WIDTH = A4[0] - 2 * _PAGE_MARGIN

_CLASS_COLORS = {
    "Vegetação Nativa": "#1a5276",
    "Pastagem": "#0E6B3D",
    "Pastagem / Veg. Herbácea": "#0E6B3D",
    "Agricultura": "#2ecc71",
    "Solo Exposto": "#f39c12",
    "Área Construída": "#e74c3c",
    "Água": "#2e86de",
}
_CROP_COLORS = {
    "Cana-de-açúcar": "#8e44ad",
    "Citros": "#f1c40f",
    "Pastagem": "#0E6B3D",
    "Soja": "#16a085",
    "Milho": "#e67e22",
}
_SATELLITE_LABELS = {
    "sentinel2": "Sentinel-2",
    "landsat8": "Landsat 8",
}
_CLASSIFIER_LABELS = {
    "auto": "Automático",
    "ensemble": "Ensemble",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "unet_cnn": "U-Net",
    "brazil_rf": "Random Forest (Brasil)",
    "brazil_svm": "SVM (Brasil)",
    "brazil_unet": "U-Net (Brasil)",
    "global_rf": "Random Forest (Global)",
    "global_svm": "SVM (Global)",
    "sp_rf": "Random Forest (SP)",
    "sp_svm": "SVM (SP)",
    "sp_ensemble": "Ensemble regional SP",
}

_BASE_STYLES = getSampleStyleSheet()
_TITLE = ParagraphStyle(
    "GeoRuralTitle",
    parent=_BASE_STYLES["Title"],
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    textColor=colors.HexColor("#1a5276"),
    alignment=0,
    spaceAfter=2,
)
_SUBTITLE = ParagraphStyle(
    "GeoRuralSubtitle",
    parent=_BASE_STYLES["Normal"],
    fontName="Helvetica-Bold",
    fontSize=13,
    leading=16,
    textColor=colors.HexColor("#243b53"),
    spaceAfter=2,
)
_SECTION = ParagraphStyle(
    "GeoRuralSection",
    parent=_BASE_STYLES["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=15,
    textColor=colors.HexColor("#1a5276"),
    spaceBefore=4,
    spaceAfter=5,
)
_SUBHEADING = ParagraphStyle(
    "GeoRuralSubheading",
    parent=_BASE_STYLES["Heading3"],
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=13,
    textColor=colors.HexColor("#243b53"),
    spaceBefore=2,
    spaceAfter=3,
)
_BODY = ParagraphStyle(
    "GeoRuralBody",
    parent=_BASE_STYLES["Normal"],
    fontName="Helvetica",
    fontSize=9,
    leading=12,
    textColor=colors.HexColor("#243b53"),
)
_SMALL = ParagraphStyle(
    "GeoRuralSmall",
    parent=_BODY,
    fontSize=8,
    leading=10.5,
    textColor=colors.HexColor("#52667a"),
)
_CAPTION = ParagraphStyle(
    "GeoRuralCaption",
    parent=_SMALL,
    alignment=TA_CENTER,
    fontSize=7.5,
)

_GRID = colors.HexColor("#c9d3dc")
_HEADER_BG = colors.HexColor("#e8eef3")
_ALT_BG = colors.HexColor("#f7f9fb")


def _fmt_number(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    text = f"{number:,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_percent(value: Any) -> str:
    return f"{_fmt_number(value, 2)}%"


def _fmt_date(value: Any) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.strptime(str(value), "%Y-%m-%d")
    except ValueError:
        return escape(str(value))
    return parsed.strftime("%d/%m/%Y")


def _fmt_datetime(value: datetime | None) -> str:
    if value is None:
        return "—"
    if value.tzinfo is not None:
        value = value.astimezone(UTC)
    return value.strftime("%d/%m/%Y %H:%M UTC")


def _esc(value: Any) -> str:
    if value is None:
        return "—"
    return escape(str(value))


def _satellite_label(value: Any) -> str:
    if not value:
        return "—"
    key = str(value).lower()
    return _SATELLITE_LABELS.get(key, str(value).upper())


def _classifier_label(value: Any) -> str:
    if not value:
        return "—"
    key = str(value)
    return _CLASSIFIER_LABELS.get(key, key.replace("_", " ").title())


def _table(
    rows: list[list[Any]],
    widths: list[float],
    *,
    align_right: bool = True,
    zebra: bool = True,
) -> Table:
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("GRID", (0, 0), (-1, -1), 0.25, _GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if align_right:
        style.append(("ALIGN", (1, 0), (-1, -1), "RIGHT"))
    if zebra and len(rows) > 2:
        style.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _ALT_BG]))
    table.setStyle(TableStyle(style))
    return table


def _metadata_table(analysis: Analysis, result: dict[str, Any]) -> Table:
    coverage = result.get("coverage") or analysis.coverage
    coverage_label = {
        "brazil": "Brasil",
        "global": "Global",
    }.get(coverage or "", "—")
    label_source = result.get("label_source") or analysis.label_source or "—"
    satellite = _satellite_label(
        result.get("satellite") or analysis.satellite
    )
    classifier = _classifier_label(
        result.get("classifier") or analysis.classifier
    )
    models = result.get("models_used") or analysis.models_used or []
    models_text = " + ".join(_classifier_label(name) for name in models) if models else "—"
    confidence = result.get("confidence_mean")
    confidence_text = (
        _fmt_percent(float(confidence) * 100)
        if confidence is not None
        else "—"
    )
    postprocess = result.get("postprocessed")
    postprocess_text = (
        "Sim" if postprocess else "Não" if postprocess is not None else "—"
    )
    max_cloud = analysis.max_cloud
    max_cloud_text = f"{max_cloud}%" if max_cloud is not None else "—"

    rows = [
        [
            Paragraph("<b>Satélite</b>", _SMALL),
            Paragraph(_esc(satellite), _BODY),
            Paragraph("<b>Classificador</b>", _SMALL),
            Paragraph(_esc(classifier), _BODY),
        ],
        [
            Paragraph("<b>Cobertura</b>", _SMALL),
            Paragraph(_esc(coverage_label), _BODY),
            Paragraph("<b>Fonte dos rótulos</b>", _SMALL),
            Paragraph(_esc(label_source), _BODY),
        ],
        [
            Paragraph("<b>Período</b>", _SMALL),
            Paragraph(
                f"{_fmt_date(result.get('start_date') or analysis.start_date)}"
                f" a {_fmt_date(result.get('end_date') or analysis.end_date)}",
                _BODY,
            ),
            Paragraph("<b>Nuvem máxima</b>", _SMALL),
            Paragraph(max_cloud_text, _BODY),
        ],
        [
            Paragraph("<b>Imagens usadas</b>", _SMALL),
            Paragraph(_fmt_number(result.get("num_images"), 0), _BODY),
            Paragraph("<b>Confiança média</b>", _SMALL),
            Paragraph(confidence_text, _BODY),
        ],
        [
            Paragraph("<b>Pós-processamento</b>", _SMALL),
            Paragraph(postprocess_text, _BODY),
            Paragraph("<b>Modelos</b>", _SMALL),
            Paragraph(_esc(models_text), _BODY),
        ],
    ]
    table = Table(rows, colWidths=[3.3 * cm, 5.2 * cm, 3.3 * cm, 5.2 * cm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, _GRID),
                ("BACKGROUND", (0, 0), (0, -1), _HEADER_BG),
                ("BACKGROUND", (2, 0), (2, -1), _HEADER_BG),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _measurements_table(result: dict[str, Any]) -> Table | None:
    measurements = result.get("measurements") or {}
    if not measurements:
        return None
    items = [
        ("Área total", f"{_fmt_number(measurements.get('total_area_ha'))} ha"),
        ("Perímetro", f"{_fmt_number(measurements.get('perimeter_m'), 1)} m"),
        ("Pixels", _fmt_number(measurements.get("total_pixels"), 0)),
        ("Talhões", _fmt_number(measurements.get("parcels"), 0)),
    ]
    cells = [
        Paragraph(f"<b>{label}</b><br/>{value}", _BODY) for label, value in items
    ]
    table = Table([cells], colWidths=[_CONTENT_WIDTH / 4] * 4)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _ALT_BG),
                ("BOX", (0, 0), (-1, -1), 0.25, _GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, _GRID),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def _class_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    class_areas = result.get("class_areas") or []
    if class_areas:
        return class_areas
    return [
        {
            "class_name": item.get("class_name"),
            "area_ha": None,
            "percent": item.get("percent"),
            "parcels": None,
        }
        for item in result.get("stats") or []
    ]


def _class_table(items: list[dict[str, Any]]) -> Table:
    rows: list[list[Any]] = [
        ["Classe", "Área (ha)", "Proporção", "Talhões"],
    ]
    for item in items:
        color = _CLASS_COLORS.get(item.get("class_name"), "#888888")
        rows.append(
            [
                Paragraph(
                    f'<font color="{color}">•</font> '
                    f"{_esc(item.get('class_name'))}",
                    _BODY,
                ),
                _fmt_number(item.get("area_ha")),
                _fmt_percent(item.get("percent")),
                _fmt_number(item.get("parcels"), 0),
            ]
        )
    return _table(
        rows,
        [8.0 * cm, 3.4 * cm, 3.0 * cm, 2.6 * cm],
    )


def _bar_chart(items: list[dict[str, Any]]) -> Drawing | None:
    active = [item for item in items if (item.get("percent") or 0) > 0]
    if not active:
        return None

    height = max(4.2 * cm, 0.75 * cm * len(active) + 1.1 * cm)
    drawing = Drawing(_CONTENT_WIDTH, height)
    chart = HorizontalBarChart()
    chart.x = 4.3 * cm
    chart.y = 0.45 * cm
    chart.width = _CONTENT_WIDTH - chart.x - 0.4 * cm
    chart.height = height - 0.9 * cm
    chart.data = [[float(item.get("percent") or 0) for item in active]]
    chart.categoryAxis.categoryNames = [
        str(item.get("class_name")) for item in active
    ]
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 7.5
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 100
    chart.valueAxis.valueStep = 20
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.labelTextFormat = "%d%%"
    chart.bars[0].strokeColor = colors.white
    for index, item in enumerate(active):
        color = _CLASS_COLORS.get(item.get("class_name"), "#888888")
        chart.bars[(0, index)].fillColor = colors.HexColor(color)
    drawing.add(chart)
    return drawing


def _load_image(
    job_id: str,
    filename: str,
    *,
    max_width: float,
    max_height: float,
) -> RLImage | None:
    path = resolve_output_file(job_id, filename)
    if path is None:
        return None
    try:
        with PILImage.open(path) as image:
            width_px, height_px = image.size
    except OSError:
        return None
    if width_px <= 0 or height_px <= 0:
        return None

    scale = min(max_width / width_px, max_height / height_px)
    return RLImage(
        str(path), width=width_px * scale, height=height_px * scale
    )


def _image_with_caption(
    job_id: str,
    filename: str,
    caption: str,
    *,
    max_width: float = _CONTENT_WIDTH,
    max_height: float = 12.5 * cm,
) -> list[Any]:
    image = _load_image(
        job_id, filename, max_width=max_width, max_height=max_height
    )
    if image is None:
        return []
    return [image, Paragraph(caption, _CAPTION)]


def _side_by_side_images(job_id: str) -> list[Any]:
    half = (_CONTENT_WIDTH - 0.4 * cm) / 2
    rgb = _load_image(
        job_id, "rgb.png", max_width=half, max_height=7.5 * cm
    )
    overlay = _load_image(
        job_id, "overlay.png", max_width=half, max_height=7.5 * cm
    )
    if rgb is None and overlay is None:
        return []
    if rgb is None or overlay is None:
        image = rgb or overlay
        caption = (
            "Imagem real" if rgb is not None else "Classificação sobreposta"
        )
        return [image, Paragraph(caption, _CAPTION)]

    images = Table([[rgb, overlay]], colWidths=[half, half])
    images.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    captions = Table(
        [
            [
                Paragraph("Imagem real", _CAPTION),
                Paragraph("Classificação sobreposta (50%)", _CAPTION),
            ]
        ],
        colWidths=[half, half],
    )
    return [images, captions]


def _topography_table(topography: dict[str, Any]) -> Table:
    rows = [
        ["Indicador", "Valor"],
        ["Elevação mínima", f"{_fmt_number(topography.get('elevation_min'), 1)} m"],
        ["Elevação média", f"{_fmt_number(topography.get('elevation_mean'), 1)} m"],
        ["Elevação máxima", f"{_fmt_number(topography.get('elevation_max'), 1)} m"],
        ["Desnível", f"{_fmt_number(topography.get('relief_m'), 1)} m"],
        ["Declividade média", f"{_fmt_number(topography.get('slope_mean'), 1)}°"],
        ["Declividade máxima", f"{_fmt_number(topography.get('slope_max'), 1)}°"],
        ["Relevo dominante", _esc(topography.get("dominant_class"))],
        [
            "Fonte do DEM",
            f"{_esc(topography.get('dem_source'))}"
            + (
                f" (~{_fmt_number(topography.get('dem_resolution_m'), 0)} m)"
                if topography.get("dem_resolution_m")
                else ""
            ),
        ],
    ]
    return _table(rows, [8.0 * cm, 9.0 * cm], align_right=False)


def _topography_classes_table(classes: list[dict[str, Any]]) -> Table:
    rows: list[list[Any]] = [["Relevo", "Área (ha)", "Proporção"]]
    for item in classes:
        rows.append(
            [
                _esc(item.get("class_name")),
                _fmt_number(item.get("area_ha")),
                _fmt_percent(item.get("percent")),
            ]
        )
    return _table(rows, [9.0 * cm, 4.0 * cm, 4.0 * cm])


def _crop_table(items: list[dict[str, Any]]) -> Table:
    rows: list[list[Any]] = [["Cultura", "Área (ha)", "Proporção"]]
    for item in items:
        color = _CROP_COLORS.get(item.get("class_name"), "#888888")
        rows.append(
            [
                Paragraph(
                    f'<font color="{color}">•</font> '
                    f"{_esc(item.get('class_name'))}",
                    _BODY,
                ),
                _fmt_number(item.get("area_ha")),
                _fmt_percent(item.get("percent")),
            ]
        )
    return _table(rows, [9.0 * cm, 4.0 * cm, 4.0 * cm])


def _on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d5dde5"))
    canvas.setLineWidth(0.5)
    canvas.line(
        _PAGE_MARGIN,
        1.55 * cm,
        A4[0] - _PAGE_MARGIN,
        1.55 * cm,
    )
    canvas.setFillColor(colors.HexColor("#6b7c8c"))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(
        _PAGE_MARGIN,
        1.15 * cm,
        "GeoRural — Relatório de Classificação",
    )
    canvas.drawRightString(
        A4[0] - _PAGE_MARGIN,
        1.15 * cm,
        f"Página {doc.page}",
    )
    canvas.restoreState()


def build_analysis_report(
    analysis: Analysis, user_name: str | None = None
) -> bytes:
    """Monta o PDF completo de uma análise concluída."""
    result = analysis.result or {}
    job_id = str(analysis.id)
    area_name = analysis.area.name if analysis.area else None
    title = analysis.name or area_name or f"Análise {job_id}"

    story: list[Any] = [
        Paragraph("Relatório de Classificação", _TITLE),
        Paragraph(_esc(title), _SUBTITLE),
        Paragraph(
            f"<b>Job:</b> {_esc(job_id)} &nbsp;&nbsp; "
            f"<b>Gerado em:</b> {_fmt_datetime(datetime.now(UTC))} "
            f"&nbsp;&nbsp; <b>Usuário:</b> {_esc(user_name)}",
            _SMALL,
        ),
        Spacer(1, 10),
        Paragraph("Resumo", _SECTION),
        _metadata_table(analysis, result),
    ]

    measurements = _measurements_table(result)
    if measurements is not None:
        story.extend([Spacer(1, 6), measurements])

    story.extend(
        [
            CondPageBreak(7 * cm),
            Paragraph("Mapa de Classificação", _SECTION),
        ]
    )
    classification = _image_with_caption(
        job_id,
        "classification.png",
        "Uso e cobertura do solo classificado.",
    )
    if classification:
        story.extend(classification)
    else:
        story.append(
            Paragraph(
                "Mapa de classificação indisponível para esta análise.",
                _SMALL,
            )
        )

    items = _class_items(result)
    if items:
        story.extend(
            [
                CondPageBreak(7 * cm),
                Paragraph("Distribuição das Classes", _SECTION),
            ]
        )
        chart = _bar_chart(items)
        if chart is not None:
            story.extend([chart, Spacer(1, 4)])
        story.append(_class_table(items))

    images = _side_by_side_images(job_id)
    if images:
        story.extend(
            [
                CondPageBreak(7 * cm),
                Paragraph("Imagens da Área", _SECTION),
                *images,
            ]
        )

    topography = result.get("topography") or {}
    topo_classes = topography.get("classes") or []
    if topography:
        story.extend(
            [
                CondPageBreak(8 * cm),
                Paragraph("Análise Topográfica", _SECTION),
                _topography_table(topography),
            ]
        )
        if topo_classes:
            story.extend(
                [Spacer(1, 5), _topography_classes_table(topo_classes)]
            )
        relief = _load_image(
            job_id,
            "relief.png",
            max_width=12 * cm,
            max_height=8 * cm,
        ) or _load_image(
            job_id,
            "slope.png",
            max_width=12 * cm,
            max_height=8 * cm,
        )
        if relief is not None:
            story.extend(
                [
                    CondPageBreak(9 * cm),
                    KeepTogether(
                        [
                            Paragraph("Mapa de Relevo", _SUBHEADING),
                            relief,
                            Paragraph(
                                "Mapa de relevo da área analisada.", _CAPTION
                            ),
                        ]
                    ),
                ]
            )

    crop_stats = result.get("crop_stats") or []
    if crop_stats:
        story.extend(
            [
                CondPageBreak(7 * cm),
                Paragraph("Tipo de Cultivo (estimado)", _SECTION),
                _crop_table(crop_stats),
            ]
        )
        crop_image = _load_image(
            job_id, "crop.png", max_width=12 * cm, max_height=7 * cm
        )
        if crop_image is not None:
            story.extend(
                [
                    CondPageBreak(8 * cm),
                    KeepTogether(
                        [
                            Paragraph("Mapa de Cultivo", _SUBHEADING),
                            crop_image,
                            Paragraph("Mapa de cultivo estimado.", _CAPTION),
                        ]
                    ),
                ]
            )

    story.extend(
        [
            CondPageBreak(5 * cm),
            Paragraph("Observações", _SECTION),
            Paragraph(
                "A confiança média é derivada das probabilidades do modelo e "
                "não é calibrada. Os resultados são apoio à decisão e não "
                "substituem vistoria de campo.",
                _BODY,
            ),
            Spacer(1, 3),
            Paragraph(
                "Relatório gerado automaticamente pelo GeoRural.",
                _SMALL,
            ),
        ]
    )

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=_PAGE_MARGIN,
        rightMargin=_PAGE_MARGIN,
        topMargin=1.7 * cm,
        bottomMargin=2.0 * cm,
        title=f"Relatório GeoRural — {job_id}",
        author="GeoRural",
        subject="Classificação de uso e cobertura do solo",
    )
    document.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return buffer.getvalue()
