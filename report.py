"""
report.py - Geração de Relatórios Técnicos de Inspeção em PDF
Defeitos AeBAD (Yang et al., 2022): ablation, groove, fracture, breakdown
"""
from __future__ import annotations

import datetime
import io
from pathlib import Path
import numpy as np


# Conhecimento científico rigoroso sobre os defeitos (Yang et al., 2022)
DEFECT_IMPACT = {
    "breakdown": (
        "Desbalanceamento dinâmico severo do rotor, perdas bruscas de eficiência "
        "aerodinâmica e risco de danos em cascata a jusante, por exemplo provocados por fragmentos soltos. "
        "Causado pelo impacto de corpos estranhos (ex.: areia, metal, aves ou granizo), provocando a quebra da pá."
    ),
    "ablation": (
        "Proveniente dos gases a alta temperatura aos quais as pás estão continuamente expostas. "
        "Reduz o desempenho térmico e mecânico do material da pá. Quando a degradação atinge determinado "
        "nível crítico e o material deixa de cumprir os requisitos estruturais, ocorre fratura."
    ),
    "groove": (
        "Pequenos defeitos na margem da pá causados por um elevado número de ciclos de tensão elástica. "
        "Com a operação contínua, estas ranhuras expandem-se, atuando como concentradores de tensão que originam a fratura da pá."
    ),
    "fracture": (
        "Áreas em falta na pá resultantes da evolução de ablação térmica crítica ou da expansão de ranhuras mecânicas. "
        "Representa uma descontinuidade estrutural grave com perda imediata de integridade funcional."
    ),
    "none": (
        "Sem descontinuidades ou anomalias detetadas acima dos limiares de tolerância. "
        "A integridade superficial cumpre os requisitos de conformidade da inspeção."
    ),
}

DEFECT_TITLES = {
    "breakdown": "Breakdown",
    "ablation": "Ablation",
    "groove": "Groove",
    "fracture": "Fracture",
    "none": "Conforme",
}


def identify_defect_type(
    image_name: str,
    is_anomalous: bool,
    pred_map: np.ndarray | None = None,
    pixel_threshold: float = 0.50,
) -> tuple[str, str]:
    """
    Identifica o tipo de falha (ablation, groove, fracture, breakdown)
    com base no nome/anotação ou características morfológicas da deteção.
    Retorna (chave_defeito, titulo_defeito).
    """
    if not is_anomalous:
        return "none", DEFECT_TITLES.get("none", "Conforme")

    detected_key = "breakdown"
    name_lower = str(image_name).lower()
    for k in ["ablation", "groove", "fracture", "breakdown"]:
        if k in name_lower:
            detected_key = k
            return detected_key, DEFECT_TITLES.get(detected_key, detected_key)

    if pred_map is not None:
        mask = pred_map >= pixel_threshold
        defect_area = int(mask.sum())
        if defect_area > 0:
            import cv2
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                largest = max(cnts, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
                hull = cv2.convexHull(largest)
                solidity = cv2.contourArea(largest) / (cv2.contourArea(hull) + 1e-5)

                if aspect_ratio > 3.5:
                    detected_key = "fracture" if solidity < 0.6 else "groove"
                elif defect_area > 8000:
                    detected_key = "breakdown"
                else:
                    detected_key = "ablation"
                return detected_key, DEFECT_TITLES.get(detected_key, detected_key)

    return detected_key, DEFECT_TITLES.get(detected_key, detected_key)


def explain_metric_case(metric_name: str, value: float) -> str:
    """
    Interpretação técnica e contextualizada do valor concreto de cada métrica
    especificamente para o caso analisado.
    """
    if np.isnan(value):
        return "Métrica não aplicável para este caso (ausência de anotação de referência)."

    pct_str = f"{value * 100:.1f}%"

    if metric_name == "Pixel AUROC":
        if value >= 0.98:
            return f"O valor de {pct_str} demonstra excelente separação global entre metal são e regiões anómalas na superfície."
        elif value >= 0.90:
            return f"O valor de {pct_str} indica boa discriminação geral, com pequenas variações em zonas de reflexão ou curvatura."
        else:
            return f"O valor de {pct_str} assinala separação reduzida nesta peça, recomendando verificação do limiar."

    elif metric_name == "Localization Precision @ FPR 5%":
        if value >= 0.95:
            return f"Com taxa de falsos alarmes limitada a 5%, a precisão de {pct_str} confirma localização confiável do núcleo do defeito."
        elif value >= 0.80:
            return f"Atinge {pct_str} de precisão de localização mantendo os falsos alarmes sob controlo."
        else:
            return f"Precisão de localização moderada ({pct_str}), sugerindo dispersão do mapa térmico em torno do defeito real."

    elif metric_name == "Pixel F1-Score (Dice)":
        if value >= 0.50:
            return f"O F1-Score de {pct_str} representa equilíbrio robusto entre a área detetada e a geometria da falha."
        elif value >= 0.20:
            return f"O valor de {pct_str} é típico na inspeção de ranhuras ou fissuras finas, onde variações marginais de contorno reduzem o Dice mas mantêm a identificação do defeito."
        else:
            return f"O F1-Score de {pct_str} reflete divergências entre o contorno detetado e a anotação manual de referência."

    elif metric_name == "IoU (Jaccard Index)":
        if value >= 0.40:
            return f"O índice de sobreposição direta de {pct_str} comprova aderência geométrica estrita à anotação real."
        elif value >= 0.15:
            return f"Com {pct_str} de sobreposição estrita (IoU), o núcleo do defeito coincide com a referência, com ligeiras discrepâncias nas bordas."
        else:
            return f"Índice de {pct_str} reflete sobreposição restrita, comum quando o modelo engloba a vizinhança imediata da descontinuidade."

    elif metric_name == "Precision":
        if value >= 0.70:
            return f"Precisão de {pct_str}: a grande maioria dos pixels assinalados corresponde a descontinuidade física real."
        elif value >= 0.30:
            return f"Precisão de {pct_str}: a deteção abrange a falha com margem prudente em torno dos seus contornos."
        else:
            return f"Precisão de {pct_str}: a marcação engloba áreas circundantes, gerando falsos positivos na periferia."

    elif metric_name == "Recall (Sensibilidade)":
        if value >= 0.85:
            return f"Recall de {pct_str}: garante que a quase totalidade da extensão da falha foi coberta pela deteção."
        elif value >= 0.60:
            return f"Recall de {pct_str}: a maioria da extensão do defeito foi detetada, restando frações marginais."
        else:
            return f"Recall de {pct_str}: fração relevante da anotação não foi assinalada pelo limiar atual."

    return f"Valor medido: {pct_str}."


# =====================================================================
# GERADOR DE PDF 1.4 PURO (Zero Dependências Externas)
# =====================================================================

def _escape_pdf_text(text: str) -> str:
    """Escapa caracteres para strings PDF WinAnsiEncoding."""
    replacements = {
        "\\": "\\\\",
        "(": "\\(",
        ")": "\\)",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def _wrap_text(text: str, max_chars: int = 75) -> list[str]:
    """Quebra texto em linhas com tamanho máximo."""
    words = text.split(" ")
    lines = []
    current = ""
    for w in words:
        if not current:
            current = w
        elif len(current) + 1 + len(w) <= max_chars:
            current += " " + w
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def generate_pdf_report(
    image_name: str,
    is_anomalous: bool,
    pixel_threshold: float,
    metrics: dict[str, float] | None = None,
    defect_type: str = "breakdown",
) -> bytes:
    """
    Gera um relatório formal de inspeção técnico em PDF (PDF 1.4 padrão).
    Totalmente auto-contido, sério, formal e sem emojis.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=40,
            rightMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0F172A"),
        )
        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748B"),
        )
        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#1E293B"),
        )
        bold_style = ParagraphStyle(
            "ReportBold",
            parent=body_style,
            fontName="Helvetica-Bold",
        )
        table_text = ParagraphStyle(
            "TableText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#334155"),
        )
        table_val = ParagraphStyle(
            "TableVal",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            alignment=2,
            textColor=colors.HexColor("#0369A1"),
        )

        elements = []

        # Título simples e sério
        elements.append(Paragraph("Relatório de inspeção", title_style))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Data e hora de emissão: {timestamp}", subtitle_style))
        elements.append(Spacer(1, 15))

        # Tabela de dados gerais
        verdict_str = "DEFEITUOSO" if is_anomalous else "NORMAL"
        verdict_color = colors.HexColor("#DC2626") if is_anomalous else colors.HexColor("#16A34A")

        info_data = [
            [
                Paragraph("<b>Ficheiro analisado:</b>", body_style),
                Paragraph(str(image_name), body_style),
                Paragraph("<b>Veredito da inspeção:</b>", body_style),
                Paragraph(f"<b><font color='{verdict_color.hexval()}'>{verdict_str}</font></b>", body_style),
            ],
            [
                Paragraph("<b>Limiar de pixel:</b>", body_style),
                Paragraph(f"{pixel_threshold:.2f}", body_style),
                Paragraph("", body_style),
                Paragraph("", body_style),
            ],
        ]
        info_table = Table(info_data, colWidths=[120, 140, 120, 135])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 15))

        # Impacto operacional direto
        impact_text = DEFECT_IMPACT.get(defect_type, DEFECT_IMPACT["breakdown"])
        elements.append(Paragraph("<b>Considerações operacionais:</b>", bold_style))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(impact_text, body_style))
        elements.append(Spacer(1, 15))

        # Tabela de métricas
        elements.append(Paragraph("<b>Métricas de validação:</b>", bold_style))
        elements.append(Spacer(1, 6))

        metrics_list = [
            ("Pixel AUROC", metrics.get("pixel_auroc", float("nan")) if metrics else float("nan")),
            ("Localization Precision @ FPR 5%", metrics.get("localization_at_5fpr", float("nan")) if metrics else float("nan")),
            ("Pixel F1-Score (Dice)", metrics.get("pixel_f1", float("nan")) if metrics else float("nan")),
            ("IoU (Jaccard Index)", metrics.get("iou", float("nan")) if metrics else float("nan")),
            ("Precision", metrics.get("precision", float("nan")) if metrics else float("nan")),
            ("Recall (Sensibilidade)", metrics.get("recall", float("nan")) if metrics else float("nan")),
        ]

        table_data = [[
            Paragraph("<b>Métrica</b>", table_text),
            Paragraph("<b>Valor</b>", table_val),
            Paragraph("<b>Interpretação técnica no caso</b>", table_text),
        ]]

        for name, val in metrics_list:
            val_str = "N/A" if np.isnan(val) else f"{val * 100:.1f}%"
            expl = explain_metric_case(name, val)
            table_data.append([
                Paragraph(name, table_text),
                Paragraph(val_str, table_val),
                Paragraph(expl, table_text),
            ])

        m_table = Table(table_data, colWidths=[140, 65, 310])
        m_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(m_table)

        doc.build(elements)
        return buf.getvalue()

    except Exception:
        return _build_native_pdf(image_name, is_anomalous, pixel_threshold, metrics, defect_type)


def _build_native_pdf(
    image_name: str,
    is_anomalous: bool,
    pixel_threshold: float,
    metrics: dict[str, float] | None,
    defect_type: str,
) -> bytes:
    """
    Constrói um PDF 1.4 binário válido sem qualquer dependência externa.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict_str = "DEFEITUOSO" if is_anomalous else "NORMAL"
    impact_text = DEFECT_IMPACT.get(defect_type, DEFECT_IMPACT["breakdown"])

    stream_lines = []

    # Barra superior sutil
    stream_lines.append("0.1 0.15 0.25 rg 40 795 515 2 re f")

    # Título do Relatório
    stream_lines.append("BT")
    stream_lines.append("/F2 18 Tf")
    stream_lines.append("40 765 Td")
    stream_lines.append(f"({_escape_pdf_text('Relatório de inspeção')}) Tj")
    stream_lines.append("ET")

    # Subtítulo com Data
    stream_lines.append("BT")
    stream_lines.append("/F1 9 Tf")
    stream_lines.append("0.4 0.45 0.55 rg")
    stream_lines.append("40 748 Td")
    stream_lines.append(f"({_escape_pdf_text(f'Data e hora de emissão: {timestamp}')}) Tj")
    stream_lines.append("ET")

    # Caixa de informações gerais
    stream_lines.append("0.96 0.97 0.98 rg 40 690 515 42 re f")
    stream_lines.append("0.8 0.83 0.88 RG 40 690 515 42 re S")

    stream_lines.append("BT")
    stream_lines.append("0 0 0 rg")
    stream_lines.append("/F2 9 Tf")
    stream_lines.append("50 714 Td (Ficheiro analisado:) Tj")
    stream_lines.append("/F1 9 Tf")
    stream_lines.append(f" 85 0 Td ({_escape_pdf_text(str(image_name)[:28])}) Tj")
    stream_lines.append("/F2 9 Tf")
    stream_lines.append(" 140 0 Td (Veredito da inspeção:) Tj")
    stream_lines.append("/F2 10 Tf")
    if is_anomalous:
        stream_lines.append(" 0.85 0.1 0.1 rg")
    else:
        stream_lines.append(" 0.1 0.6 0.2 rg")
    stream_lines.append(f" 110 0 Td ({verdict_str}) Tj")
    stream_lines.append("ET")

    stream_lines.append("BT")
    stream_lines.append("0 0 0 rg")
    stream_lines.append("/F2 9 Tf")
    stream_lines.append("50 698 Td (Limiar de pixel:) Tj")
    stream_lines.append("/F1 9 Tf")
    stream_lines.append(f" 85 0 Td ({pixel_threshold:.2f}) Tj")
    stream_lines.append("ET")

    # Considerações Operacionais
    stream_lines.append("BT")
    stream_lines.append("0 0 0 rg")
    stream_lines.append("/F2 10 Tf")
    stream_lines.append("40 660 Td (Considerações operacionais:) Tj")
    stream_lines.append("ET")

    wrapped_impact = _wrap_text(impact_text, max_chars=85)
    y_text = 645
    stream_lines.append("BT")
    stream_lines.append("/F1 8.5 Tf")
    stream_lines.append("0.15 0.2 0.25 rg")
    stream_lines.append(f"40 {y_text} Td")
    for idx, l in enumerate(wrapped_impact):
        if idx > 0:
            stream_lines.append(f"0 -12 Td")
        stream_lines.append(f"({_escape_pdf_text(l)}) Tj")
    stream_lines.append("ET")

    y_table_start = y_text - (len(wrapped_impact) * 12) - 20

    # Título da tabela de métricas
    stream_lines.append("BT")
    stream_lines.append("0 0 0 rg")
    stream_lines.append("/F2 10 Tf")
    stream_lines.append(f"40 {y_table_start} Td (Métricas de validação:) Tj")
    stream_lines.append("ET")

    y_t = y_table_start - 16

    # Cabeçalho da tabela
    stream_lines.append(f"0.92 0.94 0.96 rg 40 {y_t - 4} 515 18 re f")
    stream_lines.append(f"0.75 0.8 0.85 RG 40 {y_t - 4} 515 18 re S")

    stream_lines.append("BT")
    stream_lines.append("0.2 0.25 0.3 rg")
    stream_lines.append("/F2 8.5 Tf")
    stream_lines.append(f"48 {y_t} Td (Métrica) Tj")
    stream_lines.append("140 0 Td (Valor) Tj")
    stream_lines.append("60 0 Td (Interpretação técnica no caso) Tj")
    stream_lines.append("ET")

    y_t -= 20

    metrics_list = [
        ("Pixel AUROC", metrics.get("pixel_auroc", float("nan")) if metrics else float("nan")),
        ("Localization Precision @ FPR 5%", metrics.get("localization_at_5fpr", float("nan")) if metrics else float("nan")),
        ("Pixel F1-Score (Dice)", metrics.get("pixel_f1", float("nan")) if metrics else float("nan")),
        ("IoU (Jaccard Index)", metrics.get("iou", float("nan")) if metrics else float("nan")),
        ("Precision", metrics.get("precision", float("nan")) if metrics else float("nan")),
        ("Recall (Sensibilidade)", metrics.get("recall", float("nan")) if metrics else float("nan")),
    ]

    for m_name, m_val in metrics_list:
        val_str = "N/A" if np.isnan(m_val) else f"{m_val * 100:.1f}%"
        expl_lines = _wrap_text(explain_metric_case(m_name, m_val), max_chars=55)
        row_h = max(20, len(expl_lines) * 11 + 6)

        stream_lines.append(f"0.85 0.88 0.92 RG 40 {y_t - row_h + 14} 515 {row_h} re S")

        stream_lines.append("BT")
        stream_lines.append("0.1 0.15 0.2 rg")
        stream_lines.append("/F2 8.5 Tf")
        stream_lines.append(f"48 {y_t} Td ({_escape_pdf_text(m_name)}) Tj")
        stream_lines.append("ET")

        stream_lines.append("BT")
        stream_lines.append("0.05 0.45 0.65 rg")
        stream_lines.append("/F2 8.5 Tf")
        stream_lines.append(f"190 {y_t} Td ({val_str}) Tj")
        stream_lines.append("ET")

        stream_lines.append("BT")
        stream_lines.append("0.25 0.3 0.35 rg")
        stream_lines.append("/F1 8 Tf")
        stream_lines.append(f"250 {y_t} Td")
        for i, el in enumerate(expl_lines):
            if i > 0:
                stream_lines.append("0 -11 Td")
            stream_lines.append(f"({_escape_pdf_text(el)}) Tj")
        stream_lines.append("ET")

        y_t -= row_h

    content_stream = "\n".join(stream_lines).encode("latin-1")

    pdf_objs = []
    pdf_objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    pdf_objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    pdf_objs.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595.28 841.89] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>"
    )
    pdf_objs.append(
        f"<< /Length {len(content_stream)} >>\nstream\n".encode("ascii")
        + content_stream
        + b"\nendstream"
    )
    pdf_objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    pdf_objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")

    body = io.BytesIO()
    body.write(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(pdf_objs, 1):
        offsets.append(body.tell())
        body.write(f"{i} 0 obj\n".encode("ascii"))
        body.write(obj)
        body.write(b"\nendobj\n")

    xref_pos = body.tell()
    body.write(b"xref\n")
    body.write(f"0 {len(pdf_objs) + 1}\n".encode("ascii"))
    body.write(b"0000000000 65535 f \n")
    for off in offsets:
        body.write(f"{off:010d} 00000 n \n".encode("ascii"))

    body.write(b"trailer\n")
    body.write(f"<< /Size {len(pdf_objs) + 1} /Root 1 0 R >>\n".encode("ascii"))
    body.write(b"startxref\n")
    body.write(f"{xref_pos}\n".encode("ascii"))
    body.write(b"%%EOF")

    return body.getvalue()
