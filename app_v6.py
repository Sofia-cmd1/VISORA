
from __future__ import annotations

import hashlib
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms as T
import streamlit as st

from engine import AeroInspector
from metrics import (
    compute_single_image_metrics,
    compute_adaptive_pixel_threshold,
    compute_optimal_pixel_threshold_gt,
    compute_gt_overlay,
)
from report import (
    identify_defect_type,
    generate_pdf_report,
    DEFECT_IMPACT,
)

# ------------------------------------------------------------------
# Ícone da Aba do Browser (Aviãozinho em SVG codificado em Data URI)
# ------------------------------------------------------------------
PLANE_FAVICON = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
    "<path fill='%23000000' stroke='%2338BDF8' stroke-width='1.2' stroke-linejoin='round' "
    "d='M2 12L9 12L13 3L16 3L13 12L20 12L22 8L24 8L22 12L24 16L22 16L20 12L13 12L16 21L13 21L9 12L2 12Z'/>"
    "</svg>"
)

# ------------------------------------------------------------------
# Configuração da Página
# ------------------------------------------------------------------
st.set_page_config(
    page_title="VISORA",
    page_icon=PLANE_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Folha de Estilos
# ------------------------------------------------------------------
st.markdown("""
<style>
/* ── Ocultar cabeçalhos nativos do Streamlit ─────────────────── */
header[data-testid="stHeader"],
[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stToolbarActions"],
[data-testid="stStatusWidget"],
.stDeployButton, footer {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
}

.main {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

.main .block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 100% !important;
}

/* ── Fundo Geral ───────────────────────────────────────────── */
html, body, .stApp {
    background-color: #33445C !important;
    color: #FFFFFF !important;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 16px !important;
}

/* ── Barra Lateral com Disposição Centralizada ─────────────── */
[data-testid="stSidebar"] {
    background-color: #243348 !important;
    border-right: 1px solid #485D7C !important;
    min-width: 295px !important;
    max-width: 295px !important;
}
[data-testid="stSidebarContent"] {
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 100vh !important;
}

/* Contentor do Logótipo e Animação Centralizado Verticalmente */
.vis-sidebar-center-logo {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 20px 20px;
    width: 100%;
}
.vis-brand {
    font-size: 44px;
    font-weight: 900;
    letter-spacing: 6px;
    color: #FFFFFF;
    line-height: 1;
    margin-top: 16px;
    text-align: center;
}
.vis-brand-dot { color: #60A5FA; }

/* ── Secção do Limiar dentro da Barra Lateral ──────────────── */
.vis-sidebar-section {
    padding: 8px 6px 4px 6px;
    border-top: 1px solid #3B5375;
    margin-top: 10px;
    margin-bottom: 2px;
}
.vis-sidebar-label {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #93C5FD;
    margin-bottom: 6px;
    padding-left: 2px;
}

/* Alinhamento horizontal estrito e eliminação de quebras no bloco do limiar */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 4px !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding: 0 !important;
}

[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    min-width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(1) {
    justify-content: flex-start !important;
    flex: 2 1 auto !important;
}

[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) {
    justify-content: center !important;
    flex: 1.1 1 auto !important;
}

[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3) {
    justify-content: flex-end !important;
    flex: 0.9 1 auto !important;
}

/* Remover margens nativas do Streamlit que empurram o stToggle e stMarkdown para baixo */
[data-testid="stSidebar"] .stToggle {
    margin: 0 !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    height: 32px !important;
}

[data-testid="stSidebar"] .stToggle > label {
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
    height: 32px !important;
    display: flex !important;
    align-items: center !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    margin: 0 !important;
    padding: 0 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
}

/* Linha com 'Automático', '?' */
.vis-auto-row {
    display: flex;
    align-items: center;
    gap: 6px;
    height: 32px;
    white-space: nowrap;
}
.vis-auto-title {
    font-size: 15.5px;
    font-weight: 800;
    color: #FFFFFF !important;
    letter-spacing: 0.2px;
}
.vis-toggle-info {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: transparent;
    color: #FFFFFF !important;
    border: 1.5px solid #FFFFFF !important;
    font-size: 10.5px;
    font-weight: 800;
    cursor: help;
    user-select: none;
}

/* Retângulo com o número do limiar colocado à frente do botão de ligar/desligar */
.vis-side-val-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #172638;
    border: 1.5px solid #3B82F6;
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 15px;
    font-weight: 800;
    color: #93C5FD;
    letter-spacing: 0.5px;
    line-height: 1;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    white-space: nowrap;
}

/* Estilo limpo da linha do slider (fio preto, limites brancos) */
.stSlider label {
    color: #FFFFFF !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    color: #FFFFFF !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}
.stSlider div[data-testid="stSliderTickBar"] {
    color: #FFFFFF !important;
}
.stSlider div[data-baseweb="slider"] > div:first-child > div:first-child {
    background: #000000 !important;
    height: 5px !important;
    border-radius: 4px !important;
}
.stSlider div[data-baseweb="slider"] > div:first-child > div:nth-child(2) {
    background: #000000 !important;
    height: 5px !important;
}
.stSlider div[role="slider"] {
    background-color: #FFFFFF !important;
    border: 3px solid #000000 !important;
    width: 20px !important;
    height: 20px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4) !important;
}

/* Espaçamento elegante do slider */
[data-testid="stSidebar"] .stSlider {
    padding-top: 4px !important;
    padding-bottom: 6px !important;
}

/* ── Cabeçalho Principal ───────────────────────────────────── */
.vis-header {
    padding-bottom: 16px;
    border-bottom: 1px solid #485D7C;
    margin-bottom: 22px;
}
.vis-title {
    font-size: 32px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: 0.3px;
    line-height: 1.2;
}
.vis-title-blue { color: #93C5FD; }
.vis-subtitle {
    font-size: 16px;
    color: #E2E8F0;
    margin-top: 4px;
    font-weight: 400;
}

/* ── Botões de Modo com Letras Grandes (Câmara vs Carregar Ficheiro) ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 800 !important;
    font-size: 20px !important;
    padding: 1rem 2rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.5px !important;
}
.stButton > button[kind="primary"] {
    background: #1D4ED8 !important;
    color: #FFFFFF !important;
    border: 2.5px solid #93C5FD !important;
    box-shadow: 0 4px 16px rgba(29, 78, 216, 0.5) !important;
}
.stButton > button[kind="secondary"] {
    background: #25374E !important;
    color: #CBD5E1 !important;
    border: 1.5px solid #4B6385 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #2E435F !important;
    color: #FFFFFF !important;
    border-color: #93C5FD !important;
}

/* ── Títulos de Secção Grandes ─────────────────────────────── */
.vis-section-title {
    font-size: 20px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: 0.5px;
    margin-top: 14px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.vis-section-title::after {
    content: '';
    flex: 1;
    height: 1.5px;
    background: #4A6284;
}

/* ── Veredito Suave ────────────────────────────────────────── */
.vis-simple-verdict-fail {
    font-size: 22px;
    font-weight: 800;
    color: #F87171;
    margin: 14px 0 18px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.vis-simple-verdict-ok {
    font-size: 22px;
    font-weight: 800;
    color: #4ADE80;
    margin: 14px 0 18px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Retângulo de Upload (Texto '200MB...' a Branco) ───────── */
[data-testid="stFileUploader"] {
    width: 100% !important;
}
[data-testid="stFileUploader"] > section {
    background: #2B3D56 !important;
    border: 2px dashed #60A5FA !important;
    border-radius: 12px !important;
    min-height: 230px !important;
    padding: 40px 25px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
}
[data-testid="stFileUploader"] > section:hover {
    border-color: #93C5FD !important;
    background: #334866 !important;
}
[data-testid="stFileUploader"] section [data-testid="stFileUploadDropzoneInstructions"] {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
}
[data-testid="stFileUploader"] section button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    padding: 0.75rem 1.9rem !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    margin-top: 14px !important;
}
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section div span,
[data-testid="stFileUploader"] [data-testid="stFileUploadDropzoneInstructions"] span {
    font-size: 15px !important;
    color: #FFFFFF !important;
    font-weight: 500 !important;
    margin-top: 8px !important;
}

/* ── Tabela de Métricas ────────────────────────────────────── */
.vis-table {
    width: 100%;
    border-collapse: collapse;
    background: #2B3D56;
    border: 1px solid #485D7C;
    border-radius: 10px;
    overflow: hidden;
}
.vis-table thead tr {
    background: #223247;
    border-bottom: 2px solid #3B82F6;
}
.vis-table th {
    padding: 13px 18px;
    text-align: left;
    font-size: 13px;
    font-weight: 700;
    color: #93C5FD;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.vis-table th:last-child { text-align: right; }
.vis-table td {
    padding: 13px 18px;
    color: #FFFFFF;
    border-bottom: 1px solid #3D526F;
    font-size: 15px;
}
.vis-table td:last-child {
    text-align: right;
    font-weight: 700;
    color: #93C5FD;
    font-size: 16px;
}
.vis-table tr:last-child td { border-bottom: none; }

/* ── Bolinha de Ajuda ──────────────────────────────────────── */
.help-bubble {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #3B5375;
    color: #93C5FD;
    font-size: 11px;
    font-weight: 700;
    cursor: help;
    margin-left: 6px;
    border: 1px solid #57759F;
    vertical-align: middle;
}

/* ── Legenda de Sobreposição ───────────────────────────────── */
.vis-legend {
    background: #2B3D56;
    border: 1px solid #485D7C;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 14.5px;
    color: #F8FAFC;
    line-height: 2.2;
}
.vis-legend-dot {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 3px;
    margin-right: 8px;
    vertical-align: middle;
}

/* ── Caixa de Aviso / Info ─────────────────────────────────── */
.vis-info {
    background: #2B3D56;
    border: 1px solid #485D7C;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 15px;
    color: #E2E8F0;
    line-height: 1.6;
}

/* ── Descrição de Defeito ──────────────────────────────────── */
.vis-defect-desc {
    background: rgba(37, 99, 235, 0.15);
    border: 1px solid #60A5FA;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 15px;
    color: #FFFFFF;
    line-height: 1.6;
    margin-top: 10px;
}

/* ── Botão de Download PDF ─────────────────────────────────── */
.stDownloadButton > button {
    background: #1D4ED8 !important;
    color: #FFFFFF !important;
    border: 1px solid #60A5FA !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 0.75rem 1.8rem !important;
}
.stDownloadButton > button:hover {
    background: #2563EB !important;
    border-color: #93C5FD !important;
}

/* ── Seletor de Anomalia ───────────────────────────────────── */
.stSelectbox label {
    color: #FFFFFF !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}
.stSelectbox div[data-baseweb="select"] {
    background-color: #2B3D56 !important;
    border-color: #485D7C !important;
    font-size: 15px !important;
    border-radius: 8px !important;
}

/* ── Contador de Imagens (Pasta) ───────────────────────────── */
.vis-folder-counter {
    text-align: center;
    padding-top: 10px;
    font-size: 15px;
    color: #E2E8F0;
}
.vis-folder-counter strong { color: #FFFFFF; font-weight: 700; }
.vis-folder-counter code { color: #93C5FD; font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg",".png"}

DEFAULT_DATASET_ROOTS = [
    Path(r"C:\Users\Carolina.Freitas\Downloads\AeBAD\AeBAD\AeBAD_S"),
    Path(r"C:\Users\Carolina.Freitas\Downloads\AeBAD"),
    Path("AeBAD_S"),
    Path("ground_truth"),
    Path("sample_data"),
]

CHECKPOINT_PATHS = [
    r"models/Dinomaly_traced.pt",
    r"C:\Users\Carolina.Freitas\OneDrive - Instituto CCG ZGDV\Ambiente de Trabalho\hack\models\Dinomaly_traced.pt",
]

# ------------------------------------------------------------------
# Carregamento do Modelo
# ------------------------------------------------------------------
@st.cache_resource
def load_inspector(path: str) -> AeroInspector:
    return AeroInspector(checkpoint_path=path)

inspector = None
for cp in CHECKPOINT_PATHS:
    if Path(cp).is_file():
        try:
            inspector = load_inspector(cp)
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
        break

if inspector is None:
    st.warning("Modelo não encontrado. Coloque Dinomaly_traced.pt em models/")

# ------------------------------------------------------------------
# Classificador CNN de Tipo de Anomalia (cnn_tl_traced.pt)
# ------------------------------------------------------------------
CNN_CLASSIFIER_PATHS = [
    r"models/cnn_tl_traced.pt",
    r"C:\Users\Carolina.Freitas\OneDrive - Instituto CCG ZGDV\Ambiente de Trabalho\hack\cnn_tl_traced.pt",
]

# Ordem alfabética do ImageFolder: ablation=0, breakdown=1, fracture=2, groove=3
_CNN_CLASSES = ["ablation", "breakdown", "fracture", "groove"]

_clf_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_classifier(path: str):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(path, map_location=device)
    model.eval()
    return model, device


cnn_classifier = None
cnn_device = "cpu"
for _cp in CNN_CLASSIFIER_PATHS:
    if Path(_cp).is_file():
        try:
            cnn_classifier, cnn_device = load_classifier(_cp)
        except Exception:
            cnn_classifier = None
        break


def classify_defect(pil_image: Image.Image) -> tuple[str, float]:
    """Classifica o tipo de defeito com a CNN. Devolve (chave, confiança)."""
    if cnn_classifier is None:
        return None, 0.0
    tensor = _clf_transform(pil_image.convert("RGB")).unsqueeze(0).to(cnn_device)
    with torch.inference_mode():
        logits = cnn_classifier(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = int(probs.argmax())
    return _CNN_CLASSES[idx], float(probs[idx])


# ------------------------------------------------------------------
# Utilitários
# ------------------------------------------------------------------
def fmt_metric(value: float, as_pct: bool = True) -> str:
    if np.isnan(value):
        return "N/A"
    return f"{value:.1%}" if as_pct else f"{value:.4f}"


@st.cache_data
def build_gt_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for root in DEFAULT_DATASET_ROOTS:
        gt_dir = root / "ground_truth" if (root / "ground_truth").exists() else root
        if gt_dir.exists():
            for f in gt_dir.rglob("*"):
                if f.is_file() and not f.name.startswith(".") and not f.name.startswith("._"):
                    if f.suffix.lower() in IMAGE_EXTENSIONS:
                        nc = f.name.lower()
                        index[nc] = str(f)
                        index[f.stem.lower()] = str(f)
                        if "_mask" in nc:
                            index[nc.replace("_mask", "")] = str(f)
                            index[f.stem.lower().replace("_mask", "")] = str(f)
    return index


def find_gt_mask(image_name: str) -> Path | None:
    idx = build_gt_index()
    lookup = Path(image_name).name.lower()
    if lookup in idx:
        return Path(idx[lookup])
    stem = Path(image_name).stem.lower()
    if stem in idx:
        return Path(idx[stem])
    return None


def get_pixel_threshold_for(
    image_name: str,
    resized_map: np.ndarray,
    orig_size: tuple,
    auto: bool,
    manual_val: float,
) -> float:
    """Devolve o limiar de pixel a usar — automático ou manual."""
    if not auto:
        return manual_val
    gt = find_gt_mask(image_name)
    if gt is not None:
        try:
            gt_img = Image.open(gt).convert("L").resize(orig_size, Image.NEAREST)
            gt_msk = (np.asarray(gt_img) > 127).astype(np.uint8)
            return compute_optimal_pixel_threshold_gt(gt_msk, resized_map)
        except Exception:
            pass
    return compute_adaptive_pixel_threshold(resized_map)


# ------------------------------------------------------------------
# Componentes UI
# ------------------------------------------------------------------
def render_metrics_table(m: dict[str, float]):
    descriptions = {
        "pixel_auroc":           "Capacidade de distinguir pixels sãos de anomalias. 1.0 = separação perfeita.",
        "localization_at_5fpr":  "Precisão de localização com falsos alarmes limitados a 5%.",
        "pixel_f1":              "Equilíbrio entre capturar o defeito e evitar falsos positivos.",
        "iou":                   "Sobreposição exata entre a área detetada e a anotação real.",
        "precision":             "Dos pixels marcados como anómalos, qual a percentagem corresponde a defeito real.",
        "recall":                "Do defeito real anotado, qual a percentagem foi detetada pela IA.",
    }
    labels = [
        ("Pixel AUROC",                    "pixel_auroc"),
        ("Localization Precision @ FPR 5%", "localization_at_5fpr"),
        ("Pixel F1-Score (Dice)",           "pixel_f1"),
        ("IoU (Jaccard Index)",             "iou"),
        ("Precision",                       "precision"),
        ("Recall (Sensibilidade)",          "recall"),
    ]
    rows = "".join(
        f'<tr><td>{lbl}'
        f'<span title="{descriptions[k]}" class="help-bubble">?</span></td>'
        f'<td>{fmt_metric(m[k])}</td></tr>'
        for lbl, k in labels
    )
    st.markdown(
        f'<table class="vis-table">'
        f'<thead><tr><th>Métrica</th><th>Valor</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )


def render_verdict(is_anomalous: bool):
    """Veredito simples, direto e sem retângulo pesado."""
    if is_anomalous:
        st.markdown(
            '<div class="vis-simple-verdict-fail">● Peça defeituosa</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="vis-simple-verdict-ok">● Peça conforme</div>',
            unsafe_allow_html=True,
        )


def render_visual_diagnostic(result: dict, ncols: int = 3):
    st.markdown('<div class="vis-section-title">Diagnóstico Visual</div>', unsafe_allow_html=True)
    keys = ["heatmap", "overlay", "detections"]
    captions = ["Mapa de Anomalias", "Sobreposição Térmica", "Delimitação do Defeito"]
    if ncols == 4:
        keys = ["original"] + keys
        captions = ["Peça Original"] + captions
    cols = st.columns(ncols)
    for col, key, cap in zip(cols, keys, captions):
        with col:
            st.image(result["images"][key], caption=cap, use_container_width=True)


def render_gt_metrics(orig_np, orig_size, resized_map, pixel_threshold, image_name, key):
    gt_path = find_gt_mask(image_name)
    if gt_path is None:
        st.markdown(
            f'<div class="vis-info">Não foi encontrada máscara de referência para '
            f'<strong>{image_name}</strong>.</div>',
            unsafe_allow_html=True,
        )
        return None

    try:
        gt_img = Image.open(gt_path).convert("L").resize(orig_size, Image.NEAREST)
    except Exception:
        return None

    gt_np = (np.asarray(gt_img) > 127).astype(np.uint8)
    m = compute_single_image_metrics(
        gt_mask=gt_np, pred_map=resized_map, pixel_threshold=pixel_threshold)

    col_ov, col_tbl = st.columns([1, 2])
    with col_ov:
        ov = compute_gt_overlay(
            orig_np=orig_np, gt_mask=gt_np,
            pred_map=resized_map, pixel_threshold=pixel_threshold)
        st.image(ov, caption="Anotação real vs. Deteção da IA", use_container_width=True)
        st.markdown(
            '<div class="vis-legend">'
            '<span class="vis-legend-dot" style="background:#00E676"></span>'
            '<b>Verde</b> — Defeito detetado corretamente<br>'
            '<span class="vis-legend-dot" style="background:#00B0FF"></span>'
            '<b>Azul</b> — Defeito omitido (não previsto)<br>'
            '<span class="vis-legend-dot" style="background:#FF5252"></span>'
            '<b>Vermelho</b> — Falso alarme (inexistente)'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_tbl:
        render_metrics_table(m)

    return m


def render_report_section(image_name, is_anomalous, pixel_threshold, pred_map, metrics, key_prefix, pil_image=None):
    st.divider()
    st.markdown('<div class="vis-section-title">Relatório de Inspeção</div>', unsafe_allow_html=True)

    defect_labels = {
        "breakdown": "Breakdown", "ablation": "Ablation",
        "groove": "Groove",       "fracture": "Fracture",
    }

    col_sel, col_dl = st.columns([2, 1])
    with col_sel:
        if is_anomalous:
            # 1.ª tentativa — classificador CNN
            selected, confidence = None, 0.0
            if pil_image is not None and cnn_classifier is not None:
                with st.spinner("A identificar o tipo de anomalia..."):
                    selected, confidence = classify_defect(pil_image)

            # Recuo silencioso — heurística morfológica
            if selected is None:
                res = identify_defect_type(image_name, is_anomalous, pred_map, pixel_threshold)
                selected  = res[0] if isinstance(res, (tuple, list)) else res
                confidence = 0.0

            label_display = defect_labels.get(selected, selected)
            conf_str = f" &nbsp;<span style='color:#94A3B8;font-size:13px;'>({confidence:.0%})</span>" if confidence > 0 else ""
            st.markdown(
                f'<div style="color:#93C5FD;font-weight:700;font-size:15px;margin-bottom:8px;">'
                f'Tipo de anomalia detetado:&nbsp;'
                f'<span style="color:#FFFFFF;font-weight:800;">{label_display}</span>{conf_str}'
                f'</div>',
                unsafe_allow_html=True,
            )
            if selected in DEFECT_IMPACT:
                st.markdown(
                    f'<div class="vis-defect-desc">{DEFECT_IMPACT[selected]}</div>',
                    unsafe_allow_html=True,
                )
        else:
            selected = "none"

    pdf_bytes = generate_pdf_report(
        image_name=image_name, is_anomalous=is_anomalous,
        pixel_threshold=pixel_threshold, metrics=metrics,
        defect_type=selected if is_anomalous else "none",
    )
    with col_dl:
        st.write("")
        st.download_button(
            label="Descarregar PDF",
            data=pdf_bytes,
            file_name=f"relatorio_{Path(image_name).stem}.pdf",
            mime="application/pdf",
            key=f"dl_{key_prefix}",
        )


# ==================================================================
# INICIALIZAÇÃO DE ESTADO
# ==================================================================
if "input_mode" not in st.session_state:
    st.session_state["input_mode"] = "file"

if "best_pixel_threshold" not in st.session_state:
    st.session_state["best_pixel_threshold"] = 0.50

if "show_metrics_threshold" not in st.session_state:
    st.session_state["show_metrics_threshold"] = False

# ------------------------------------------------------------------
# Controle Dinâmico do Limiar na Barra Lateral:
# - No modo Câmara: aparece automaticamente se houver foto capturada
# - No modo Ficheiro: só aparece após clicar em "Calcular Métricas"
# ------------------------------------------------------------------
is_cam_mode = st.session_state.get("input_mode") == "camera"
has_cam_photo = (st.session_state.get("camera_input") is not None) or ("camera_fwd" in st.session_state)

if is_cam_mode:
    show_threshold_active = has_cam_photo
else:
    show_threshold_active = st.session_state.get("show_metrics_threshold", False)

# ==================================================================
# BARRA LATERAL (VISORA CENTRALIZADA)
# ==================================================================
with st.sidebar:
    # Ajustar a altura do espaçador: se o limiar estiver visível, puxa tudo para cima
    if show_threshold_active:
        spacer_h = "2vh"
    else:
        spacer_h = "10vh"
    st.markdown(f'<div style="height: {spacer_h};"></div>', unsafe_allow_html=True)

    # Logótipo VISORA e Animação Centralizados
    st.markdown("""
<div class="vis-sidebar-center-logo">
  <svg width="245" height="66" viewBox="0 0 245 66" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <style>
        .el{animation:eye-blink 8s ease-in-out infinite;transform-origin:46px 33px;}
        .ep{animation:eye-gaze 5s ease-in-out infinite;}
        .pa{animation:plane-glide 5s ease-in-out infinite;}
        @keyframes eye-blink{0%,42%,58%,100%{transform:scaleY(1);}50%{transform:scaleY(.04);}}
        @keyframes eye-gaze{0%{transform:translateX(0);}50%{transform:translateX(8px);}100%{transform:translateX(0);}}
        @keyframes plane-glide{0%{transform:translate(0,0);}50%{transform:translate(16px,-8px);}100%{transform:translate(0,0);}}
      </style>
      <linearGradient id="lg1" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#FFFFFF"/><stop offset="100%" stop-color="#93C5FD"/>
      </linearGradient>
    </defs>
    <g class="el">
      <path d="M6 33 Q46 10 86 33" stroke="url(#lg1)" stroke-width="2.6" stroke-linecap="round" fill="none"/>
      <path d="M6 33 Q46 56 86 33" stroke="url(#lg1)" stroke-width="2.6" stroke-linecap="round" fill="none"/>
    </g>
    <g class="ep">
      <circle cx="46" cy="33" r="13" stroke="#60A5FA" stroke-width="1.8" fill="#1C2D42"/>
      <circle cx="46" cy="33" r="6" fill="#3B82F6"/>
      <circle cx="48" cy="31" r="2.2" fill="#FFFFFF"/>
    </g>
    <line x1="60" y1="32" x2="150" y2="26" stroke="#60A5FA" stroke-width="1.3"
          stroke-dasharray="3.5 4" opacity=".7"/>
    <g class="pa">
      <g transform="translate(146,14) scale(.78)">
        <path d="M0 10 L16 10 L25 1 L30 1 L24 10 L40 10 L45 5 L48 5 L45 10 L52 10
                 Q56 10 54 12 Q52 14 45 14 L40 14 L24 23 L30 23 L25 14 L16 14 L0 14 Z"
              fill="#1C2D42" stroke="#93C5FD" stroke-width="1.5" stroke-linejoin="round"/>
      </g>
    </g>
  </svg>
  <div class="vis-brand">VIS<span class="vis-brand-dot">O</span>RA</div>
</div>
""", unsafe_allow_html=True)

    # O Limiar de Deteção fica alojado no interior da barra lateral
    if show_threshold_active:
        st.markdown(
            '<div class="vis-sidebar-section">'
            '<div class="vis-sidebar-label">Limiar de Deteção</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        pixel_thresh_val = float(st.session_state.get("best_pixel_threshold", 0.50))
        current_val = float(np.clip(pixel_thresh_val, 0.10, 0.90))

        if is_cam_mode:
            # ── MODO CÂMARA: Apenas ajuste manual (slider), sem "Automático", sem badge e sem toggle
            cam_k = st.session_state.get("camera_last", "def")
            pixel_threshold = st.slider(
                label="Limiar",
                min_value=0.10,
                max_value=0.90,
                value=current_val,
                step=0.01,
                key=f"cam_sl_{cam_k}",
                label_visibility="collapsed",
                help="Ajuste fino da sensibilidade de deteção ao nível do pixel.",
            )
            auto_mode = False
            st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        else:
            # ── MODO FICHEIRO: Mantém 'Automático (?)', '[ 0.xx ]' e botão toggle
            col_lbl, col_val, col_swt = st.columns([10, 6, 5])
            with col_swt:
                auto_mode = st.toggle(
                    label="",
                    value=True,
                    key="auto_threshold",
                    label_visibility="collapsed",
                )
            with col_lbl:
                st.markdown(
                    '<div class="vis-auto-row">'
                    '<span class="vis-auto-title">Automático</span>'
                    '<span class="vis-toggle-info" title="Calcula o limiar de pixel ideal para cada imagem. Desative para ajuste manual.">?</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with col_val:
                if auto_mode:
                    st.markdown(
                        f'<span class="vis-side-val-badge">{pixel_thresh_val:.2f}</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.write("")

            if not auto_mode:
                st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
                file_k = st.session_state.get("single_last_file", "def")
                pixel_threshold = st.slider(
                    "Valor do limiar",
                    min_value=0.10,
                    max_value=0.90,
                    value=current_val,
                    step=0.05,
                    key=f"file_sl_{file_k}",
                    help="Ajuste fino da sensibilidade de deteção ao nível do pixel.",
                )
            else:
                pixel_threshold = pixel_thresh_val

            st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    else:
        auto_mode = True
        pixel_threshold = float(st.session_state.get("best_pixel_threshold", 0.50))


# ==================================================================
# CABEÇALHO PRINCIPAL
# ==================================================================
st.markdown(
    '<div class="vis-header">'
    '<div class="vis-title">Inspeção <span class="vis-title-blue">Inteligente de Peças</span></div>'
    '<div class="vis-subtitle">Análise não destrutiva por visão computacional</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ==================================================================
# SELETOR DE MODO (Câmara / Carregar Ficheiro) 
# ==================================================================
col_mode_cam, col_mode_file = st.columns(2)

with col_mode_cam:
    cam_active = st.session_state["input_mode"] == "camera"
    if st.button(
        "Câmara",
        key="btn_mode_cam",
        use_container_width=True,
        type="primary" if cam_active else "secondary",
    ):
        st.session_state["input_mode"] = "camera"
        st.session_state["show_metrics_threshold"] = False
        st.rerun()

with col_mode_file:
    file_active = st.session_state["input_mode"] == "file"
    if st.button(
        "Carregar Ficheiro",
        key="btn_mode_file",
        use_container_width=True,
        type="primary" if file_active else "secondary",
    ):
        st.session_state["input_mode"] = "file"
        st.session_state["show_metrics_threshold"] = False
        st.session_state["single_metrics_active"] = False
        st.session_state.pop("camera_fwd", None)
        st.session_state.pop("camera_last", None)
        st.rerun()

st.markdown('<hr style="margin: 16px 0 22px 0;">', unsafe_allow_html=True)


# ==================================================================
# MODO — CÂMARA (Inspeção Imediata e Automática)
# ==================================================================
if st.session_state["input_mode"] == "camera":
    cam_img = st.camera_input(label="", key="camera_input", label_visibility="collapsed")

    if cam_img is not None and inspector is not None:
        # Gerar identificador único baseado no conteúdo da foto capturada
        img_bytes = cam_img.getvalue()
        file_id = f"cam_{hashlib.md5(img_bytes).hexdigest()}"
        input_image = Image.open(cam_img)

        file_changed = st.session_state.get("camera_last") != file_id
        if file_changed or "camera_fwd" not in st.session_state:
            st.session_state["camera_last"] = file_id
            with st.spinner("A analisar a estrutura da peça..."):
                raw_score, resized_map, inf_time, orig_size, orig_np = inspector.forward_pass(input_image)
                thresh = compute_adaptive_pixel_threshold(resized_map)
                st.session_state["best_pixel_threshold"] = thresh

                st.session_state["camera_fwd"] = {
                    "raw_score": raw_score, "resized_map": resized_map,
                    "inf_time": inf_time,   "orig_size":  orig_size,
                    "orig_np":  orig_np,    "thresh":     thresh,
                }
            st.rerun()

        if "camera_fwd" in st.session_state:
            c = st.session_state["camera_fwd"]
            # No modo câmara o limiar é controlado pelo slider manual
            thresh = pixel_threshold
            result = inspector.apply_thresholds(
                raw_score=c["raw_score"], resized_map=c["resized_map"],
                orig_np=c["orig_np"],    pixel_threshold=thresh,
                inference_time_ms=c["inf_time"],
            )

            render_verdict(result["is_anomalous"])
            render_visual_diagnostic(result, ncols=3)

            st.markdown(
                '<div class="vis-info" style="margin-top:16px;">'
                'Para capturas em tempo real com câmara não existe máscara de referência. '
                'O diagnóstico visual e o veredito encontram-se disponíveis.'
                '</div>',
                unsafe_allow_html=True,
            )
            render_report_section(
                image_name="captura_camara.jpg",
                is_anomalous=result["is_anomalous"],
                pixel_threshold=thresh,
                pred_map=c["resized_map"],
                metrics=None,
                key_prefix="camera",
                pil_image=input_image,
            )
    else:
        st.session_state["show_metrics_threshold"] = False
        st.session_state.pop("camera_fwd", None)
        st.session_state.pop("camera_last", None)


# ==================================================================
# MODO — FICHEIRO (1 ficheiro individual ou vários ficheiros em pasta)
# Inspeção Imediata e Automática após o Upload
# ==================================================================
else:
    uploaded = st.file_uploader(
        label="",
        type=["png", "jpg"],
        accept_multiple_files=True,
        key="file_upload_main",
        label_visibility="collapsed",
    )

    if uploaded:
        uploaded = [f for f in uploaded
                    if not f.name.startswith(".") and not f.name.startswith("._")]

    # ── UMA ÚNICA IMAGEM ─────────────────────────────────────────
    if uploaded and len(uploaded) == 1:
        uf          = uploaded[0]
        file_id     = getattr(uf, "file_id", None) or f"{uf.name}_{getattr(uf,'size',0)}"
        input_image = Image.open(uf)

        file_changed = st.session_state.get("single_last_file") != file_id
        if file_changed or "single_fwd" not in st.session_state:
            st.session_state["single_metrics_active"] = False
            st.session_state["show_metrics_threshold"] = False
            with st.spinner("A analisar a estrutura da peça..."):
                raw_score, resized_map, inf_time, orig_size, orig_np = inspector.forward_pass(input_image)
                thresh = get_pixel_threshold_for(uf.name, resized_map, orig_size, auto=True, manual_val=0.50)
                st.session_state["best_pixel_threshold"] = thresh
                st.session_state["single_fwd"] = {
                    "raw_score": raw_score, "resized_map": resized_map,
                    "inf_time": inf_time,   "orig_size":  orig_size,
                    "orig_np":  orig_np,    "thresh":     thresh,
                }
                st.session_state["single_last_file"] = file_id

        if ("single_fwd" in st.session_state
                and st.session_state.get("single_last_file") == file_id):
            c      = st.session_state["single_fwd"]
            thresh = c["thresh"] if auto_mode else pixel_threshold
            result = inspector.apply_thresholds(
                raw_score=c["raw_score"], resized_map=c["resized_map"],
                orig_np=c["orig_np"],    pixel_threshold=thresh,
                inference_time_ms=c["inf_time"],
            )

            # Veredito suave
            render_verdict(result["is_anomalous"])

            # Diagnóstico Visual
            render_visual_diagnostic(result, ncols=3)

            # Métricas de Validação
            st.divider()
            st.markdown('<div class="vis-section-title">Métricas de Validação</div>',
                        unsafe_allow_html=True)
            if st.button("Calcular Métricas", key="calc_single"):
                st.session_state["single_metrics_active"] = True
                st.session_state["show_metrics_threshold"] = True
                st.rerun()

            if st.session_state.get("single_metrics_active"):
                m = render_gt_metrics(
                    orig_np=c["orig_np"], orig_size=c["orig_size"],
                    resized_map=c["resized_map"], pixel_threshold=thresh,
                    image_name=uf.name, key="single",
                )
                if m:
                    st.session_state[f"single_metrics_data_{uf.name}"] = m

            # Relatório de Inspeção
            metrics_dict = st.session_state.get(f"single_metrics_data_{uf.name}")
            render_report_section(
                image_name=uf.name, is_anomalous=result["is_anomalous"],
                pixel_threshold=thresh, pred_map=c["resized_map"],
                metrics=metrics_dict, key_prefix="single",
                pil_image=input_image,
            )

    # ── MÚLTIPLAS IMAGENS (Navegação por pasta integrada) ────────
    elif uploaded and len(uploaded) > 1:
        all_images = sorted(uploaded, key=lambda x: x.name)

        if "folder_idx" not in st.session_state:
            st.session_state["folder_idx"] = 0

        max_idx = len(all_images) - 1
        idx = min(st.session_state["folder_idx"], max_idx)
        st.session_state["folder_idx"] = idx

        # Barra de Navegação entre ficheiros
        col_prev, col_info, col_next = st.columns([1, 4, 1])
        with col_prev:
            if st.button("Anterior", use_container_width=True, key="prev_img", disabled=(idx <= 0)):
                st.session_state["folder_idx"] = max(0, idx - 1)
                st.session_state["show_metrics_threshold"] = False
                st.rerun()
        with col_info:
            st.markdown(
                f'<div class="vis-folder-counter">'
                f'Imagem <strong>{idx + 1}</strong> de <strong>{len(all_images)}</strong>'
                f'&ensp;&middot;&ensp;<code>{all_images[idx].name}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_next:
            if st.button("Seguinte", use_container_width=True, key="next_img", disabled=(idx >= max_idx)):
                st.session_state["folder_idx"] = min(max_idx, idx + 1)
                st.session_state["show_metrics_threshold"] = False
                st.rerun()

        new_idx = st.slider("", 0, max_idx, idx, key="folder_slider", label_visibility="collapsed")
        if new_idx != idx:
            st.session_state["folder_idx"] = new_idx
            st.session_state["show_metrics_threshold"] = False
            st.rerun()

        current_file = all_images[st.session_state["folder_idx"]]
        try:
            current_image = Image.open(current_file).convert("RGB")
        except Exception as e:
            st.error(f"Erro ao abrir a imagem: {e}")
            st.stop()

        if inspector is None:
            st.stop()

        fwd_key = f"folder_fwd_{current_file.name}"
        if fwd_key not in st.session_state:
            with st.spinner("A analisar a estrutura da peça..."):
                rs, rm, it, os_, on_ = inspector.forward_pass(current_image)
                thresh = get_pixel_threshold_for(current_file.name, rm, os_, auto=True, manual_val=0.50)
                st.session_state["best_pixel_threshold"] = thresh
                st.session_state[fwd_key] = {
                    "raw_score": rs, "resized_map": rm, "inf_time": it,
                    "orig_size": os_, "orig_np": on_, "thresh": thresh,
                }

        cf     = st.session_state[fwd_key]
        thresh = cf["thresh"] if auto_mode else pixel_threshold
        result = inspector.apply_thresholds(
            raw_score=cf["raw_score"], resized_map=cf["resized_map"],
            orig_np=cf["orig_np"],    pixel_threshold=thresh,
            inference_time_ms=cf["inf_time"],
        )

        render_verdict(result["is_anomalous"])
        render_visual_diagnostic(result, ncols=4)

        # Métricas de Validação
        st.divider()
        st.markdown('<div class="vis-section-title">Métricas de Validação</div>',
                    unsafe_allow_html=True)
        metrics_active_key = f"folder_metrics_active_{current_file.name}"
        if st.button("Calcular Métricas", key=f"calc_folder_{current_file.name}"):
            st.session_state[metrics_active_key] = True
            st.session_state["show_metrics_threshold"] = True
            st.rerun()

        if st.session_state.get(metrics_active_key):
            m_f = render_gt_metrics(
                orig_np=cf["orig_np"], orig_size=cf["orig_size"],
                resized_map=cf["resized_map"], pixel_threshold=thresh,
                image_name=current_file.name, key=f"folder_{current_file.name}",
            )
            if m_f:
                st.session_state[f"folder_metrics_data_{current_file.name}"] = m_f

        folder_metrics = st.session_state.get(f"folder_metrics_data_{current_file.name}")
        render_report_section(
            image_name=current_file.name, is_anomalous=result["is_anomalous"],
            pixel_threshold=thresh, pred_map=cf["resized_map"],
            metrics=folder_metrics, key_prefix=f"folder_{current_file.name}",
            pil_image=current_image,
        )
    else:
        st.session_state["show_metrics_threshold"] = False
