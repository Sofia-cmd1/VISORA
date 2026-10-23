"""
metrics.py - Cálculo de Métricas de Deteção de Anomalias
Compatível com: AeBAD-S, VisA, MVTec AD
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve


# ------------------------------------------------------------------
# Image-Level Metrics
# ------------------------------------------------------------------

def compute_image_auroc(gt_labels: list[int], pred_scores: list[float]) -> float:
    """AUROC ao nível da imagem."""
    if len(set(gt_labels)) < 2:
        return float("nan")
    return float(roc_auc_score(gt_labels, pred_scores))


def compute_best_f1_and_threshold(
    gt_labels: list[int], pred_scores: list[float]
) -> tuple[float, float]:
    """
    Encontra o limiar de score que maximiza o F1-Score.
    Retorna (melhor_f1, melhor_threshold).
    """
    if len(set(gt_labels)) < 2:
        return float("nan"), float("nan")
    precisions, recalls, thresholds = precision_recall_curve(gt_labels, pred_scores)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(f1_scores[best_idx]), float(thresholds[best_idx])


def compute_f1_at_threshold(
    gt_labels: list[int], pred_scores: list[float], threshold: float
) -> float:
    """F1-Score dado um limiar de decisão fixo."""
    preds = [1 if s >= threshold else 0 for s in pred_scores]
    return float(f1_score(gt_labels, preds, zero_division=0))


# ------------------------------------------------------------------
# Pixel-Level Metrics
# ------------------------------------------------------------------

def compute_pixel_auroc(
    gt_masks: list[np.ndarray], pred_maps: list[np.ndarray]
) -> float:
    """AUROC ao nível do pixel."""
    all_gt = np.concatenate([m.flatten().astype(np.uint8) for m in gt_masks])
    all_pred = np.concatenate([m.flatten().astype(np.float32) for m in pred_maps])
    if len(set(all_gt.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(all_gt, all_pred))


def compute_localization_at_fpr(
    gt_masks: list[np.ndarray],
    pred_maps: list[np.ndarray],
    fpr_limit: float = 0.05,
) -> float:
    """
    TPR (recall de pixel) quando a taxa de falsos positivos é <= fpr_limit.
    Padrão industrial: fpr_limit=0.05 (5%).
    """
    all_gt = np.concatenate([m.flatten().astype(np.uint8) for m in gt_masks])
    all_pred = np.concatenate([m.flatten().astype(np.float32) for m in pred_maps])
    if len(set(all_gt.tolist())) < 2:
        return float("nan")
    fprs, tprs, _ = roc_curve(all_gt, all_pred)
    idx = np.searchsorted(fprs, fpr_limit, side="right") - 1
    idx = max(0, min(idx, len(tprs) - 1))
    return float(tprs[idx])


def compute_pro_score(
    gt_masks: list[np.ndarray],
    pred_maps: list[np.ndarray],
    fpr_limit: float = 0.30,
    num_points: int = 100,
) -> float:
    """
    Per-Region Overlap (PRO) Score.
    Avalia o overlap médio por região de defeito conectada,
    integrado até fpr_limit (padrão: 30%).
    """
    try:
        import cv2
    except ImportError:
        return float("nan")

    all_gt = np.concatenate([m.flatten().astype(np.uint8) for m in gt_masks])
    all_pred = np.concatenate([m.flatten().astype(np.float32) for m in pred_maps])

    if len(set(all_gt.tolist())) < 2:
        return float("nan")

    _, _, thresholds_global = roc_curve(all_gt, all_pred)
    step = max(1, len(thresholds_global) // num_points)
    thresholds_sample = thresholds_global[::-1][::step]

    pro_values = []
    fpr_values = []

    for thresh in thresholds_sample:
        overlaps = []
        total_normal_pixels = 0
        total_fp = 0

        for gt_mask, pred_map in zip(gt_masks, pred_maps):
            pred_binary = (pred_map >= thresh).astype(np.uint8)
            gt_binary = gt_mask.astype(np.uint8)

            normal_pixels = int((gt_binary == 0).sum())
            fp_pixels = int(((pred_binary == 1) & (gt_binary == 0)).sum())
            total_normal_pixels += normal_pixels
            total_fp += fp_pixels

            n_labels, labels = cv2.connectedComponents(gt_binary)
            for label_id in range(1, n_labels):
                region = labels == label_id
                if region.sum() == 0:
                    continue
                overlap = float(pred_binary[region].sum()) / float(region.sum())
                overlaps.append(overlap)

        fpr = total_fp / (total_normal_pixels + 1e-8)
        if fpr > fpr_limit:
            break

        fpr_values.append(fpr)
        pro_values.append(np.mean(overlaps) if overlaps else 0.0)

    if len(pro_values) < 2:
        return float("nan")

    return float(np.trapz(pro_values, fpr_values) / fpr_limit)


# ------------------------------------------------------------------
# Defect Severity Score
# ------------------------------------------------------------------

def compute_defect_severity_score(
    pred_map_normalized: np.ndarray,
    pixel_threshold: float = 0.50,
) -> dict:
    """
    Defect Severity Score (DSS) = defect_area_ratio (%) x max_anomaly_intensity
    """
    mask = pred_map_normalized >= pixel_threshold
    total_pixels = pred_map_normalized.size
    defect_pixels = int(mask.sum())
    area_ratio = (defect_pixels / total_pixels) * 100.0
    max_intensity = float(pred_map_normalized.max())
    dss = area_ratio * max_intensity

    if dss < 0.5:
        level = "Conforme"
    elif dss < 2.0:
        level = "Alerta"
    else:
        level = "Rejeição"

    return {
        "dss": round(dss, 4),
        "area_ratio": round(area_ratio, 3),
        "max_intensity": round(max_intensity, 4),
        "level": level,
    }


# ------------------------------------------------------------------
# Single Image Evaluation (Ultra-Rápido)
# ------------------------------------------------------------------

def compute_single_image_metrics(
    gt_mask: np.ndarray,
    pred_map: np.ndarray,
    pixel_threshold: float = 0.50,
) -> dict[str, float]:
    """
    Calcula instantaneamente as métricas de validação para uma única imagem
    comparando a sua previsão com a sua máscara de Ground Truth.
    """
    gt_binary = (gt_mask > 0).astype(np.uint8)
    pred_binary = (pred_map >= pixel_threshold).astype(np.uint8)

    has_defect = bool(gt_binary.sum() > 0)

    # 1. Pixel AUROC da imagem
    px_auroc = compute_pixel_auroc([gt_binary], [pred_map]) if has_defect else float("nan")

    # 2. PRO Score (Per-Region Overlap)
    pro = compute_pro_score([gt_binary], [pred_map]) if has_defect else float("nan")

    # 3. Localization @ FPR 5%
    loc_5fpr = compute_localization_at_fpr([gt_binary], [pred_map], fpr_limit=0.05) if has_defect else float("nan")

    # 4. Métricas de segmentação direta (IoU, Dice/F1, Precision, Recall)
    intersection = float((pred_binary & gt_binary).sum())
    pred_total = float(pred_binary.sum())
    gt_total = float(gt_binary.sum())
    union = float((pred_binary | gt_binary).sum())

    iou = (intersection / union) if union > 0 else (1.0 if not has_defect and pred_total == 0 else 0.0)
    dice_f1 = (2.0 * intersection / (pred_total + gt_total)) if (pred_total + gt_total) > 0 else (1.0 if not has_defect and pred_total == 0 else 0.0)
    precision = (intersection / pred_total) if pred_total > 0 else 0.0
    recall = (intersection / gt_total) if gt_total > 0 else (1.0 if not has_defect else 0.0)

    return {
        "pixel_auroc": px_auroc,
        "pro_score": pro,
        "localization_at_5fpr": loc_5fpr,
        "iou": iou,
        "pixel_f1": dice_f1,
        "precision": precision,
        "recall": recall,
    }


def compute_adaptive_pixel_threshold(pred_map: np.ndarray) -> float:
    """
    Calcula matematicamente o limiar de pixel ótimo e específico para a imagem pelo método de Otsu,
    que minimiza a variância intra-classe entre o material são e a anomalia.
    """
    import cv2
    gray_8u = np.uint8(np.clip(pred_map * 255.0, 0, 255))
    otsu_val, _ = cv2.threshold(gray_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = float(otsu_val) / 255.0
    return float(np.clip(thresh, 0.20, 0.85))


def compute_optimal_pixel_threshold_gt(gt_mask: np.ndarray, pred_map: np.ndarray) -> float:
    """
    Encontra o limiar de pixel exato que maximiza o F1-Score (Dice) em relação ao Ground Truth.
    """
    from sklearn.metrics import precision_recall_curve
    gt_flat = (gt_mask > 0).astype(np.uint8).flatten()
    pred_flat = pred_map.flatten()
    if gt_flat.sum() == 0:
        return compute_adaptive_pixel_threshold(pred_map)
    precisions, recalls, thresholds = precision_recall_curve(gt_flat, pred_flat)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores[:-1])) if len(f1_scores) > 1 else 0
    return float(np.clip(thresholds[best_idx], 0.15, 0.90))


def compute_gt_overlay(
    orig_np: np.ndarray,
    gt_mask: np.ndarray,
    pred_map: np.ndarray,
    pixel_threshold: float,
    alpha: float = 0.50,
) -> np.ndarray:
    """
    Gera a sobreposição entre a imagem original, o Ground Truth real e a anomalia prevista pela IA:
    - Verde: anomalia prevista corretamente (Verdadeiro Positivo / TP)
    - Azul: parte do defeito real não prevista pelo modelo (Falso Negativo / FN)
    - Vermelho: defeitos inexistentes previstos pelo modelo (Falso Positivo / FP)
    """
    import cv2

    if orig_np.ndim == 2:
        base_rgb = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
    elif orig_np.shape[2] == 4:
        base_rgb = cv2.cvtColor(orig_np, cv2.COLOR_RGBA2RGB)
    else:
        base_rgb = orig_np.copy()

    orig_h, orig_w = base_rgb.shape[:2]

    # Assegurar dimensões coerentes
    if gt_mask.shape[:2] != (orig_h, orig_w):
        gt_mask = cv2.resize(gt_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    if pred_map.shape[:2] != (orig_h, orig_w):
        pred_map = cv2.resize(pred_map.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    gt_binary = (gt_mask > 0)
    pred_binary = (pred_map >= pixel_threshold)

    # Classificação de regiões
    tp = pred_binary & gt_binary        # Verde: deteção correta
    fn = (~pred_binary) & gt_binary      # Azul: defeito omitido
    fp = pred_binary & (~gt_binary)      # Vermelho: falso alarme

    overlay = base_rgb.copy()

    COLOR_TP = np.array([0, 230, 90], dtype=np.float32)   # Verde
    COLOR_FN = np.array([0, 150, 255], dtype=np.float32)  # Azul
    COLOR_FP = np.array([255, 40, 40], dtype=np.float32)  # Vermelho

    if np.any(tp):
        overlay[tp] = np.clip((1.0 - alpha) * overlay[tp] + alpha * COLOR_TP, 0, 255).astype(np.uint8)
    if np.any(fn):
        overlay[fn] = np.clip((1.0 - alpha) * overlay[fn] + alpha * COLOR_FN, 0, 255).astype(np.uint8)
    if np.any(fp):
        overlay[fp] = np.clip((1.0 - alpha) * overlay[fp] + alpha * COLOR_FP, 0, 255).astype(np.uint8)

    # Contornos nítidos para delimitação cirúrgica
    def draw_mask_contours(img: np.ndarray, mask_bool: np.ndarray, color_rgb: tuple[int, int, int], thickness: int = 2):
        if not np.any(mask_bool):
            return
        cnts, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, color_rgb, thickness, cv2.LINE_AA)

    draw_mask_contours(overlay, fn, (0, 130, 255), thickness=2)
    draw_mask_contours(overlay, fp, (255, 30, 30), thickness=2)
    draw_mask_contours(overlay, tp, (0, 255, 90), thickness=2)

    return overlay



# ------------------------------------------------------------------
# Batch Runner — processa um dataset completo
# ------------------------------------------------------------------

def evaluate_dataset(
    inspector,
    image_paths: list,
    gt_label_list: list[int],
    gt_mask_paths: list | None = None,
    score_threshold: float = 0.65,
    pixel_threshold: float = 0.50,
) -> dict:
    from PIL import Image
    import numpy as np

    pred_scores = []
    pred_maps = []
    gt_masks = []
    per_image_results = []

    for i, img_path in enumerate(image_paths):
        pil_img = Image.open(img_path).convert("RGB")
        result = inspector.inspect(pil_img, score_threshold=score_threshold, pixel_threshold=pixel_threshold)

        pred_scores.append(result["anomaly_score"])

        hm_gray = np.asarray(result["images"]["heatmap"].convert("L"), dtype=np.float32) / 255.0
        pred_maps.append(hm_gray)

        if gt_mask_paths and i < len(gt_mask_paths) and gt_mask_paths[i] is not None:
            mask_pil = Image.open(gt_mask_paths[i]).convert("L")
            mask_pil = mask_pil.resize(pil_img.size, Image.NEAREST)
            mask_np = (np.asarray(mask_pil) > 127).astype(np.uint8)
        else:
            w, h = pil_img.size
            mask_np = np.zeros((h, w), dtype=np.uint8)
        gt_masks.append(mask_np)

        dss_info = compute_defect_severity_score(hm_gray, pixel_threshold)

        per_image_results.append({
            "image": str(img_path),
            "gt_label": gt_label_list[i],
            "pred_score": round(result["anomaly_score"], 4),
            "pred_label": int(result["is_anomalous"]),
            "correct": int(result["is_anomalous"]) == gt_label_list[i],
            "dss": dss_info["dss"],
            "dss_level": dss_info["level"],
            "defect_area_pct": dss_info["area_ratio"],
            "inference_ms": result["inference_time_ms"],
        })

    has_pixel_gt = any(m.sum() > 0 for m in gt_masks)

    image_auroc = compute_image_auroc(gt_label_list, pred_scores)
    best_f1, best_thresh = compute_best_f1_and_threshold(gt_label_list, pred_scores)
    f1_at_current = compute_f1_at_threshold(gt_label_list, pred_scores, score_threshold)
    pixel_auroc = compute_pixel_auroc(gt_masks, pred_maps) if has_pixel_gt else float("nan")
    loc_at_5fpr = compute_localization_at_fpr(gt_masks, pred_maps, fpr_limit=0.05) if has_pixel_gt else float("nan")
    pro = compute_pro_score(gt_masks, pred_maps) if has_pixel_gt else float("nan")

    avg_inference = float(np.mean([r["inference_ms"] for r in per_image_results]))

    return {
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "best_f1": best_f1,
        "best_threshold": best_thresh,
        "f1_at_current_threshold": f1_at_current,
        "pro_score": pro,
        "localization_at_5fpr": loc_at_5fpr,
        "avg_inference_ms": round(avg_inference, 2),
        "n_images": len(image_paths),
        "per_image": per_image_results,
    }
