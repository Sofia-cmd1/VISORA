"""
engine.py - Módulo de inferência aeroespacial para Dinomaly (TorchScript)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch
from PIL import Image


class AeroInspector:
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Modelo não encontrado em: {self.checkpoint_path}")

        print(f"[AeroInspector] Carregando modelo em {self.device}...")
        self.model = torch.jit.load(str(self.checkpoint_path), map_location=self.device)
        self.model.eval()
        print("[AeroInspector] Modelo carregado com sucesso!")

    def _prepare_image(self, pil_image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        image_rgb = pil_image.convert("RGB")
        orig_w, orig_h = image_rgb.size

        img_array = np.asarray(image_rgb, dtype=np.float32)
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0) / 255.0
        return tensor.to(self.device), (orig_w, orig_h)

    def _normalize_heatmap(self, anomaly_map: np.ndarray) -> np.ndarray:
        amin, amax = float(anomaly_map.min()), float(anomaly_map.max())
        if amax - amin < 1e-8:
            return np.zeros_like(anomaly_map, dtype=np.float32)
        return (anomaly_map - amin) / (amax - amin)

    def forward_pass(self, pil_image: Image.Image) -> Tuple[float, np.ndarray, float, Tuple[int, int], np.ndarray]:
        """
        Executa apenas o forward do modelo neural e retorna:
        (raw_score, resized_normalized_map, inference_time_ms, (orig_w, orig_h), orig_np)
        """
        tensor, (orig_w, orig_h) = self._prepare_image(pil_image)
        orig_np = np.asarray(pil_image.convert("RGB"))

        t_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(tensor)
        inference_time_ms = (time.perf_counter() - t_start) * 1000.0

        raw_score = None
        raw_map = None

        score_candidates = []
        map_candidates = []

        for item in outputs:
            if not isinstance(item, torch.Tensor):
                continue

            # Garante que não é booleano (pred_label = True vira 1.0 se não for filtrado)
            if item.dtype == torch.bool:
                continue

            item_np = item.detach().cpu().numpy()

            # Score deve ser tensor de ponto flutuante escalar ou 1D
            if item.is_floating_point() and (item.ndim <= 1 or item.numel() == 1):
                val = float(item_np.reshape(-1)[0])
                score_candidates.append(val)
            elif item.is_floating_point() and item.ndim >= 2:
                sq = np.squeeze(item_np)
                if sq.ndim == 2:
                    map_candidates.append(sq)

        if map_candidates:
            raw_map = map_candidates[0]
        else:
            raise RuntimeError("Não foi possível extrair o anomaly map do modelo.")

        norm_map = self._normalize_heatmap(raw_map)
        resized_map = cv2.resize(norm_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Se encontrou um score float genuíno e não-trivial (diferente de 1.0 estático de label)
        if score_candidates:
            non_ones = [s for s in score_candidates if s < 0.99999]
            if non_ones:
                raw_score = non_ones[0]
            else:
                raw_score = score_candidates[0]

        # Se o score do modelo vier saturado ou nulo, calculamos o score contínuo real da anomalia
        # baseado na intensidade dos percentis mais altos da anomalia (Top 1% de ativação)
        if raw_score is None or raw_score >= 0.9999:
            raw_score = float(np.percentile(resized_map, 99.5))

        return raw_score, resized_map, inference_time_ms, (orig_w, orig_h), orig_np

    def apply_thresholds(
        self,
        raw_score: float,
        resized_map: np.ndarray,
        orig_np: np.ndarray,
        pixel_threshold: float,
        inference_time_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Aplica o limiar de pixel em tempo real sobre o mapa de anomalia,
        recalculando dinamicamente a máscara, contornos e o veredito.
        """
        orig_h, orig_w = orig_np.shape[:2]

        # 1. Geração do Heatmap colorido (Jet Colormap)
        heatmap_colored = cv2.applyColorMap(np.uint8(resized_map * 255), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # 2. Geração da máscara de defeito usando o pixel_threshold atual
        mask = resized_map >= pixel_threshold

        # 3. Overlay com realce vermelho no defeito
        overlay = orig_np.copy()
        overlay[mask] = (0.60 * overlay[mask] + 0.40 * np.array([255, 30, 30])).astype(np.uint8)

        # 4. Deteção de Contornos e Bounding Boxes
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detection_view = orig_np.copy()

        defect_details = []
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 10:  # Ignora micro-ruídos isolados
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            defect_details.append({"id": idx + 1, "area_px": int(area), "bbox": (x, y, w, h)})
            cv2.rectangle(detection_view, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 5. O veredito é determinado pela existência de regiões anómalas confirmadas
        is_anomalous = len(defect_details) > 0

        # Severidade padronizada com teto claro de 0.0 a 10.0
        peak_intensity = float(resized_map.max())
        total_defect_px = sum(d["area_px"] for d in defect_details)
        extent_factor = min(1.0, total_defect_px / 1500.0) if defect_details else 0.0
        severity_score = (peak_intensity * 6.0) + (extent_factor * 4.0) if is_anomalous else (peak_intensity * 2.0)
        severity_score = round(float(np.clip(severity_score, 0.0, 10.0)), 2)

        return {
            "is_anomalous": is_anomalous,
            "anomaly_score": raw_score,
            "pixel_threshold_used": pixel_threshold,
            "severity_score": severity_score,
            "inference_time_ms": round(inference_time_ms, 2),
            "defect_regions_count": len(defect_details),
            "defects_list": defect_details,
            "images": {
                "original": Image.fromarray(orig_np),
                "heatmap": Image.fromarray(heatmap_rgb),
                "overlay": Image.fromarray(overlay),
                "detections": Image.fromarray(detection_view),
                "mask": Image.fromarray((mask * 255).astype(np.uint8)),
            },
        }

    def inspect(
        self,
        pil_image: Image.Image,
        pixel_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Executa a inspeção completa."""
        raw_score, resized_map, inf_time, _, orig_np = self.forward_pass(pil_image)
        return self.apply_thresholds(
            raw_score=raw_score,
            resized_map=resized_map,
            orig_np=orig_np,
            pixel_threshold=pixel_threshold,
            inference_time_ms=inf_time,
        )
