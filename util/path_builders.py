from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List


class PathBuilder:
    def __init__(self, base_dir: str, model_path: str, replace_path: str = "") -> None:
        self.base_dir = base_dir
        self.model_path = model_path
        self.replace_path = replace_path or ""

    @property
    def model_name(self) -> str:
        return os.path.basename(self.model_path.rstrip("/\\"))

    def model_output_dir(self) -> str:
        return os.path.join(self.base_dir, self.model_name)

    def attention_dir(self) -> str:
        return os.path.join(self.model_output_dir(), "output_attentions")

    def images_dir(self) -> str:
        return os.path.join(self.model_output_dir(), "images")

    def results_dir(self) -> str:
        return os.path.join(self.model_output_dir(), "results")

    def result_json_path(self) -> str:
        return os.path.join(self.results_dir(), "result.json")

    def image_relative_path(self, image_path: str) -> str:
        if self.replace_path:
            try:
                rel = Path(image_path).resolve().relative_to(Path(self.replace_path).resolve())
                return rel.as_posix()
            except Exception:
                cleaned = image_path.replace(self.replace_path, "")
                return cleaned.lstrip("\\/").replace("\\", "/")
        return image_path.lstrip("\\/").replace("\\", "/")

    def attention_file_id(self, image_relative_path: str) -> str:
        normalized = image_relative_path.replace("\\", "/").lstrip("/")
        file_id = normalized.replace("/", "__")
        return file_id.replace(":", "")

    def attention_file_path(self, image_relative_path: str) -> str:
        file_id = self.attention_file_id(image_relative_path)
        return os.path.join(self.attention_dir(), f"{file_id}.pth")

    def pth_file_paths(self, file_names: Iterable[str]) -> List[str]:
        return [
            os.path.join(self.attention_dir(), file_name)
            for file_name in file_names
            if file_name.endswith(".pth")
        ]

    def visualization_base_path(self, image_relative_path: str) -> str:
        return os.path.join(self.images_dir(), image_relative_path)

    def visualization_check_path(self, save_name: str, return_aggregate: bool) -> str:
        if return_aggregate:
            return save_name.replace(".png", "_final_aggreated_image.png")
        return save_name.replace(".png", "_final_valid_image.png")

    def segmentation_metric_paths(self, return_aggregate: bool) -> Dict[str, str]:
        results_dir = self.results_dir()
        if return_aggregate:
            return {
                "median": os.path.join(results_dir, "seg_score_aggreated.xlsx"),
                "median_zero": os.path.join(results_dir, "seg_score_aggreated_zero.xlsx"),
                "median_new": os.path.join(results_dir, "seg_score_aggreated_new.xlsx"),
            }
        return {
            "median": os.path.join(results_dir, "seg_score_median_new.xlsx"),
            "median_zero": os.path.join(results_dir, "seg_score_median_zero_new.xlsx"),
        }

    def anomaly_score_path(self, result_json_path: str) -> str:
        return os.path.join(os.path.dirname(result_json_path), "anomaly_score.xlsx")

    @staticmethod
    def parent_dir(path: str) -> str:
        return os.path.dirname(path)
