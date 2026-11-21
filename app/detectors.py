"""YOLO 适配器骨架。模型加载代码已给出，推理结果转换由学员完成。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .protocol import WasteType


@dataclass
class Detection:
    waste_type: WasteType
    class_name: str
    confidence: float
    box: Tuple[int, int, int, int]


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class YoloDetector(Detector):
    """加载指定权重，并把模型原始类别映射到四类垃圾。"""

    DEFAULT_CLASS_MAP = {
        "bottolad": WasteType.RECYCLABLE,
        "plasticbottle": WasteType.RECYCLABLE,
        "box": WasteType.RECYCLABLE,
        "can": WasteType.RECYCLABLE,
        "papercup": WasteType.RECYCLABLE,
        "yangleduo": WasteType.RECYCLABLE,
        "potato": WasteType.KITCHEN,
        "carrot": WasteType.KITCHEN,
        "qincarrot": WasteType.KITCHEN,
        "redcarrot": WasteType.KITCHEN,
        "whitcarrot": WasteType.KITCHEN,
        "battery": WasteType.HAZARDOUS,
        "drug": WasteType.HAZARDOUS,
        "cobblestone": WasteType.OTHER,
        "porcelain": WasteType.OTHER,
    }

    def __init__(
        self,
        weights: Optional[str] = None,
        model=None,
        confidence: float = 0.50,
        device: Optional[str] = None,
    ) -> None:
        if not 0.0 < confidence <= 1.0:
            raise ValueError("置信度阈值必须在 (0, 1] 范围内")
        if model is None:
            if not weights:
                raise ValueError("必须指定 YOLO 权重文件")
            weight_path = Path(weights)
            if not weight_path.is_file():
                raise FileNotFoundError("YOLO 权重不存在：{}".format(weight_path))
            try:
                import torch
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("未安装 YOLO 运行环境") from exc
            self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = YOLO(str(weight_path))
        else:
            self.model = model
            self.device = device or "cpu"

        self.confidence = confidence
        model_names = getattr(self.model, "names", {})
        unknown = sorted(set(model_names.values()) - set(self.DEFAULT_CLASS_MAP))
        if unknown:
            raise ValueError("权重中存在未配置类别：{}".format(", ".join(unknown)))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """TODO 3：调用 ``model.predict`` 并转换 boxes、类别和置信度。"""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )
        detections = []  # type: List[Detection]

        for result in results:
            names = getattr(result, "names", None) or self.model.names
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                confidence = float(box.conf.item())
                if confidence < self.confidence:
                    continue

                class_id = int(box.cls.item())
                class_name = names[class_id]
                waste_type = self.DEFAULT_CLASS_MAP[class_name]
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = (int(round(value)) for value in xyxy)
                detections.append(
                    Detection(
                        waste_type=waste_type,
                        class_name=class_name,
                        confidence=confidence,
                        box=(x1, y1, x2, y2),
                    )
                )

        return sorted(detections, key=lambda item: item.confidence, reverse=True)
