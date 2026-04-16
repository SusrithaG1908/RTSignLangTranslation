"""
pipeline_config.py
Single source-of-truth for every model pipeline.

Instead of copy-pasting dictionaries across train / benchmark / webcam
scripts, import PIPELINE_REGISTRY and look up configs by name or number.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple


IMG_SIZE_CNN = (128, 128)
IMG_SIZE_MOBILENET = (224, 224)


@dataclass
class PipelineConfig:
    """All settings for one model pipeline."""
    name: str
    model_filename: str          # relative to MODELS_DIR
    labels_filename: str         # relative to MODELS_DIR
    img_size: Tuple[int, int]
    use_mediapipe: bool
    is_mobilenet: bool
    test_data_subdir: str        # "test" under data/ or data_mp/
    display_number: str = ""     # "1"–"4" shown in webcam menu

    # Resolved at runtime (set by PipelineRegistry.resolve())
    model_path: Optional[Path] = field(default=None, repr=False)
    labels_path: Optional[Path] = field(default=None, repr=False)
    test_dir: Optional[Path] = field(default=None, repr=False)

    @property
    def preprocess_mode(self) -> str:
        return "mobilenet" if self.is_mobilenet else "cnn"


class PipelineRegistry:
    """
    Central registry of all pipelines.

    Usage
    -----
    from core.pipeline_config import PipelineRegistry
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    registry = PipelineRegistry(PROJECT_ROOT)

    cfg = registry.get("CNN_Raw_v2")
    cfg = registry.get_by_number("1")
    all_configs = registry.all()
    """

    _DEFINITIONS = [
        PipelineConfig(
            name="CNN_Raw_v2",
            model_filename="cnn_raw_v2.h5",
            labels_filename="class_labels_cnn_raw_v2.json",
            img_size=IMG_SIZE_CNN,
            use_mediapipe=False,
            is_mobilenet=False,
            test_data_subdir="data/test",
            display_number="1",
        ),
        PipelineConfig(
            name="CNN_MediaPipeCrop_v2",
            model_filename="cnn_mp_v2.h5",
            labels_filename="class_labels_cnn_mp_v2.json",
            img_size=IMG_SIZE_CNN,
            use_mediapipe=True,
            is_mobilenet=False,
            test_data_subdir="data_mp/test",
            display_number="2",
        ),
        PipelineConfig(
            name="MobileNet_TL_10%_v2",
            model_filename="mobilenet_mp_10%_v2.h5",
            labels_filename="class_labels_mobilenet_mp_10%_v2.json",
            img_size=IMG_SIZE_MOBILENET,
            use_mediapipe=True,
            is_mobilenet=True,
            test_data_subdir="data_mp/test",
            display_number="3",
        ),
        PipelineConfig(
            name="MobileNet_TL_25%_v2",
            model_filename="mobilenet_mp_25%_v2.h5",
            labels_filename="class_labels_mobilenet_mp_25%_v2.json",
            img_size=IMG_SIZE_MOBILENET,
            use_mediapipe=True,
            is_mobilenet=True,
            test_data_subdir="data_mp/test",
            display_number="4",
        ),
    ]

    def __init__(self, project_root: Path):
        self._project_root = project_root
        self._configs: Dict[str, PipelineConfig] = {}
        self._by_number: Dict[str, PipelineConfig] = {}

        models_dir = project_root / "models"
        for cfg in self._DEFINITIONS:
            cfg.model_path = models_dir / cfg.model_filename
            cfg.labels_path = models_dir / cfg.labels_filename
            cfg.test_dir = project_root / cfg.test_data_subdir
            self._configs[cfg.name] = cfg
            if cfg.display_number:
                self._by_number[cfg.display_number] = cfg

    # ------------------------------------------------------------------
    def get(self, name: str) -> PipelineConfig:
        if name not in self._configs:
            raise KeyError(f"Unknown pipeline '{name}'. Available: {list(self._configs)}")
        return self._configs[name]

    def get_by_number(self, number: str) -> PipelineConfig:
        if number not in self._by_number:
            raise KeyError(f"No pipeline with number '{number}'. Available: {list(self._by_number)}")
        return self._by_number[number]

    def all(self):
        return list(self._configs.values())

    def print_menu(self):
        print("\nSelect pipeline:")
        for cfg in self._configs.values():
            label = "MobileNet" if cfg.is_mobilenet else "CNN"
            mp_tag = "+ MediaPipe Crop" if cfg.use_mediapipe else "(Raw ROI)"
            pct = cfg.name.split("_")[2] if cfg.is_mobilenet else ""
            suffix = f" {pct}" if pct else ""
            print(f"  {cfg.display_number}️⃣  {label}{suffix} {mp_tag}")
