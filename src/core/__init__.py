"""
core/__init__.py
Convenience re-exports so scripts can do:

    from core import HandCropper, ImagePreprocessor, SignClassifier, ...
"""

from .hand_cropper    import HandCropper, CropResult
from .preprocessor    import ImagePreprocessor, PreprocessResult
from .classifier      import SignClassifier, Prediction
from .pipeline_config import PipelineConfig, PipelineRegistry
from .dataset         import DatasetDownloader, DatasetOrganizer
from .tts_speaker     import TTSSpeaker
from .word_builder    import WordBuilder, WordBuilderState

__all__ = [
    "HandCropper", "CropResult",
    "ImagePreprocessor", "PreprocessResult",
    "SignClassifier", "Prediction",
    "PipelineConfig", "PipelineRegistry",
    "DatasetDownloader", "DatasetOrganizer",
    "TTSSpeaker",
    "WordBuilder", "WordBuilderState",
]
