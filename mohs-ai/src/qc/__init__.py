"""QC training and inference modules."""

__all__ = ["QCOnnxInferencer", "train_main"]


def __getattr__(name: str):
    if name == "QCOnnxInferencer":
        from .infer_qc import QCOnnxInferencer as _QCOnnxInferencer
        return _QCOnnxInferencer
    if name == "train_main":
        from .train_qc import main as _train_main
        return _train_main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
