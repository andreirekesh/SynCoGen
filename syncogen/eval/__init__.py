"""Post-hoc evaluation of generated molecules (validity, diversity, novelty,
QED, SA score, xTB/MMFF energies, and optional retrosynthesis)."""

__all__ = ["evaluate"]


def __getattr__(name):  # PEP 562: lazy export so the package imports cheaply
    if name == "evaluate":
        from syncogen.eval.metrics import evaluate

        return evaluate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
