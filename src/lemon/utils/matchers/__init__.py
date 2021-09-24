_lazy_subimports = {"TransformerMatcher": "_transformer_matcher", "MagellanMatcher": "_magellan"}


__all__ = ["TransformerMatcher", "MagellanMatcher"]  # type: ignore


def __getattr__(name):
    import importlib

    if name in _lazy_subimports:
        module = importlib.import_module("." + _lazy_subimports[name], __name__)
        return module.__dict__[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
