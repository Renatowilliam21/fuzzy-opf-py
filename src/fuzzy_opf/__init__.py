# Disable opfython's per-module TimedRotatingFileHandler *before* importing
# anything that touches opfython (below): opfython.utils.logging.get_logger
# gives every module (opfython.core.opf, opfython.models.supervised, ...)
# its OWN file handler, all pointed at the same "opfython.log". On Windows,
# when midnight rollover fires on one handler, the others still hold the
# file open and os.rename fails with WinError 32 -- harmless but noisy
# (fills the console with tracebacks on long-running experiments) and lets
# the log grow unbounded (seen at 21MB in one run). Console output (the
# other handler get_logger adds) is untouched; only the file handler is
# replaced with a no-op.
import logging as _logging
import opfython.utils.logging as _opf_logging

_opf_logging.get_timed_file_handler = _logging.NullHandler

from .model import FuzzyOPF
from .ensemble import EnsembleFuzzyOPF
from .tuning import genetic_search, pso_search, cem_search, random_search, bayesian_search, nsga2_search, TuningResult, ParetoResult
from .datasets import (
    load_dataset,
    standardize,
    stratified_split,
    oversample_minority_classes,
    smote_oversample,
    opf_us_undersample,
    apply_balance,
)

__all__ = [
    "FuzzyOPF",
    "EnsembleFuzzyOPF",
    "genetic_search",
    "pso_search",
    "cem_search",
    "random_search",
    "bayesian_search",
    "nsga2_search",
    "TuningResult",
    "ParetoResult",
    "load_dataset",
    "standardize",
    "stratified_split",
    "oversample_minority_classes",
    "smote_oversample",
    "opf_us_undersample",
    "apply_balance",
]
