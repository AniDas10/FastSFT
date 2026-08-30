"""Canonical stage names, shared by every module that refers to a stage."""

DATA_GENERATOR = "data_generator"
DATA_FORMATTER = "data_formatter"
FINE_TUNER = "fine_tuner"

STAGE_ORDER = [DATA_GENERATOR, DATA_FORMATTER, FINE_TUNER]
STAGE_NAMES = tuple(STAGE_ORDER)
