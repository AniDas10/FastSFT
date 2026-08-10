"""Tier-0 pin tests for fastsft.stages.constants -- canonical stage names/order.

STAGE_ORDER's order is load-bearing for DistillationPipeline (--start-stage
indexes into it) and for main.py's CLI validation; a silent reorder would
misroute --start-stage without any loud failure.
"""

from fastsft.stages.constants import (
    DATA_FORMATTER,
    DATA_GENERATOR,
    FINE_TUNER,
    STAGE_NAMES,
    STAGE_ORDER,
)


def test_stage_name_values():
    assert DATA_GENERATOR == "data_generator"
    assert DATA_FORMATTER == "data_formatter"
    assert FINE_TUNER == "fine_tuner"


def test_stage_order_is_generate_format_train():
    assert STAGE_ORDER == [DATA_GENERATOR, DATA_FORMATTER, FINE_TUNER]


def test_stage_names_matches_stage_order_as_tuple():
    assert STAGE_NAMES == tuple(STAGE_ORDER)
