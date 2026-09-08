"""Capture existing content-free metrics only inside an explicit curated evaluation."""

import json
import logging
from contextlib import contextmanager

from elderhelp.observability import request_context


@contextmanager
def capture(request_id, trace):
    events = []

    class Collector(logging.Handler):
        def emit(self, record):
            value = json.loads(record.getMessage())
            if value.get("request_id") == str(request_id):
                events.append(value)

    logger = logging.getLogger("elderhelp.metrics")
    previous = logger.level
    logger.setLevel(logging.INFO)
    handler = Collector()
    logger.addHandler(handler)
    token = request_context.set(str(request_id))
    try:
        yield
    finally:
        request_context.reset(token)
        logger.removeHandler(handler)
        logger.setLevel(previous)
        trace["usage_events"] = events
        trace["paid_cost_budget_usd"] = 0
