"""distilabel logging plumbing for the model layer -- kept out of base.py so it
stays the Model class + catalog fetch. Only Model.run_pipeline needs this.
"""

import logging
from logging.handlers import QueueHandler


def detach_stale_queue_handlers() -> None:
    """Removes any QueueHandler left on the root logger.

    distilabel's stop_logging() (utils/logging.py) closes its multiprocessing
    queue after every Pipeline.run() but never detaches the QueueHandler it
    attached to the root logger. Left in place, the next unrelated log call
    anywhere in the process (e.g. huggingface_hub's HTTP-warning logger)
    raises "Queue is closed" inside logging's Handler.emit().
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, QueueHandler):
            root_logger.removeHandler(handler)
