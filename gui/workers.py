import logging
import traceback

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

_log = logging.getLogger('ciaw')


class PipelineWorker(QThread):
    finished = pyqtSignal(str, np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, fn, op_name, state, **kwargs):
        super().__init__()
        self.fn = fn
        self.op_name = op_name
        self.state = state
        self.kwargs = kwargs

    def run(self):
        try:
            image = self.state.get_base_image()
            result = self.fn(image, **self.kwargs)
            self.finished.emit(self.op_name, result)
        except Exception as e:
            tb = traceback.format_exc()
            _log.error('[Worker %s] %s: %s\n%s', self.op_name, type(e).__name__, e, tb)
            self.error.emit(str(e))
