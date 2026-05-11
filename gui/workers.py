import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


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
        image = self.state.get_base_image()
        try:
            result = self.fn(image, **self.kwargs)
            self.finished.emit(self.op_name, result)
        except Exception as e:
            self.error.emit(str(e))
