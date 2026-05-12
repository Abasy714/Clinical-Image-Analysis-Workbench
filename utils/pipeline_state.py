# STATUS: IMPLEMENTED
"""
Manages the sequential image enhancement pipeline state.
Maintains an ordered stack of (operation_name, image_array) pairs with undo and reset support.
"""

import numpy as np
from collections import deque

MAX_STACK_SIZE = 20

class PipelineState:
    """
    Manages the sequential image enhancement pipeline.
    Stores (op_name: str, image: np.ndarray) pairs in a deque.
    """

    def __init__(self):
        self._stack: deque = deque(maxlen=MAX_STACK_SIZE)
        self._original: np.ndarray | None = None
        self._checkpoints: dict = {}
        self._mode: str = 'cumulative'  # 'cumulative' or 'independent'

    def set_original(self, image: np.ndarray):
        self._original = image.copy()
        self._stack.clear()
        self._checkpoints.clear()

    def has_image(self) -> bool:
        """Return True after a real image has been loaded."""
        return self._original is not None

    def push(self, op_name: str, image: np.ndarray):
        if image is None:
            raise ValueError("Cannot push an empty image into the pipeline.")
        self._stack.append((op_name, image.copy()))

    def undo(self) -> np.ndarray | None:
        if self._stack:
            self._stack.pop()
        return self.current()

    def reset(self) -> np.ndarray | None:
        self._stack.clear()
        return self._original

    def current(self) -> np.ndarray | None:
        if self._stack:
            return self._stack[-1][1]
        return self._original

    def get_stack_names(self) -> list:
        return [name for name, _ in self._stack]

    def get_stack_entries(self) -> list:
        """Return list of (op_name, image) tuples from bottom to top."""
        return list(self._stack)

    def get_original(self) -> np.ndarray | None:
        """Return the original image set by set_original, or None if empty."""
        return self._original

    def current_op(self) -> str:
        """Return the name of the top operation, or 'Original' if stack empty."""
        if self._stack:
            return self._stack[-1][0]
        return "Original"

    def set_mode(self, mode: str):
        """
        Set pipeline mode.
        'cumulative': each op applies on previous result (default)
        'independent': each op applies on original image
        """
        assert mode in ('cumulative', 'independent'), f"Invalid mode: {mode}"
        self._mode = mode

    def get_mode(self) -> str:
        """Return current pipeline mode ('cumulative' or 'independent')."""
        return self._mode

    def get_base_image(self) -> np.ndarray | None:
        """
        Return the image that the NEXT operation should apply to.
        Cumulative: top of stack (or original if empty)
        Independent: always original
        """
        if self._mode == 'independent':
            return self.get_original()
        return self.current()

    def save_checkpoint(self, slot: str):
        img = self.current()
        if img is None:
            raise ValueError("No image loaded.")
        op = self._stack[-1][0] if self._stack else "original"
        self._checkpoints[slot] = (op, img.copy())

    def restore_checkpoint(self, slot: str) -> np.ndarray | None:
        if slot not in self._checkpoints:
            return None
        op_name, image = self._checkpoints[slot]
        self.push(f"restore:{slot}({op_name})", image)
        return image

    def get_checkpoints(self) -> dict:
        return {slot: op for slot, (op, _) in self._checkpoints.items()}
