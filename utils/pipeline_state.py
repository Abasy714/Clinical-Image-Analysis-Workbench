# STATUS: IMPLEMENTED
"""
Manages the sequential image enhancement pipeline state.
Maintains an ordered stack of (operation_name, image_array) pairs with undo and reset support.
"""

# numpy — image array storage and copying
# collections.deque — efficient stack with max length for memory safety

# CONSTANTS
# MAX_STACK_SIZE = 20

# FUNCTIONS / CLASSES
# class PipelineState:
#   def __init__: initialize empty stack, store reference to original image
#   def set_original(image): store the base image, clear stack
#   def push(op_name, image): append (op_name, image.copy()) to stack
#   def undo() -> np.ndarray: pop last entry, return previous image (or original if stack empty)
#   def reset() -> np.ndarray: clear stack, return original image
#   def current() -> np.ndarray: return top of stack or original
#   def get_stack_names() -> list[str]: return list of operation names for UI display

import numpy as np
from collections import deque

MAX_STACK_SIZE = 20

_EMPTY = np.zeros((512, 512), dtype=np.uint8)


class PipelineState:
    """
    Manages the sequential image enhancement pipeline.
    Stores (op_name: str, image: np.ndarray) pairs in a deque.
    """

    def __init__(self):
        self._stack: deque = deque(maxlen=MAX_STACK_SIZE)
        self._original: np.ndarray | None = None
        self._checkpoints: dict = {}

    def set_original(self, image: np.ndarray):
        self._original = image.copy()
        self._stack.clear()

    def push(self, op_name: str, image: np.ndarray):
        self._stack.append((op_name, image.copy()))

    def undo(self) -> np.ndarray:
        if self._stack:
            self._stack.pop()
        return self.current()

    def reset(self) -> np.ndarray:
        self._stack.clear()
        return self._original if self._original is not None else _EMPTY

    def current(self) -> np.ndarray:
        if self._stack:
            return self._stack[-1][1]
        return self._original if self._original is not None else _EMPTY

    def get_stack_names(self) -> list:
        return [name for name, _ in self._stack]

    def save_checkpoint(self, slot: str):
        img = self.current()
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
