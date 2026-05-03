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
