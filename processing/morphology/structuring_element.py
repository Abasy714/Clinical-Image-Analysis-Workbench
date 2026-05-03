"""
Generates binary structuring elements (SE) for morphological operations.
Supports square and cross shapes at user-defined sizes.
"""

# numpy — array construction (ones, zeros)

# CONSTANTS
# SE_SHAPES = ['square', 'cross']

# FUNCTIONS
# def get_square_se(size: int) -> np.ndarray:
#   — return size x size array of ones
# def get_cross_se(size: int) -> np.ndarray:
#   — return size x size array with ones only on center row and center column

# --- UTIL USAGE GUIDE ---
# from utils.error_handler import wrap_errors
#
# @wrap_errors                        # decorate get_square_se and get_cross_se
# Note: always return dtype=np.uint8 arrays — morphology functions expect binary 0/1 input
# Note: size must be odd — add a guard: if size % 2 == 0: raise ValueError