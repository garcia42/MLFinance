from enum import IntEnum
import sys
import math

# Memory debugging configuration
MEMDEBUG = True

# System constants
MAXPOSNUM = sys.maxsize
MAXNEGNUM = -sys.maxsize - 1
PI = math.pi

# Key codes
KEY_ESCAPE = 27
KEY_CTRLQ = 17  # Total abort

# Error codes
class ErrorCode(IntEnum):
    OK = 0
    NO_ERROR = 0
    ESCAPE = 1
    ABORT = 2
    INSUFFICIENT_MEMORY = 3
    SYNTAX = 4
    FTI = 5

# Variables - using IntEnum for better organization and type safety
class Variables(IntEnum):
    RSI = 10
    DETRENDED_RSI = 12
    STOCHASTIC_RSI = 14
    STOCHASTIC = 16
    MA_DIFF = 18
    MACD = 20
    PPO = 22
    LINEAR_TREND = 24
    QUADRATIC_TREND = 26
    CUBIC_TREND = 28
    PRICE_INTENSITY = 30
    ADX = 34
    AROON_UP = 36
    AROON_DOWN = 38
    AROON_DIFF = 40
    CLOSE_MINUS_MA = 50
    LINEAR_DEVIATION = 52
    QUADRATIC_DEVIATION = 54
    CUBIC_DEVIATION = 56
    PRICE_CHANGE_OSCILLATOR = 60
    PRICE_VARIANCE_RATIO = 62
    CHANGE_VARIANCE_RATIO = 64
    INTRADAY_INTENSITY = 70
    MONEY_FLOW = 72
    REACTIVITY = 74
    PRICE_VOLUME_FIT = 76
    VOLUME_WEIGHTED_MA_RATIO = 78
    NORMALIZED_ON_BALANCE_VOLUME = 80
    DELTA_ON_BALANCE_VOLUME = 82
    NORMALIZED_POSITIVE_VOLUME_INDEX = 84
    NORMALIZED_NEGATIVE_VOLUME_INDEX = 86
    VOLUME_MOMENTUM = 88
    ENTROPY = 100
    MUTUAL_INFORMATION = 102
    FTI_LOWPASS = 110
    FTI_BEST_PERIOD = 112
    FTI_BEST_FTI = 114
    FTI_BEST_WIDTH = 116

# Program limitations
MAX_NAME_LENGTH = 15
MAX_VARS = 8192

# Memory management functions if MEMDEBUG is True
if MEMDEBUG:
    class MemoryManager:
        def __init__(self):
            self.allocations = {}
            
        def malloc(self, size):
            # Implement tracked memory allocation
            memory = bytearray(size)
            self.allocations[id(memory)] = size
            return memory
            
        def free(self, ptr):
            # Implement tracked memory deallocation
            if id(ptr) in self.allocations:
                del self.allocations[id(ptr)]
                
        def realloc(self, ptr, size):
            # Implement tracked memory reallocation
            new_memory = bytearray(size)
            if ptr is not None and id(ptr) in self.allocations:
                old_size = self.allocations[id(ptr)]
                new_memory[:min(old_size, size)] = ptr[:min(old_size, size)]
                del self.allocations[id(ptr)]
            self.allocations[id(new_memory)] = size
            return new_memory
            
        def memtext(self, text):
            # Logging function for memory operations
            print(f"Memory operation: {text}")
            
        def memclose(self):
            # Cleanup function
            if self.allocations:
                print(f"Warning: {len(self.allocations)} memory allocations not freed")
            self.allocations.clear()

    # Create global memory manager instance
    memory_manager = MemoryManager()
    
    # Define memory management functions
    malloc = memory_manager.malloc
    free = memory_manager.free
    realloc = memory_manager.realloc
    memtext = memory_manager.memtext
    memclose = memory_manager.memclose
else:
    # Use standard memory management
    malloc = lambda size: bytearray(size)
    free = lambda ptr: None
    realloc = lambda ptr, size: bytearray(size)
    memtext = lambda text: None
    memclose = lambda: None