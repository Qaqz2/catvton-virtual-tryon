"""
Simplified functions.py — pure PyTorch, no C++ compilation needed.
Replaces the original which used torch.utils.cpp_extension.load().
"""

ACT_RELU = "relu"
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"

__all__ = ["ACT_RELU", "ACT_LEAKY_RELU", "ACT_ELU", "ACT_NONE"]
