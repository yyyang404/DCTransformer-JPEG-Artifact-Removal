r"""
The vendored metrics package in this repository only includes PSNR.

Inputs use format :math:`(N, C, H, W)` and produce outputs of format
:math:`(N)` by averaging spatially and over channels. The batch dimension is
not averaged. Inputs should be images in [0, 1].
"""

from ._psnr import *

__all__ = ["psnr"]
