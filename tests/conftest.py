"""测试环境：强制无窗口绘图，避免 CI / 无桌面环境弹窗。"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
