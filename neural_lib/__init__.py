"""Backward-compatibility alias module routing 'neural_lib' to 'doraneural'."""

import sys
from doraneural import *
from doraneural import __version__, __all__

sys.modules["neural_lib"] = sys.modules["doraneural"]
