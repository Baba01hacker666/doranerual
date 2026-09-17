"""Alias module routing 'doranerual' to 'doraneural'."""

import sys
from doraneural import *
from doraneural import __version__, __all__

# Ensure submodule lookups work seamlessly
sys.modules["doranerual"] = sys.modules["doraneural"]
