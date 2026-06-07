import os
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

parent_dir = parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # Set root directory to import modules
sys.path.insert(0, parent_dir)

from numcompute.io import *

TEST_DATA = ''