# Common imports
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import functools
import re
import math
import random
import datetime

from typing import List, Callable, Union, Dict, Tuple, Optional, Iterator

#File I/O wrappers without thread locking.
gfile = tf.io.gfile