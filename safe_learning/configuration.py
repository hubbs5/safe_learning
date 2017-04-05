"""General configuration class for dtypes."""

from __future__ import absolute_import, print_function, division

import tensorflow as tf


class Configuration(object):
    """Configuration class."""

    def __init__(self):
        """Initialization."""
        super(Configuration, self).__init__()

        # Dtype for computations
        self.dtype = tf.float64

        # Batch size for stability verification
        self.gp_batch_size = 100

    @property
    def np_dtype(self):
        """Return the numpy dtype."""
        return self.dtype.as_numpy_dtype
