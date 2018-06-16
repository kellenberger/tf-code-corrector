"""Base Class for batch generators"""

import numpy as np

class BatchGenerator:

    def pad_array_with_zeros(self, array):
        """
        Args:
            array: a 2D-array of shape (n,)
        Returns:
            An array of shape (n, max_length) where:
                max_length: length of the longest array in a
        """
        max_length = 0
        sequence_lengths = []
        for l in array:
            sequence_lengths.append(len(l))
            if len(l) > max_length:
                max_length = len(l)

        b = np.zeros((len(array), max_length))
        for i in range(len(array)):
            b[i][:sequence_lengths[i]] = array[i]

        return b
