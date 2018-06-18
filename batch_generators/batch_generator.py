"""Base Class for batch generators"""

import numpy as np

class BatchGenerator:

    def _pad_char_array(self, array):
        """
        Args:
            array: a 2D-array of shape (n,)
        Returns:
            An array of shape (n, max_length, 1) where:
                max_length: length of the longest array in a
                shorter arrays are pre-padded
        """
        pad_code = 128           # padding symbol

        max_length = 0
        sequence_lengths = []
        for l in array:
            sequence_lengths.append(len(l))
            if len(l) > max_length:
                max_length = len(l)

        b = np.empty((len(array), max_length, 1), dtype='int32')
        b.fill(pad_code)
        for i in range(len(array)):
            b[i][-sequence_lengths[i]:] = [[ord(a)] for a in array[i]]

        return b, sequence_lengths

    def _unpack_bits(self, array):
        """
        Args:
            array: a 3D-array of shape (n, m, 1)
        Returns:
            An array of shape (n, m, x) where all entries of the argument
            are unpacked into a bit array and:
                x: number of bits needed to represent array.dtype
        """
        return np.unpackbits(array, axis=2)
