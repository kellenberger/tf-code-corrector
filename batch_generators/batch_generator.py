class BatchGenerator:

    def pad_array_with_zeros(array):
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
