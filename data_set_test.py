import tensorflow as tf
import numpy as np
import time
import datetime
import os

def main(_):
    data_directory = '/data/cvg/sven/java_github_corpus'
    java_files = []

    with open(os.path.join(data_directory, 'trainJava.csv'), 'r') as train_projects:
        for project in train_projects:
            if os.path.exists(os.path.join(data_directory, project.strip())):
                if(len(java_files) > 100):
                    break
                for subdir, _, files in os.walk(os.path.join(data_directory, project.strip())):
                    for file in files:
                        java_files.append(os.path.join(data_directory, project.strip(), subdir, file))

    print("number of train files: {}".format(len(java_files)))

    def map_function(line):
        t = tf.map_fn(lambda elem:
                tf.py_func(lambda char: np.array(ord(char), dtype=np.int32), [elem], tf.int32), tf.string_split([line], '').values, tf.int32)
        dec_inp = tf.concat([[0], t], 0)
        dec_out = tf.concat([t, [1000]], 0)
        def drop_c():
            drop_char = tf.random_uniform([1], minval = 0, maxval = tf.size(t), dtype=tf.int32)
            return tf.concat([tf.slice(t, [0], drop_char), tf.slice(t, drop_char+1, tf.subtract(tf.size(t), drop_char+1))], 0)

        enc_inp = tf.cond(tf.random_uniform([]) > 0.9, drop_c, lambda: t)
        return enc_inp, tf.expand_dims(tf.size(enc_inp), 0), dec_inp, dec_out, tf.expand_dims(tf.size(dec_inp),0)


    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_tensor_slices(java_files)

        dataset = dataset.flat_map(
            lambda filename: (
                tf.data.TextLineDataset(filename)
                .filter(lambda line:
                            tf.not_equal(
                                tf.size(tf.string_split([tf.py_func(lambda l: l.strip(), [line], tf.string)],"")),
                                tf.constant(0, dtype=tf.int32))
                       )
            )
        )
        dataset = dataset.shuffle(10000)
        dataset = dataset.map(map_function, num_parallel_calls = 4)
        pad = tf.constant(128, dtype=tf.int32)
        dataset = dataset.padded_batch(128, padded_shapes=([None], [1], [None], [None], [1]),
                                       padding_values=(pad, tf.constant(0, dtype=tf.int32), pad, pad, tf.constant(0, dtype=tf.int32)))
        print('data set construction complete')
        iterator = dataset.make_one_shot_iterator()
        print('iterator initialized')

    sess = tf.Session()
    for i in range(25):
        start_time = time.time()
        a, b, c, d, e = sess.run(iterator.get_next())
        end_time = time.time()
        print("{}, {}s".format(i, end_time - start_time))
        print(np.reshape(b, [128]))
        print(np.reshape(e, [128]))
    sess.close()

if __name__ == "__main__":
    tf.app.run()
