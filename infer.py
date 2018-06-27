""" TF Code Corrector Implementation """
import tensorflow as tf
import numpy as np
import random
import os

from models.train_model import TrainModel
from models.evaluation_model import EvaluationModel

tf.app.flags.DEFINE_string("data_directory", "", "Directory of the data set")
tf.app.flags.DEFINE_string("output_directory", "", "Output directory for checkpoints and tests")
tf.app.flags.DEFINE_string("checkpoint_path", "", "Path to checkpoint")
tf.app.flags.DEFINE_integer("max_sequence_length", 200, "Max length of input sequence")
tf.app.flags.DEFINE_integer("pad_id", 128, "Code of padding character")
tf.app.flags.DEFINE_integer("sos_id", 2, "Code of start-of-sequence character")
tf.app.flags.DEFINE_integer("eos_id", 3, "Code of end-of-sequence character")
tf.app.flags.DEFINE_integer("batch_size", 128, "Bath size for training input")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers of the network")
tf.app.flags.DEFINE_integer("num_units", 256, "Number of units in each layer")
tf.app.flags.DEFINE_integer("num_iterations", 12000, "Number of iterations in training")
tf.app.flags.DEFINE_integer("eval_steps", 1000, "Step size for evaluation")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer")

FLAGS = tf.app.flags.FLAGS

def main(_):
    infer_graph = tf.Graph()

    with infer_graph.as_default():
        infer_iterator, infer_file = create_iterator()
        infer_model = EvaluationModel(FLAGS, infer_iterator)

    infer_sess = tf.Session(graph=infer_graph)

    initialize_iterator(infer_iterator, infer_file, 'testJava.csv', infer_sess)
    infer_model.saver.restore(infer_sess, FLAGS.checkpoint_path)

    infered = False
    while(not infered):
        try:
            infer_model.eval(infer_sess)
            infered = True
        except tf.errors.OutOfRangeError:
            initialize_iterator(infer_iterator, infer_file, 'testJava.csv', infer_sess)


def initialize_iterator(iterator, file_placeholder, projects_file, sess):
    with open(os.path.join(FLAGS.data_directory, projects_file), 'r') as project_data:
        projects = np.array(project_data.read().splitlines())
        project = projects[random.randint(0, len(projects)-1)]
        project = os.path.join(FLAGS.data_directory, project+'.java')

    sess.run(iterator.initializer, feed_dict={file_placeholder: project})

def create_iterator():
    java_file = tf.placeholder(tf.string, shape=[])

    def map_function(line):
        t = tf.py_func(lambda string: string.strip(), [line], tf.string)
        t = tf.map_fn(lambda elem:
                tf.py_func(lambda char: np.array(ord(char), dtype=np.int32), [elem], tf.int32), tf.string_split([t], '').values, tf.int32)
        dec_inp = tf.concat([[FLAGS.sos_id], t], 0)
        dec_out = tf.concat([t, [FLAGS.eos_id]], 0)
        def drop_c():
            drop_char = tf.random_uniform([1], minval = 0, maxval = tf.size(t), dtype=tf.int32)
            return tf.concat([tf.slice(t, [0], drop_char), tf.slice(t, drop_char+1, tf.subtract(tf.size(t), drop_char+1))], 0)

        enc_inp = tf.cond(tf.logical_and(tf.random_uniform([]) > 0.9, tf.size(t) > 1), drop_c, lambda: t)
        return enc_inp, tf.expand_dims(tf.size(enc_inp), 0), dec_inp, dec_out, tf.expand_dims(tf.size(dec_inp),0)


    with tf.device('/cpu:0'):
        dataset = tf.data.TextLineDataset(java_file).filter(lambda line:
                                    tf.logical_and(
                                        tf.not_equal(
                                            tf.size(tf.string_split([tf.py_func(lambda l: l.strip(), [line], tf.string)],"")),
                                            tf.constant(0, dtype=tf.int32)
                                        ),
                                        tf.less(
                                            tf.size(tf.string_split([tf.py_func(lambda l: l.strip(), [line], tf.string)],"")),
                                            tf.constant(FLAGS.max_sequence_length, dtype=tf.int32)
                                        )
                                    )
                               )
        dataset = dataset.shuffle(10000)
        dataset = dataset.map(map_function, num_parallel_calls = 4)
        pad = tf.constant(FLAGS.pad_id, dtype=tf.int32)
        dataset = dataset.apply(tf.contrib.data.padded_batch_and_drop_remainder(
                                    FLAGS.batch_size,
                                    padded_shapes=([None], [1], [None], [None], [1]),
                                    padding_values=(pad, tf.constant(0, dtype=tf.int32), pad, pad, tf.constant(0, dtype=tf.int32))))
        dataset = dataset.prefetch(FLAGS.batch_size)
        return dataset.make_initializable_iterator(), java_file


if __name__ == "__main__":
    tf.app.run()
