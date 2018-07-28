""" TF Code Corrector Implementation """
import tensorflow as tf
import numpy as np
import random
import os
import glob
import time
import json
import datetime
import javalang
import sys

from models.train_model import TrainModel
from models.evaluation_model import EvaluationModel
from corruptors import java_corruptor

tf.app.flags.DEFINE_string("data_directory", "", "Directory of the data set")
tf.app.flags.DEFINE_string("output_directory", "", "Output directory for checkpoints and tests")
tf.app.flags.DEFINE_integer("max_sequence_length", 1000, "Max length of input sequence")
tf.app.flags.DEFINE_integer("sequence_length_step", 250, "Step size in which the sequence length is increased")
tf.app.flags.DEFINE_integer("sequence_length_increase_iterations", 5000, "Number of iterations after which the sequence_length is increased")
tf.app.flags.DEFINE_integer("simple_corruption_threshold", 20000, "Number of total iterations after which the difficulty of the corruption is increased")
tf.app.flags.DEFINE_integer("pad_id", 1, "Code of padding character")
tf.app.flags.DEFINE_integer("sos_id", 2, "Code of start-of-sequence character")
tf.app.flags.DEFINE_integer("eos_id", 3, "Code of end-of-sequence character")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size for training input")
tf.app.flags.DEFINE_integer("num_layers", 4, "Number of layers of the network")
tf.app.flags.DEFINE_integer("num_units", 256, "Number of units in each layer")
tf.app.flags.DEFINE_integer("num_iterations", 40000, "Number of iterations in training")
tf.app.flags.DEFINE_integer("eval_steps", 1000, "Step size for evaluation")
tf.app.flags.DEFINE_integer("random_seed", 1234, "Seed for all random operations")
tf.app.flags.DEFINE_integer("num_gpus", 1, "number of GPUs to use")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer")

FLAGS = tf.app.flags.FLAGS

def main(_):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S/")
    FLAGS.output_directory = os.path.join(FLAGS.output_directory, timestamp)
    os.makedirs(FLAGS.output_directory)

    with open(os.path.join(FLAGS.output_directory, 'hparams.json'), 'w') as hparam:
        json.dump(FLAGS.flag_values_dict(), hparam)

    random.seed(FLAGS.random_seed)

    train_graph = tf.Graph()
    eval_graph = tf.Graph()

    with train_graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        train_iterator, train_placeholders = create_iterator()
        train_model = TrainModel(FLAGS, train_iterator)
        initializer = tf.global_variables_initializer()

    with eval_graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        eval_iterator, eval_placeholders = create_iterator()
        eval_model = EvaluationModel(FLAGS, eval_iterator)

    train_sess = tf.Session(graph=train_graph)
    eval_sess = tf.Session(graph=eval_graph)

    train_sess.run(initializer)
    initialize_iterator(train_iterator, 'train', FLAGS.sequence_length_step, train_placeholders, train_sess, simple=True)
    print("Max sequence length set to {}".format(FLAGS.sequence_length_step))
    sys.stdout.flush()
    initialize_iterator(eval_iterator, 'eval', FLAGS.max_sequence_length, eval_placeholders, eval_sess)

    for i in range(FLAGS.num_iterations):
        trained = False
        while(not trained):
            try:
                train_model.train(train_sess, i+1)
                trained = True
            except tf.errors.OutOfRangeError:
                step = (i+1) // FLAGS.sequence_length_increase_iterations
                step += 1
                current_sequence_length = step * FLAGS.sequence_length_step
                if current_sequence_length > FLAGS.max_sequence_length:
                    current_sequence_length = FLAGS.max_sequence_length
                initialize_iterator(train_iterator, 'train', current_sequence_length, train_placeholders, train_sess, simple=True if i < FLAGS.simple_corruption_threshold else False)


        if (i+1) % FLAGS.eval_steps == 0:
            checkpoint_path = train_model.saver.save(train_sess, FLAGS.output_directory, global_step=i+1)
            eval_model.saver.restore(eval_sess, checkpoint_path)
            evaluated = False
            while(not evaluated):
                try:
                    eval_model.eval(eval_sess)
                    evaluated = True
                except tf.errors.OutOfRangeError:
                    initialize_iterator(eval_iterator, 'eval', FLAGS.max_sequence_length, eval_placeholders, eval_sess)

        if (i+1) % FLAGS.sequence_length_increase_iterations == 0:
            step = (i+1) // FLAGS.sequence_length_increase_iterations
            step += 1
            new_sequence_length = step * FLAGS.sequence_length_step
            if new_sequence_length <= FLAGS.max_sequence_length:
                initialize_iterator(train_iterator, 'train', new_sequence_length, train_placeholders, train_sess, simple=True if i < FLAGS.simple_corruption_threshold else False)
                print("Max sequence length increased to {}".format(new_sequence_length))
                sys.stdout.flush()


def initialize_iterator(iterator, file_name, sequence_length, placeholders, sess, simple=False):
    file_placeholder = placeholders[0]
    sequence_length_placeholder = placeholders[1]
    corrupt_simple = placeholders[2]
    file = random.choice(glob.glob(os.path.join(FLAGS.data_directory, file_name) + '*.java'))
    sess.run(iterator.initializer, feed_dict={file_placeholder: file, sequence_length_placeholder: sequence_length, corrupt_simple: simple})

def create_iterator():
    java_file = tf.placeholder(tf.string, shape=[], name='file_placeholer')
    max_sequence_length = tf.placeholder(tf.int32, shape=[], name='max_sequence_length')
    simple_corruption = tf.placeholder(tf.bool, shape=[], name='simple_corruption')

    def map_function(line):
        t = tf.py_func(lambda string: string.strip(), [line], tf.string)
        t = tf.map_fn(lambda elem:
                tf.py_func(lambda char: np.array(ord(char), dtype=np.int32), [elem], tf.int32), tf.string_split([t], '').values, tf.int32)
        dec_inp = tf.concat([[FLAGS.sos_id], t], 0)
        dec_out = tf.concat([t, [FLAGS.eos_id]], 0)

        enc_inp = tf.cond(simple_corruption,
                        lambda: tf.identity(line),
                        lambda: tf.py_func(java_corruptor.corrupt, [line], tf.string)
                        )
        enc_inp = tf.map_fn(lambda elem:
                tf.py_func(lambda char: np.array(ord(char), dtype=np.int32), [elem], tf.int32), tf.string_split([enc_inp], '').values, tf.int32)

        return enc_inp, tf.expand_dims(tf.size(enc_inp), 0), dec_inp, dec_out, tf.expand_dims(tf.size(dec_inp),0)


    with tf.device('/cpu:0'):
        def valid_and_ascii(s):
            try:
                s = unicode(s, 'utf-8')
            except:
                return False
            return True

        dataset = tf.data.TextLineDataset(java_file).filter(lambda line:
                        tf.logical_and(
                            tf.py_func(lambda l: valid_and_ascii(l), [line], tf.bool),
                            tf.logical_and(
                                tf.not_equal(
                                    tf.size(tf.string_split([tf.py_func(lambda l: l.strip(), [line], tf.string)],"")),
                                    tf.constant(0, dtype=tf.int32)
                                ),
                                tf.less(
                                    tf.size(tf.string_split([tf.py_func(lambda l: l.strip(), [line], tf.string)],"")),
                                    max_sequence_length
                                )
                            )
                        )
        )
        dataset = dataset.shuffle(1000)
        dataset = dataset.map(map_function, num_parallel_calls = 4)
        pad = tf.constant(FLAGS.pad_id, dtype=tf.int32)
        dataset = dataset.apply(tf.contrib.data.padded_batch_and_drop_remainder(
                                    FLAGS.batch_size,
                                    padded_shapes=([None], [1], [None], [None], [1]),
                                    padding_values=(pad, tf.constant(0, dtype=tf.int32), pad, pad, tf.constant(0, dtype=tf.int32))))
        dataset = dataset.prefetch(FLAGS.batch_size)
        return dataset.make_initializable_iterator(), (java_file, max_sequence_length, simple_corruption)


if __name__ == "__main__":
    tf.app.run()
