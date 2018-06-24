""" TF Code Corrector Implementation """
import tensorflow as tf
import numpy as np
import os

from models.train_model import TrainModel
from models.evaluation_model import EvaluationModel

tf.app.flags.DEFINE_string("data_directory", "", "Directory of the data set")
tf.app.flags.DEFINE_string("output_directory", "", "Output directory for checkpoints and tests")
tf.app.flags.DEFINE_integer("num_projects", 100, "number of projects which are sample simultaneously")
tf.app.flags.DEFINE_integer("pad_id", 128, "Code of padding character")
tf.app.flags.DEFINE_integer("sos_id", 2, "Code of start-of-sequence character")
tf.app.flags.DEFINE_integer("eos_id", 3, "Code of end-of-sequence character")
tf.app.flags.DEFINE_integer("batch_size", 128, "Bath size for training input")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers of the network")
tf.app.flags.DEFINE_integer("num_units", 256, "Number of units in each layer")
tf.app.flags.DEFINE_integer("num_iterations", 10000, "Number of iterations in training")
tf.app.flags.DEFINE_integer("eval_steps", 100, "Step size for evaluation")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer")

FLAGS = tf.app.flags.FLAGS

def main(_):
    train_graph = tf.Graph()
    eval_graph = tf.Graph()

    with train_graph.as_default():
        train_iterator, train_files = create_iterator(FLAGS)
        train_model = TrainModel(FLAGS, train_iterator)
        initializer = tf.global_variables_initializer()

    with eval_graph.as_default():
        eval_iterator, eval_files = create_iterator(FLAGS)
        eval_model = EvaluationModel(FLAGS, eval_iterator)

    train_sess = tf.Session(graph=train_graph)
    eval_sess = tf.Session(graph=eval_graph)

    train_sess.run(initializer)
    initialize_iterator(train_iterator, train_files,
            os.path.join(FLAGS.data_directory, 'trainJava.csv'), FLAGS.num_projects, train_sess)
    initialize_iterator(eval_iterator, eval_files,
            os.path.join(FLAGS.data_directory, 'testJava.csv'), FLAGS.num_projects, eval_sess)

    for i in range(FLAGS.num_iterations):
        try:
            train_model.train(train_sess, i)
        except tf.errors.OutOfRangeError:
            initialize_iterator(train_iterator, train_files,
                    os.path.join(FLAGS.data_directory, 'trainJava.csv'), FLAGS.num_projects, train_sess)
            train_model.train(train_sess, i)


        if i % FLAGS.eval_steps == 0:
            checkpoint_path = train_model.saver.save(train_sess, FLAGS.output_directory, global_step=i)
            eval_model.saver.restore(eval_sess, checkpoint_path)
            try:
                eval_model.eval(eval_sess)
            except tf.errors.OutOfRangeError:
                initialize_iterator(eval_iterator, eval_files,
                        os.path.join(FLAGS.data_directory, 'testJava.csv'), FLAGS.num_projects, eval_sess)
                eval_model.eval(eval_sess)


def initialize_iterator(iterator, file_placeholder, projects_file, num_projects, sess):
    with open(projects_file) as project_data:
        projects = np.array(project_data.read().splitlines())
        projects = projects[np.random.randint(len(projects), size=num_projects)]

    sess.run(iterator.initializer, feed_dict={file_placeholder: projects})

def create_iterator(FLAGS):
    java_files = tf.placeholder(tf.string, shape=[FLAGS.num_projects])

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
        pad = tf.constant(FLAGS.pad_id, dtype=tf.int32)
        dataset = dataset.apply(tf.contrib.data.padded_batch_and_drop_remainder(
                                    FLAGS.batch_size,
                                    padded_shapes=([None], [1], [None], [None], [1]),
                                    padding_values=(pad, tf.constant(0, dtype=tf.int32), pad, pad, tf.constant(0, dtype=tf.int32))))
        dataset = dataset.prefetch(FLAGS.batch_size)
        return dataset.make_initializable_iterator(), java_files


if __name__ == "__main__":
    tf.app.run()
