""" TF Code Corrector Implementation """
import tensorflow as tf

from batch_generators.java_batch_generator import JavaBatchGenerator
from models.train_model import TrainModel
from models.evaluation_model import EvaluationModel

tf.app.flags.DEFINE_string("data_directory", "", "Directory of the data set")
tf.app.flags.DEFINE_string("output_directory", "", "Output directory for checkpoints and tests")
tf.app.flags.DEFINE_string("batch_generator", "Java", "Batch Generator which is to be used; "
                                                    "must be one of: Java, Text")
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
      train_model = TrainModel(FLAGS)
      initializer = tf.global_variables_initializer()

    with eval_graph.as_default():
      eval_model = EvaluationModel(FLAGS)

    train_sess = tf.Session(graph=train_graph)
    eval_sess = tf.Session(graph=eval_graph)

    train_sess.run(initializer)

    for i in range(FLAGS.num_iterations):

      train_model.train(train_sess, i)

      if i % FLAGS.eval_steps == 0:
        checkpoint_path = train_model.saver.save(train_sess, FLAGS.output_directory, global_step=i)
        eval_model.saver.restore(eval_sess, checkpoint_path)
        eval_model.eval(eval_sess)


if __name__ == "__main__":
    tf.app.run()
