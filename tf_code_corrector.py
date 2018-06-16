""" TF Code Corrector Implementation """
import tensorflow as tf

tf.app.flags.DEFINE_string("data_directory", "", "Directory of the data set")
tf.app.flags.DEFINE_string("output_directory", "", "Output directory for checkpoints and tests")
tf.app.flags.DEFINE_string("batch_generator", "Java", "Batch Generator which is to be used; "
                                                    "must be one of: Java, Text")
tf.app.flags.DEFINE_integer("batch_size", 128, "Bath size for training input")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers of the network")
tf.app.flags.DEFINE_integer("num_units", 256, "Number of units in each layer")

FLAGS = tf.app.flags.FLAGS

def main(_):
    input = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, None), name='input')
    target = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, None), name='target')

    if FLAGS.batch_generator == "Java":
        batch_generator = JavaBatchGenerator(FLAGS.data_directory)
    elif FLAGS.batch_generator == "Text":
        raise NotImplementedError("TextBatchGenerator is not implemented yet")
    else:
        raise ValueError("batch_generator argument not recognized; must be one of: "
                         "Java, Text")

    rnn_layers = [tf.nn.rnn_cell.LSTMCell(FLAGS.num_units) for i in range(FLAGS.num_layers)]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell = multi_rnn_cell,
                                                        inputs = input,
                                                        dtype = tf.float32)

if __name__ == "__main__":
    tf.app.run()
