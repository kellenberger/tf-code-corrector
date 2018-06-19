""" TF Code Corrector Implementation """
import tensorflow as tf

tf.app.flags.DEFINE_string("data_directory", "", "Directory of the data set")
tf.app.flags.DEFINE_string("output_directory", "", "Output directory for checkpoints and tests")
tf.app.flags.DEFINE_string("batch_generator", "Java", "Batch Generator which is to be used; "
                                                    "must be one of: Java, Text")
tf.app.flags.DEFINE_integer("batch_size", 128, "Bath size for training input")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers of the network")
tf.app.flags.DEFINE_integer("num_units", 256, "Number of units in each layer")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer")

FLAGS = tf.app.flags.FLAGS

def main(_):
    encoder_input = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, None, 1), name='encoder_input')
    sequence_lengths = tf.placeholder(tf.int32, shape=(FLAGS.batch_size), name='sequence_lengths')
    decoder_input = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, None, 1), name='decoder_input')
    target_output = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, None), name='target_output')
    target_lengths = tf.placeholder(tf.int32, shape=(FLAGS.batch_size), name="target_lengths")

    pad_code = tf.constant(128, dtype = tf.int32)

    target_weights = tf.to_float(tf.map_fn(
                                    lambda x: tf.map_fn(
                                                lambda y: tf.logical_not(tf.equal(y, pad_code)),
                                                x,
                                                dtype=tf.bool),
                                    target_output,
                                    dtype= tf.bool))

    if FLAGS.batch_generator == "Java":
        batch_generator = JavaBatchGenerator(FLAGS.data_directory)
    elif FLAGS.batch_generator == "Text":
        raise NotImplementedError("TextBatchGenerator is not implemented yet")
    else:
        raise ValueError("batch_generator argument not recognized; must be one of: "
                         "Java, Text")

    projection_layer = tf.layers.Dense(256, use_bias = False) # 256 characters can be represented in UTF-8

    encoder_layers = [tf.nn.rnn_cell.LSTMCell(FLAGS.num_units) for i in range(FLAGS.num_layers)]
    encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_layers)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell = encoder_cell,
                                                        inputs = tf.to_float(encoder_input),
                                                        sequence_length = sequence_lengths,
                                                        dtype = tf.float32)

    decoder_layers = [tf.nn.rnn_cell.LSTMCell(FLAGS.num_units) for i in range(FLAGS.num_layers)]
    decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_layers)
    helper = tf.contrib.seq2seq.TrainingHelper(tf.to_float(decoder_input), target_lengths)
    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, encoder_state,
        output_layer=projection_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
    logits = outputs.rnn_output

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_output, logits=logits)
    train_loss = (tf.reduce_sum(crossent * target_weights) / FLAGS.batch_size)

    train_perplexity = tf.exp(tain_loss)

    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(
        gradients, FLAGS.max_gradient_norm)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    update_step = optimizer.apply_gradients(
        zip(clipped_gradients, params))

if __name__ == "__main__":
    tf.app.run()
