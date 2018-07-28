"""Train Model"""
import tensorflow as tf
import time
import sys

from batch_generators.java_batch_generator import JavaBatchGenerator

class TrainModel:

    def __init__(self, FLAGS, iterator):

        encoder_input, sequence_lengths, decoder_input, target_output, target_lengths = iterator.get_next()
        sequence_lengths = tf.reshape(sequence_lengths, [FLAGS.batch_size])
        target_lengths = tf.reshape(target_lengths, [FLAGS.batch_size])
        encoder_input = tf.reverse(encoder_input, [1])

        encoder_input = tf.reshape(encoder_input, [FLAGS.batch_size, -1, 1])
        decoder_input = tf.reshape(decoder_input, [FLAGS.batch_size, -1, 1])

        encoder_input = tf.cast(encoder_input, tf.float32)
        decoder_input = tf.cast(decoder_input, tf.float32)

        pad_code = tf.constant(FLAGS.pad_id, dtype = tf.int32)

        target_weights = tf.to_float(tf.map_fn(
                                        lambda x: tf.map_fn(
                                                    lambda y: tf.logical_not(tf.equal(y, pad_code)),
                                                    x,
                                                    dtype=tf.bool),
                                        target_output,
                                        dtype= tf.bool))

        projection_layer = tf.layers.Dense(128, use_bias = False) # 128 characters can be represented in ASCII


        encoder_layers = []
        for i in range(FLAGS.num_layers):
            encoder_layers.append(tf.contrib.rnn.DeviceWrapper(tf.nn.rnn_cell.LSTMCell(FLAGS.num_units), "/gpu:%d" % (i % FLAGS.num_gpus)))
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_layers)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell = encoder_cell,
                                                            inputs = encoder_input,
                                                            dtype = tf.float32)

        decoder_layers = []
        for i in range(FLAGS.num_layers):
            decoder_layers.append(tf.contrib.rnn.DeviceWrapper(tf.nn.rnn_cell.LSTMCell(FLAGS.num_units), "/gpu:%d" % (i % FLAGS.num_gpus)))
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_layers)

        # Create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            FLAGS.num_units, encoder_outputs,
            scale=True)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=FLAGS.num_units)

        decoder_initial_state = decoder_cell.zero_state(FLAGS.batch_size, dtype=tf.float32).clone(
          cell_state=encoder_state)

        helper = tf.contrib.seq2seq.TrainingHelper(decoder_input, target_lengths)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, decoder_initial_state,
            output_layer=projection_layer)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        train_loss = (tf.reduce_sum(crossent * target_weights) / FLAGS.batch_size)


        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params, colocate_gradients_with_ops=True)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, FLAGS.max_gradient_norm)

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self.update_step = optimizer.apply_gradients(
            zip(clipped_gradients, params))

        self.train_loss = train_loss

        self.saver = tf.train.Saver()
        self.start_time = None

    def train(self, session, i):
        if not self.start_time:
            self.start_time = time.time()
        _, loss = session.run([self.update_step, self.train_loss])
        if i % 100 == 0:
            end_time = time.time()
            print("iteration {}, loss: {:.2f}, minutes: {:.2f}".format(i, loss, (end_time - self.start_time)/60))
            sys.stdout.flush()
            self.start_time = None
