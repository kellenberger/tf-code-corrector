"""Evaluation Model"""
import tensorflow as tf
import sys

from batch_generators.java_batch_generator import JavaBatchGenerator

class EvaluationModel:

    def __init__(self, FLAGS, iterator):
        batch_size = tf.placeholder_with_default(FLAGS.batch_size, shape=[])

        encoder_input, sequence_lengths, decoder_input, target_output, target_lengths = iterator.get_next()
        sequence_lengths = tf.reshape(sequence_lengths, [batch_size])
        target_lengths = tf.reshape(target_lengths, [batch_size])

        if FLAGS.reverse_input:
            encoder_input = tf.reverse(encoder_input, [1])

        encoder_input = tf.reshape(encoder_input, [batch_size, -1, 1])
        encoder_input = tf.cast(encoder_input, tf.float32)

        projection_layer = tf.layers.Dense(128, use_bias = False) # 128 characters can be represented in ASCII

        if FLAGS.cell_type == "lstm":
            encoder_layers = [tf.nn.rnn_cell.LSTMCell(FLAGS.num_units) for i in range(FLAGS.num_layers)]
        elif FLAGS.cell_type == "gru":
            encoder_layers = [tf.nn.rnn_cell.GRUCell(FLAGS.num_units) for i in range(FLAGS.num_layers)]
        elif FLAGS.cell_type == "rnn":
            encoder_layers = [tf.nn.rnn_cell.BasicRNNCell(FLAGS.num_units) for i in range(FLAGS.num_layers)]
        else:
            raise ValueError("Unknown cell type %s!" % FLAGS.cell_type)
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_layers)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell = encoder_cell,
                                                            inputs = encoder_input,
                                                            sequence_length = None if FLAGS.reverse_input else sequence_lengths,
                                                            dtype = tf.float32)

        if FLAGS.cell_type == "lstm":
            decoder_layers = [tf.nn.rnn_cell.LSTMCell(FLAGS.num_units) for i in range(FLAGS.num_layers)]
        elif FLAGS.cell_type == "gru":
            decoder_layers = [tf.nn.rnn_cell.GRUCell(FLAGS.num_units) for i in range(FLAGS.num_layers)]
        elif FLAGS.cell_type == "rnn":
            decoder_layers = [tf.nn.rnn_cell.BasicRNNCell(FLAGS.num_units) for i in range(FLAGS.num_layers)]
        else:
            raise ValueError("Unknown cell type %s!" % FLAGS.cell_type)
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_layers)

        if FLAGS.use_attention:
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                FLAGS.num_units, encoder_outputs,
                memory_sequence_length = None if FLAGS.reverse_input else sequence_lengths,
                scale=True)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism,
                attention_layer_size=FLAGS.num_units)

            decoder_initial_state = decoder_cell.zero_state(batch_size, dtype=tf.float32).clone(
              cell_state=encoder_state)
        else:
            decoder_initial_state = encoder_state

        # Helper
        def sample_fn(output):
            return tf.reshape(tf.argmax(output, axis=-1, output_type=tf.int32), [batch_size, 1])

        def end_fn(sample_ids):
            return tf.reshape(tf.equal(sample_ids, tf.constant(FLAGS.eos_id, tf.int32)), [batch_size])

        def next_inputs_fn(sample_ids):
            return tf.cast(sample_ids, tf.float32)

        start_inputs = tf.fill([batch_size, 1], FLAGS.sos_id)
        start_inputs = tf.cast(start_inputs, tf.float32)

        helper = tf.contrib.seq2seq.InferenceHelper(
                sample_fn=sample_fn,
                sample_shape=[1],
                sample_dtype=tf.int32,
                start_inputs=start_inputs,
                end_fn=end_fn,
                next_inputs_fn=next_inputs_fn)

        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, decoder_initial_state,
            output_layer=projection_layer)
        # Dynamic decoding
        maximum_iterations = tf.round(tf.reduce_max(sequence_lengths) * 2)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations = maximum_iterations)
        self.translations = outputs.sample_id

        self.encoder_input = encoder_input
        self.target_output = target_output

        self.reverse_input = FLAGS.reverse_input

        self.saver = tf.train.Saver(max_to_keep=10)

    def eval(self, session, silent=False):
        translations, target, input = session.run([self.translations, self.target_output, self.encoder_input])
        if not silent:
            s = ''
            inp = reversed(input[0]) if self.reverse_input else input[0]
            for c in inp:
                if c == 1:
                    continue
                s+= chr(c)
            print("Source: {}".format(s))
            s = ''
            for c in target[0]:
                s += chr(c)
            print("Target: {}".format(s))
            s = ''
            for c in translations[0]:
                s += chr(c)
            print("Actual: {}".format(s))
            sys.stdout.flush()
        return translations
