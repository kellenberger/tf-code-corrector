"""Evaluation Model"""
import tensorflow as tf
import sys
import math

from batch_generators.java_batch_generator import JavaBatchGenerator

class EvaluationModel:

    def __init__(self, FLAGS, iterator):
        batch_size = tf.placeholder_with_default(FLAGS.batch_size, shape=[])

        encoder_input, sequence_lengths, decoder_input, target_output, target_lengths = iterator.get_next()
        sequence_lengths = tf.reshape(sequence_lengths, [batch_size])
        target_lengths = tf.reshape(target_lengths, [batch_size])
        encoder_input = tf.reverse(encoder_input, [1])

        encoder_input = tf.reshape(encoder_input, [batch_size, -1, 1])
        encoder_input = tf.cast(encoder_input, tf.float32)

        projection_layer = tf.layers.Dense(128, use_bias = False) # 128 characters can be represented in ASCII

        layers_per_gpu = int(math.ceil(FLAGS.num_layers / float(FLAGS.num_gpus)))

        encoder_layers = []
        layers_to_go = FLAGS.num_layers
        for i in range(FLAGS.num_gpus):
            layers = min(layers_per_gpu, layers_to_go)
            for j in range(layers):
                encoder_layers.append(tf.contrib.rnn.DeviceWrapper(tf.nn.rnn_cell.LSTMCell(FLAGS.num_units), "/gpu:%d" % i))
            layers_to_go -= layers
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_layers)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell = encoder_cell,
                                                            inputs = encoder_input,
                                                            dtype = tf.float32)

        decoder_layers = []
        layers_to_go = FLAGS.num_layers
        for i in range(FLAGS.num_gpus):
            layers = min(layers_per_gpu, layers_to_go)
            for j in range(layers):
                decoder_layers.append(tf.contrib.rnn.DeviceWrapper(tf.nn.rnn_cell.LSTMCell(FLAGS.num_units), "/gpu:%d" % i))
            layers_to_go -= layers
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_layers)

        if FLAGS.use_attention:
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                FLAGS.num_units, encoder_outputs,
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

        self.saver = tf.train.Saver(max_to_keep=10)

    def eval(self, session, silent=False):
        translations, target, input = session.run([self.translations, self.target_output, self.encoder_input])
        if not silent:
            s = ''
            for c in reversed(input[0]):
                if c == 128:
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
