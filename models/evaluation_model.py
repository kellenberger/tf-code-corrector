"""Evaluation Model"""
import tensorflow as tf

from batch_generators.java_batch_generator import JavaBatchGenerator

class EvaluationModel:

    def __init__(self, FLAGS, iterator):

        encoder_input, sequence_lengths, decoder_input, target_output, target_lengths = iterator.get_next()
        encoder_input = tf.reverse(encoder_input, [1])
        sequence_lengths = tf.reshape(sequence_lengths, [FLAGS.batch_size])
        target_lengths = tf.reshape(target_lengths, [FLAGS.batch_size])

        # Embedding
        embedding = tf.get_variable("embedding", [256, 10], dtype=tf.float32)
        encoder_emb_inp = tf.nn.embedding_lookup(embedding, encoder_input)
        decoder_emb_inp = tf.nn.embedding_lookup(embedding, decoder_input)

        projection_layer = tf.layers.Dense(256, use_bias = False) # 256 characters can be represented in UTF-8

        encoder_layers = [tf.nn.rnn_cell.LSTMCell(FLAGS.num_units) for i in range(FLAGS.num_layers)]
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_layers)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell = encoder_cell,
                                                            inputs = encoder_emb_inp,
                                                            sequence_length = sequence_lengths,
                                                            dtype = tf.float32)

        decoder_layers = [tf.nn.rnn_cell.LSTMCell(FLAGS.num_units) for i in range(FLAGS.num_layers)]
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_layers)

        # Create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            FLAGS.num_units, encoder_outputs,
            memory_sequence_length=sequence_lengths)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=FLAGS.num_units)

        decoder_initial_state = decoder_cell.zero_state(FLAGS.batch_size, dtype=tf.float32).clone(
          cell_state=encoder_state)

        # Helper
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding,
            tf.fill([FLAGS.batch_size], FLAGS.sos_id), FLAGS.eos_id)

        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, decoder_initial_state,
            output_layer=projection_layer)
        # Dynamic decoding
        maximum_iterations = tf.round(tf.reduce_max(sequence_lengths) * 2)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations = maximum_iterations)
        self.translations = outputs.sample_id

        self.target_output = target_output

        self.saver = tf.train.Saver()

    def eval(self, session):
        translations, target = session.run([self.translations, self.target_output])
        for i in range(10):
            s = ''
            for c in target[i]:
                s += chr(c)
            print("Target: {}".format(s))
            s = ''
            for c in translations[i]:
                s += chr(c)
            print("Actual: {}".format(s))
