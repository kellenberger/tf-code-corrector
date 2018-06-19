"""Train Model"""
import tensorflow as tf

from batch_generators.java_batch_generator import JavaBatchGenerator

class TrainModel:

    def __init__(self, FLAGS):

        encoder_input = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, None), name='encoder_input')
        sequence_lengths = tf.placeholder(tf.int32, shape=(FLAGS.batch_size), name='sequence_lengths')
        decoder_input = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, None), name='decoder_input')
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
            self.batch_generator = JavaBatchGenerator(FLAGS.data_directory).train_batch_generator(FLAGS.batch_size)
        elif FLAGS.batch_generator == "Text":
            raise NotImplementedError("TextBatchGenerator is not implemented yet")
        else:
            raise ValueError("batch_generator argument not recognized; must be one of: "
                             "Java, Text")

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
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, target_lengths)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, encoder_state,
            output_layer=projection_layer)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        train_loss = (tf.reduce_sum(crossent * target_weights) / FLAGS.batch_size)


        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, FLAGS.max_gradient_norm)

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self.update_step = optimizer.apply_gradients(
            zip(clipped_gradients, params))

        self.train_loss = train_loss

        self.encoder_input = encoder_input
        self.sequence_lengths = sequence_lengths
        self.decoder_input = decoder_input
        self.target_output = target_output
        self.target_lengths = target_lengths

        self.saver = tf.train.Saver()

    def train(self, session, i):
        encoder_input, sequence_lengths, decoder_input, target_output, target_lengths = self.batch_generator.next()
        _, loss = session.run([self.update_step, self.train_loss],
                            feed_dict = {
                                self.encoder_input: encoder_input,
                                self.sequence_lengths: sequence_lengths,
                                self.decoder_input: decoder_input,
                                self.target_output: target_output,
                                self.target_lengths: target_lengths
                            })
        print("iteration {}, loss: {:.2f}".format(i, loss))
