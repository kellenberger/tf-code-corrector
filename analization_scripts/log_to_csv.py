import os
import tensorflow as tf

tf.app.flags.DEFINE_string("log_file", "", "Log file to read loss from")

FLAGS = tf.app.flags.FLAGS

def main(_):
    with open(FLAGS.log_file, 'r') as log, \
        open(FLAGS.log_file.split('.')[0] + '.csv', 'w') as csv:
        log_data = log.read().split(' ')
        for i, token in enumerate(log_data):
            if token == "loss:":
                loss = log_data[i+1]
                loss = loss[:-1]
                csv.write(loss + ";\n")


if __name__ == "__main__":
    tf.app.run()
