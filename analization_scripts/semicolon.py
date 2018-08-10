import os
import tensorflow as tf

tf.app.flags.DEFINE_string("eval_file", "", "Evaluation file of the return type corruption")
tf.app.flags.DEFINE_string("test_files_dir", "", "Directory of test files")

FLAGS = tf.app.flags.FLAGS

def main(_):
    with open(FLAGS.eval_file, 'r') as eval_file, \
        open(os.path.join(FLAGS.test_files_dir, 'semicolon.src'), 'r') as source_file, \
        open(os.path.join(FLAGS.test_files_dir, 'semicolon.tgt'), 'r') as target_file:
        while True:
            src = source_file.readline().strip()
            tgt = target_file.readline().strip()
            eval = eval_file.readline().strip()
            if not src or not tgt or not eval:
                break

            for i, c in enumerate(tgt):
                if c != src[i]:
                    if tgt != eval:
                        print("target: {}, eval: {}".format(tgt[i-5:i+5], eval[i-5:i+5]))
                    break


if __name__ == "__main__":
    tf.app.run()
