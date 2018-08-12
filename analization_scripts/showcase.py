import os
import tensorflow as tf
import re
import random

tf.app.flags.DEFINE_string("eval_file", "", "Evaluation file")
tf.app.flags.DEFINE_string("test_files_dir", "", "Directory of test files")
tf.app.flags.DEFINE_integer("eol_id", 4, "end-of-line id")

FLAGS = tf.app.flags.FLAGS

def main(_):
    targets = []
    sources = []
    evals = []
    with open(FLAGS.eval_file, 'r') as eval_file, \
        open(os.path.join(FLAGS.test_files_dir, FLAGS.eval_file.split('.')[0].split('/')[-1] + '.src'), 'r') as source_file, \
        open(os.path.join(FLAGS.test_files_dir, FLAGS.eval_file.split('.')[0].split('/')[-1] + '.tgt'), 'r') as target_file:
        for line in target_file:
            targets.append(line)

        for line in source_file:
            sources.append(line)

        for line in eval_file:
            evals.append(line)

    zipped = list(zip(targets, sources, evals))
    random.shuffle(zipped)

    correct_example = False
    incorrect_example = False
    for tgt, src, eval in zipped:
        if tgt == eval and not correct_example:
            correct_example = True
            print("CORRECT:")
            print(re.sub(chr(FLAGS.eol_id), "\n", src))
            print(re.sub(chr(FLAGS.eol_id), "\n", eval))
            print("\n-------------\n")
        if tgt != eval and not incorrect_example:
            incorrect_example = True
            print("INCORRECT:")
            print(re.sub(chr(FLAGS.eol_id), "\n", tgt))
            print(re.sub(chr(FLAGS.eol_id), "\n", src))
            print(re.sub(chr(FLAGS.eol_id), "\n", eval))
            print("\n-------------\n")
        if correct_example and incorrect_example:
            break



if __name__ == "__main__":
    tf.app.run()
