import os
import tensorflow as tf
from collections import Counter

tf.app.flags.DEFINE_string("eval_file", "", "Evaluation file of the return type corruption")
tf.app.flags.DEFINE_string("test_files_dir", "", "Directory of test files")

FLAGS = tf.app.flags.FLAGS

def main(_):
    correct = Counter()
    occurrences = Counter()

    with open(FLAGS.eval_file, 'r') as eval_file, \
        open(os.path.join(FLAGS.test_files_dir, 'method_return.src'), 'r') as source_file, \
        open(os.path.join(FLAGS.test_files_dir, 'method_return.tgt'), 'r') as target_file:
        while True:
            src = source_file.readline()
            tgt = target_file.readline()
            eval = eval_file.readline()
            if not src or not tgt or not eval:
                break

            type = ''
            for i, c in enumerate(src):
                if c != tgt[i]:
                    type = tgt[i:].split(' ')[0]
                    break

            occurrences[type] += 1
            if tgt == eval:
                correct[type] += 1

    values = []
    for t, n in occurrences.most_common():
        values.append((t, correct[t], n, (correct[t] / float(n)) * 100))

    for v in sorted(values, key=lambda x: -x[3]):
        print("{} {}/{}, {:.2f}%".format(v[0], v[1], v[2], v[3]))


if __name__ == "__main__":
    tf.app.run()
