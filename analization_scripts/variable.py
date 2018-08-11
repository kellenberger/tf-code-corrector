import os
import tensorflow as tf
import re
from collections import Counter

tf.app.flags.DEFINE_string("eval_file", "", "Evaluation file of the return type corruption")
tf.app.flags.DEFINE_string("test_files_dir", "", "Directory of test files")
tf.app.flags.DEFINE_integer("eol_id", 4, "end-of-line id")

FLAGS = tf.app.flags.FLAGS

def main(_):
    total_n = 0
    corrected_n = 0
    total = Counter()
    corrected = Counter()
    with open(FLAGS.eval_file, 'r') as eval_file, \
        open(os.path.join(FLAGS.test_files_dir, 'variable.src'), 'r') as source_file, \
        open(os.path.join(FLAGS.test_files_dir, 'variable.tgt'), 'r') as target_file:
        while True:
            src = source_file.readline().strip()
            tgt = target_file.readline().strip()
            eval = eval_file.readline().strip()
            if not src or not tgt or not eval:
                break

            correct_variable = ''
            misspelled_variable = ''
            for i, c in enumerate(tgt):
                if c != src[i]:
                    for word in re.finditer(r'\b[a-zA-Z0-9_]+\b', tgt):
                        if word.start() <= i and word.end() >= i:
                            correct_variable = tgt[word.start():word.end()]
                            break
                    for word in re.finditer(r'\b[a-zA-Z0-9_]+\b', src):
                        if word.start() <= i and word.end() >= i:
                            misspelled_variable = src[word.start():word.end()]
                            break
                    break

            length = len(correct_variable)
            if length == 0:
                continue
            variable_count = len(list(re.finditer(r'\b' + correct_variable + r'\b', tgt)))
            correct_count = len(list(re.finditer(r'\b' + correct_variable + r'\b', eval)))
            incorrect_count = len(list(re.finditer(r'\b' + misspelled_variable + r'\b', eval)))

            total[str(length)] += 1
            total_n += 1
            if variable_count == correct_count or variable_count == incorrect_count:
                corrected[str(length)] += 1
                corrected_n += 1

    for l, n in sorted(total.most_common(), key=lambda x: int(x[0])):
        print("{}: {}/{}, {:.2f}%".format(l, corrected[l], n, (corrected[l] / float(n)) * 100))

    print("corrected_variable: {}/{}, {:.2f}%".format(corrected_n, total_n, (corrected_n / float(total_n)) * 100))



if __name__ == "__main__":
    tf.app.run()
