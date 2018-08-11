import os
import tensorflow as tf
import re
import javalang
from collections import Counter

tf.app.flags.DEFINE_string("eval_file", "", "Evaluation file of the return type corruption")
tf.app.flags.DEFINE_string("test_files_dir", "", "Directory of test files")
tf.app.flags.DEFINE_integer("eol_id", 4, "end-of-line id")

FLAGS = tf.app.flags.FLAGS

def main(_):
    total = Counter()
    correct = Counter()
    with open(FLAGS.eval_file, 'r') as eval_file, \
        open(os.path.join(FLAGS.test_files_dir, 'switch.src'), 'r') as source_file, \
        open(os.path.join(FLAGS.test_files_dir, 'switch.tgt'), 'r') as target_file:
        while True:
            src = source_file.readline().strip()
            tgt = target_file.readline().strip()
            eval = eval_file.readline().strip()
            if not src or not tgt or not eval:
                break

            split_src = src.split(chr(FLAGS.eol_id))
            switch_index = 0
            for i, line in enumerate(tgt.split(chr(FLAGS.eol_id))):
                if line != split_src[i]:
                    switch_index = i
                    break

            s = "class A {\n" + re.sub(chr(FLAGS.eol_id), "\n", tgt) + "\n}"
            tree = javalang.parse.parse(s)

            first_line = ''
            second_line = ''
            for _, node in tree.filter(javalang.tree.LocalVariableDeclaration):
                line = node.position[0] - 2
                if line == switch_index:
                    first_line = type(node)
                elif line == switch_index + 1:
                    second_line = type(node)

            for _, node in tree.filter(javalang.tree.StatementExpression):
                node = node.children[1]
                if not isinstance(node, javalang.tree.Assignment) and not isinstance(node, javalang.tree.MethodInvocation):
                    continue
                if hasattr(node, 'position') and node.position:
                    line = node.position[0] - 2
                elif hasattr(node.children[0], 'position') and node.children[0].position:
                    line = node.children[0].position[0] - 2
                else:
                    continue

                if line == switch_index:
                    first_line = type(node)
                elif line == switch_index + 1:
                    second_line = type(node)

            switch = first_line.__name__ + '<->' + second_line.__name__
            total[switch] += 1
            if eval == tgt:
                correct[switch] += 1

        for l, n in sorted(total.most_common(), key=lambda x: -x[1]):
            print("{}: {}/{}, {:.2f}%".format(l, correct[l], n, (correct[l] / float(n)) * 100))


if __name__ == "__main__":
    tf.app.run()
