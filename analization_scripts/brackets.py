import os
import tensorflow as tf

tf.app.flags.DEFINE_string("eval_file", "", "Evaluation file of the brackets corruption")

FLAGS = tf.app.flags.FLAGS

def main(_):
    total = 0
    balanced_brackets = 0
    matching_brackets = 0
    with open(FLAGS.eval_file, 'r') as eval_file:
        for line in eval_file:
            total += 1

            round = 0
            square = 0
            curly = 0
            for c in line:
                if c == '(':
                    round += 1
                elif c == ')':
                    round -= 1
                elif c == '{':
                    curly += 1
                elif c == '}':
                    curly -= 1
                elif c == '[':
                    square += 1
                elif c == ']':
                    square -= 1

            if round == 0 and square == 0 and curly == 0:
                balanced_brackets += 1

            brackets = []
            error = False
            for c in line:
                if c == '(' or c == '{' or c == '[':
                    brackets.append(c)
                elif c == ')':
                    if not len(brackets) == 0 and brackets[-1] == '(':
                        brackets.pop()
                    else:
                        error = True
                        break;
                elif c == '}':
                    if not len(brackets) == 0 and brackets[-1] == '{':
                        brackets.pop()
                    else:
                        error = True
                        break;
                elif c == ']':
                    if not len(brackets) == 0 and brackets[-1] == '[':
                        brackets.pop()
                    else:
                        error = True
                        break;

            if not error and len(brackets) == 0:
                matching_brackets += 1

    print("balanced brackets {}/{}, {:.2f}%".format(balanced_brackets, total, (balanced_brackets / float(total)) * 100))
    print("matching brackets {}/{}, {:.2f}%".format(matching_brackets, total, (matching_brackets / float(total)) * 100))


if __name__ == "__main__":
    tf.app.run()
