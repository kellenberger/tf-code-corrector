import os
import tensorflow as tf

tf.app.flags.DEFINE_string("eval_file", "", "Evaluation file of the uncorrupted")
tf.app.flags.DEFINE_string("test_files_dir", "", "Directory of test files")
tf.app.flags.DEFINE_integer("eol_id", 4, "end-of-line id")

FLAGS = tf.app.flags.FLAGS

def main(_):
    total = 0
    tolerance = 0
    wrong_switch = 0
    with open(FLAGS.eval_file, 'r') as eval_file, \
        open(os.path.join(FLAGS.test_files_dir, 'semicolon.tgt'), 'r') as target_file:
        while True:
            tgt = target_file.readline().strip()
            eval = eval_file.readline().strip()
            if not tgt or not eval:
                break

            total += 1

            for i, c in enumerate(tgt):
                if c != eval[i]:
                    print("target: {}".format(tgt[i:]))
                    print("eval:   {}".format(eval[i:]))
                    print()
                    break

            incorrect = False
            for i, c in enumerate(tgt):
                if c != eval[i] and ord(c) != ord(eval[i])-1 and ord(c) != ord(eval[i])+1:
                    incorrect = True
                    break

            if not incorrect:
                tolerance += 1

            eval_lines = eval.split(chr(FLAGS.eol_id))
            for i, tgt_line in enumerate(tgt.split(chr(FLAGS.eol_id))):
                if tgt_line != eval_lines[i]:
                    if i < len(eval_lines) - 1:
                        new_eval = eval_lines[:i] + [eval_lines[i+1]] + [eval_lines[i]] + eval_lines[i+2:]
                        new_eval = chr(FLAGS.eol_id).join(new_eval)
                        incorrect = False
                        for i, c in enumerate(tgt):
                            if c != new_eval[i] and ord(c) != ord(new_eval[i])-1 and ord(c) != ord(new_eval[i])+1:
                                incorrect = True
                                break

                        if not incorrect:
                            wrong_switch += 1
                    break

    print("tolerance: {}/{}, {:.2f}%".format(tolerance, total, (tolerance / float(total)) * 100))
    print("wrong switch: {}/{}, {:.2f}%".format(wrong_switch, total, (wrong_switch / float(total)) * 100))


if __name__ == "__main__":
    tf.app.run()
