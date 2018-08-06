"""Creates a test file for each introduced error"""
import os
import random
import tensorflow as tf
import sys
import glob
from ..corruptors import java_corruptor

tf.app.flags.DEFINE_string("java_directory", "", "Java directory path")
tf.app.flags.DEFINE_integer("max_sequence_length", 300, "Maximal length for "
                                                    "considered sequences")
tf.app.flags.DEFINE_integer("lines_per_file", 8192, "Lines in each test file")

FLAGS = tf.app.flags.FLAGS

def create_uncorrupted_file(dir, file_name, project_files):
    line_count = 0
    with open(os.path.join(dir, file_name + '.src'), 'w') as source_file, \
            open(os.path.join(dir, file_name + '.tgt'), 'w') as target_file:
        while True:
            project = random.choice(project_files)
            print("Line Count: {}".format(line_count))
            sys.stdout.flush()
            if line_count == FLAGS.lines_per_file:
                break
            with open(project, 'r') as project_data:
                lines = project_data.readlines()
                suitable_lines = [line for line in lines if len(line) <= FLAGS.max_sequence_length]
                random.shuffle(suitable_lines)
                for line in suitable_lines:
                    if line_count == FLAGS.lines_per_file:
                        break
                    tgt_line = line.strip()
                    src_line = tgt_line
                    if src_line:
                        source_file.write(src_line + "\n")
                        target_file.write(tgt_line + "\n")
                        line_count += 1

def create_corrupted_file(dir, file_name, uncorrupted_file, corruption_fn):
    with open(os.path.join(dir, uncorrupted_file + '.tgt'), 'r') as uncorrupted_data, \
            open(os.path.join(dir, file_name + '.src'), 'w') as source_file, \
            open(os.path.join(dir, file_name + '.tgt'), 'w') as target_file:
        for line in uncorrupted_data:
            tgt_line = line.strip()

            s = java_corruptor._prepare(tgt_line)
            s = corruption_fn(s)
            src_line = java_corruptor._unprepare(s)
            if tgt_line == src_line:
                print 'unable to corrupt'
            source_file.write(src_line + "\n")
            target_file.write(tgt_line + "\n")


def main(_):
    test_directory = os.path.join(FLAGS.java_directory, 'test_files')
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    test_projects = glob.glob(os.path.join(FLAGS.java_directory, 'test*.java'))

    print('Create Uncorrupted')
    create_uncorrupted_file(test_directory, 'uncorrupted', test_projects)
    print('Create Brackets')
    create_corrupted_file(test_directory, 'brackets', 'uncorrupted', java_corruptor._remove_bracket)
    print('Create Semicolon')
    create_corrupted_file(test_directory, 'semicolon', 'uncorrupted', java_corruptor._remove_semicolon)
    print('Create Variable')
    create_corrupted_file(test_directory, 'variable', 'uncorrupted', java_corruptor._misspell_variable)
    print('Create Switch')
    create_corrupted_file(test_directory, 'switch', 'uncorrupted', java_corruptor._switch_statement_lines)
    print ('Method return')
    create_corrupted_file(test_directory, 'method_return', 'uncorrupted', java_corruptor._change_method_return)


if __name__ == "__main__":
    tf.app.run()
