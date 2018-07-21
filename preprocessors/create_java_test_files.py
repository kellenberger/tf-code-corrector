"""Creates a test file for each introduced error"""
import os
import random
import tensorflow as tf
import sys
import glob
from ..corruptors import java_corruptor

tf.app.flags.DEFINE_string("java_directory", "", "Java directory path")
tf.app.flags.DEFINE_integer("max_sequence_length", 1000, "Maximal length for "
                                                    "considered sequences")
tf.app.flags.DEFINE_integer("lines_per_file", 8192, "Lines in each test file")

FLAGS = tf.app.flags.FLAGS

def add_typo(line):
    if len(line) <= 1:
        return None

    change_char = random.randint(0, len(line)-2)
    return line[:change_char] + line[change_char+1] + line[change_char] + line[change_char+2:]

def remove_bracket(line):
    if len(line) <= 1:
        return None

    brackets = ['(', ')', '[', ']', '{', '}']
    bracket_indices = [i for i, c in enumerate(line) if c in brackets]
    if not bracket_indices:
        return None

    drop_index = random.choice(bracket_indices)
    return line[:drop_index] + line[drop_index+1:]

def remove_semicolon(line):
    if len(line) <= 1:
        return None

    semicolon_indices = [i for i, c in enumerate(line) if c == ';']
    if not semicolon_indices:
        return None

    drop_index = random.choice(semicolon_indices)
    return line[:drop_index] + line[drop_index+1:]

def do_nothing(line):
    return line

def create_test_file(dir, file_name, project_files, pertubation_fn, unchanged_allowed = False):
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
                lines = [line for line in lines if len(line) <= FLAGS.max_sequence_length]
                if len(lines) < 500:
                    continue
                random_lines = random.sample(lines, 500)
                for line in random_lines:
                    if line_count == FLAGS.lines_per_file:
                        break
                    tgt_line = line.strip()
                    src_line = pertubation_fn(line.strip())
                    if src_line and (src_line != tgt_line or unchanged_allowed):
                        source_file.write(src_line + "\n")
                        target_file.write(tgt_line + "\n")
                        line_count += 1


def main(_):
    with open(os.path.join(FLAGS.java_directory, 'eval.java'), 'w') as eval_file:
        for i in range(256):
            eval_file.write("public class A { public int getOne(){ return 1; } }\n")

    test_directory = os.path.join(FLAGS.java_directory, 'test_files')
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    test_projects = glob.glob(os.path.join(FLAGS.java_directory, 'test*.java'))

    print('Create Brackets')
    create_test_file(test_directory, 'brackets', test_projects, java_corruptor._remove_bracket)
    print('Create Semicolon')
    create_test_file(test_directory, 'semicolon', test_projects, java_corruptor._remove_semicolon)
    print('Create Variable')
    create_test_file(test_directory, 'variable', test_projects, java_corruptor._misspell_variable)
    print('Create Switch')
    create_test_file(test_directory, 'switch', test_projects, java_corruptor._switch_statement_lines)
    print('Create Unperturbed')
    create_test_file(test_directory, 'unperturbed', test_projects, do_nothing, unchanged_allowed=True)

if __name__ == "__main__":
    tf.app.run()
