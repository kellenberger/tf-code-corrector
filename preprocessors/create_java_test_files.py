"""Creates a test file for each introduced error"""
import os
import random

tf.app.flags.DEFINE_string("java_directory", "", "Java directory path")
tf.app.flags.DEFINE_integer("max_sequence_length", 100, "Maximal length for "
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

def create_test_file(dir, file_name, project_files, pertubation_fn):
    line_count = 0
    random.shuffle(project_files)
    with open(os.path.join(dir, file_name + '.src'), 'w') as source_file, \
            open(os.path.join(dir, file_name + '.tgt'), 'w') as target_file:
        for project in project_files:
            if line_count == FLAGS.lines_per_file:
                break
            with(open(project, 'r') as project_data:
                lines = project_data.readlines()
                random_lines = random.sample(lines, 100)
                for line in random_lines:
                    if line_count == FLAGS.lines_per_file:
                        break
                    tgt_line = line.strip()
                    src_line = pertubation_fn(line.strip())
                    if src_line:
                        source_file.write(src_line + "\n")
                        target_file.write(tgt_line + "\n")
                        line_count += 1


def main(_):
    test_directory = os.path.join(FLAGS.java_directory, 'test_files')
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    test_projects = []
    with open(os.path.join(FLAGS.java_directory, 'testJava.csv'), 'r') as source_projects:
        for project in source_projects:
            test_projects.append(os.path.join(FLAGS.java_directory, project.strip() + '.java'))

    create_test_file(test_directory, 'typos', test_projects, add_typo)
    create_test_file(test_directory, 'brackets', test_projects, remove_bracket)
    create_test_file(test_directory, 'semicolon', test_projects, remove_semicolon)
    create_test_file(test_directory, 'unperturbed', test_projects, do_nothing)

if __name__ == "__main__":
    tf.app.run()
