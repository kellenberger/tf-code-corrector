"""Preprocesses Java Github Corpus."""
import os
import tensorflow as tf
from shutil import copyfile
import re

tf.app.flags.DEFINE_string("java_directory", "", "Java directory path")
tf.app.flags.DEFINE_string("split_directory", "", "Split dirctory path")
tf.app.flags.DEFINE_string("out_directory", "", "Directory to write processed data to")

FLAGS = tf.app.flags.FLAGS


def main(_):
    assert FLAGS.java_directory
    assert FLAGS.split_directory
    assert FLAGS.out_directory

    with open(os.path.join(FLAGS.out_directory, 'eval.java'), 'w') as eval_file:
        for i in range(256):
            eval_file.write("public class A { public int getOne(){ return 1; } }\n")

    _write_files_to_new_location('trainJava.csv', 'train_')
    _write_files_to_new_location('testJava.csv', 'test_')


def _write_files_to_new_location(source_file, output_name):
    project_count = 0

    with open(os.path.join(FLAGS.split_directory, source_file), 'r') as source_projects:
        for line in source_projects:
            project_count += 1

    with open(os.path.join(FLAGS.split_directory, source_file), 'r') as source_projects:
        file_count = 0
        char_count = 0
        output_file = open(os.path.join(FLAGS.out_directory, output_name + str(file_count) + '.java'), 'w')
        for i, project in enumerate(source_projects):
            print('Source project: {}/{}'.format(i, project_count))
            project = project.strip()
            for subdir, _, files in os.walk(os.path.join(FLAGS.java_directory, project)):
                for file in files:
                    if file.endswith('.java') and not file.startswith('.'):
                        with open(os.path.join(subdir, file), 'r') as file_data:
                            content = file_data.read()
                            content = _remove_comments(content).strip()
                            content = re.sub('\s+', ' ', content)
                            if content:
                                output_file.write(content)
                                output_file.write("\n")
                                char_count += len(content)
                                if char_count >= 100000000:  # ~ File size 100MB
                                    file_count += 1
                                    char_count = 0
                                    output_file.close()
                                    output_file = open(os.path.join(FLAGS.out_directory, output_name + str(file_count) + '.java'), 'w')

def _remove_comments(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

if __name__ == "__main__":
    tf.app.run()
