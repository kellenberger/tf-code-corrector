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

    _write_files_to_new_location('trainJava.csv')
    _write_files_to_new_location('testJava.csv')


def _write_files_to_new_location(source_file):
    project_count = 0

    with open(os.path.join(FLAGS.split_directory, source_file), 'r') as source_projects:
        for line in source_projects:
            project_count += 1

    with open(os.path.join(FLAGS.split_directory, source_file), 'r') as source_projects:
        for i, project in enumerate(source_projects):
            print('Source project: {}/{}'.format(i, project_count))
            project = project.strip()
            with open(os.path.join(FLAGS.out_directory, project + '.java'), 'w') as project_file:
                for subdir, _, files in os.walk(os.path.join(FLAGS.java_directory, project)):
                    for file in files:
                        if file.endswith('.java') and not file.startswith('.'):
                            with open(os.path.join(subdir, file), 'r') as file_data:
                                content = file_data.read()
                                content = _remove_comments(content).strip()
                                if content:
                                    project_file.write(content)
                                    project_file.write("\n\n")
    copyfile(os.path.join(FLAGS.split_directory, source_file), os.path.join(FLAGS.out_directory, source_file))

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
