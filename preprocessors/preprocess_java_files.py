"""Preprocesses Java Github Corpus."""
import os
import tensorflow as tf
from shutil import copyfile

tf.app.flags.DEFINE_string("java_directory", "", "Java directory path")
tf.app.flags.DEFINE_string("split_directory", "", "Split dirctory path")
tf.app.flags.DEFINE_string("out_directory", "", "Directory to write processed data to")

FLAGS = tf.app.flags.FLAGS


def main(_):
    assert FLAGS.java_directory
    assert FLAGS.split_directory
    assert FLAGS.out_directory

    project_count = 0

    with open(os.path.join(FLAGS.split_directory, 'trainJava.csv'), 'r') as train_projects:
        for line in train_projects:
            project_count += 1

    with open(os.path.join(FLAGS.split_directory, 'trainJava.csv'), 'r') as train_projects:
        for i, project in enumerate(train_projects):
            print('Train project: {}/{}'.format(i, project_count))
            project = project.strip()
            os.makedirs(os.path.join(FLAGS.out_directory, project))
            for subdir, _, files in os.walk(os.path.join(FLAGS.java_directory, project)):
                for file in files:
                    if file.endswith('.java') and not file.startswith('.'):
                        copyfile(os.path.join(subdir, file), os.path.join(FLAGS.out_directory, project, file))

    project_count = 0

    with open(os.path.join(FLAGS.split_directory, 'testJava.csv'), 'r') as test_projects:
        for line in test_projects:
            project_count += 1

    with open(os.path.join(FLAGS.split_directory, 'testJava.csv'), 'r') as test_projects:
        for i, project in enumerate(test_projects):
            print('Test project: {}/{}'.format(i, project_count))
            project = project.strip()
            os.makedirs(os.path.join(FLAGS.out_directory, project))
            for subdir, _, files in os.walk(os.path.join(FLAGS.java_directory, project)):
                for file in files:
                    if file.endswith('.java') and not file.startswith('.'):
                        copyfile(os.path.join(subdir, file), os.path.join(FLAGS.out_directory, project, file))

if __name__ == "__main__":
    tf.app.run()
