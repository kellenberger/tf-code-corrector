"""Preprocesses Java Github Corpus."""
import os
import tensorflow as tf
from shutil import copyfile
import re
import sys
import javalang
from ..corruptors import java_corruptor

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
            sys.stdout.flush()
            project = project.strip()
            for subdir, _, files in os.walk(os.path.join(FLAGS.java_directory, project)):
                for file in files:
                    if file.endswith('.java') and not file.startswith('.'):
                        with open(os.path.join(subdir, file), 'r') as file_data:
                            content = file_data.read()
                            content = _remove_comments(content).strip()
                            content = re.sub('\s+', ' ', content)
                            if len([char for char in content if ord(char)>127]) > 0:
                                continue
                            methods = _get_methods(content)
                            for method in methods:
                                if method and java_corruptor.corruptable(method):
                                    output_file.write(method)
                                    output_file.write("\n")
                                    char_count += len(method)
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

def _find_closing_bracket(text):
    open_brackets = 0
    for i in range(len(text)):
        char = text[i]
        if char == '{':
            open_brackets += 1
        elif char == '}':
            open_brackets -= 1
            if open_brackets == 0:
                return i
    return -1

def _get_methods(text):
    try:
        tree = javalang.parse.parse(text)
    except:
        return []

    methods = []
    try:
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            if node.return_type:
                method_name = node.return_type.name
            else:
                method_name = "void"
            method_name += " " + node.name

            method_index = text.find(method_name)
            if method_index == -1:
                continue

            pre_method = text[:method_index][::-1]
            semi = pre_method.find(';')
            bracket = pre_method.find('}')
            opening_bracket = pre_method.find('{')
            if bracket == -1 or (opening_bracket != -1 and opening_bracket < bracket):
                bracket = pre_method.find('{')

            if semi == -1 and bracket == -1:
                continue
            if semi == -1:
                start_index = bracket
            elif bracket == -1:
                start_index = semi
            else:
                start_index = min(semi, bracket)

            start_index = method_index - (start_index - 1)

            open_end_method = text[start_index:]
            end_index = _find_closing_bracket(open_end_method)
            if end_index == -1:
                continue

            methods.append(open_end_method[:end_index + 1])
    except RuntimeError:
        return []

    return methods

if __name__ == "__main__":
    tf.app.run()
