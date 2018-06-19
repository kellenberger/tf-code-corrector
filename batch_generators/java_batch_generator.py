"""Batch Generator for the Java Github Corpus"""
import numpy as np
import os
import random
import javalang

import batch_generator

class JavaBatchGenerator(batch_generator.BatchGenerator):

    def __init__(self, data_directory):
        self.data_directory = data_directory

        self.projects = []
        with open(os.path.join(self.data_directory, 'trainJava.csv'), 'r') as train_projects:
            for project in train_projects:
                if os.path.exists(os.path.join(self.data_directory, project.strip())):
                    self.projects.append(project.strip())

    def train_batch_generator(self, batch_size = 128):
        """
        Args:
            batch_size: Size of the batch_size
        Returns:
            A tuple of the form (input_batch, input_sequence_lengths,
            target_batch, target_sequence_lengths), where:
                input_batch: batch of input sequences
                input_sequence_lengths: array containing the lengths of
                    the input sequences
                target_batch: batch of respective target sequences
                target_sequence_lengths: array containing the lengths of
                    the target sequences
        """
        while True:
            selected_lines = []
            random_projects = np.random.choice(self.projects, size=batch_size)
            for random_project in random_projects:
                file = random.choice(os.listdir(os.path.join(self.data_directory, random_project)))
                with open(os.path.join(self.data_directory, random_project, file), 'r') as random_file:
                    lines = random_file.read().split("\n")
                    selected_line = ''
                    while len(selected_line) == 0:
                        selected_line = random.choice(lines).strip()
                    selected_lines.append(selected_line)

            target_batch = [[ char for char in line] for line in selected_lines]
            target_input_batch, target_output_batch, target_sequence_lengths = self._pad_output_array(target_batch)

            input_batch = []
            for line in selected_lines:
                char_array = [char for char in line]
                if len(char_array) > 1 and random.random() > 0.9:
                    drop_char = random.randint(0, len(char_array)-1)
                    if(drop_char >= len(char_array)):
                        print(drop_char)
                        print(char_array    )
                    del char_array[drop_char]
                input_batch.append(char_array[::-1])
            input_batch, input_sequence_lengths = self._pad_input_array(input_batch)

            yield input_batch, input_sequence_lengths, target_input_batch, target_output_batch, target_sequence_lengths
