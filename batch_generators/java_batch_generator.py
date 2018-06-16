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
            A tuple of the form (input_batch, target_batch), where:
                input_batch: batch of input sequences
                target_batch: batch of respective target sequences
        """
        while True:
            selected_lines = []
            random_projects = np.random.choice(self.projects, size=batch_size)
            for random_project in random_projects:
                file = random.choice(os.listdir(os.path.join(self.data_directory, random_project)))
                with open(os.path.join(self.data_directory, random_project, file), 'r') as random_file:
                    lines = random_file.read().split("\n")
                    selected_line = None
                    while not selected_line:
                        selected_line = random.choice(lines).strip()
                    selected_lines.append(selected_line)

            input_batch = []
            for line in selected_lines:
                char_array = [char for char in line]
                if len(line) >= 1 and random.random() > 0.9:
                    drop_char = random.randint(0, len(line))
                    del char_array[drop_char]
                input_batch.append(char_array)
            yield self.pad_array_with_zeros(input_batch)
