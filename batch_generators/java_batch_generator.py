import numpy as np

def train_batch_generator(data_directory, batch_size = 128):
    """
    Args:
        batch_size: Size of the batch_size
    Returns:
        A tuple of the form (input_batch, target_batch), where:
            input_batch: batch of input sequences
            target_batch: batch of respective target sequences
    """
    projects = []
    with open(os.path.join(data_directory, 'trainJava.csv'), 'r') as train_projects:
        print('load projects')
        for project in train_projects:
            if os.path.exists(os.path.join(data_directory, project.strip())):
                projects.append(project.strip())
    while True:
        input_batch = []
        random_projects = np.random.choice(projects, size=batch_size)
        for random_project in random_projects:
            file = random.choice(os.listdir(os.path.join(data_directory, random_project)))
            with open(os.path.join(data_directory, random_project, file), 'r') as random_file:
                lines = random_file.read().split("\n")
                selected_line = None
                while not selected_line:
                    selected_line = random.choice(lines).strip()
                input_batch.append(selected_line)
        yield input_batch
