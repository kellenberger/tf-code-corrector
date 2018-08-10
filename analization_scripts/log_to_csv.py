def log_to_csv(file):
    with open(file, 'r') as log, \
        open(file.split('.')[0] + '.csv', 'w') as csv:
        log_data = log.read().split(' ')
        for i, token in enumerate(log_data):
            if token == "loss:":
                loss = log_data[i+1]
                loss = loss[:-1]
                csv.write(loss + ";\n")
