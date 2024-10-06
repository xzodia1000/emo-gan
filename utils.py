import ast
import logging
import os
from tqdm import tqdm


def save_to_file(filename, content):
    with open(filename, "w") as file:
        file.write(content)


def read_train_results(output_directory):
    with open(os.path.join(output_directory, "train_results.txt"), "r") as file:
        lines = file.readlines()
        train_loss = ast.literal_eval(lines[0].strip().split(": ")[1])
        val_loss = ast.literal_eval(lines[1].strip().split(": ")[1])
        try:
            acc = ast.literal_eval(lines[2].strip().split(": ")[1])
            rec = ast.literal_eval(lines[3].strip().split(": ")[1])
        except:
            acc = None
            rec = None

    return train_loss, val_loss, acc, rec


def print_logs(hparams, log, output_directory):
    if hparams.print_log == "print":
        tqdm.write(log)
    else:
        filepath = os.path.join(output_directory, hparams.log_filename)
        logging.getLogger().setLevel(logging.INFO)

        # Configure logging
        logging.basicConfig(
            filename=filepath,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        logging.info(log)
