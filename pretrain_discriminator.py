import argparse
import os
import torch
from torch.optim import Adam
from tqdm import tqdm

from early_stopper import EarlyStopper
from hparams import create_hparams
from loss_functions import EmotionClassifierLoss
from utils import print_logs, save_to_file
from model_utils import (
    create_output_directory,
    init_classifier_weights,
    load_checkpoint,
    load_classifier,
    parse_batch_classifier,
    prepare_dataloaders,
    save_checkpoint,
    validate_classifier,
)


def train(output_directory, checkpoint_path, device, hparams):
    early_stopper = EarlyStopper(patience=15, min_delta=0.001)

    train_losses, val_losses, accs, recalls = [], [], [], []
    train_loader, _, val_loader, _ = prepare_dataloaders(hparams)

    model = load_classifier(hparams, device)
    model.apply(init_classifier_weights)

    learning_rate = hparams.learning_rate
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=hparams.weight_decay,
        betas=hparams.betas,
    )
    criterion = EmotionClassifierLoss()

    start_epoch = 0

    if checkpoint_path is not None:
        model, optimizer, learning_rate, start_epoch = load_checkpoint(
            hparams, checkpoint_path, model, optimizer, output_directory
        )

    for epoch in tqdm(range(start_epoch, hparams.epochs)):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x, y = parse_batch_classifier(batch, hparams)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        total_loss /= len(train_loader)
        val_loss, acc, prec, rec = validate_classifier(
            model, val_loader, criterion, hparams
        )

        train_losses.append(total_loss)
        val_losses.append(total_loss)
        accs.append(acc)
        recalls.append(rec)

        if (epoch + 1) % hparams.checkpoint_interval == 0:
            save_checkpoint(
                hparams, model, optimizer, learning_rate, epoch, output_directory
            )
            print_logs(
                hparams,
                f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}",
                output_directory,
            )

        if early_stopper.early_stop(validation_loss=val_loss):
            print_logs(
                hparams,
                f"Early Stopped at {epoch+1} with Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}",
                output_directory,
            )
            break

    torch.save(model.state_dict(), os.path.join(output_directory, "final_model.pth"))
    save_to_file(
        os.path.join(output_directory, "train_results.txt"),
        f"Train Loss: {train_losses}\nVal Loss: {val_losses}\nAcc: {accs}\nRecall: {recalls}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_directory", type=str, help="directory to save checkpoints"
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="checkpoint path",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="local",
        required=False,
        help="runtime device type (local | dmog)",
    )
    parser.add_argument(
        "-p",
        "--print_log",
        type=str,
        default="print",
        required=False,
        help="print to terminal or log",
    )

    args = parser.parse_args()

    create_output_directory(args.output_directory)

    if args.device == "dmog":
        hparams = create_hparams(local=False, log_output=args.print_log)
    else:
        hparams = create_hparams(local=True, log_output=args.print_log)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_logs(hparams, f"Classifier training on {device}", args.output_directory)
    print_logs(hparams, f"Number of epochs: {hparams.epochs}", args.output_directory)

    train(args.output_directory, args.checkpoint_path, device, hparams)
