import os
import argparse
import torch
from tqdm import tqdm
from torch.optim import Adam
from early_stopper import EarlyStopper
from hparams import create_hparams
from loss_functions import GeneratorLoss

from model_utils import (
    create_output_directory,
    load_checkpoint,
    load_generator,
    parse_batch_generator,
    prepare_dataloaders,
    save_checkpoint,
    validate_generator,
)
from utils import print_logs, save_to_file


def train(output_directory, checkpoint_path, device, hparams):
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    early_stopper = EarlyStopper(patience=20, min_delta=0.0001)

    train_losses, val_losses = [], []
    train_loader, _, val_loader, _ = prepare_dataloaders(hparams)

    model = load_generator(hparams, device)

    learning_rate = hparams.learning_rate
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=hparams.weight_decay,
    )

    criterion = GeneratorLoss()

    start_epoch = 0

    if checkpoint_path is not None:
        model, optimizer, learning_rate, start_epoch = load_checkpoint(
            hparams, checkpoint_path, model, optimizer, output_directory
        )

    for epoch in tqdm(range(start_epoch, hparams.epochs)):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x, y = parse_batch_generator(batch)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            _ = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams.grad_clip_thresh
            )

            total_loss += loss.item()

        total_loss /= len(train_loader)
        val_loss = validate_generator(model, val_loader, criterion)

        train_losses.append(total_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % hparams.checkpoint_interval == 0:
            save_checkpoint(
                hparams, model, optimizer, learning_rate, epoch, output_directory
            )
            print_logs(
                hparams,
                f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}",
                output_directory,
            )

        if early_stopper.early_stop(validation_loss=val_loss):
            print_logs(
                hparams,
                f"Early stopping at epoch {epoch+1} with Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}",
                output_directory,
            )
            break

    torch.save(model.state_dict(), os.path.join(output_directory, "final_model.pth"))
    save_to_file(
        os.path.join(output_directory, "train_results.txt"),
        f"Train Loss: {train_losses}\nVal Loss: {val_losses}",
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

    print_logs(hparams, f"Generator training on {device}", args.output_directory)
    print_logs(hparams, f"Number of epochs: {hparams.epochs}", args.output_directory)

    train(args.output_directory, args.checkpoint_path, device, hparams)