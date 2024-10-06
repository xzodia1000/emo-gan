import os
import argparse
import torch
from tqdm import tqdm
from torch.optim import Adam
from hparams import create_hparams
from loss_functions import EmotionClassifierLoss, GeneratorLoss
from itertools import zip_longest

from model_utils import *
from utils import print_logs, save_to_file


def train_single_batch(
    batch,
    hparams,
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    g_criterion,
    d_criterion,
):
    """Train the generator and discriminator with a single emotion dataset"""

    d_x, d_y = parse_batch_classifier(batch, hparams)
    g_x, g_y = parse_batch_generator(batch)

    g_optimizer.zero_grad()
    d_optimizer.zero_grad()

    g_outputs = generator(g_x)  # Output from generator
    g_loss = g_criterion(g_outputs, g_y)  # Loss with ground truth
    g_mel, g_mel_postnet, _, _ = g_outputs

    # Parse the output for the discriminator
    g_mel = parse_batch_classifier_inference(g_mel, hparams)
    g_mel_postnet = parse_batch_classifier_inference(g_mel_postnet, hparams)

    fake_pred_for_g = discriminator(g_mel)  # Predictions from discriminator

    # Loss for generator with discriminator predictions
    g_loss += d_criterion(fake_pred_for_g, d_y)
    fake_pred_postnet_for_g = discriminator(g_mel_postnet)
    g_loss += d_criterion(fake_pred_postnet_for_g, d_y)

    # Loss for discriminator with generator predictions and real data
    fake_pred = discriminator(g_mel.detach())
    d_fake_loss = d_criterion(fake_pred, d_y)
    fake_pred_postnet = discriminator(g_mel_postnet.detach())
    d_fake_loss += d_criterion(fake_pred_postnet, d_y)

    real_pred = discriminator(d_x)
    d_real_loss = d_criterion(real_pred, d_y)

    d_loss = d_fake_loss + d_real_loss / 2

    # Update the generator and discriminator
    g_loss.backward()
    g_optimizer.step()
    d_loss.backward()
    d_optimizer.step()

    return g_loss.item(), d_loss.item()


def train_mixed_batch(
    batch,
    hparams,
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    g_criterion,
    d_criterion,
    update_discriminator=False,
):
    """Train the generator and discriminator with a mixed emotion dataset"""

    _, d_y = parse_batch_classifier(batch, hparams)
    g_x, g_y = parse_batch_generator(batch)

    g_optimizer.zero_grad()
    d_optimizer.zero_grad()

    g_outputs = generator(g_x, real_mels=False)  # Output from generator
    g_mel, g_mel_postnet, _, _ = g_outputs

    # Parse the output for the discriminator
    g_mel = parse_batch_classifier_inference(g_mel, hparams)
    g_mel_postnet = parse_batch_classifier_inference(g_mel_postnet, hparams)

    # Loss for generator with discriminator predictions
    fake_pred_for_g = discriminator(g_mel)
    g_loss = d_criterion(fake_pred_for_g, d_y)
    fake_pred_postnet_for_g = discriminator(g_mel_postnet)
    g_loss += d_criterion(fake_pred_postnet_for_g, d_y)

    g_loss.backward()
    g_optimizer.step()

    if not update_discriminator:
        # Do not update the discriminator
        return g_loss.item(), None

    # Loss for discriminator with generator predictions and real labels
    fake_pred = discriminator(g_mel.detach())
    d_fake_loss = d_criterion(fake_pred, d_y)
    fake_pred_postnet = discriminator(g_mel_postnet.detach())
    d_fake_loss += d_criterion(fake_pred_postnet, d_y)
    d_loss = d_fake_loss

    d_loss.backward()
    d_optimizer.step()

    return g_loss.item(), d_loss.item()


def train(output_directory, checkpoint_number, device, hparams):
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    single_train_loader, mixed_train_loader, single_val_loader, mixed_val_loader = (
        prepare_dataloaders(hparams)
    )

    train_single_g_loss, train_single_d_loss, val_single_g_loss, val_single_d_loss = (
        [],
        [],
        [],
        [],
    )
    train_mixed_g_loss, train_mixed_d_loss, val_mixed_g_loss = [], [], []
    accs, recalls = [], []

    generator = load_generator(hparams, device)
    discriminator = load_classifier(hparams, device)
    generator = load_final_model(hparams.generator_checkpoint, generator)
    discriminator = load_final_model(hparams.discriminator_checkpoint, discriminator)

    g_learning_rate = hparams.learning_rate
    d_learning_rate = hparams.learning_rate
    g_optimizer = Adam(
        generator.parameters(),
        lr=g_learning_rate,
        weight_decay=hparams.weight_decay,
        betas=hparams.betas,
    )
    d_optimizer = Adam(
        discriminator.parameters(),
        lr=d_learning_rate,
        weight_decay=hparams.weight_decay,
        betas=hparams.betas,
    )

    g_criterion = GeneratorLoss()
    d_criterion = EmotionClassifierLoss()

    start_epoch = 0

    if checkpoint_number is not None:
        generator_path = os.path.join(
            output_directory,
            "generator",
            f"checkpoint_epoch_{checkpoint_number}.pth.tar",
        )
        generator, g_optimizer, start_epoch = load_checkpoint(
            hparams,
            generator_path,
            generator,
            g_optimizer,
            output_directory,
        )

        discriminator_path = os.path.join(
            output_directory,
            "discriminator",
            f"checkpoint_epoch_{checkpoint_number}.pth.tar",
        )
        discriminator, d_optimizer, start_epoch = load_checkpoint(
            hparams,
            checkpoint_number,
            discriminator_path,
            d_optimizer,
            output_directory,
        )

    for epoch in tqdm(range(start_epoch, hparams.epochs)):
        generator.train()
        discriminator.train()

        interval = hparams.epochs / 10
        mixed_ratio = min(1.0, epoch / interval * 0.1)
        single_ratio = 1.0 - mixed_ratio

        single_batches = int(single_ratio * len(single_train_loader))
        mixed_batches = int(mixed_ratio * len(mixed_train_loader))

        tsgl, tsdl = 0, 0
        tmgl, tmdl = 0, 0

        for i, ((single_batch), (mixed_batch)) in enumerate(
            zip_longest(single_train_loader, mixed_train_loader)
        ):
            if i < single_batches:
                g_loss, d_loss = train_single_batch(
                    single_batch,
                    hparams,
                    generator,
                    discriminator,
                    g_optimizer,
                    d_optimizer,
                    g_criterion,
                    d_criterion,
                )

                tsgl += g_loss
                tsdl += d_loss

            if i < mixed_batches:
                update_discriminator = True if mixed_ratio > 0.7 else False
                g_loss, d_loss = train_mixed_batch(
                    mixed_batch,
                    hparams,
                    generator,
                    discriminator,
                    g_optimizer,
                    d_optimizer,
                    g_criterion,
                    d_criterion,
                    update_discriminator,
                )

                tmgl += g_loss
                tmdl += d_loss if d_loss is not None else 0

        if single_batches > 0:
            tsgl /= single_batches
            tsdl /= single_batches
            train_single_g_loss.append(tsgl)
            train_single_d_loss.append(tsdl)

        if mixed_batches > 0:
            tmgl /= mixed_batches
            tmdl /= mixed_batches
            train_mixed_g_loss.append(tmgl)
            train_mixed_d_loss.append(tmdl)

        if (epoch + 1) % hparams.checkpoint_interval == 0:

            vsgl = validate_generator(generator, single_val_loader, g_criterion)
            vsdl, acc, prec, rec = validate_classifier(
                discriminator, single_val_loader, d_criterion, hparams
            )
            vmgl = validate_mixed_generator(
                generator, discriminator, mixed_val_loader, d_criterion, hparams
            )

            val_single_g_loss.append(vsgl)
            val_single_d_loss.append(vsdl)
            val_mixed_g_loss.append(vmgl)
            accs.append(acc)
            recalls.append(rec)

            save_checkpoint(
                hparams,
                generator,
                g_optimizer,
                g_learning_rate,
                epoch,
                os.path.join(output_directory, "generator"),
            )
            save_checkpoint(
                hparams,
                discriminator,
                d_optimizer,
                d_learning_rate,
                epoch,
                os.path.join(output_directory, "discriminator"),
            )

            print_logs(
                hparams,
                f"Epoch {epoch+1}\nGenerator: Train Loss: {tsgl:.4f}, Val Loss: {vsgl:.4f}\nDiscriminator: Train Loss: {tsdl:.4f}, Val Loss: {vsdl:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}\nMixed Generator: Train Loss: {tmgl:.4f} Val Loss: {vmgl:.4f}\nMixed Discriminator: Train Loss: {tmdl:.4f}",
                output_directory,
            )

    torch.save(
        generator.state_dict(), os.path.join(output_directory, "final_generator.pth")
    )
    torch.save(
        discriminator.state_dict(),
        os.path.join(output_directory, "final_discriminator.pth"),
    )
    save_to_file(
        os.path.join(output_directory, "train_results.txt"),
        f"Single Generator Train Loss: {train_single_g_loss}\nSingle Generator Val Loss: {val_single_g_loss}\nSingle Discriminator Train Loss: {train_single_d_loss}\nSingle Discriminator Val Loss: {val_single_d_loss}\nMixed Generator Train Loss: {train_mixed_g_loss}\nMixed Generator Val Loss: {val_mixed_g_loss}\nMixed Discriminator Train Loss: {train_mixed_d_loss}\nAccuracy: {accs}\nRecall: {recalls}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_directory", type=str, help="directory to save checkpoints"
    )
    parser.add_argument(
        "-c",
        "--checkpoint_number",
        type=str,
        default=None,
        required=False,
        help="checkpoint number",
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
    create_output_directory(os.path.join(args.output_directory, "generator"))
    create_output_directory(os.path.join(args.output_directory, "discriminator"))

    if args.device == "dmog":
        hparams = create_hparams(local=False, log_output=args.print_log)
    else:
        hparams = create_hparams(local=True, log_output=args.print_log)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_logs(hparams, f"Training on {device}", args.output_directory)
    print_logs(hparams, f"Number of epochs: {hparams.epochs}", args.output_directory)

    train(args.output_directory, args.checkpoint_number, device, hparams)
