import os
import numpy as np
import torch
import python_speech_features as ps
from sklearn.metrics import precision_score, accuracy_score, recall_score
from torch.utils.data import DataLoader


from data_class import TextMelLoader, TextMelCollate
from models.generator.model import Generator
from models.discriminator_classifier.model import EmotionClassifier
from utils import print_logs


############################
#   Data Helper Functions  #
############################


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    mixed_trainset = None
    mixed_valset = None
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_loader = DataLoader(
        trainset,
        num_workers=1,
        shuffle=True,
        batch_size=hparams.batch_size,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        valset,
        num_workers=1,
        shuffle=True,
        batch_size=hparams.batch_size,
        collate_fn=collate_fn,
    )

    if hparams.mixed_training_files is not None:
        mixed_trainset = TextMelLoader(hparams.mixed_training_files, hparams, True)
        mixed_valset = TextMelLoader(hparams.mixed_validation_files, hparams, True)

        mixed_train_loader = DataLoader(
            mixed_trainset,
            num_workers=1,
            shuffle=True,
            batch_size=hparams.batch_size,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn,
        )
        mixed_val_loader = DataLoader(
            mixed_valset,
            num_workers=1,
            shuffle=True,
            batch_size=hparams.batch_size,
            collate_fn=collate_fn,
        )

    return train_loader, mixed_train_loader, val_loader, mixed_val_loader


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


############################
# General Helper Functions #
############################


def create_output_directory(output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)


def save_checkpoint(hparams, model, optimizer, learning_rate, epoch, output_directory):
    filepath = os.path.join(output_directory, f"checkpoint_epoch_{epoch+1}.pth.tar")
    print_logs(
        hparams,
        f"Saving model and optimizer state at epoch {epoch+1} to {filepath}",
        output_directory,
    )
    torch.save(
        {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        filepath,
    )


def load_final_model(folder_path, model):
    assert os.path.isfile(os.path.join(folder_path, "final_model.pth"))
    state_dict = torch.load(
        os.path.join(folder_path, "final_model.pth"), map_location="cpu"
    )
    model.load_state_dict(state_dict)
    return model


def load_checkpoint(hparams, checkpoint_path, model, optimizer, output_directory):
    assert os.path.isfile(checkpoint_path)
    print_logs(hparams, f"Loading checkpoint {checkpoint_path}", output_directory)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    learning_rate = checkpoint_dict["learning_rate"]
    start_epoch = checkpoint_dict["epoch"]
    print_logs(
        hparams,
        f"Loaded checkpoint {checkpoint_path} from epoch {start_epoch}",
        output_directory,
    )
    return model, optimizer, learning_rate, start_epoch


def load_model_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    return model


############################
#   Classifier Functions   #
############################


def load_classifier(hparams, device):
    model = EmotionClassifier(hparams).to(device)
    return model


def parse_batch_classifier(batch, hparams):
    (
        _,
        _,
        emotion_vector,
        mel_padded,
        _,
        _,
        _,
    ) = batch
    emotion_vector = to_gpu(emotion_vector).float()
    mel_padded = to_gpu(process_classifier_input(mel_padded, hparams)).float()

    return (mel_padded), (emotion_vector)


def parse_batch_classifier_inference(mel, hparams):
    mel_padded = to_gpu(process_classifier_input(mel, hparams)).float()
    return mel_padded


def process_classifier_input(data_class_batch, hparams):
    post_output = data_class_batch
    post_output = post_output.cpu().detach().numpy()
    size = post_output.shape[0]
    post_output_new = np.zeros(
        (size, hparams.n_channels, hparams.win_length, hparams.n_mel_channels)
    )  # [32,3,800,80]
    for index, item in enumerate(post_output):
        item_new = process_mel(item, hparams)  # [3,800,80]
        post_output_new[index, :, :, :] = item_new
    post_output_new = torch.tensor(post_output_new, dtype=torch.float32)
    return post_output_new


def process_mel(mel_input, hparams):

    # mel_input [80,344]
    mel_input = mel_input.T  # [344,80]
    delta1 = ps.delta(mel_input, 2)
    delta2 = ps.delta(delta1, 2)

    time = mel_input.shape[0]
    mel = np.pad(
        mel_input,
        ((0, hparams.win_length - time), (0, 0)),
        "constant",
        constant_values=0,
    )  # [800,80]
    delta1 = np.pad(
        delta1, ((0, hparams.win_length - time), (0, 0)), "constant", constant_values=0
    )
    delta2 = np.pad(
        delta2, ((0, hparams.win_length - time), (0, 0)), "constant", constant_values=0
    )

    mel_output = np.zeros(
        (hparams.n_channels, hparams.win_length, hparams.n_mel_channels)
    )
    mel_output[0, :, :] = mel
    mel_output[1, :, :] = delta1
    mel_output[2, :, :] = delta2

    return mel_output


def init_classifier_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0.1)
    elif type(m) == torch.nn.Conv2d:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0.1)


def validate_classifier(model, val_loader, criterion, hparams):
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in val_loader:
            x, y = parse_batch_classifier(batch, hparams)

            outputs = model(x)
            loss = criterion(outputs, y)

            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = np.round(probabilities)
            all_preds.extend(predictions)
            all_true.extend(y.cpu().numpy())

            total_val_loss += loss.item()

    all_true = np.array(all_true)
    all_preds = np.array(all_preds)

    accuracy = accuracy_score(all_preds, all_true)
    precision = precision_score(all_true, all_preds, average="macro", zero_division=1)
    recall = recall_score(all_true, all_preds, average="macro", zero_division=1)

    return total_val_loss / len(val_loader), accuracy, precision, recall


############################
#   Generator Functions    #
############################


def load_generator(hparams, device):
    model = Generator(hparams).to(device)
    return model


def parse_batch_generator(batch):
    (
        text_padded,
        text_lengths,
        emotion_vector,
        input_mel_padded,
        real_mel_padded,
        stop_token_padded,
        mel_lengths,
    ) = batch

    text_padded = to_gpu(text_padded).long()
    text_lengths = to_gpu(text_lengths).long()
    emotion_vector = to_gpu(emotion_vector).float()
    input_mel_padded = to_gpu(input_mel_padded).float()
    real_mel_padded = to_gpu(real_mel_padded).float()
    stop_token_padded = to_gpu(stop_token_padded).float()
    mel_lengths = to_gpu(mel_lengths).long()

    return (
        (text_padded, text_lengths, emotion_vector, input_mel_padded, mel_lengths),
        (real_mel_padded, stop_token_padded, mel_lengths),
    )


def parse_test_batch_generator(batch):
    (
        text_padded,
        text_lengths,
        emotion_vector,
        input_mel_padded,
        real_mel_padded,
        stop_token_padded,
        mel_lengths,
        text,
        emotion,
        speaker,
    ) = batch

    text_padded = to_gpu(text_padded).long()
    text_lengths = to_gpu(text_lengths).long()
    emotion_vector = to_gpu(emotion_vector).float()
    input_mel_padded = to_gpu(input_mel_padded).float()
    real_mel_padded = to_gpu(real_mel_padded).float()
    stop_token_padded = to_gpu(stop_token_padded).float()
    mel_lengths = to_gpu(mel_lengths).long()

    return (
        (text_padded, text_lengths, emotion_vector, input_mel_padded, mel_lengths),
        (real_mel_padded, stop_token_padded, mel_lengths),
        (text, emotion, speaker),
    )


def validate_generator(model, val_loader, criterion):
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            x, y = parse_batch_generator(batch)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_val_loss += loss.item()

    return total_val_loss / len(val_loader)


def validate_mixed_generator(generator, discriminator, val_loader, criterion, hparams):
    generator.eval()
    discriminator.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            x, _ = parse_batch_generator(batch)

            g_mel, g_mel_postnet, _, _ = generator(x, real_mels=False)

            g_mel = parse_batch_classifier_inference(g_mel, hparams)
            g_mel_postnet = parse_batch_classifier_inference(g_mel_postnet, hparams)

            fake_pred = discriminator(g_mel.detach())
            loss = criterion(fake_pred, x[2])
            fake_pred_postnet = discriminator(g_mel_postnet.detach())
            loss += criterion(fake_pred_postnet, x[2])

            total_val_loss += loss.item()

    return total_val_loss / len(val_loader)
