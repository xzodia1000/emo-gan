import torch

from models.generator.model import Generator
from hparams import create_hparams


def test_forward_pass_w_mels():
    hparams = create_hparams()
    model = Generator(hparams)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = 2
    seq_len = 10

    text_input = torch.randint(0, hparams.n_symbols, (batch_size, seq_len))
    text_lengths = torch.LongTensor([seq_len, seq_len - 2])
    emotion_input = torch.rand(batch_size, hparams.n_emotions)
    mels = torch.rand(batch_size, hparams.n_mel_channels, seq_len)
    gate_padded = torch.zeros(batch_size, seq_len)
    output_lengths = torch.LongTensor([seq_len, seq_len - 2])

    batch = (text_input, text_lengths, emotion_input, mels, gate_padded, output_lengths)
    inputs, _ = model.parse_batch(batch)

    try:
        with torch.no_grad():
            outputs = model.forward(inputs)
            print(
                "Forward pass successful with mels. Output shapes:",
                [o.shape for o in outputs],
            )
            assert True
    except Exception as e:
        print("Error during forward pass (with mels):", str(e))
        assert False


def test_forward_pass_wo_mels():
    hparams = create_hparams()
    model = Generator(hparams)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = 2
    seq_len = 10

    text_input = torch.randint(0, hparams.n_symbols, (batch_size, seq_len))
    text_lengths = torch.LongTensor([seq_len, seq_len - 2])
    emotion_input = torch.rand(batch_size, hparams.n_emotions)
    mels = torch.rand(batch_size, hparams.n_mel_channels, seq_len)
    gate_padded = torch.zeros(batch_size, seq_len)
    output_lengths = torch.LongTensor([seq_len, seq_len - 2])

    batch = (text_input, text_lengths, emotion_input, mels, gate_padded, output_lengths)
    inputs, _ = model.parse_batch(batch)

    try:
        with torch.no_grad():
            outputs = model.forward(inputs, real_mels=False)
            print(
                "Forward pass successful without mels. Output shapes:",
                [o.shape for o in outputs],
            )
            assert True
    except Exception as e:
        print("Error during forward pass (without mels):", str(e))
        assert False


test_forward_pass_w_mels()
# test_forward_pass_wo_mels()
