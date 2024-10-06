import librosa
import numpy as np


def recover_wav(
    real_mel,
    pred_mel,
    n_fft=2048,
    win_length=800,
    hop_length=200,
    n_iter=32,
):
    # Get the mel spectrogram
    real_mel = real_mel.data.cpu().numpy()[0]
    pred_mel = pred_mel.data.cpu().numpy()[0]

    # Denormalize the mel spectrogram
    # based on the mean of the training set
    mean = np.mean(real_mel, axis=1)
    mel_denormalized = pred_mel + mean[:, None] + mean[:, None]
    mel = np.exp(mel_denormalized)

    # Invert the mel spectrogram
    # to get the waveform
    filters = librosa.filters.mel(
        sr=16000,
        n_fft=n_fft,
        n_mels=80,
        norm=1,
    )
    inv_filters = np.linalg.pinv(filters)
    mel = np.dot(inv_filters, mel)

    # Use the Griffin-Lim algorithm
    y = librosa.griffinlim(
        mel,
        n_iter=n_iter,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        pad_mode="reflect",
    )
    return y
