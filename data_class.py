import ast
import random
import librosa
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from speechpy.processing import cmvn

from text import text_to_sequence


def load_dataset(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def average_mels(mel1, mel2):
    max_time = max(mel1.size(1), mel2.size(1))

    mel1 = F.pad(mel1, (0, max_time - mel1.size(1)), "constant", 0)
    mel2 = F.pad(mel2, (0, max_time - mel2.size(1)), "constant", 0)

    return (mel1 + mel2) / 2


class TextMelLoader(torch.utils.data.Dataset):
    """
    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
    """

    def __init__(self, dataset_entries, hparams, mixed_training=False, normalize=True):
        self.dataset_entries = load_dataset(dataset_entries)
        self.mixed_training = mixed_training
        self.normalize = normalize
        self.audio_file_root = hparams.audio_file_root
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.n_mel_channels = hparams.n_mel_channels
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.noise_std = hparams.noise_std

        random.seed(hparams.seed)
        random.shuffle(self.dataset_entries)

    def get_dataset_entry(self, dataset_entries):
        text, emotion_vector, audiopath, extra_feature = (
            dataset_entries[0],
            dataset_entries[1],
            dataset_entries[2],
            dataset_entries[3],
        )
        text = self.get_text(text)
        emotion_vector = self.get_vector(emotion_vector)

        if self.mixed_training:
            mel1 = self.get_mel(self.audio_file_root + audiopath, False)
            mel2 = self.get_mel(self.audio_file_root + extra_feature, False)
            input_mel = average_mels(mel1, mel2)
            real_mel = input_mel
        else:
            input_mel = self.get_mel(
                self.audio_file_root + audiopath, int(extra_feature)
            )
            real_mel = (
                self.get_mel(self.audio_file_root + audiopath, False)
                if int(extra_feature) == 1
                else input_mel
            )

        return (text, input_mel, real_mel, emotion_vector)

    # Converting audio to mel spectrogram
    def get_mel(self, filename, augment):
        audio, _ = librosa.load(filename, sr=self.sampling_rate)
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # Converting to audio using STFT to spectrogram
        spec = librosa.core.stft(
            y=audio,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann",
            center=True,
            pad_mode="reflect",
        )

        # Getting magnitude of the complex number
        spec = librosa.magphase(spec)[0]

        # Converting to mel spectrogram
        melspec = librosa.feature.melspectrogram(
            S=spec,
            sr=self.sampling_rate,
            n_mels=self.n_mel_channels,
            power=1.0,
            fmin=0.0,
            fmax=None,
            htk=False,
            norm=1,
        )

        # Converting to log scale
        melspec = np.log(melspec).astype(np.float32)

        if self.normalize:
            # Apply global CMVN
            melspec = torch.from_numpy(cmvn(melspec))
        else:
            melspec = torch.from_numpy(melspec)

        if augment:
            # Add white Gaussian noise
            noise = torch.randn(melspec.size()) * self.noise_std
            melspec += noise

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def get_vector(self, vector_string):
        vector_list = ast.literal_eval(vector_string)
        return vector_list

    def __getitem__(self, index):
        return self.get_dataset_entry(self.dataset_entries[index])

    def __len__(self):
        return len(self.dataset_entries)


class TextMelCollate:
    """Zero-pads model inputs and targets based on number of frames per setep"""

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """

        text_lengths = torch.IntTensor([len(x[0]) for x in batch])
        mel_lengths = torch.IntTensor([x[1].size(1) for x in batch])
        mel_bin = batch[0][1].size(0)

        max_text_len = torch.max(text_lengths).item()
        max_mel_len = torch.max(mel_lengths).item()

        if max_mel_len % self.n_frames_per_step != 0:
            max_mel_len += self.n_frames_per_step - max_mel_len % self.n_frames_per_step
            assert max_mel_len % self.n_frames_per_step == 0

        text_input_padded = torch.LongTensor(len(batch), max_text_len)
        input_mel_padded = torch.FloatTensor(len(batch), mel_bin, max_mel_len)
        real_mel_padded = torch.FloatTensor(len(batch), mel_bin, max_mel_len)
        stop_token_padded = torch.FloatTensor(len(batch), max_mel_len)

        text_input_padded.zero_()
        input_mel_padded.zero_()
        real_mel_padded.zero_()
        stop_token_padded.zero_()

        emotion_vectors = torch.FloatTensor([x[3] for x in batch])

        for i in range(len(batch)):
            text = batch[i][0]
            input_mel = batch[i][1]
            real_mel = batch[i][2]

            text_input_padded[i, : text.size(0)] = text
            input_mel_padded[i, :, : input_mel.size(1)] = input_mel
            real_mel_padded[i, :, : real_mel.size(1)] = real_mel
            stop_token_padded[i, input_mel.size(1) - self.n_frames_per_step :] = 1

        text_lengths, _ = torch.sort(text_lengths, descending=True)

        return (
            text_input_padded,
            text_lengths,
            emotion_vectors,
            input_mel_padded,
            real_mel_padded,
            stop_token_padded,
            mel_lengths,
        )
