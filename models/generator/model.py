import torch
from torch import nn
from math import sqrt


from models.generator.decoder import Decoder
from models.generator.encoder import TextEncoder
from models.generator.layers import PostNet
from models.generator.utils import get_mask_from_lengths


class Generator(nn.Module):
    def __init__(self, hparams):
        super(Generator, self).__init__()

        self.text_embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim
        )
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std

        self.text_embedding.weight.data.uniform_(-val, val)

        self.emotion_embedding = nn.Linear(
            hparams.n_emotions, hparams.emotions_embedding_dim
        )
        nn.init.xavier_uniform_(self.emotion_embedding.weight)

        self.text_encoder = TextEncoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = PostNet(hparams)

    def forward(self, inputs, real_mels=True):
        text_inputs, text_lengths, emotion_vector, mels, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        max_len = torch.max(text_lengths).item()

        outputs = []

        # create mask for text inputs
        text_embed = self.text_embedding(text_inputs).transpose(1, 2)
        # encode text inputs
        encoder_outputs = self.text_encoder(text_embed, text_lengths)
        outputs.append(encoder_outputs)

        emotion_vector = emotion_vector.unsqueeze(1)
        # pass emotion vector through linear layer
        emotion_embed = self.emotion_embedding(emotion_vector)
        emotion_embed = emotion_embed.expand(-1, max_len, -1)
        outputs.append(emotion_embed)

        # concatenate encoder outputs and emotion embeddings
        outputs = torch.cat(outputs, dim=-1)

        mel_outputs, gate_outputs, alignments = self.decoder(
            outputs, mels, text_lengths, real_mels
        )

        mel_outputs_postnet = self.postnet(mel_outputs)

        outputs = (mel_outputs, mel_outputs_postnet, gate_outputs, alignments)

        return outputs

    def inference(self, input_text, emotion_vector):
        text_embed = self.text_embedding(input_text).transpose(1, 2)
        encoder_outputs = self.text_encoder.inference(text_embed)

        if emotion_vector.dim() == 1:
            emotion_vector = emotion_vector.unsqueeze(0)
        emotion_embed = self.emotion_embedding(emotion_vector)

        if emotion_embed.dim() == 2:
            emotion_embed = emotion_embed.unsqueeze(1)
            emotion_embed = emotion_embed.expand(-1, encoder_outputs.size(1), -1)

        outputs = torch.cat([encoder_outputs, emotion_embed], dim=-1)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)

        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments
