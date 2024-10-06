import torch
from torch import nn
from torch.nn import functional as F

from models.generator.layers import ConvNorm, LinearNorm
from models.generator.utils import sort_batch


class TextEncoder(nn.Module):
    """Encoder module:
    - Three 1-d convolution banks
    - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super(TextEncoder, self).__init__()

        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.encoder_kernel_size = hparams.encoder_kernel_size
        self.encoder_n_convolutions = hparams.encoder_n_convolutions

        convolutions = []
        for _ in range(self.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    self.encoder_embedding_dim,
                    self.encoder_embedding_dim,
                    kernel_size=self.encoder_kernel_size,
                    stride=1,
                    padding=int((self.encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(self.encoder_embedding_dim),
            )
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(
            self.encoder_embedding_dim,
            int(self.encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs
