import torch
from torch import nn
import torch.nn.functional as F


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        param=None,
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        nn.init.xavier_uniform_(
            self.conv.weight,
            gain=nn.init.calculate_gain(w_init_gain, param=param),
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                LinearNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class PostNet(nn.Module):
    """Postnet
    - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    hparams.n_mel_channels,
                    hparams.postnet_dim,
                    kernel_size=hparams.postnet_kernel_size,
                    stride=1,
                    padding=int((hparams.postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(hparams.postnet_dim),
            )
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        hparams.postnet_dim,
                        hparams.postnet_dim,
                        kernel_size=hparams.postnet_kernel_size,
                        stride=1,
                        padding=int((hparams.postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(hparams.postnet_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    hparams.postnet_dim,
                    hparams.n_mel_channels,
                    kernel_size=hparams.postnet_kernel_size,
                    stride=1,
                    padding=int((hparams.postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(hparams.n_mel_channels),
            )
        )
        self.dropout = hparams.postnet_dropout
        self.predict_spectrogram = hparams.predict_spectrogram

    def forward(self, input):
        # input [B, mel_bins, T]

        x = input
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(
                torch.tanh(self.convolutions[i](x)), self.dropout, self.training
            )
        x = F.dropout(self.convolutions[-1](x), self.dropout, self.training)

        if self.predict_spectrogram:
            o = x + self.projection(input.transpose(1, 2)).transpose(1, 2)
        else:
            o = x + input

        return o


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2,
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense = LinearNorm(
            attention_n_filters, attention_dim, bias=False, w_init_gain="tanh"
        )

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class AttentionV2(nn.Module):
    def __init__(
        self,
        attention_rnn_dim,
        embedding_dim,
        attention_dim,
        attention_location_n_filters,
        attention_location_kernel_size,
    ):
        super(AttentionV2, self).__init__()
        self.query_layer = LinearNorm(
            attention_rnn_dim, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.memory_layer = LinearNorm(
            embedding_dim, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters, attention_location_kernel_size, attention_dim
        )
        self.score_mask_value = -float(1e20)

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat:  prev. and cumulative att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(
            torch.tanh(processed_query + processed_attention_weights + processed_memory)
        )

        energies = energies.squeeze(-1)
        return energies

    def forward(
        self,
        attention_hidden_state,
        memory,
        processed_memory,
        attention_weights_cat,
        mask,
        log_alpha,
    ):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        log_energy = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )

        # log_energy =

        if mask is not None:
            mask_bool = mask.bool()
            log_energy.data.masked_fill_(mask_bool, self.score_mask_value)

        # attention_weights = F.softmax(alignment, dim=1)

        # content_score = log_energy.unsqueeze(1) #[B, MAX_TIME] -> [B, 1, MAX_TIME]
        # log_alpha = log_alpha.unsqueeze(2) #[B, MAX_TIME] -> [B, MAX_TIME, 1]

        # log_total_score = log_alpha + content_score

        # previous_attention_weights = attention_weights_cat[:,0,:]

        log_alpha_shift_padded = []
        max_time = log_energy.size(1)
        for sft in range(2):
            shifted = log_alpha[:, : max_time - sft]
            shift_padded = F.pad(shifted, (sft, 0), "constant", self.score_mask_value)
            log_alpha_shift_padded.append(shift_padded.unsqueeze(2))

        biased = torch.logsumexp(torch.cat(log_alpha_shift_padded, 2), 2)

        log_alpha_new = biased + log_energy

        attention_weights = F.softmax(log_alpha_new, dim=1)

        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights, log_alpha_new
