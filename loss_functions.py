import torch
import torch.nn as nn

from models.generator.utils import get_mask_from_lengths


class EmotionClassifierLoss(nn.Module):
    def __init__(self):
        super(EmotionClassifierLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, model_output, targets):
        # Assuming targets are the true labels for emotions
        loss = self.criterion(model_output, targets)
        return loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.L1Loss = nn.L1Loss(reduction="none")
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction="none")
        self.n_frames_per_step = 2

    def parse_targets(self, targets):
        """
        mel_target [batch_size, mel_bins, T]
        stop_target [batch_size, T]
        """
        mel_target, stop_target, mel_lengths = targets

        B = stop_target.size(0)
        stop_target = stop_target.reshape(B, -1, self.n_frames_per_step)
        stop_target = stop_target[:, :, 0]

        return mel_target, stop_target, mel_lengths

    def forward(self, model_output, targets, real_mels=True):
        (
            mel_outputs,
            mel_outputs_postnet,
            stop_outputs,
            _,
        ) = model_output

        mel_target, stop_target, output_lengths = self.parse_targets(targets)

        mel_mask = (
            get_mask_from_lengths(output_lengths, mel_outputs.size(2))
            .unsqueeze(1)
            .expand(-1, mel_target.size(1), -1)
            .float()
        )

        mel_step_lengths = torch.ceil(
            output_lengths.float() / self.n_frames_per_step
        ).long()
        stop_mask = get_mask_from_lengths(
            mel_step_lengths, int(mel_target.size(2) / self.n_frames_per_step)
        ).float()

        recon_loss = (
            torch.sum(self.L1Loss(mel_outputs, mel_target) * mel_mask)
            / torch.sum(mel_mask)
            if real_mels
            else 0
        )

        recon_postnet_loss = (
            torch.sum(self.L1Loss(mel_outputs_postnet, mel_target) * mel_mask)
            / torch.sum(mel_mask)
            if real_mels
            else 0
        )

        stop_loss = torch.sum(
            self.BCEWithLogitsLoss(stop_outputs, stop_target) * stop_mask
        ) / torch.sum(stop_mask)

        return recon_loss + recon_postnet_loss + stop_loss
