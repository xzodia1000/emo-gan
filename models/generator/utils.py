import torch


def sort_batch(data, lengths):
    """
    sort data by length
    sorted_data[initial_index] == data
    """
    sorted_lengths, sorted_index = lengths.sort(0, descending=True)
    sorted_data = data[sorted_index]
    _, initial_index = sorted_index.sort(0, descending=False)

    return sorted_data, sorted_lengths, initial_index


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask
