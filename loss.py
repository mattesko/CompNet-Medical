import torch

def dice(outputs, labels):

    outputs, labels = outputs.float(), labels.float()
    intersect = torch.dot(outputs.contiguous().view(-1), labels.contiguous().view(-1))
    union = torch.add(torch.sum(outputs), torch.sum(labels))
    dice = 1 - (2 * intersect + 1e-5) / (union + 1e-5)
    return dice
