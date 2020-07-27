import torch


def dice(outputs, labels, smooth=1e-5):
    outputs, labels = outputs.float(), labels.float()
    intersect = torch.dot(outputs.contiguous().view(-1),
                          labels.contiguous().view(-1))
    union = torch.add(torch.sum(outputs), torch.sum(labels))
    dice = (2 * intersect + smooth) / (union + smooth)
    return dice


def dice_loss(outputs, labels, smooth=1e-5):
    """Compute the dice loss
    Args:
        outputs (Tensor): The model's predictions
        labels (Tensor): The target labels (aka ground truth predictions)
        smooth (float/int): A smoothness factor
    Returns:
        dice loss (Tensor)
    """
    return 1 - dice(outputs, labels, smooth)


# def tversky(outputs, labels, smooth=1e-5, alpha=0.7):
