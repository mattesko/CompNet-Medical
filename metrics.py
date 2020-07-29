import torch


def dice_score(outputs, labels, smooth=1e-5):
    """Compute the Dice/F1 score. If a division by zero occurs, 
    returns a score of 0
    Args:
        outputs (Tensor): The model's predictions
        labels (Tensor): The target labels (aka ground truth predictions)
        smooth (float/int): A smoothness factor
    Returns:
        Dice score (Tensor)
    """
    outputs, labels = outputs.float(), labels.float()
    intersect = torch.dot(outputs.contiguous().view(-1),
                          labels.contiguous().view(-1))
    union = torch.add(torch.sum(outputs), torch.sum(labels))
    dice = (2 * intersect + smooth) / (union + smooth)
    return dice if not torch.isnan(dice) else torch.Tensor([0.0])


def dice_loss(outputs, labels, smooth=1e-5):
    """Compute the dice/F1 loss
    Args:
        outputs (Tensor): The model's predictions
        labels (Tensor): The target labels (aka ground truth predictions)
        smooth (float/int): A smoothness factor
    Returns:
        Dice loss (Tensor) = 1 - Dice score
    """
    return 1 - dice_score(outputs, labels, smooth)


def jaccard_score(outputs, labels, smooth=1e-5):
    """Compute the Jaccard/IoU score. If a division by zero occurs,
    returns a score of 0.
    Args:
        outputs (Tensor): The model's predictions
        labels (Tensor): The target labels (aka ground truth predictions)
        smooth (float/int): A smoothness factor
    Returns:
        Jaccard score (Tensor)
    """
    outputs, labels = outputs.float(), labels.float()
    intersect = torch.dot(outputs.contiguous().view(-1),
                          labels.contiguous().view(-1))
    union = torch.add(torch.sum(outputs), torch.sum(labels))
    jaccard = (intersect + smooth) / (union + smooth)
    return jaccard if not torch.isnan(jaccard) else torch.Tensor([0.0])


def jaccard_loss(outputs, labels, smooth=1e-5):
    """Compute the Jaccard/IoU loss
    Args:
        outputs (Tensor): The model's predictions
        labels (Tensor): The target labels (aka ground truth predictions)
        smooth (float/int): A smoothness factor
    Returns:
        Jaccard loss (Tensor) = 1 - Jaccard Score
    """
    return 1 - jaccard_score(outputs, labels, smooth)


# def tversky(outputs, labels, smooth=1e-5, alpha=0.7):
