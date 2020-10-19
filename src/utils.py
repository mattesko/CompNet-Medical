import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
# import torchvision.transforms.functional as F


def create_canvas(arr1, arr2, show=False, title1='Predicted Segmentation Map',
                 title2='Ground Truth Segmentation Map'):
    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
    ax1.set_title(title1)
    ax1.imshow(np.squeeze(arr1), cmap="gray")
    ax1.axis('off')

    ax2.set_title(title2)
    ax2.imshow(np.squeeze(arr2), cmap="gray")
    ax2.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    if not show:
        plt.close()
    return im


# def stack_masks(origin, mask1=None, mask2=None):
#     """Stack two masks ontop of one another to help compare their boundaries"""
#     img = F.to_pil_image(origin + 0.5).convert("RGB")
#     if mask1 is not None:
#         mask1 =  F.to_pil_image(torch.cat([
#             torch.zeros_like(origin),
#             torch.stack([mask1.float()]),
#             torch.zeros_like(origin)
#         ]))
#         img = Image.blend(img, mask1, 0.2)

#     if mask2 is not None:
#         mask2 =  F.to_pil_image(torch.cat([
#             torch.stack([mask2.float()]),
#             torch.zeros_like(origin),
#             torch.zeros_like(origin)
#         ]))
#         img = Image.blend(img, mask2, 0.2)
    
#     return img
