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
    if not show:
        plt.close()

    ax2.set_title(title2)
    ax2.imshow(np.squeeze(arr2), cmap="gray")
    ax2.axis('off')
    if not show:
        plt.close()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)

    if not show:
        plt.close(fig=fig)
    
    return im


def synthetic_occlusion(test_slices, textured=False, color=1):
    synthetic_images = []
    
    for i in range(0, len(test_slices)-2, 3):
    
        im = test_slices[i].copy()
        # im = im[110:390, 50:380]
        # im[150:200, 25:125] = im.min()
        if textured:
            im[200:225, 100:150] = color
            im[200:225, 150:200] = im.min()

            im[225:250, 100:150] = im.min()
            im[225:250, 150:200] = color

            im[250:275, 100:200] = color
            im[250:275, 150:200] = im.min()

            im[275:300, 100:200] = im.min()
            im[275:300, 150:200] = color
        else:
            im[200:300, 100:200] = color
        synthetic_images.append(im)

        im = test_slices[i+1].copy()
        # im = im[115:370, 50:420]
        # im[50:100, 150:250] = im.min()
        if textured:
            im[200:225, 100:150] = color
            im[200:225, 150:200] = im.min()

            im[225:250, 100:150] = im.min()
            im[225:250, 150:200] = color

            im[250:275, 100:200] = color
            im[250:275, 150:200] = im.min()

            im[275:300, 100:200] = im.min()
            im[275:300, 150:200] = color
        else:
            im[200:300, 100:200] = color
        synthetic_images.append(im)

        im = test_slices[i+2].copy()
        # im = im[115:385, 50:420]
        # im[120:190, 40:135] = im.min()
        if textured:
            im[150:175, 200:250] = color
            im[150:175, 250:300] = im.min()
            im[175:200, 200:250] = im.min()
            im[175:200, 250:300] = color
        else:
            im[150:200, 200:300] = color
        synthetic_images.append(im)

    im = np.tile(0.0, (512, 512, 3)).astype(np.float32)
    synthetic_images.append(im)
    return synthetic_images


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
