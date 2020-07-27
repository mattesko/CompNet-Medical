import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def create_canvas(arr1, arr2, show=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
    ax1.set_title('Predicted Segmentation Map')
    ax1.imshow(np.squeeze(arr1 * 255), cmap="gray")
    ax1.axis('off')
    
    ax2.set_title('Ground Truth Segmentation Map')
    ax2.imshow(np.squeeze(arr2 * 255), cmap="gray")
    ax2.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    if not show:
        plt.close()
    return im