from comet_ml import Experiment
import torch
import torch.nn.functional as F
import numpy as np
import warnings
import kornia.augmentation as K

from src.metrics import dice_score, jaccard_score
from src.utils import create_canvas
from src.config import directories
from src.dataset import Resize


def train_one_epoch(net, dataloader, optimizer, criterion, use_dice_loss=False,
                    num_accumulated_steps=1, device=None, input_dtype=torch.double,
                    target_dtype=torch.long, **kwargs):
    """Train the network on the entire dataset once
    Args:
        net (torch.nn.Module): The neural network to train
        dataloader (torch.utils.data.Dataloader): The batched data to train on
        optimizer (torch.optim): The optimizer for the network (Adam, etc.)
        criterion (callable): The loss function. Expects pairs of outputs and
        targets as inputs
        use_dice_loss (bool): Whether you're using the Dice loss for training
            Must be set to true if the criterion is a Dice loss function
            (default: False)
        num_accumulated_steps (int): The number of times the loss gradients are
            accumulated before a step in the optimizer is taken.
            Helpful to increase this to speed up training if your batch size is
            low (default: 1)

    Returns:
        network (torch.nn.Module), running_loss (float32)
    """
    net.train()
    torch.set_grad_enabled(True)
    optimizer.zero_grad()
    running_loss = 0.0

    for i, data in enumerate(dataloader):

        input_images, targets = data
        
        if device:
            input_images = input_images.to(device, input_dtype)
            targets = targets.to(device, target_dtype)
        
        outputs = net(input_images)

        if use_dice_loss:
            outputs = F.log_softmax(outputs, dim=1)
            outputs = outputs[:, 1, :, :].unsqueeze(dim=1)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze(dim=1)
            loss = criterion(outputs, targets)

        loss.backward()
        running_loss += loss.detach().cpu().numpy()

        if (i+1) % num_accumulated_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            running_loss /= num_accumulated_steps

    if (i+1) % num_accumulated_steps != 0:
        optimizer.step()
        running_loss /= (num_accumulated_steps - i % num_accumulated_steps)

    running_loss = running_loss.item()
    return net, running_loss


def validate(net, dataloader, epoch, device=None, input_dtype=torch.double,
             target_dtype=torch.long, use_dice_loss=False, experiment=None,
             batch_freq=50, epoch_freq=25, threshold=0.5, **kwargs):
    """Gather validation metrics (Dice, Jaccard) on neural network
    """
    net.eval()
    torch.set_grad_enabled(False)
    dice_mean = torch.zeros((1), device=device)
    jaccard_mean = torch.zeros((1), device=device)

    for i, data in enumerate(dataloader):

        input_images, targets = data
        
        if device:
            input_images = input_images.to(device, input_dtype)
            targets = targets.to(device, target_dtype)
        
        outputs = net(input_images)

        if use_dice_loss:
            outputs = F.log_softmax(outputs, dim=1)
        else:
            outputs = F.softmax(outputs, dim=1)
        outputs = F.threshold(outputs[:, 1, :, :].unsqueeze(dim=1), threshold, 0)
        outputs = torch.round(outputs)

        score = dice_score(outputs, targets)
        dice_mean = dice_mean + (score - dice_mean) / (i + 1)
        score = jaccard_score(outputs, targets)
        jaccard_mean = jaccard_mean + (score - jaccard_mean) / (i + 1)

        if experiment:
            if i % batch_freq == 0 and (epoch+1) % epoch_freq == 0:
                outputs, targets = outputs.data.cpu().numpy()*255, targets.data.cpu().numpy()*255
                for idx, (out, gt) in enumerate(zip(outputs, targets)):
                    with warnings.catch_warnings():
                        img = create_canvas(out, gt, show=False)
                        warnings.filterwarnings("ignore",category=DeprecationWarning)
                        experiment.log_image(img, name=f'epoch_{epoch:03d}_batch_{i:03d}_idx_{idx}_segmap', overwrite=True, 
                                             image_format="png", image_scale=1.0, image_shape=None, image_colormap="gray",
                                             image_channels="first", copy_to_tmp=False, step=epoch)

    return dice_mean.item(), jaccard_mean.item()


if __name__ == '__main__':
    import os
    from datetime import datetime

    from sklearn.model_selection import train_test_split
    import h5py

    from torch.nn import CrossEntropyLoss
    from torch import optim
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from src.dataset import Chaos2DSegmentationDataset, get_image_pair_filepaths, ClassificationDataset
    from src.models import UNet
    from src.metrics import dice_loss, dice_score
    
    date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    experiment = Experiment(api_key="P5seMqEJjqZ8mDA7QYSuK3yUJ",
                            project_name="chaos-liver-segmentation",
                            workspace="matthew42", auto_metric_logging=False)

    params = {
        "lr": 0.0001,
        "batch_size": 16,
        "split_train_val": 0.8,
        "epochs": 45,
        "use_dice_loss": False,
        "cache": True,
        "random_seed": 42,
        "shuffle_data": True,
        "scheduler": "StepLR",
        "step_size": 15,
        "gamma": 0.75,
        "threshold": 0.9,
        "pretrained": True,
    }

    is_cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda_available else "cpu")
    input_images_dtype = torch.double
    targets_dtype = torch.long
    if is_cuda_available: torch.cuda.empty_cache()
        
    
    size = (256, 256)
    crop = (224, 224)
    cache_input_transform = transforms.Compose([
        transforms.Lambda(lambda x: np.stack((x, x, x), axis=2)),
        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
        transforms.Lambda(lambda x: x.astype(np.float32)),
#         transforms.ToPILImage(),
#         transforms.Grayscale(3),
#         transforms.Resize((128, 128)),
#         transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
        Resize(size),
        K.CenterCrop(size=crop),
    ])

    cache_target_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.astype(np.uint8)),
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255),
        transforms.Lambda(lambda x: x.long()),
    ])

#    input_transform = transforms.Compose([
#        K.RandomAffine(0, shear=(-5, 5)),
#        K.RandomHorizontalFlip(),
#        transforms.Lambda(lambda x: x.squeeze()),
#    ])
    input_transform = None

#    target_transform = transforms.Compose([
#        K.RandomAffine(0, shear=(-5, 5)),
#        K.RandomHorizontalFlip(),
#        transforms.Lambda(lambda x: x.squeeze()),
#        transforms.Lambda(lambda x: x*255),
#        transforms.Lambda(lambda x: x.long()),
#    ])
    target_transform = None

    # Load data for training and validation
#     image_pair_filepaths = get_image_pair_filepaths(data_dir)[:]
#     train_filepaths, val_filepaths = train_test_split(image_pair_filepaths,
#                                                       train_size=params['split_train_val'],
#                                                       random_state=params['random_seed'],
#                                                       shuffle=params["shuffle_data"])

#     train_dataset = Chaos2DSegmentationDataset(train_filepaths, input_transform=input_transform,
#                                                gt_transform=target_transform, cache=params['cache'],
#                                                cache_input_transform=cache_input_transform,
#                                                cache_gt_transform=cache_target_transform,
#                                                device=device)

#     val_dataset = Chaos2DSegmentationDataset(val_filepaths, cache=params['cache'],
#                                              cache_input_transform=cache_input_transform,
#                                              cache_gt_transform=cache_target_transform,
#                                              device=device)
    data_dir = directories['chaos']
    hdf5_path = os.path.join(data_dir, 'train.hdf5')
#     hdf5_path = os.path.join(data_dir, 'train_augmented.hdf5')
    hf = h5py.File(hdf5_path, 'r')

    images, targets = hf['images'], hf['masks']

    images = [cache_input_transform(im) for im in images]
    targets = [cache_target_transform(t) for t in targets]

    X_train, X_test, y_train, y_test = train_test_split(images, targets, 
                                                        train_size=params['split_train_val'],
                                                        random_state=params['random_seed'],
                                                        shuffle=params["shuffle_data"])
    train_dataset = ClassificationDataset(X_train, y_train, input_transform,
                                         target_transform)
    val_dataset = ClassificationDataset(X_test, y_test,
            target_transform=transforms.Compose([transforms.Lambda(lambda x: x*255), transforms.Lambda(lambda x: x.long()),])
            )

    num_train, num_val = len(train_dataset), len(val_dataset)
    params['num_samples'] = num_train + num_val
    params['target_transform'] = target_transform.__str__()
    params['input_transform'] = input_transform.__str__()

    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'])
    val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'])

    # Instantiate model, optimizer, and criterion

    unet = UNet(dice=params['use_dice_loss'], pretrained=params['pretrained'])
    if is_cuda_available: unet = unet.to(device, dtype=input_images_dtype)

    optimizer = optim.Adam(unet.parameters(), lr=params['lr'])
    if params['scheduler'] == 'StepLR': 
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                              step_size=params['step_size'], gamma=params['gamma'])
    elif params['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # cross-entropy loss: weighting of negative vs positive pixels
    loss_weight = torch.DoubleTensor([0.01, 0.99])
    if is_cuda_available: loss_weight = loss_weight.to(device)
    criterion = dice_loss if params['use_dice_loss'] else CrossEntropyLoss(weight=loss_weight,
                                                                           reduction='mean')

    image, target = train_dataset[0]
    image = image.clone().permute(1, 2, 0).numpy()
    target = target.clone().numpy()
    img = create_canvas(image, target, show=False,
                               title1='Example Input', title2='Example Target')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        experiment.log_image(img, name=f'example_input_target', overwrite=True,
                    image_format="png", image_scale=1.0, image_shape=None, image_colormap="gray",
                    copy_to_tmp=False)
    
    experiment.log_parameters(params)
    num_accumulated_steps = 128 // params['batch_size']

    with experiment.train():

        print(f'Number of training images:\t{num_train}\nNumber of validation images:\t{num_val}')
        for epoch in range(params['epochs']):

            unet, running_loss = train_one_epoch(unet, train_dataloader, optimizer,
                                                 criterion, 
                                                 num_accumulated_steps=num_accumulated_steps,
                                                 device=device, **params)

            if params['use_dice_loss']:
                print(f'[Epoch {epoch+1:03d} Training]\tDice Loss:\t\t{running_loss:.4f}')
            else:
                print(f'[Epoch {epoch+1:03d} Training]\tCross-Entropy Loss:\t{running_loss:.4f}')
            experiment.log_metric("Running Loss", running_loss, epoch=epoch, step=epoch, include_context=False)

            f1_mean, jaccard_mean = validate(unet, val_dataloader, epoch,
                                             device=device,
                                             batch_freq=25, epoch_freq=5,
                                             experiment=experiment, **params)

            if params['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(f1_mean)
            else:
                scheduler.step()
            print(f'[Epoch {epoch+1:03d} Validation]\tAverage F1 Score:\t{f1_mean:.4f}\tAverage Jaccard/IoU:\t{jaccard_mean:.4f}\n')

            experiment.log_metric('Validation Average F1 Score', f1_mean,
                                  epoch=epoch, include_context=False)
            experiment.log_metric('Validation Average Jaccard/IoU', jaccard_mean,
                                  epoch=epoch, include_context=False)
                
            
    filepath = os.path.join(directories['checkpoints'], f'unet_liver_{date_time}.pth')
#     torch.save(unet.state_dict(), filename)
    torch.save({
#         'epoch': epoch,
        'model_state_dict': unet.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
        }, filepath)
    experiment.log_asset(filepath, copy_to_tmp=False)
    experiment.end()
    hf.close()
