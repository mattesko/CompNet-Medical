%%time
softmax = nn.Softmax(dim=1)
# with experiment.train():
for epoch in tqdm(range(epochs), desc=f'Training {epochs} epochs'):

    running_loss = 0.0
    unet.train()
    
    if epoch == low_lr_epoch:
        for param_group in optimizer.param_groups:
            lr = lr / 10
            param_group['lr'] = lr

    for i, data in enumerate(train_dataloader):

        input_images, gt_images = data

        if is_cuda_available:
            input_images = input_images.to(device, dtype=input_images_dtype)
            gt_images = gt_images.to(device, dtype=gt_images_dtype)

        outputs = unet(input_images)

        if use_dice_loss:
            outputs = outputs[:,1,:,:].unsqueeze(dim=1)
            loss = criterion(outputs, gt_images)
        else:
            gt_images = gt_images.squeeze(dim=1)
            loss = criterion(outputs, gt_images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
#         experiment.log_metric('Loss', loss.item(), step=i)
    if use_dice_loss:
        print(f'[Epoch {epoch+1:03d}] Training Dice Loss: {running_loss/(i+1):.4f}')
    else:
        print(f'[Epoch {epoch+1:03d}] Training Cross-Entropy Loss: {running_loss/(i+1):.4f}')
#     experiment.log_metric(f"{'Dice' if use_dice_loss else 'Cross-Entropy'} Running Loss", running_loss, 
#                           epoch=epoch, include_context=False)

    unet.eval()
    val_accuracy = 0.0
    all_accuracy = []
    all_dice = []
    all_f1 = []

    for i, data in enumerate(val_dataloader):
        accuracy = 0.0
        intersect = 0.0
        union = 0.0

        input_images, gt_images = data
        if is_cuda_available:
            input_images = input_images.to(device, dtype=input_images_dtype)
            gt_images = gt_images.to(device, dtype=gt_images_dtype)
        outputs = unet(input_images)
        
#         pdb.set_trace()
        # round outputs to either 0 or 1
        if not use_dice_loss: outputs = softmax(outputs)
        outputs = outputs[:, 1, :, :].unsqueeze(dim=1).round()
        
        outputs, gt_images = outputs.data.cpu().numpy(), gt_images.data.cpu().numpy()

#         accuracy += (outputs == gt_images).sum() / float(outputs.size)

        # dice
#         intersect += ((outputs + gt_images) == 2).sum()
#         union += np.sum(outputs == 1) + np.sum(gt_images == 1)

#         all_accuracy.append(accuracy / float(i+1))
#         all_dice.append(1 - (2 * intersect + 1e-5) / (union + 1e-5))
        f1 = f1_score(gt_images.reshape(-1), outputs.reshape(-1), zero_division=1)
        all_f1.append(f1)

        if i % 100 == 0:
            for idx, (out, gt) in enumerate(zip(outputs, gt_images)):
#                 with warnings.catch_warnings():
#                     warnings.filterwarnings("ignore",category=DeprecationWarning)
#                     experiment.log_image(out * 255, name=f'epoch_{epoch}_batch_{i}_idx_{idx}_segmap', overwrite=True, 
#                                          image_format="png", image_scale=1.0, image_shape=None, image_colormap="gray",
#                                          image_channels="first", copy_to_tmp=False, step=i)
                create_canvas(out, gt, show=True)

#     print(f'[Epoch {epoch+1:03d}] Validation Accuracy: {np.mean(all_accuracy)}. Validation Dice Score: {np.mean(all_dice)}\n')\
#     print(f'[Epoch {epoch+1:03d}] Validation Average Dice Score: {np.mean(all_dice)}\n')\
    print(f'[Epoch {epoch+1:03d}] Validation Average F1 Score: {np.mean(all_f1)}\n')

#     experiment.log_metric('Validation Accuracy', np.mean(all_accuracy), 
#                           epoch=epoch, include_context=False)
#     experiment.log_metric('Validation Dice Score', np.mean(all_dice), 
#                           epoch=epoch, include_context=False)
#     experiment.log_metrics({
#         'Validation Accuracy': np.mean(all_accuracy),
#         'Validation Dice Score': np.mean(all_dice)
#     }, epoch=epoch)
# experiment.end()


arr1 = np.squeeze(outputs[1], axis=0)
arr2 = np.squeeze(targets[1], axis=0)
height, _ = arr1.shape
division = np.tile(255//2, (height, 10))
canvas = np.concatenate((arr1*255, arr2*255), axis=1)
plt.imshow(canvas, cmap="gray")
plt.axis('off')










dir_img = '../CompositionalNets/data/rsna-pneumonia-detection-challenge/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()
    
    
    
import torch
from torch import nn
import torch.nn.functional as F

from torchvision import models


class UNet(nn.Module):

    def __init__(self, dice=False, pretrained=False):

        super(UNet, self).__init__()
         
        if pretrained:
            encoder = models.vgg11(pretrained=True)
            self.conv1_input = encoder[0] # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            self.conv2_input = encoder[3] # Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            self.conv3_input = encoder[6] # Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv3 =       encoder[8] # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            self.conv4_input = encoder[11] # Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv4 =       encoder[13] # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            self.conv6 =       encoder[16] # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        else:
            self.conv1_input =      nn.Conv2d(1, 64, 3, padding=1)
            self.conv2_input =      nn.Conv2d(64, 128, 3, padding=1)
            self.conv3_input =      nn.Conv2d(128, 256, 3, padding=1)
            self.conv3 =            nn.Conv2d(256, 256, 3, padding=1)
            self.conv4_input =      nn.Conv2d(256, 512, 3, padding=1)
            self.conv4 =            nn.Conv2d(512, 512, 3, padding=1)
            self.conv6 =            nn.Conv2d(512, 512, 3, padding=1)
        
        self.conv1_input =      nn.Conv2d(1, 64, 3, padding=1)
        self.conv1 =            nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_input =      nn.Conv2d(64, 128, 3, padding=1)
        self.conv2 =            nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_input =      nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 =            nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_input =      nn.Conv2d(256, 512, 3, padding=1)
        self.conv4 =            nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_input =      nn.Conv2d(512, 1024, 3, padding=1)
        self.conv5 =            nn.Conv2d(1024, 1024, 3, padding=1)

        self.conv6_up =         nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6_input =      nn.Conv2d(1024, 512, 3, padding=1)
        self.conv6 =            nn.Conv2d(512, 512, 3, padding=1)
        self.conv7_up =         nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7_input =      nn.Conv2d(512, 256, 3, padding=1)
        self.conv7 =            nn.Conv2d(256, 256, 3, padding=1)
        self.conv8_up =         nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8_input =      nn.Conv2d(256, 128, 3, padding=1)
        self.conv8 =            nn.Conv2d(128, 128, 3, padding=1)
        self.conv9_up =         nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9_input =      nn.Conv2d(128, 64, 3, padding=1)
        self.conv9 =            nn.Conv2d(64, 64, 3, padding=1)
        self.conv9_output =     nn.Conv2d(64, 2, 1)

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def switch(self, dice):

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def forward(self, x):

        layer1 = F.relu(self.conv1_input(x))
        layer1 = F.relu(self.conv1(layer1))

        layer2 = F.max_pool2d(layer1, 2)
        layer2 = F.relu(self.conv2_input(layer2))
        layer2 = F.relu(self.conv2(layer2))

        layer3 = F.max_pool2d(layer2, 2)
        layer3 = F.relu(self.conv3_input(layer3))
        layer3 = F.relu(self.conv3(layer3))

        layer4 = F.max_pool2d(layer3, 2)
        layer4 = F.relu(self.conv4_input(layer4))
        layer4 = F.relu(self.conv4(layer4))

        layer5 = F.max_pool2d(layer4, 2)
        layer5 = F.relu(self.conv5_input(layer5))
        layer5 = F.relu(self.conv5(layer5))

        layer6 = F.relu(self.conv6_up(layer5))
        layer6 = torch.cat((layer4, layer6), 1)
        layer6 = F.relu(self.conv6_input(layer6))
        layer6 = F.relu(self.conv6(layer6))

        layer7 = F.relu(self.conv7_up(layer6))
        layer7 = torch.cat((layer3, layer7), 1)
        layer7 = F.relu(self.conv7_input(layer7))
        layer7 = F.relu(self.conv7(layer7))

        layer8 = F.relu(self.conv8_up(layer7))
        layer8 = torch.cat((layer2, layer8), 1)
        layer8 = F.relu(self.conv8_input(layer8))
        layer8 = F.relu(self.conv8(layer8))

        layer9 = F.relu(self.conv9_up(layer8))
        layer9 = torch.cat((layer1, layer9), 1)
        layer9 = F.relu(self.conv9_input(layer9))
        layer9 = F.relu(self.conv9(layer9))
#         layer9 = self.final(self.conv9_output(layer9), dim=1)
        layer9 = self.conv9_output(layer9)

        return layer9
    

class PretrainedUnet(nn.Module):
    
    def __init__(self, dice=False):
        
        super().__init__()
        encoder = models.vgg11(pretrained=True)
        self.conv1_input = encoder[0] # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.conv2_input = encoder[3] # Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.conv3_input = encoder[6] # Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = encoder[8] # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.conv4_input = encoder[11] # Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = encoder[13] # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.conv5_input = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.conv6_up = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6_input = nn.Conv2d(512 + 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = encoder[16] # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.conv7_up =         nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7_input = nn.Conv2d(256 + 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.conv8_up =         nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8_input = nn.Conv2d(128 + 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.conv9_up = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9_input = nn.Conv2d(64 + 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv9 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv9_out = nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
#         self.conv512_512 = encoder[18] # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def switch(self, dice):

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax
            
    def forward(self, x):
        layer1 = F.relu(self.conv1_input(x))
        layer1 = F.relu(self.conv1(layer1))

        layer2 = F.max_pool2d(layer1, 2)
        layer2 = F.relu(self.conv2_input(layer2))
        layer2 = F.relu(self.conv2(layer2))

        layer3 = F.max_pool2d(layer2, 2)
        layer3 = F.relu(self.conv3_input(layer3))
        layer3 = F.relu(self.conv3(layer3))

        layer4 = F.max_pool2d(layer3, 2)
        layer4 = F.relu(self.conv4_input(layer4))
        layer4 = F.relu(self.conv4(layer4))

        layer5 = F.max_pool2d(layer4, 2)
        layer5 = F.relu(self.conv5_input(layer5))
        layer5 = F.relu(self.conv5(layer5))

        layer6 = F.relu(self.conv6_up(layer5))
        layer6 = torch.cat((layer4, layer6), 1)
        layer6 = F.relu(self.conv6_input(layer6))
        layer6 = F.relu(self.conv6(layer6))

        layer7 = F.relu(self.conv7_up(layer6))
        layer7 = torch.cat((layer3, layer7), 1)
        layer7 = F.relu(self.conv7_input(layer7))
        layer7 = F.relu(self.conv7(layer7))

        layer8 = F.relu(self.conv8_up(layer7))
        layer8 = torch.cat((layer2, layer8), 1)
        layer8 = F.relu(self.conv8_input(layer8))
        layer8 = F.relu(self.conv8(layer8))

        layer9 = F.relu(self.conv9_up(layer8))
        layer9 = torch.cat((layer1, layer9), 1)
        layer9 = F.relu(self.conv9_input(layer9))
        layer9 = F.relu(self.conv9(layer9))
        layer9 = self.final(self.conv9_output(layer9), dim=1)

        return layer9