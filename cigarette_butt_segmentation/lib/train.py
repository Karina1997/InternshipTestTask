import copy
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from lib.metrics import dice_loss


def calc_loss(pred, target, metrics, batch_size, bce_weight=0.5):
    target = target.view(batch_size, 1, 512, 512)
    pred = torch.sigmoid(pred)

    bce = F.binary_cross_entropy(pred, target)

    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, all_losses, all_bce, all_dice, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    all_losses[phase].append(metrics['loss'] / epoch_samples)
    all_bce[phase].append(metrics['bce'] / epoch_samples)
    all_dice[phase].append(metrics['dice'] / epoch_samples)
    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, dataloaders, device, batch_size, num_epochs=25):
    if torch.cuda.is_available():
        model.cuda()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    all_losses = {'train' : [], 'val' : []}
    all_bce = {'train' : [], 'val' : []}
    all_dice = {'train' : [], 'val' : []}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                inputs = inputs.float()

                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs.float(), labels.float(), metrics, batch_size)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, all_losses, all_bce, all_dice, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, [all_losses, all_bce, all_dice]
