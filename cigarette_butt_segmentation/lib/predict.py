import torch


def get_pred_masks(model, test_data_loader, device):
    pred_masks_dict = {}
    for img_id, image in test_data_loader:
        img = image.to(device)
        res = model(img.float())
        pred_mask = torch.sigmoid(res).cpu()
        pred_masks_dict[img_id] = pred_mask.view(512, 512).detach().apply_(round).numpy() * 255

    pred_masks_dict = dict(sorted(pred_masks_dict.items()))
    return pred_masks_dict
