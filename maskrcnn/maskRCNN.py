import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.nn.functional as F
from PIL import Image
from CustomDataset import CustomDataset
from torch.utils.data import DataLoader
import os


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


# Load a pre-trained Mask R-CNN model
def train():

    num_classes = 2
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    save_path = "./"

    dataset_train = CustomDataset(root="../dataset/f01/")

    data_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0,
                                collate_fn=lambda x: tuple(zip(*x)))

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print("EPOCH[{}] losses = {:.4f} lr = {:.4f}".format(epoch, losses, optimizer.param_groups[0]["lr"]))
        
    # Update the learning rate
    lr_scheduler.step()
    torch.save(model, os.path.join(save_path, "output.pth"))

if __name__ == '__main__':
    train()