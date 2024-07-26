import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from m_dataset import CustomDataset
from DC_loss import Dice
import torch
import cv2 as cv
import os

device = "cuda:0"
threshold = 0.5
imgs_path = "./dataset/f03/image"
mask_path = "./dataset/f03/label"
model_path = "./work_9/output.pth"
save_path = "./work_9"


def prepare_plot(origImage, origMask, predMask, filename):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.savefig(os.path.join(save_path, "pred_"+filename))


def make_predictions(model, imagePath, maskPath=mask_path):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
        orig = image.copy()
        # find the filename and generate the path to ground truth
        # mask
        filename = imagePath.split(os.path.sep)[-1]
        groundTruthPath = os.path.join(maskPath, filename)
        gtMask = cv.imread(groundTruthPath, cv.IMREAD_GRAYSCALE)

    # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor()])
        #image = np.transpose(image, (2, 0, 1))
        image = transform(image)
        image = image.unsqueeze(dim=0)
        image = image.to(device)
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image)
        predMask = torch.squeeze(predMask)
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        # filter out the weak predictions and convert them to integers
        predMask = (predMask > threshold) * 255
        predMask = predMask.astype(np.uint8)
        acc = Dice(predMask, gtMask)
        # prepare a plot for visualization
        prepare_plot(orig, gtMask, predMask, filename)
        return acc


def main():
    unet = torch.load(model_path).to(device)
    print("[INFO] loading up test image paths...")
    avg_Acc = 0
    for img in os.listdir(imgs_path):
        # make predictions and visualize the results
        img = os.path.join(imgs_path, img)
        acc = make_predictions(unet, img)
        print("Img:{} Accuracy:{:.4f}%".format(img, acc*100))
        avg_Acc = avg_Acc + acc
    
    print("Average accuracy: {:.4f}%".format((avg_Acc/len(imgs_path))*100))


if __name__ == '__main__':
    main()
