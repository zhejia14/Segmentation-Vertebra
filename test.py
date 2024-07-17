import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
import os

device = "mps"
threshold = 0.5
imgs_path = "./dataset/f03/image"
mask_path = "./dataset/f03/label"
model_path = "./work_2/output.pth"
save_path = "./work_2"


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
        image = cv.imread(imagePath)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        # resize the image and make a copy of it for visualization
        orig = image.copy()
        # find the filename and generate the path to ground truth
        # mask
        filename = imagePath.split(os.path.sep)[-1]
        groundTruthPath = os.path.join(maskPath, filename)
        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        gtMask = cv.imread(groundTruthPath, 0)

    # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(device)
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        # filter out the weak predictions and convert them to integers
        predMask = (predMask > threshold) * 255
        predMask = predMask.astype(np.uint8)
        # prepare a plot for visualization
        prepare_plot(orig, gtMask, predMask, filename)


def main():
    unet = torch.load(model_path).to(device)
    print("[INFO] loading up test image paths...")
    
    for img in os.listdir(imgs_path):
        # make predictions and visualize the results
        img = os.path.join(imgs_path, img)
        print(img)
        make_predictions(unet, img)


if __name__ == '__main__':
    main()
