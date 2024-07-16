from m_dataset import CustomDataset, combine_fold
from Unet_model import UNet
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os


def train():
    lr_setting = 0.0001
    batch_size = 2
    epoch = 1000
    device = "mps"
    save_path = "./work_2"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor()])

    combine_fold("./dataset/f01", "./dataset/f02", 1, 2)

    trainDS = CustomDataset(imgs_path="./dataset/f02/image", mask_path="./dataset/f02/label", transform=transform)

    testDS = CustomDataset(imgs_path="./dataset/f03/image", mask_path="./dataset/f03/label", transform=transform)

    trainLoader = DataLoader(trainDS, shuffle=True,
                             batch_size=batch_size, num_workers=0)
    testLoader = DataLoader(testDS, shuffle=False,
                            batch_size=batch_size, num_workers=0)

    unet = UNet().to(device)

    lossFunc = BCEWithLogitsLoss()
    optimizer = Adam(unet.parameters(), lr=lr_setting)

    trainSteps = len(trainDS) // batch_size
    testSteps = len(testDS) // batch_size

    H = {"train_loss": [], "test_loss": []}

    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(epoch)):
        # set the model in training mode
        unet.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        # loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # perform a forward pass and calculate the training loss
            pred = unet(x)
            loss = lossFunc(pred, y)
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()
            # loop over the validation set
            for (x, y) in testLoader:
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # make the predictions and calculate the validation loss
                pred = unet(x)
                totalTestLoss += lossFunc(pred, y)
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, epoch))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(save_path)
    # serialize the model to disk
    torch.save(unet, save_path+"output.pth")


def main():
    train()


if __name__ == '__main__':
    main()
