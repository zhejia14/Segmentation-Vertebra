from My_dataset import CustomDataset, combine_fold, AugmentedDataset
from Unet_model import UNet
from DC_loss import DiceLoss
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, SGD
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch ,gc
import time
import os
import my_DEBUG
import Augment_Transform
import wandb


def train():
    lr_setting = 0.001
    batch_size = 1
    epoch = 5000
    device = "cuda:0"
    save_path = "./work_11"
    model_save_step = 500
    model_save_times = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor()])

    combine_path = combine_fold("./dataset/f01", "./dataset/f02", 1, 2)

    trainDS = CustomDataset(imgs_path=os.path.join(combine_path, "image"), mask_path=os.path.join(combine_path, "label"), transform=transform)
    augmented_1 = AugmentedDataset(trainDS,  augment_transform=Augment_Transform.augment_RandomFlip)
    #augmented_2 = AugmentedDataset(trainDS, augment_transform=Augment_Transform.augment_VerticalFlip)

    testDS = CustomDataset(imgs_path="./dataset/f03/image", mask_path="./dataset/f03/label", transform=transform)

    CombineDataset = ConcatDataset([trainDS, augmented_1])
    #CombineDataset = ConcatDataset([CombineDataset, augmented_2])
    trainLoader = DataLoader(CombineDataset, shuffle=True,
                             batch_size=batch_size, num_workers=0)
    testLoader = DataLoader(testDS, shuffle=False,
                            batch_size=batch_size, num_workers=0)

    unet = UNet().to(device)

    lossFunc1 = BCEWithLogitsLoss()
    lossFunc2 = DiceLoss()
    #optimizer = Adam(unet.parameters(), lr=lr_setting)
    optimizer = SGD(unet.parameters(), lr=lr_setting, momentum=0.9)
    #scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=3000)
    trainSteps = len(CombineDataset) // batch_size
    testSteps = len(testDS) // batch_size

    H = {"train_loss": [], "test_loss": []}

    print("Size of training dataset:{}".format(len(CombineDataset)))
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(epoch)):
       
        unet.train()
       
        totalTrainLoss = 0
        totalTestLoss = 0
        totalTrainDiceLoss = 0
       
        for (i, (x, y)) in enumerate(trainLoader):

            (x, y) = (x.to(device), y.to(device))
            pred = unet(x)
            loss = lossFunc1(pred, y) + lossFunc2(pred, y)
            dice_loss = lossFunc2(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
           
            totalTrainLoss += loss
            totalTrainDiceLoss += dice_loss
        
        with torch.no_grad():

            unet.eval()
            
            for (x, y) in testLoader:
                (x, y) = (x.to(device), y.to(device))
                pred = unet(x)
                totalTestLoss += lossFunc2(pred, y)
        
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps
        avgTrainDiceLoss = totalTrainDiceLoss / trainSteps
       
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        
        print("[INFO] EPOCH: {}/{}".format(e + 1, epoch))
        print("Train loss: {:.6f}, Train Dice Loss: {:.4f}, Test loss: {:.4f}, lr: {:.4f}".format(avgTrainLoss, avgTrainDiceLoss, avgTestLoss, optimizer.param_groups[0]["lr"]))
        wandb.log({"Train loss": avgTrainLoss})
        wandb.log({"Train Dice loss": avgTrainDiceLoss, "Test Dice loss": avgTestLoss})
        wandb.log({"Epoch": e})
        if e!= 0 and e % model_save_step == 0:
            model_save_times = model_save_times + model_save_step
            torch.save(unet, os.path.join(save_path, str(model_save_times)+"_output.pth"))
        gc.collect()
        torch.cuda.empty_cache()
    
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))
    wandb.finish()
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(save_path)
    torch.save(unet, os.path.join(save_path, "final_output.pth"))
    


def main():
    wandb.init(
        config={
            "learning_rate": 0.001,
            "batch_size": 1,
            "optimizer": "SGD",
            "epochs": 5000,
            "Augmented_Describe": "RandomFlip"
        }
    )
    train()


if __name__ == '__main__':
    main()
