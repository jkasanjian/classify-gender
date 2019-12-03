import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# define model parameters
NUM_EPOCHS = 90  # original paper
BATCH_SIZE = 100
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 2  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use
# modify this to point to your data directory
INPUT_ROOT_DIR = 'data/train'
TRAIN_IMG_DIR = 'data/validation'

class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes=2):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        # initialize bias
        self.init_bias()  

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.
        Args:
            x (Tensor): input tensor
        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        return self.classifier(x)

if __name__ == '__main__':
    # create model
    alexnet = AlexNet(num_classes=NUM_CLASSES)
    print('AlexNet created')
    print(alexnet)


    # create dataset and transform image to 227 x 227
    train_transform = transforms.Compose([
        transforms.RandomSizedCrop(227),
        transforms.CenterCrop(IMAGE_DIM),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transform= train_transform)

    
    print('Dataset created')

    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    
    print('Dataloader created')

    
    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    # setting proposed by the original paper - which doesn't train....
    #     optimizer = optim.SGD(
    #     params=alexnet.parameters(),
    #     lr=LR_INIT,
    #     momentum=MOMENTUM,
    #     weight_decay=LR_DECAY)
    print('Optimizer created')

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')

    # start training!!
    print('Starting training...')
    total_steps = 0
    i = 0
    batch = 1
    acc = []
    for epochs in range(1): #should be NUM_EPOCHS
        lr_scheduler.step()
        for imgs,labels in dataloader:
            
            # calculate the loss
            output = alexnet(imgs)
            loss = F.cross_entropy(output, labels)
            
            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if total_steps % 10 == 0:
            #     with torch.no_grad():
            #         _, preds = torch.max(output, 1)
            #         accuracy = torch.sum(preds == labels).item()
            #     print("Epoch:", epochs, "Accuracy:", accuracy, "Step", total_steps)
            total_steps += 1
            #i +=1
        # with torch.no_grad():
        #     _, preds = torch.max(output, 1)
        #     accuracy = torch.sum(preds == labels).item()
        #     print("Epoch:", epochs, "Accuracy:", accuracy/100, "Step", total_steps)

        alexnet.save_state_dict('Trained')

# Evaluate testing dataset, must be compatible for DatasetLoader, make sure BATCH SIZE is 1 for incoming dataset 
def eval(dataset, aNet):
    correct = 0
    alexnet = aNet.load_state_dict('Trained')
    for x,y in dataset:
        output = alexnet(x)
        ##Need to check whether torch.no_grad makes a difference one at a time. Should we have it for every iteration?
        with torch.no_grad:
            _, pred = torch.max(output,1)
            if(pred == y):
                correct += 1
    return (correct/len(dataset))




if __name__ == '__main__':
    
    # create model
    alexnet = AlexNet(num_classes=NUM_CLASSES)
    print('AlexNet created')
    print(alexnet)

    # create dataset and transform image to 227 x 227
    train_transform = transforms.Compose([
        transforms.RandomSizedCrop(227),
        transforms.CenterCrop(IMAGE_DIM),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transform= train_transform)
    print('Dataset created')

    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    print('Dataloader created')

    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    # setting proposed by the original paper - which doesn't train....
    #     optimizer = optim.SGD(
    #     params=alexnet.parameters(),
    #     lr=LR_INIT,
    #     momentum=MOMENTUM,
    #     weight_decay=LR_DECAY)
    print('Optimizer created')

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')

    # start training!!
    print('Starting training...')
    total_steps = 0
    i = 0
    batch = 1
    acc = []
    for epochs in range(1): #should be NUM_EPOCHS
        lr_scheduler.step()
        for imgs,labels in dataloader:
            
            # calculate the loss
            output = alexnet(imgs)
            loss = F.cross_entropy(output, labels)
            
            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if total_steps % 10 == 0:
            #     with torch.no_grad():
            #         _, preds = torch.max(output, 1)
            #         accuracy = torch.sum(preds == labels).item()
            #     print("Epoch:", epochs, "Accuracy:", accuracy, "Step", total_steps)
            total_steps += 1
            #i +=1
        # with torch.no_grad():
        #     _, preds = torch.max(output, 1)
        #     accuracy = torch.sum(preds == labels).item()
        #     print("Epoch:", epochs, "Accuracy:", accuracy/100, "Step", total_steps)

        alexnet.save_state_dict('Trained')

# Evaluate testing dataset, must be compatible for DatasetLoader, make sure BATCH SIZE is 1 for incoming dataset 
def eval(dataset, aNet):
    correct = 0
    alexnet = aNet.load_state_dict('Trained')
    for x,y in dataset:
        output = alexnet(x)
        # Need to check whether torch.no_grad makes a difference one at a time. Should we have it for every iteration?
        with torch.no_grad:
            _, pred = torch.max(output,1)
            if(pred == y):
                correct += 1
    return (correct/len(dataset))
