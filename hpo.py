#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms

import argparse

NUM_DOG_TYPES=133


def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset))
    )

def train(model, train_loader, criterion, optimizer, args):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    model.train()
    epochs = args.epoch
    for e in range(epochs):
        running_loss=0
        correct=0
        for data, target in train_loader:
            optimizer.zero_grad()
            pred = model(data)             #No need to reshape data since CNNs take image inputs
            loss = criterion(pred, target)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            print(f"Loss {running_loss/len(train_loader.dataset)}, Accuracy {100*(correct/len(train_loader.dataset))}%")
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    
    for parameter in model.parameters():
        parameter.require_gradient = False #Shutting off gradient calculation in order to not touch hidden layers
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(nn.Linear(num_features, NUM_DOG_TYPES))
    return model
    
def create_data_loaders(s3Bucket, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    #https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2
    #https://pytorch.org/vision/stable/transforms.html
    
    transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_uri = f"s://{s3Bucket}/dogImages/train"
    test_uri = f"s://{s3Bucket}/dogImages/test"
    val_uri = f"s://{s3Bucket}/dogImages/val"

    train_loader = datasets.ImageFolder(train_uri, transform=transform, batch_size=batch_size["train"], shuffle=True)
    # val_loader = datasets.ImageFolder(val_uri, transform=transform, batch_size=batch_size["val"],shuffle=True)
    test_loader = datasets.ImageFolder(test_uri, transform=transform, batch_size=batch_size["test"], shuffle=False)
    return train_loader, test_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, test_loader = create_data_loaders(args.s3bucket, {"train":args.batch_size, "test":args.test_batch_size})

    model=train(model, train_loader, loss_criterion, optimizer, args)

    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader,loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    outPath = f"s://{args.s3bucket}/dogImages"
    torch.save(model, "proj3_cnn.pt")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--s3bucket",
        type=str,
        default=None,
        metavar="N",
        help="input s3 bucket name (default: None)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    args=parser.parse_args()
#     train_kwargs = {"batch_size": args.batch_size}
#     test_kwargs = {"batch_size": args.test_batch_size}
    
    main(args)
