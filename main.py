import torch
import torch.nn as nn

import torch.optim

import torchvision
import torchvision.transforms as transforms

from data import get_data_loader
from train import trainModel
from model import TumorDetector


def testModel(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Testing in {device}')
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
            model.eval()

            running_loss = 0
            running_error = 0
            correct = 0
            total = 0

            for i, data in enumerate(test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                running_loss += criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                running_error += (predicted != labels).long().sum().item()
                correct += (predicted == labels).long().sum().item()
                total += labels.size(0)

            test_loss = running_loss/len(val_loader)
            test_error = running_error/len(val_loader.dataset)
            test_acc = correct/total

    print(f'Test | Loss: {test_loss:.3f}, Error: {test_error:.3f}, Acc: {test_acc:.2%}')

if __name__ == '__main__':
    torch.manual_seed(1000)
    classes, train_loader, val_loader, test_loader = get_data_loader()
    model = TumorDetector()


    batch_size = 128
    lr = 0.01
    epochs = 30

    train_model = 0
    if train_model:
        trainModel(model, train_loader, val_loader, test_loader, batch_size, lr, epochs)
    
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = 'model_TumorDetector_bs128_lr0.01_epoch6'
        model.load_state_dict(torch.load(model_path, map_location= device))
        testModel(model, test_loader)