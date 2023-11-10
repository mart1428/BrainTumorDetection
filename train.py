import torch
import torch.nn as nn

import torch.optim

import time

def get_model_name(name, batch_size, lr, epoch):
    path = 'model_{0}_bs{1}_lr{2}_epoch{3}'.format(name, batch_size, lr, epoch)
    return path

def trainModel(model, train_loader, val_loader, test_loader, batch_size = 128, lr = 0.001, epochs = 30):
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Training in {device}')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay= 0.00001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        
        running_loss = 0
        running_error = 0
        correct = 0
        total = 0

        model.train()

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            running_error += (predicted != labels).long().sum().item()
            correct += (predicted == labels).long().sum().item()

            total += labels.size(0)

        train_loss = running_loss/len(train_loader)
        train_error = running_error/len(train_loader.dataset)
        train_acc = correct/total

        with torch.no_grad():
            model.eval()

            running_loss = 0
            running_error = 0
            correct = 0
            total = 0

            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                running_loss += criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                running_error += (predicted != labels).long().sum().item()
                correct += (predicted == labels).long().sum().item()
                total += labels.size(0)

            val_loss = running_loss/len(val_loader)
            val_error = running_error/len(val_loader.dataset)
            val_acc = correct/total

        print(f'Epoch {epoch+1}: Train | Loss: {train_loss:.3f}, Error: {train_error:.3f}, Acc: {train_acc:.2%} || Val | Loss: {val_loss:.3f}, Error: {val_error:.3f}, Acc: {val_acc:.2%}')
        end_time = time.time() - start_time
        print(f'Time after Epoch {epoch}: {end_time:.2f}s')
        model_path = get_model_name(model.name, batch_size, lr, epoch + 1)
        torch.save(model.state_dict(), model_path)

