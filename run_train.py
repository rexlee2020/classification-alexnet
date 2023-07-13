from tqdm import tqdm
from utils import CIFAR10Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms





def run_training(my_model):

    if torch.cuda.is_available():
        device = 'cuda'
        print("Using cuda")
    else:
        device = 'cpu'
        print("Using cpu")
        
    # Define transforms for data normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create train and validation datasets
    trainset = CIFAR10Dataset(root_dir="cifar-10-batches-py", transform=transform, train=True)
    valset = CIFAR10Dataset(root_dir="cifar-10-batches-py", transform=transform, train=False)

    # Create train and validation dataloaders
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)



    # net = AlexNet()
    
    net = my_model
    net = net.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    num_epochs = 1000
    best_accuracy = 0.0
    patience = 3  # Number of epochs to wait for improvement
    counter = 0  # Counter for early stopping
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=patience)

    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch', disable=True) as pbar:
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 200 == 199:
                    pbar.set_postfix({'Loss': running_loss/200})
                    running_loss = 0.0
                pbar.update(1)

        # Validation loop
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch : {epoch}, Validation accuracy: {accuracy:.2f}%')

        # Check if the current accuracy is better than the previous best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            counter = 0  # Reset the counter
        else:
            counter += 1

        # Check if early stopping criteria is met
        if counter >= patience:
            lr_print = optimizer.param_groups[0]['lr']
            print(f'Decreasing lr to {lr_print}! Validation accuracy stopped improving.')
            
            # Decrease learning rate by 1/10
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            
            lr_scheduler.step(best_accuracy)  # Adjust learning rate scheduler based on best accuracy
            counter = 0  # Reset the counter
            
            if optimizer.param_groups[0]['lr'] < 1e-6:
                print('Learning rate reached minimum threshold. Stopping training.')
                break

    print('Training finished')

    # Evaluate the network on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(valloader), desc='Testing', unit='batch') as pbar:
            for data in valloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.update(1)

    accuracy = 100 * correct / total
    # print(f'Accuracy of the network on the {len(valset)} validation images: {accuracy:.2f}%')
    return accuracy
