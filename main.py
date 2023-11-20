import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet121, resnet18
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from CovidNet import CovidNet
from VGG16 import VGG16
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data transformations for training and testing
transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

# Set the batch size and number of workers for data loading
batch_size = 32
num_workers = 2

# Create data loaders for the train and test datasets
train_dataset = ImageFolder(root="./data/covid_dataset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

validation_dataset = ImageFolder(root="./data/covid_dataset/validation", transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

models = ["CovidNet", "DenseNet", "ResNet", "VGG16"]

for currModel in models:
    model = CovidNet()

    if currModel == "DenseNet":
        model = densenet121(pretrained=True)

    elif currModel[0] == "ResNet":
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 3)

    elif currModel == "VGG16":
        model = VGG16()

    model.to(device)

    print(f"\n\nFOR MODEL {currModel}:")

    with open(currModel, "w") as file:
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Train and evaluate the model in each epoch
        num_epochs = 30
        accuracyPerEpoch = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Evaluate the model on the test data
            correct = 0
            total = 0
            with torch.no_grad():
                for data in validation_loader:
                    images, labels = data
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            accuracyPerEpoch.append(accuracy)

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy on the test data: {accuracy}%", file=file)

        # Calculate mean
        mean_value = np.mean(accuracyPerEpoch)

        # Calculate standard deviation
        std_deviation = np.std(accuracyPerEpoch)

        print("Mean:", mean_value, file=file)
        print("Standard Deviation:", std_deviation, file=file)

print("Finished!")
