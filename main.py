import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from CovidNet import CovidNet
from VGG16 import VGG16

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


###### Initialize the model

### Use CovidNet
model = CovidNet()

### Use VGG16
# model = VGG16()

### Use DenseNet
# model = torchvision.models.densenet121(pretrained=True)

## Adjust the first convolutional layer to accept one channel
## model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.classifier = nn.Linear(model.classifier.in_features, 3)

### Use ResNet
# model = torchvision.models.resnet18(pretrained=True)
## model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# model.fc = nn.Linear(model.fc.in_features, 3)

model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 30
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
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

print("Training finished.")

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

print(f"Accuracy on the test data: {100 * correct / total}%")
