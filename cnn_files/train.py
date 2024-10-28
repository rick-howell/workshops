# This file trains the CNN on the MNIST data set

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Step 1: Load the data
data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Step 2: Create a DataLoader
data_loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
    
# Step 3: Create the CNN
import model
cnn = model.CNN()
print(cnn)

# Step 4: Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

# Step 5: Train the CNN
num_epochs = 2
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
            running_loss = 0.0

print('Finished Training')

# Step 6: Save the model
torch.save(cnn.state_dict(), 'mnist_cnn_weights')

# Step 7: Run Tests
data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)

correct = 0
total = 0

with torch.no_grad():
    for data in data_loader:
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
