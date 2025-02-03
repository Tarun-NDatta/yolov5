import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os

#data transformations
transform = transforms.Compose([
    transforms.RandomRotation(10),  
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

total_start_time = time.time()

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, 28, 28)
            sample_output = self.conv3(self.conv2(self.conv1(sample_input)))
            flattened_size = sample_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model_path = "cnn_model.pth"

if os.path.exists(model_path):
    print("Loading pre-trained model...")
    model.load_state_dict(torch.load(model_path, weights_only=True))  
else:
    print("Training model...")
    num_epochs = 10

    for epoch in range(num_epochs):
        epoch_start_time = time.time()  

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        epoch_end_time = time.time() 
        epoch_time = epoch_end_time - epoch_start_time  

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
              f"Accuracy: {100 * correct / total:.2f}%, Time Taken: {epoch_time:.2f} sec")

    torch.save(model.state_dict(), model_path)
    print("Model saved.")

total_end_time = time.time()
total_training_time = total_end_time - total_start_time
print(f"Total Training Time: {total_training_time:.2f} sec")

# Inference Time Tracking
inference_start_time = time.time()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

inference_end_time = time.time()
inference_time = inference_end_time - inference_start_time

print(f'Accuracy on test images: {100 * correct / total:.2f}%')
print(f'Total Inference Time: {inference_time:.4f} sec')

num_images = 10
plt.figure(figsize=(10, 6))

pred_start_time = time.time()

data_iter = iter(test_loader)
images, labels = next(data_iter)
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

pred_end_time = time.time()
prediction_time = pred_end_time - pred_start_time

for i in range(num_images):
    plt.subplot(1, num_images, i+1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(f'Pred: {predicted[i]}\nTrue: {labels[i]}')
    plt.axis('off')

plt.show()
print(f"Time Taken for Prediction of {num_images} images: {prediction_time:.4f} sec")
