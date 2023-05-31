import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameter setting
num_epochs = 10
num_classes = 10
batch_size = 128
learning_rate = 0.05
DOWNLOAD = True
# Fashion MNIST data set
train_dataset = torchvision.datasets.FashionMNIST(
 root='./FashionMnist',  # Data paths
 train=True,  # for training
 transform=torchvision.transforms.ToTensor(),  # turn to tensor [0,1]
 download=DOWNLOAD, #Whether to download the dataset
)

test_dataset = torchvision.datasets.FashionMNIST(
 root='./FashionMnist',
 train=False, # For testing
 transform=torchvision.transforms.ToTensor(),
 download=DOWNLOAD)

# Data Loader
# Features and labels in the data are packaged separately by batch
# shuffle indicates whether to randomly shuffle the data
# num_workers indicates whether to start a new thread to speed up the loading of data,
# but windows often reports an error if set to any other number, so it is better to set it to 0.

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, input_channel=1, output_channel=10):
        '''
        This part defines 5 convolution layers and 3 fully connected layers to realize multi-classification of images
        :param input_channel:
        :param output_channel:
        '''
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),  # batch Normalisation
            nn.ReLU(),  # reluActivation functions
            nn.MaxPool2d(kernel_size=3, stride=2),  # Maximum pooling layer
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv5 = torch.nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 2048),  ## Here it is calculated based on the input image size,
            # only the pooling operation will shrink the image, the 28*28 image is pooled three times and the image size is 2*2
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, output_channel),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)   #Change of dimension, equivalent to straightening operation

        output = self.classifier(x)

        return output


model = ConvNet(input_channel=1, output_channel=10).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward propagation
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Reverse propagation and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Defining test functions
def test_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Test models
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
test_accuracy = test_model(model, test_loader, device)
print(f"Test accuracy: {test_accuracy:.2f}%")
# Defining forward hook functions
def hook_fn(module, input, output):
    global layer_activations
    layer_activations = output

# Register forward hooks
hook_module = model.classifier
handle = hook_module.register_forward_hook(hook_fn)

def extract_cnn_activations(test_dataset, cnn_model, indices):

    activations = []

    with torch.no_grad():
        for i in indices:
            image, _ = test_dataset[i]  # Just extract the image, no labels required
            image = image.to(device)

            image = image.unsqueeze(0)
            _ = cnn_model(image)
            output = layer_activations.cpu().numpy().flatten()

            activations.append(output)

    return activations

# Get the sample index for each category
indices_per_class = [[] for _ in range(10)]
for i, (_, label) in enumerate(test_dataset):
    if len(indices_per_class[label]) < 100:
        indices_per_class[label].append(i)

# Activation using index extraction
activations_cnn = [extract_cnn_activations(test_dataset, model, indices) for indices in indices_per_class]
#Calculating the average correlation coefficient
def compute_mean_correlation(activations_i, activations_j):
    n_i = len(activations_i)
    n_j = len(activations_j)
    correlation_matrix = np.corrcoef(np.vstack([activations_i, activations_j]))
    return correlation_matrix[:n_i, n_i:].mean()
def generate_matrix():
    # Create an empty matrix of 10*10 for save correlation
    correlation_matrix_cnn = np.zeros((10, 10))
    # generate 10*10 matrix
    for i in range(10):
        for j in range(10):
            correlation_matrix_cnn[i, j] = compute_mean_correlation(activations_cnn[i], activations_cnn[j])
    print(correlation_matrix_cnn)
    handle.remove()
    # draw iamge Generate Figure 2
    plt.figure(figsize=(10, 10))
    plt.imshow(correlation_matrix_cnn, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(correlation_matrix_cnn.shape[1]), fontsize=12)
    plt.yticks(range(correlation_matrix_cnn.shape[0]), fontsize=12)
    plt.title("Correlation Matrix", fontsize=16)
    plt.show()

