import numpy as np
import torch
import torchvision  # Data set
import torch.nn as nn  # Build a network
import torch.utils.data as Data  # load data
import time   # timekeeping
import matplotlib.pyplot as plt  # drawing
torch.cuda.empty_cache()#Clean up unwanted memory to avoid running out of space
# hyper_parameters setting
DOWNLOAD = True  # 'True' means the data set needs to be downloaded and 'False' means it is already downloaded.
BATCH_SIZE = 128   #Set to suit your computer 16 32 64 etc.
EPOCH = 10  # epoch times
learning_rate = 0.05  #learning rate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Automatic selection of training device, cpu / gpu (if supported)

## load dataset

train_data = torchvision.datasets.FashionMNIST(
 root='./FashionMnist',  # Data storage directory
 train=True,  #  use for training
 transform=torchvision.transforms.ToTensor(),  # Convert to [0,1] tensor
 download=DOWNLOAD, #Whether to download the dataset
)

test_data = torchvision.datasets.FashionMNIST(
 root='./FashionMnist',
 train=False, # use for testing
 transform=torchvision.transforms.ToTensor(),
 download=DOWNLOAD)

print(train_data[0]) # Print view of data is a tensor image data and label

# Features and labels in the data are packaged separately by batch
# shuffle indicates whether to randomly shuffle the data
# num_workers indicates whether to start a new thread to speed up the loading of data,
# but windows often reports an error if set to any other number, so it is better to set it to 0.

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class MyCNN(torch.nn.Module):
    # set network structure
    '''
    This part defines 5 convolution layers and 3 fully connected layers to realize multi-classification of images
    '''
    def __init__(self, input_channel=1, output_channel=10):
        """
                Initialize the CNN model with the given input_channel and output_channel.
                :param input_channel: The number of input channels (default: 1)
                :param output_channel: The number of output channels (default: 10)
        """
        super(MyCNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),  # batch Normalisation
            nn.ReLU(),  # reluActivation functions
            nn.MaxPool2d(kernel_size=3, stride=2),  #Maximum pooling layer
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
            nn.Linear(256 * 2 * 2, 2048),  # Here it is calculated based on the input image size,
            # only the pooling operation will shrink the image, the 28*28 image is pooled three times and the image size is 2*2
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, output_channel),
        )

    def forward(self, x):
        """
               Perform the forward pass on the input tensor x.
               :param x: The input tensor
               :return: The output tensor
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  #Change of dimension, equivalent to straightening operation
        output = self.classifier(x)

        return output




def train_test():
    # optimizer Optimizer selection and definition of loss function
    # SGD stochastic gradient descent optimization with cross-entropy loss is chosen here
    optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, weight_decay=0.001)
    # optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    # Some data preparation for plotting, saving and exporting
    train_epochs_loss = []
    train_acc = []# saving training acc
    test_epochs_loss = []
    test_acc = []# saving testing acc
    best_acc = 0

    sum = 0
    # training and testing
    print('Training and Testing ...\n')
    for epoch in range(EPOCH):
        cnn.train()  # Get into training mode
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time() #init
        for step, (tr_x, tr_y) in enumerate(train_loader):
            tr_x, tr_y = tr_x.to(device), tr_y.to(device)
            output = cnn(tr_x)  # The output of a well-trained network
            loss = loss_func(output, tr_y)  # Calculating losses
            optimizer.zero_grad() # Gradient first set to 0
            loss.backward()  # Calculate the gradient of each parameter according to the backward propagation of the loss function
            optimizer.step()  # Parameter update

            train_l_sum += loss.item()  # Total accumulated losses
            train_acc_sum += (output.argmax(dim=1) == tr_y).sum().item()  # Cumulative number of correct predictions
            n += tr_y.shape[0]

        train_epochs_loss.append((train_l_sum/(step+1)))
        train_acc.append(train_acc_sum / n)
        #initialising test results
        test_corrects = 0.0
        test_num = 0

        cnn.eval()  # 测试形态，不能忘
        acc_sum, num, te_loss = 0.0, 0, 0.0
        with torch.no_grad():  # 这一句也很重要，有时可以减少内存占用（具体原因未知）
            for bat_idx, (X, y) in enumerate(test_loader):
                te_out = cnn(X.to(device))
                acc_sum += (te_out.argmax(dim=1) == y.to(device)).sum().item()
                te_loss += loss_func(te_out, y.to(device)).item()
                num += y.shape[0]
        test_acc_epoch = acc_sum / num
        test_loss = te_loss / (bat_idx + 1)
        test_acc.append(test_acc_epoch)


        #Print training and test to see the results of this epoch
        print('epoch: %d, loss %.4f, train_acc: %.3f,  test_acc: %.3f,  test_loss: %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / (step+1), train_acc_sum / n,  test_acc_epoch, test_loss, time.time() - start))
        test_epochs_loss.append(test_loss)
        if test_acc_epoch > best_acc:

            best_acc = test_acc_epoch

    #draw image Generate Figure 1
    plt.subplot(121)
    plt.plot(train_acc[:],'-o',label="train_acc")
    plt.plot(test_acc[:],'-o',label="test_acc")
    plt.title('epochs_accuracy')
    plt.legend()
    # Generate the accuracy and loss plot.
    plt.subplot(122)
    plt.plot(train_epochs_loss[:],'-o',label="train_loss")
    plt.plot(test_epochs_loss[:],'-o',label="test_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.savefig('eqochs_acc_loss.png')
    plt.show()
if __name__ == '__main__':
    cnn = MyCNN(input_channel=1, output_channel=10)  # Creating a network, not to be forgotten
    print(cnn)  # Print network structure
    cnn.to(device)  # Loading the network to the device
    train_test()