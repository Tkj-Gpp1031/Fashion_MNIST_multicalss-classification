from matplotlib import pyplot as plt
from torchvision.models import resnet18
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Data pre-processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.Grayscale(num_output_channels=3),#turn image from gray to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#normalize image
])
# Load the Fashion MNIST test set
test_dataset = FashionMNIST(root="./FashionMNIST",  # Data paths
                             train=False,  # No training data set is used
                             transform=transform,   # Transform PIL.Image or numpy.array data types to torch.

                             download=False,  # If you have already downloaded the data, you do not need to download it again here
                             )
#Get the last hidden layer active by forward hooking
def hook_fn(module, input, output):
    global layer_activations
    layer_activations = output


#Get activations
def extract_resnet18_activations(test_dataset, pretrained_resnet18, indices):
    global layer_activations
    activations = []
    with torch.no_grad():
        for i in indices:
            image, _ = test_dataset[i]  # 只需提取图像，无需标签
            image = image.to(device)

            image = image.unsqueeze(0)
            _ = pretrained_resnet18(image)
            output = layer_activations.cpu().numpy().flatten()

            activations.append(output)
    return activations


#Calculating the average correlation coefficient
def compute_mean_correlation(activations_i, activations_j):
    n_i = len(activations_i)
    n_j = len(activations_j)
    correlation_matrix = np.corrcoef(np.vstack([activations_i, activations_j]))
    return correlation_matrix[:n_i, n_i:].mean()
def generate_matrix():
    #Create an empty matrix of 10*10 for save correlation
    correlation_matrix_resnet18 = np.zeros((10, 10))
    # generate 10*10 matrix
    for i in range(10):
        for j in range(10):
            correlation_matrix_resnet18[i, j] = compute_mean_correlation(activations_resnet18[i], activations_resnet18[j])
    print(correlation_matrix_resnet18)
    handle.remove()
    #draw iamge Generate Figure 2
    plt.figure(figsize=(10, 10))
    plt.imshow(correlation_matrix_resnet18, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(correlation_matrix_resnet18.shape[1]), fontsize=12)
    plt.yticks(range(correlation_matrix_resnet18.shape[0]), fontsize=12)
    plt.title("Correlation Matrix", fontsize=16)
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load ResNet-18 model
    pretrained_resnet18 = resnet18()
    pretrained_resnet18 = pretrained_resnet18.to(device)
    #  Register forward hooks
    hook_module = pretrained_resnet18.layer4[1].conv2
    handle = hook_module.register_forward_hook(hook_fn)

    # Get the sample index for each category
    indices_per_class = [[] for _ in range(10)]
    for i, (_, label) in enumerate(test_dataset):
        if len(indices_per_class[label]) < 100:
            indices_per_class[label].append(i)

    # Activation using index extraction
    activations_resnet18 = [extract_resnet18_activations(test_dataset, pretrained_resnet18, indices) for indices in
                            indices_per_class]
    #
    generate_matrix()