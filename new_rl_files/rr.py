import torch
import torch.nn as nn

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.layer2 = nn.Tanh()
        self.layer3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.layer4 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.layer5 = nn.Tanh()
        self.layer6 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.layer7 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.layer8 = nn.Tanh()
        self.layer9 = nn.Flatten()

    def forward(self, x):
        print(f"XSHAPE {x.shape}")
        x = self.layer1(x)
        print("After Conv2d(1, 6, 5):", x.shape)
        x = self.layer2(x)
        x = self.layer3(x)
        print("After AvgPool2d(2, 2):", x.shape)
        x = self.layer4(x)
        print("After Conv2d(6, 16, 5):", x.shape)
        x = self.layer5(x)
        x = self.layer6(x)
        print("After AvgPool2d(2, 2):", x.shape)
        x = self.layer7(x)
        print("After Conv2d(16, 120, 5):", x.shape)
        x = self.layer8(x)
        x = self.layer9(x)
        print("After Flatten:", x.shape)
        return x

# Create a random input tensor with shape (1, 33, 33)
input_tensor = torch.randn( 1, 33, 33)

# Instantiate the network
net = Net()

# Perform a forward pass
output = net(input_tensor)


import torch

# Example tensor with shape (33, 33, 1)
tensor = torch.randn(33, 33, 1)

# Reshape using .permute()
reshaped_tensor = tensor.permute(2, 0, 1)  # permute to (1, 33, 33)



print("Original shape:", tensor.shape)
print("Reshaped shape:", reshaped_tensor.shape)


print(tensor)

print(reshaped_tensor)