"""Standalone script, same as the Jupyter notebook"""
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# Download and normalize the 60,000 training images, and 10,000 test images
training_data = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transforms.ToTensor())
testing_data = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transforms.ToTensor())

# Initialize data loaders to load the images and labels for the network
train_loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=64, shuffle=True)

# Define the network architecture
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(784, 20) # define hidden layer dimensions: 784 inputs, 20 outputs
        self.fc2 = nn.Linear(20, 10) # define output layer dimensions: 20 inputs, 10 outputs

    def forward(self, inputs):
        input_layer = inputs.view(inputs.shape[0], -1) # flatten the 28x28 images into a 784-long vector
        hidden_layer = F.sigmoid(self.fc1(input_layer)) # apply sigmoid to hidden layer
        output_layer = F.log_softmax(self.fc2(hidden_layer), dim=1) # apply softmax to output layer
        return output_layer

# Initialize network, error/loss function, and weight weights_optimizer
model = NeuralNet()
loss_function = nn.NLLLoss() # Calculates the loss (error) of the network
weights_optimizer = optim.SGD(model.parameters(), lr=0.05) # Updates the weights of the network

# Training loop
for epoch in range(10): # loop over the whole dataset 10 times
    running_loss = 0.0
    for images, labels in train_loader:
        # reset the optimizer
        weights_optimizer.zero_grad()

        # forward pass
        outputs = model(images)
        loss = loss_function(outputs, labels)

        # backpropagation
        loss.backward()
        weights_optimizer.step()

        # print statistics
        running_loss += loss.item()
    print('Epoch: %d loss: %.3f' %
          (epoch + 1, running_loss / len(train_loader)))

print('Finished Training')



