import torch as T
import torch.nn as nn #Where all of our layers come from
import torch.nn.functional as F  #Basis for layers and activation functions
import torch.optim as optim # Where optimizers live, like Adam

# Gettuing datasets. Torchvision has a bunch. lets grab mnist
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor # Help transform the raw data from dataset into tensors


import numpy as np
import matplotlib.pyplot as plt

class CNN(nn.Module):
    # All NN modules made must derive from the nn classes.
    # gvies access to parameters of layers, etc.

    def __init__(self,
                 learning_rate,
                 epochs,
                 batch_size,
                 num_classes=10):
        
        super(CNN, self).__init__()
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Init learning parameters
        self.loss_history = []
        self.acc_history = []
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


        ################ Begin defining our network #####################
        self.conv1 = nn.Conv2d(1, 32, 3) # (1 channel --> images are B&W)
                                            # (32 convulutional filters at 3x3 size)
        # define our batch normalization
        # takes in 32 filters, as the output from above
        self.bn1 = nn.BatchNorm2d(32) 
        
        # 32 channel inputs, outputs 32 filters, in 3x3 window
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)

        # Define pooling step. The images are 28x28 sized and 
        # we've been sliding a 3x3 window across it in each layer.
        # Each time we do this, we are performing matrix multiplication.
        # We've now output a set of 32 different filters.
        # Here, we now slide a 2x2 filter on those filters we have to
        # "pull" the "max" value from each slide on each filter.
        # It's a way to reduce dimensionality
        self.maxpool1 = nn.MaxPool2d(2)

        # Output 64 filters
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.bn6 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)

        # Need input dimensions. It'll reduce size of feature set by two
        # for each maxpool. There's a better way to do it and figure it out.
        # (custom). Must be defined here first
        input_dims = self.calc_input_dims()

        self.fc1 = nn.Linear(input_dims, self.num_classes)

        ##############END OF NETWORK ###############
        
        #  Define optimizer. self.paraneters is inherited from nn
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        self.loss = nn.CrossEntropyLoss()

        # Send NN to device
        self.to(self.device)

        # Need to grab data (custom)
        self.get_data()

    def calc_input_dims(self):
        """ Figure out how much each conv layer removes from our image 
        size. That way, we can account for it in our calculations."""

        # 4D tensor to match our input. It's zeros to help us with our shape reduction
        batch_data = T.zeros((1,1,28,28)) #one batch, one channel, 28x28 size. 
        batch_data = self.conv1(batch_data)
        # batch_data = self.bn1(batch_data)
        batch_data = self.conv2(batch_data)
        # batch_data = self.bn2(batch_data)
        batch_data = self.conv3(batch_data)

        batch_data = self.maxpool1(batch_data) # This will definitely reduce dimensionality
        batch_data = self.conv4(batch_data)
        batch_data = self.conv5(batch_data)
        batch_data = self.conv6(batch_data)
        batch_data = self.maxpool2(batch_data)

        return int(np.prod(batch_data.size()))
    
    def forward(self, batch_data):

        # pass data to GPU, making it a CUDA tensor
        batch_data = T.tensor(batch_data).to(self.device) #Note: lowercase tensor() preserves dtype. Tensor() does not

        ##########Copy the self.calc_input_dims() method above ###########
        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv2(batch_data)
        batch_data = self.bn2(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool1(batch_data) # This will definitely reduce dimensionality
        
        batch_data = self.conv4(batch_data)
        batch_data = self.bn4(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv5(batch_data)
        batch_data = self.bn5(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv6(batch_data)
        batch_data = self.bn6(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool2(batch_data)

        # flatten
        batch_data = batch_data.view(batch_data.size()[0], -1) #maybe self.flatten instead?

        # We dont want to perform softmax because 
        # the crossentropyloss will perform the softmaxing
        classes = self.fc1(batch_data)

        return classes
    
    def get_data(self):
        """ Pull testing data from the pytorch repo of data."""
        ## Pull our training data
        mnist_train_data = MNIST('gym-form-checker/example-mnist/mnist',
                                 train=True,
                                 download=True,
                                 transform=ToTensor())
        
        # like train_test_split, but only train_split! Here, we just define the 
        # batch size.
        self.train_data_loader = T.utils.data.DataLoader(mnist_train_data,
                                                         batch_size=self.batch_size,
                                                         shuffle=True,
                                                         num_workers=6,)
        
        ## Pull our testing data
        mnist_test_data = MNIST('gym-form-checker/example-mnist/mnist',
                                 train=False,
                                 download=True,
                                 transform=ToTensor())
        
        # like train_test_split, but only train_split! Here, we just define the 
        # batch size.
        self.test_data_loader = T.utils.data.DataLoader(mnist_test_data,
                                                         batch_size=self.batch_size,
                                                         shuffle=True,
                                                         num_workers=6,)
        
    def _train(self):
        """ Train our model. train already exists, thats why we call it
        _train, not train.
        
        NOTE: THIS DOES NOT DO ANY TRAINIGN OR UPDATING WEIGHTS OF NN. 
        TELLS PYTORCH THAT YOU ARE ABOUT TO ENTER THE TRAINING MODE.
        Yes, there is a Training and Testing Mode.
        
        This way, it doesnt update the statistics for the batch norm layers
        that we have defined already."""
        self.train()
        for i in range(self.epochs):
            epoch_loss = 0
            epoch_accuracy = []

            # enumerate over our data
            for j, (input_data, label) in enumerate(self.train_data_loader):
                # Gradients accumulate over each trainign step. Zero out for each trainign step
                self.optimizer.zero_grad()
                label = label.to(self.device) # Send out label to our device
                
                # Doesn't need to be cast to device. Already done in self.forward
                prediction = self.forward(input_data)

                # calculate loss
                loss = self.loss(prediction, label)

                # For our sake, let's see the prediction. dim=1 is the batch
                prediction = F.softmax(prediction, dim=1)

                classes = T.argmax(prediction, dim=1)
                wrong = T.where(classes != label,
                                T.tensor([1.]).to(self.device), #Right, gets 1
                                T.tensor([0.]).to(self.device)) #Wrong, gets 0
                
                current_accuracy = 1 - T.sum(wrong) / self.batch_size #scale accuracy by batch size

                # Append accuracy. .item() gives su the value IN the tensor to dereference it
                epoch_accuracy.append(current_accuracy.item())
                self.acc_history.append(current_accuracy.item())
                epoch_loss += loss.item()

                # Learning, backpropagate
                loss.backward()
                self.optimizer.step() # Perform one optimization step

            # Check epoch score
            print(f'Finished epoch {i}|\tTotal Loss: {epoch_loss}|\tAccuracy: {np.mean(epoch_accuracy)*100:.3f}')

            # Append loss history
            self.loss_history.append(epoch_loss)


    def _test(self):
        """ Test our model. Test already exists, thats why we call it
        _Test, not Test."""

        epoch_loss = 0
        epoch_accuracy = []

        # enumerate over our data
        for j, (input_data, label) in enumerate(self.test_data_loader):
            # We aren't updating gradients, so we don't need to zero anything
            # self.optimizer.zero_grad()
            label = label.to(self.device) # Send out label to our device
            
            # Doesn't need to be cast to device. Already done in self.forward
            prediction = self.forward(input_data)

            # calculate loss
            loss = self.loss(prediction, label)

            # For our sake, let's see the prediction. dim=1 is the batch
            prediction = F.softmax(prediction, dim=1)

            classes = T.argmax(prediction, dim=1)
            wrong = T.where(classes != label,
                            T.tensor([1.]).to(self.device), #Right, gets 1
                            T.tensor([0.]).to(self.device)) #Wrong, gets 0
            
            current_accuracy = 1 - T.sum(wrong) / self.batch_size #scale accuracy by batch size

            # Append accuracy. .item() gives su the value IN the tensor to dereference it
            epoch_accuracy.append(current_accuracy.item())
            # self.acc_history.append(current_accuracy.item()) # Not plotting
            epoch_loss += loss.item()

            # DO NOT BACKPROPAGATE
            # loss.backward()
            # self.optimizer.step() # Perform one optimization step

        # Check epoch score
        print(f'Finished batch {j}|\tTotal Loss: {epoch_loss}|\tAccuracy: {np.mean(epoch_accuracy)*100:.3f}')

        # Not plotting
        # self.loss_history.append(epoch_loss)

if __name__ == '__main__':
    network = CNN(learning_rate=0.01,
                  batch_size=128,
                  epochs=25)
    network._train()
    plt.plot(network.loss_history)
    plt.show()
    plt.plot(network.acc_history)
    plt.show()
    network._test()