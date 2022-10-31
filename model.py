import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    
    def __init__(self, in_dim, out_dim, num_hidden_layers, layer_size):
        super().__init__()

        self.num_layers = num_hidden_layers * 2 + 3 # *2 accounts for ReLU layers, +3 is input layer, input relu layer, output layer

        self.in_dim = in_dim
        self.out_dim = out_dim        

        self.layer_size = layer_size

        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Linear(self.in_dim, self.layer_size))
        self.num_hidden_layers = num_hidden_layers

        for i in range(1,self.num_hidden_layers):
            self.layer_list.append(nn.Linear(self.layer_size, self.layer_size))
            

        self.layer_list.append(nn.Linear(self.layer_size, self.out_dim))
        
    def forward(self, x):

        x = x.view(-1, self.in_dim)

        for i in range(self.num_hidden_layers):
            x = F.relu(self.layer_list[i](x))

        return self.layer_list[self.num_hidden_layers](x)

# Rewrite this class with AlexNet equivalent
class CNN(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv1 = nn.Conv2d(3, 8, (3, 3), stride=2, padding='same')
        self.conv2 = nn.Conv2d(8, 16, (3, 3), stride=2, padding='same')
        self.conv3 = nn.Conv2d(16, 32, (3, 3), stride=2, padding='same')

        inputs = torch.zeros(in_dim)
        outputs = self.conv3(self.conv2(self.conv1(inputs)))
        # lin_input = 1
        # for i in range(len(outputs.shape)):
        #     lin_input *= outputs.shape[i]

        print(outputs.shape)

        # Hardcoded input number for first layer
        self.lin1 = nn.Linear(32*20*20, 500)
        self.lin2 = nn.Linear(500, self.out_dim)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # flatten convolutional layer into vector
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x


class CNN_small(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv1 = nn.Conv2d(3, 8, (3, 3), stride=2, padding='same')
        self.conv2 = nn.Conv2d(8, 16, (3, 3), stride=2, padding='same')
        self.conv3 = nn.Conv2d(16, 32, (3, 3), stride=2, padding='same')

        # inputs = torch.zeros(in_dim)
        # outputs = self.conv3(self.conv2(self.conv1(inputs)))
        # lin_input = 1
        # for i in range(len(outputs.shape)):
        #     lin_input *= outputs.shape[i]

        # print(outputs.shape)

        # Hardcoded input number for first layer
        self.lin1 = nn.Linear(32*20*20, 500)
        self.lin2 = nn.Linear(500, self.out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # flatten convolutional layer into vector
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x