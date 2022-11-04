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

# Contains rewritten small CNN for task 1
class CNN(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv1 = nn.Conv2d(3, 16, (5, 5), stride=1, padding='valid')
        self.conv2 = nn.Conv2d(16, 32, (5, 5), stride=1, padding='valid')
        self.conv3 = nn.Conv2d(32, 64, (3, 3), stride=1, padding='valid')

        # inputs = torch.zeros(in_dim)
        # outputs = self.conv3(self.conv2(self.conv1(inputs)))
        # lin_input = 1
        # for i in range(len(outputs.shape)):
        #     lin_input *= outputs.shape[i]

        # print(outputs.shape)

        self.pooling = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)

        # Hardcoded input number for first layer
        self.lin1 = nn.Linear(64*3*3, 500)
        self.lin2 = nn.Linear(500, self.out_dim)


    def forward(self, x):
        x = self.pooling(self.dropout(F.relu(self.conv1(x))))
        x = self.pooling(self.dropout(F.relu(self.conv2(x))))
        x = self.dropout(F.relu(self.conv3(x)))

        # flatten convolutional layer into vector
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x

# Provided small CNN
class CNN_small(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim        

        self.fc_layer_neurons = 200

        self.layer1_filters = 32

        self.layer1_kernel_size = (4,4)
        self.layer1_stride = 1
        self.layer1_padding = 0

        #NB: these calculations assume:
        #1) padding is 0;
        #2) stride is picked such that the last step ends on the last pixel, i.e., padding is not used
        self.layer1_dim_h = (self.in_dim[1] - self.layer1_kernel_size[0]) / self.layer1_stride + 1
        self.layer1_dim_w = (self.in_dim[2] - self.layer1_kernel_size[1]) / self.layer1_stride + 1        

        self.conv1 = nn.Conv2d(3, self.layer1_filters, self.layer1_kernel_size, stride=self.layer1_stride, padding=self.layer1_padding)

        self.fc_inputs = int(self.layer1_filters * self.layer1_dim_h * self.layer1_dim_w)

        self.lin1 = nn.Linear(self.fc_inputs, self.fc_layer_neurons)

        self.lin2 = nn.Linear(self.fc_layer_neurons, self.out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        # flatten convolutional layer into vector
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x