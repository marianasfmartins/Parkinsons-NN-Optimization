'''Considering that our activation function is ReLU, our chosen initialization methods are He Normal (Kaiming Normal) and He Uniform (Kaiming Uniform). 
This is because ReLU substitutes all negative values with zero, so it loses half of the values. Therefore, the He/Kaiming method solves this problem since it starts with the weights having double the strength.
The He Uniform will ensure no weight will have an absurdly high value, and the He Normal will make the weights work in the shape of a normal distribution.'''

import torch.nn as nn
#Nota: entram 22 features (numero de features que temos) e sai 1 (previsao de parkingstone ou nao)
def HeNormalInitializationMethod():
    layer = nn.Linear(in_features=22, out_features=1)
    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
    return layer

def HeUniformInitializationMethod():    
    layer = nn.Linear(in_features=22, out_features=1)
    nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
    return layer


