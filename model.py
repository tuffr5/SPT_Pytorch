import torch
import torch.nn as nn
from convlstm import ConvBLSTM

def init_weights(net):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.xavier_uniform_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif classname.find('BatchNorm') != -1:
            nn.init.ones_(m.weight.data)
            nn.init.zeros_(m.bias.data)
            
    net.apply(init_func)
        
        
class SamplingNet(nn.Module):
    """Conditional sampling network
    """
    def __init__(self, last_dim):
        super(SamplingNet, self).__init__()
        
        self.net = nn.Sequential(ConvBLSTM(1, 64, kernel_size=(3, 3), num_layers=1, batch_first=True),
                                nn.BatchNorm3d(64),
                                nn.Dropout3d(0.25),
                                ConvBLSTM(64, 80, kernel_size=(3, 3), num_layers=1, batch_first=True),
                                nn.BatchNorm3d(80),
                                nn.Dropout3d(0.3),
                                ConvBLSTM(80, 80, kernel_size=(3, 3), num_layers=1, batch_first=True),
                                nn.BatchNorm3d(80,),
                                nn.Dropout3d(0.3),
                                ConvBLSTM(80, 80, kernel_size=(3, 3), num_layers=1, batch_first=True),
                                nn.BatchNorm3d(80),
                                nn.Dropout3d(0.3),
                                nn.Conv3d(80, 256, kernel_size=(3, 3, 3), padding='same'),
                                nn.Sigmoid(),
                                nn.Dropout3d(0.3),
                                nn.Conv3d(256, 64, kernel_size=(3, 3, 3), padding='same'),
                                nn.Sigmoid(),
                                nn.Conv3d(64, 1, kernel_size=(3, 3, 3), padding='same'),
                                nn.Sigmoid(),
                                nn.MaxPool3d(kernel_size=(last_dim, 1, 1)))
        init_weights(self.net)


    def forward(self, x):
        return self.net(x)