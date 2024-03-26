import torch
import torch.nn as nn
import numpy as np

class DualModel(nn.Module):
    def __init__(self, model):

        super(DualModel, self).__init__()
        self.default_cfg = model.default_cfg
        self.num_classes = model.num_classes
        


        #self.model = nn.Sequential(*list(model.children())[:-1])
        #self.fc = list(model.children())[-1]        
        self.model = model
        self.fc = model.fc
        model.fc = nn.Identity()
        
        self.dense_fc = nn.Linear(self.fc.in_features,4096)
        self.relu = nn.LeakyReLU()
        
        self.pre_fc = nn.Linear(self.fc.in_features,64)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # state size. 64 x 4 x 4
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # state size. 32 x 8 x 8
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # state size. 16 x 16 x 16
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            # state size. 8 x 32 x 32
            nn.ConvTranspose2d(8, 1, 4, 2, 1)
        )
        self.sig = nn.Sigmoid()
        self.post_fc = nn.Linear(4096,4096)
        self.sharedmodules = nn.ModuleList([self.model])
        self.taskmodules = nn.ModuleList([self.fc, self.dense_fc,self.deconvs,self.pre_fc,self.post_fc])
        
    def forward(self,x,on=False):

        x = self.model(x)
        #print(x.shape)
        #x = torch.mean(x, dim=1)
        if on:
            x1 =  self.fc(x)
            #x2 = self.sig(self.dense_fc(x))
            x2 = self.post_fc(self.deconvs(self.pre_fc(x).unsqueeze(-1).unsqueeze(-1)).squeeze().squeeze().reshape(-1,4096))
            return x1, x2
        return self.fc(x)



class DualLoss(nn.Module):
    """This is label smoothing loss function.
    """
    def __init__(self,loss,weights):
        super(DualLoss, self).__init__()
        self.dense_loss =  nn.BCEWithLogitsLoss()
        self.categorical_loss = loss
        self.dense_labels = torch.tensor(np.random.choice([0, 1], size=(196,64*64)).astype("float32"))

        self.weights = weights

    def forward(self,output,target,seperate=False):
        dense_target = []
        for t in target:
            dense_target.append(self.dense_labels[t])
        dense_target = torch.stack(dense_target).cuda()
        #print(dense_target.shape,output[1].shape)
        loss1 = self.categorical_loss(output[0],target)*self.weights[0]
        loss2 = self.dense_loss(output[1],dense_target)*self.weights[1]
        if seperate:
            return [loss1,loss2]
        return loss1 + loss2

