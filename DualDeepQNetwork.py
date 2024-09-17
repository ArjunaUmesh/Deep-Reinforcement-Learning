 # -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:40:46 2024

@author: Arjuna.Umesh
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import date
import gym

class DualDeepQNetwork(nn.Module) :
    
    def __init__(self,lr,n_actions,name,input_dims,chkpt_dir) :
        super(DualDeepQNetwork,self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+"__"+str(date.today()))
        
        #self.conv1 = nn.Conv2d(input_dims[0],32,8,stride=4)
        #self.conv2 = nn.Conv2d(32,64,4,stride=2)
        #self.conv3 =nn.Conv2d(64,64,3,stride=1)
        self.fc1 = nn.Linear(input_dims,512)
        self.fc2 = nn.Linear(512,256)
        #SPLIT INTO VALUE AND ADAVANTAGE STREAMS  :
        self.V = nn.Linear(256,1)
        self.A = nn.Linear(256,n_actions)
        
        
        self.optimizer = optim.RMSprop(self.parameters(),lr=lr)
        self.loss= nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        

    def forward(self,state) :
        
        
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))
        
        Values = self.V(flat2)
        Advantages = self.A(flat2)
        
        return Values,Advantages
        
    
    def save_checkpoint(self) :
        print("saving checkpoint..")
        
        torch.save(self.state_dict(),self.checkpoint_file)
        
    def load_checkpoint(self) :
        print("Loading checkpoint")
        self.load_state_dict(torch.load(self.checkpoint_file))
        
# env = gym.make("CartPole-v1")
# state,info = env.reset()

# DDQNetwork = DualDeepQNetwork(lr=0.9,
#                              n_actions=2,
#                              name="DualDQN",
#                              input_dims=4,
#                              chkpt_dir="Models/")

# state = torch.tensor(state,dtype=torch.float32)
# value,adv = DDQNetwork.forward(state)
# print(value)
# print(adv)