import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
    
    def __init__(self,
                 beta,
                 input_dims,
                 fc1_dims,
                 fc2_dims,
                 n_actions,
                 name,
                 checkpoint_dir = "models") : 
        super(CriticNetwork,self).__init__()
        
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,self.name + '_ddpg')
        
        self.fc1 = nn.Linear(*self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        ######
        self.action_value = nn.Linear(self.n_actions,self.fc2_dims)
        ######
        
        self.q = nn.Linear(self.fc2_dims,1)
        
        
        ##INITIALIZATION OF DEFINED WEIGHTS TO THE NETWORK  
        
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1,f1)
        self.fc1.bias.uniform_(-f1,f1)
        
        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2,f2)
        self.fc2.bias.uniform_(-f2,f2)
        
        f3 = 0.003
        self.q.weight.data.uniform_(-f3,f3)
        self.q.bias.uniform_(-f1,f3)
        
        f4 = 1.0 / np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4,f4)
        self.fcaction_value2.bias.uniform_(-f4,f4)
        
        self.optimizer =  optim.Adam(self.parameters,lr = self.beta,weight_decay = 0.01)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_avaialble() else 'cpu')
        self.to(self.device)
        
        
    def forwward(self,state,action) :
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        
        action_value = self.action_value(action)
        
        state_action_value = F.relu(torch.add(state_value,action_value))
        
        state_action_value = self.q(state_action_value)
        
        return state_action_value
    
    
    def save_checkpoint(self) :
        print("Sacing checkpoint : ")
        torch.save(self.state_dict(),self.checkpoint_file)
        
    def load_checkpoint(self) : 
        print("Loading checkpoint :")
        torch.load_state_dict(torch.load(self.checkpoint_file))
        
        
class ActorNetwork(nn.Module) : 
    
    def __init__(self,alpha,input_dims,fc1_dims,fc2_dims,n_actions,name,checkpoint_dir = "Models"):
        
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name= name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = self.checkpoint_dir + name + "_ddpg"
        
        
        self.fc1 = nn.Linear(*self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        self.mu = nn.Linear(self.fc2_dims,self.n_actions)
        
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1,f1)
        self.fc1.bias.uniform_(-f1,f1)
        
        
        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2,f2)
        self.fc2.bias.uniform_(-f2,f2)
        
        f3 = 0.003
        self.mu.weight.data.unifrom_(-f3,f3)
        self.mu.bias.uniform_(-f3,f3)
        
        self.optimizer = optim.Adam(self.parameters(),lr = self.alpha)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_avaialble() else 'cpu')
        
        self.to(self.device)
        
        
    def forward(self,state) : 
        
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = torch.tanh(self.mu(x))
        
        return x
    
    
    def save_checkpoint(self) :
        print("Saving checkpoint : ")
        torch.save(self.state_dict(),self.checkpoint_file)
        
    def load_checkpoint(self) : 
        print("Loading checkpoint :")
        torch.load_state_dict(torch.load(self.checkpoint_file))
        
        
        
        
        
        
    