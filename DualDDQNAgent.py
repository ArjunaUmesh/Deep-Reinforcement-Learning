# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:27:56 2024

@author: Arjuna.Umesh
"""

import numpy as np
import torch
from DualDeepQNetwork import DualDeepQNetwork
from replay_buffer import ReplayBuffer
import gym

class DualDDQNAgent() :
    
    def __init__(self,
                 gamma,
                 epsilon,
                 lr,
                 n_actions,
                 input_dims,
                 mem_size,
                 batch_size,
                 eps_min = 0.01,
                 eps_dec = 5e-7,
                 replace = 1000,
                 algo = None,
                 env_name = None,
                 chkpt_dir = 'Models/') :
        
        self.gamma = gamma
        self.epsilon =epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_count = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        
        
        self.q_eval = DualDeepQNetwork(self.lr,self.n_actions,
                                   input_dims = self.input_dims,
                                   name="["+self.env_name+"]__["+self.algo+"]__[q_online]",
                                   chkpt_dir = self.chkpt_dir)
        
        self.q_next = DualDeepQNetwork(self.lr,self.n_actions,
                                   input_dims = self.input_dims,
                                   name="["+self.env_name+"]__["+self.algo+"]__[q_target]",
                                   chkpt_dir = self.chkpt_dir)
        
    def choose_action(self,observation) :
        
        random_number = np.random.random()
        
        if(random_number > self.epsilon) :
            state = torch.tensor([observation],dtype=torch.float).to(self.q_eval.device)
            #from the feedforward we choose only the advantages
            #actions = self.q_eval.forward(state)
            values,advantages = self.q_eval.forward(state)
            action = torch.argmax(advantages).item()
        else :
            action = np.random.choice(self.action_space)
            
        return action
    
    
    def store_transition(self,state,action,reward,next_state,terminal) :
        self.memory.store_transition(state, action, reward, next_state, terminal)
        
    def sample_memory(self) :
        states,actions,rewards,next_states,terminals = self.memory.sample_buffer(self.batch_size)
        
        states = torch.tensor(states).to(self.q_eval.device)
        rewards = torch.tensor(rewards).to(self.q_eval.device)
        terminals = torch.tensor(terminals).to(self.q_eval.device)
        actions = torch.tensor(actions).to(self.q_eval.device)
        next_states = torch.tensor(next_states).to(self.q_eval.device)
        
        return states,actions,rewards,next_states,terminals
    
    def replace_target_network(self) :
        if self.learn_step_counter % self.replace_target_count== 0 :
            print("Replacing target network parameters : ")
            self.q_next.load_state_dict(self.q_eval.state_dict())
            
    def decrement_epsilon(self) :
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        
    def save_models(self) :
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()
    
    def load_models(self) :
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint() 
    
    def learn(self) :
        if self.memory.mem_count < self.batch_size : 
            
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        
        states,actions,rewards,next_states,terminals = self.sample_memory()
        
        indices = np.arange(self.batch_size)
       
       
        state_values,state_advantages = self.q_eval.forward(states)
        #Value and advantage for the new states as per the target network
        next_state_values,next_state_advantages = self.q_next.forward(next_states)
        
        next_state_values_eval,next_state_advantages_eval = self.q_eval.forward(next_states)
        
       
        #COMBINATION OF THE VALUES AND ADVANTAGES TO COM[PUTE Q : 
        q_pred = torch.add(state_values,(state_advantages - state_advantages.mean(dim=1,keepdim=True)))[indices,actions]
        #as per the target network
        q_next = torch.add(next_state_values,(next_state_advantages - next_state_advantages.mean(dim=1,keepdim=True)))
        q_eval = torch.add(next_state_values_eval,(next_state_advantages_eval - next_state_advantages_eval.mean(dim=1,keepdim=True)))
        
        
        max_actions = torch.argmax(q_eval,dim=1)
        
        q_next[terminals] = 0.0
        q_target = rewards + self.gamma*q_next[indices,max_actions]
         
        loss = self.q_eval.loss(q_pred,q_target).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        
        self.learn_step_counter += 1
        
        self.decrement_epsilon()
