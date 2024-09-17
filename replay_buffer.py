# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:30:03 2024

@author: Arjuna.Umesh
"""

import numpy as np

#This class is defined for maintaing a memory buffer of all visited states to sample from 
class ReplayBuffer() :
    
    def __init__(self,max_size,input_shape,n_actions) :
        
        self.mem_size = max_size
        self.mem_count = 0
        self.state_memory = np.zeros((self.mem_size,input_shape),dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size,input_shape),dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size,dtype = np.int64)
        self.reward_memory = np.zeros(self.mem_size,dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size,dtype = np.bool_)
        
        
    def store_transition(self,state,action,reward,next_state,terminal) :
        
        #define the index of the next empty slot in the buffer or overwrite a the beginning
        index =  self.mem_count % self.mem_size
        
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index] = terminal
        self.mem_count += 1

    
    def sample_buffer(self,batch_size) :
        
        max_mem = min(self.mem_count,self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace = False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        terminals = self.terminal_memory[batch]
        
        return states,actions,rewards,next_states,terminals
        