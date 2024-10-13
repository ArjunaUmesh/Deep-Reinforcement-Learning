import numpy as np

class ReplayBuffer :
    
    def __init__(self,max_size,input_shape,n_actions) :
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size,*input_shape))
        self.next_state_memory = np.zeros((self.mem_size,*input_shape))
        self.action_memory = np.zeros((self.mem_size,n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size,dtype = np.bool_)
        
    def store_transitions(self,state,action,reward,next_state,terminal):
        index = self.mem_counter % self.mem_size
        
        self.state_memory[index] = state 
        self.next_state_memory[index] = next_state 
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal
         
        self.mem_counter += 1
        
    def sample_buffer(self,batch_size) : 
        
        max_mem = min(self.mem_counter,self.mem_size) 
        
        batch = np.random.choice(max_mem,batch_size)
        
        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]
        
        return states,actions,rewards,next_states,terminals
    