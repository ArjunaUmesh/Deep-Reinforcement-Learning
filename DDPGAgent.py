import torch
from Actor_Critic_Network import ActorNetwork,CriticNetwork
from Ornstein_Uhlenbeck import OUActionNoise
from Replay_Buffer import ReplayBuffer
import numpy as np
import torch.nn.functional as F
import gymnasium as gym


class DDPGAgent():
    def __init__(self,alpha,beta,input_dims,tau,n_actions,gamma=0.99,max_size = 20000,fc1_dims = 400,fc2_dims=300, batch_size=64,epsilon = 1.0,) :
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        
        self.memory = ReplayBuffer(max_size,input_dims,n_actions)
        
        self.noise = OUActionNoise(mu = np.zeros(n_actions))

        self.actor = ActorNetwork(alpha,input_dims,fc1_dims,fc2_dims,n_actions = n_actions,name = "Actor_Network") 
        self.critic = CriticNetwork(beta,input_dims,fc1_dims,fc2_dims,n_actions = n_actions,name = "Critic_Network")
        self.target_actor = ActorNetwork(alpha,input_dims,fc1_dims,fc2_dims,n_actions = n_actions,name = "Target_Actor_Network")
        self.target_critic = CriticNetwork(beta,input_dims,fc1_dims,fc2_dims,n_actions = n_actions,name = "Target_Critic_Network")
        
        self.epsilon = epsilon
        
        self.update_network_parameters(tau=1)
        
    def choose_actions(self,state) :
        self.actor.eval()
        state = torch.tensor([state],dtype = torch.float32).to(self.actor.device)
        
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + self.epsilon*torch.tensor(self.noise(),dtype = torch.float32).to(self.actor.device)
        
        self.actor.train()
        
        return mu_prime.cpu().detach().numpy()[0]
    
    def dec_epsilon(self) : 
        self.epsilon = self.epsilon*0.9 if self.epsilon > 0.1 else self.epsilon
    
    def save_models(self) : 
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
        
    def load_models(self) :
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        
    def learn(self) : 
        
        if(self.memory.mem_counter < self.batch_size) : 
            #print("returning")
            return
            ###TEST 
            #env = gym.make("Walker2d-v4")
            #state,info = env.reset()
            #for i in range(self.batch_size) : 
                #action = self.choose_actions(state)
                #print(action)
                #next_state,reward,terminal,truncated,info = env.step(action)
                #self.memory.store_transitions(state, action, reward, next_state, terminal)
                #state = next_state
            #print(self.memory.mem_counter)
        
        states,actions,rewards,next_states,terminals = self.memory.sample_buffer(self.batch_size)
        
        states = torch.tensor([states],dtype = torch.float32).to(self.actor.device)
        next_states = torch.tensor([next_states],dtype = torch.float32).to(self.actor.device)
        actions = torch.tensor([actions],dtype = torch.float32).to(self.actor.device)
        rewards = torch.tensor([rewards],dtype = torch.float32).to(self.actor.device)
        terminals = torch.tensor([terminals]).to(self.actor.device)
        
        target_actions = self.target_actor.forward(next_states)
        next_states_critic_values = self.target_critic.forward(next_states,target_actions)
        current_states_critic_values= self.critic.forward(states,actions)        
        #TEST
        #print(current_states_critic_values)
        
        next_states_critic_values[terminals] = 0
        next_states_critic_values = next_states_critic_values.view(-1)
        
        target = rewards + self.gamma*(next_states_critic_values)
        target = target.view(self.batch_size,1)
        
        #TEST
        #print(target.shape)
        self.critic.optimizer.zero_grad()
        #TEST
        #print(current_states_critic_values[0].shape)
        critic_loss = F.mse_loss(target,current_states_critic_values[0])
        critic_loss.backward()
        self.critic.optimizer.step()
        
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states,self.actor.forward(states))
        
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        
        self.update_network_parameters()
        
        
        
    def update_network_parameters(self,tau=None) : 
        if tau is None:
            tau = self.tau
            
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)
        
        for name in actor_state_dict : 
            target_actor_state_dict[name] = tau*(actor_state_dict[name]).clone() + (1-tau)*(target_actor_state_dict[name]).clone()
        
        for name in critic_state_dict : 
            target_critic_state_dict[name] = tau*(critic_state_dict[name]).clone() + (1-tau)*(target_critic_state_dict[name]).clone()
        
        
        self.target_critic.load_state_dict(target_critic_state_dict)
        self.target_actor.load_state_dict(target_actor_state_dict)
        
        
###TEST
# agent = DDPGAgent(alpha = 0.001,
#                   beta = 0.01,
#                   input_dims = (17,),
#                   tau = 0.001,
#                   n_actions = 6)
        
# env = gym.make("Walker2d-v4")
# state,info = env.reset()

# #print(state)

# action = agent.choose_actions(state)
# agent.learn()




