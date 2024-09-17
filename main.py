import numpy as np
from DualDDQNAgent import DualDDQNAgent
from util import make_env,plot_learning_curve
import gym
from datetime import date
import time

if __name__ == '__main__' :
    load_checkpoint = True
    #env = make_env(env_name = "PongNoFrameskip-v4")
    if load_checkpoint :
        env = gym.make("CartPole-v1",render_mode = "human")
        env.reset()
        env.render()
    else : 
        env = gym.make("CartPole-v1")
    best_score = -np.inf
    
    if(load_checkpoint):
        epsilon_init=0.0
    else :
        epsilon_init=1.0
    if load_checkpoint : 
        n_games = 500
    else : 
        n_games = 5000
    agent = DualDDQNAgent(gamma = 0.99,
                     epsilon = epsilon_init, 
                     lr = 0.01, 
                     n_actions = env.action_space.n, 
                     input_dims = 4,#env.observation_space.shape,
                     mem_size = 5000,
                     batch_size = 32,
                     eps_min = 0.1,
                     eps_dec = 1e-5,
                     replace = 1000,
                     chkpt_dir='Models/',
                     algo = 'DualDQNAgent',
                     env_name = 'CartPole-v1')
    
    if load_checkpoint :
        agent.load_models()
        
    fname = "[__" + agent.algo + "]__[" + agent.env_name +"__]" + "__" + str(date.today())
 
    figure_file = 'Plots/'+fname +".png"
    
    n_steps = 0
    scores,eps_history,steps_array= [], [] , []
    
    for i in range(n_games) : 
        terminal = False
        score = 0
        
        observation,info = env.reset()
        
        
        while not terminal :
            action = agent.choose_action(observation)
            next_state, reward, terminal, _, _ = env.step(action)
            score += reward
            
            if not load_checkpoint :
                agent.store_transition(state = observation, 
                                        action = action,
                                        reward = reward,
                                        next_state = next_state,
                                        terminal = terminal)
                agent.learn()
                
            observation = next_state
            n_steps += 1
        
            print(score)
    
        scores.append(score)
        steps_array.append(n_steps)
        
        avg_score = np.mean(scores[-100:])
        
        
        print("Episode : ",i,
              "Score : ",score,
              "Average Score : ",avg_score,
              "Best Score : ",best_score,
              "Epsilon : ",agent.epsilon,
              "Number of steps : ",n_steps)
        
        if(avg_score>best_score) : 
            if not load_checkpoint :
                agent.save_models()
                pass
            best_score = avg_score
            
        eps_history.append(agent.epsilon)
        
    plot_learning_curve(x = steps_array,
                        scores = scores,
                        epsilon = eps_history,
                        filename = figure_file)
    
    
    
                
    
    