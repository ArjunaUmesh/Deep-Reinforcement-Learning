import gymnasium as gym
import numpy as np
from DDPGAgent import DDPGAgent
import matplotlib.pyplot as plt
from datetime import date

def plot_learning_curve(scores,x,filename ) :
    running_avg = np.zeros(len(scores))
    
    for i in range(len(running_avg)) : 
        running_avg[i] = np.mean(scores[-100:])
    plt.plot(x,scores)#running_avg)
    plt.title("Scores")
    
    plt.savefig(filename)
        

if __name__ == "__main__" :

    load_models = False
    if(not load_models) : 
        # env = gym.make("Walker2d-v4")
        # env.reset()

        env = gym.make("Walker2d-v4",render_mode="human")
        env.reset()
        env.render()
    else : 
        env = gym.make("Walker2d-v4",render_mode="human")
        env.reset()
        env.render()
    if(load_models):
        epsilon_init=0.0
    else :
        epsilon_init=1.0
        
    agent = DDPGAgent(alpha = 0.005,
                      beta = 0.005,
                      input_dims = (17,),#env.observation_space.shape,
                      tau = 0.001,
                      n_actions = env.action_space.shape[0],
                      gamma=0.99,
                      max_size = 20000,
                      fc1_dims = 400,
                      fc2_dims=300, 
                      batch_size=64,
                      epsilon = epsilon_init)
    
    if(load_models) : 
        agent.load_models
    
    if load_models : 
        n_games = 100
    else : 
        n_games = 1000

    
    figure_file = "Plots/Walker2d_something_something_debug" + "__" + str(date.today())+ '.png'
        
    best_score = env.reward_range[0]
    score_history = []
    steps = 0
    for i in range(n_games) :
        state,info = env.reset()
        terminal=False
        score = 0
        agent.noise.reset()
        
        while not terminal : 

            action = agent.choose_actions(state)
            next_state,reward,terminal,info,truncated = env.step(action)
            steps += 1
            
            if not load_models : 
                agent.memory.store_transitions(state,action,reward,next_state,terminal)
                agent.learn()
            score += reward
            state = next_state
            if(load_models) : 
                print("Score : ",score)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if(i%20==0):
            agent.dec_epsilon()
        
        if avg_score>best_score  :
            best_score = avg_score
            if not load_models : 
                print("Saving Models")
                agent.save_models()
  
        print("Episode : ",i,"Steps : ",steps," Score : ",score)
        print(" Average Score : ",avg_score," Best Score : ",best_score,"Epsilon : ",agent.epsilon)
        print("___________________________________________________________________________")
        
    x = [i+1 for i in range(n_games)]
    
    print(x)
    print(score_history)
    plot_learning_curve(x=x,
                        scores = score_history,
                        filename=figure_file)

    #RANDOM
    ###########################################
    # env = gym.make("Walker2d-v4",render_mode="human")
    # env.reset()
    # env.render()
    # state,info = env.reset()
    # for i in range(1000):
    #     state,info =  env.reset()
    #     terminal = False
    #     score = 0
    #     while not terminal :
    #         action = env.action_space.sample()
    #         state,reward,terminal,truncates,info = env.step(action)
    #         score += reward
    #     print("Episode ",i," Score : ",score)
    # score += reward
    # state = next_state
    # if(load_models) : 
    #     print("Score : ",score)
    ###########################################
