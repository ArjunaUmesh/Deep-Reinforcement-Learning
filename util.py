import numpy as np
import matplotlib.pyplot as plt
import gym
import collections
from gymnasium.wrappers import GrayScaleObservation
import cv2
        
class RepeatActionAndMaxFrame(gym.Wrapper) :
    
    def __init__(self,env = None,repeat = 4, clip_reward = False,num_ops = 0,fire_first = False) : 
        
        super(RepeatActionAndMaxFrame,self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros((2,)+self.shape) #To keep track of the last 2 frames and choose the one with max values in order to resolve issues due to flicker
        self.clip_reward = clip_reward
        self.num_ops = num_ops
        self.fire_first = fire_first
        
    #modifiying/Overloading the step function to take k=4 steps with the same action and to retian the state with max values across two frames
    def step(self,action) : 
        
        total_reward = 0.0
        terminal = False
        for i in range(self.repeat) :
            next_state,reward,terminal,truncated,info = self.env.step(action)
            if self.clip_reward : 
                reward = np.clip(np.array([reward],-1,1))[0]
            total_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = next_state
            if terminal : 
                break
        #Max fram to combat flickering frames 
        max_frame = np.maximum(self.frame_buffer[0],self.frame_buffer[1])
        truncated = False
        return max_frame, total_reward,terminal,truncated,info
    
    
    def reset(self) : 
        
        state,info = self.env.reset()
        num_ops = np.random.randint(self.num_ops)+1 if self.num_ops>0 else 0 
        #Number of steps to skip : 
        for i in range(num_ops) :
            _,_,terminal,_ = self.env.step(0)
            if terminal : 
                self.env.reset()
        if self.fire_first :
            assert self.env.unwrapped.get_Action_meanings()[1] == 'FIRE'
            state,_,_,_,_ = self.env.step(1)
        
        self.frame_buffer = np.zeros((2,)+self.shape)
        self.frame_buffer[0] = state
        
        return state,info
    
#Preprocess OpenAI gym Atari Screen images

class PreprocessFrame(gym.ObservationWrapper):
    
    def __init__(self,shape,env=None) : 
        super(PreprocessFrame,self).__init__(env)
        #channel to be at the beginning in pytorch as opposed to the end in gym
        self.shape = (shape[2],shape[0],shape[1]) #(1,84,84)#(shape[2],shape[0],shape[1])
        #Observation value spaces to be wihtin 0 and 1  - greyscale within 0 and 1
        self.observation_space = gym.spaces.Box(low=0.0,high=1.0,
                                                shape = self.shape,dtype=np.float32)
        
    
            
        
    def observation(self,state)  :
        
        #convert the sate to gray scale 
        new_frame = cv2.cvtColor(state,cv2.COLOR_RGB2GRAY)
        # resize : 
        resized_screen = cv2.resize(new_frame,self.shape[1:],
                                    interpolation = cv2.INTER_AREA)
        #reshaping and conversion to array as well as swapping axes
        new_state = np.array(resized_screen,dtype=np.uint8).reshape(self.shape)
        
        #rescaling by division by 255 
        new_state = new_state/255.0
        
        return new_state
    

    
class StackFrames(gym.ObservationWrapper) : 
    
    def __init__(self,env,repeat) : 
        super(StackFrames,self).__init__(env)
        self.observation_space = gym.spaces.Box(
                                                env.observation_space.low.repeat(repeat,axis=0),
                                                env.observation_space.high.repeat(repeat,axis=0),
                                                dtype = np.float32
                                                )
        self.stack = collections.deque(maxlen = repeat)
        
        
    def reset(self) :
        self.stack.clear()
        #print("Reset : ",self.env.reset())
        state,info = self.env.reset()
        
        for i in range(self.stack.maxlen) :
            self.stack.append(state)
            
        return np.array(self.stack).reshape(self.observation_space.low.shape),info
    
    def observation(self,state) : 
        
        self.stack.append(state)
        
        return np.array(self.stack).reshape(self.observation_space.low.shape)
        
    
#Make environement to tie everything together 
def make_env(env_name,shape = (84,84,1),repeat = 4, clip_rewards = False, num_ops = 0, fire_first = False) : 
    
    env = gym.make(env_name)
    #Calling the constructors of each of the defined classes for [RepeatActionAndMaxFrame, PreprocessFrame, StackFrames]

    env = PreprocessFrame(shape,env)
    env = RepeatActionAndMaxFrame(env,repeat,clip_rewards,num_ops,fire_first)
    env = StackFrames(env,repeat)
    
    return env


def plot_learning_curve(x,scores,epsilon,filename) : 
    fig = plt.figure()
    
    ax = fig.add_subplot(111,label='1')
    ax2 = fig.add_subplot(111,label = '2',frame_on=False)
    
    ax.plot(x,epsilon,color="C0")
    ax.set_xlabel("Training iterations",color="C0")
    ax.set_ylabel("Epsilon",color="C0")
    ax.tick_params(axis='x',color="C0")
    ax.tick_params(axis='y',color="C0")
    
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N) : 
        running_avg[t] = np.mean(scores[max(0,t-100):t+1])
        
    ax2.scatter(x,running_avg,color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Score",color="C1")
    ax2.yaxis.set_label_position('right')
    ax.tick_params(axis='y',color="C1")
    
    plt.savefig(filename)
    
