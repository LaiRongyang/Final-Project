# -*- coding: utf-8 -*-


# import
import numpy as np
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from DRL_environment import Environment
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import time




from collections import namedtuple
#
Tr = namedtuple('tr',('name_a','value_b'))
#Tr_object=Tr('名称为A',100)
##print(Tr_object)
##print(Tr_object.value_b)
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


GAMMA = 0.99 
MAX_TASKS = 100
NUM_EPISODES = 500
eval_profit_list = []
avg_profit_list=[]
eval_tasks_list=[]
avg_tasks_list=[]
eval_ap_list=[]
avg_ap_list=[]

# DRAW CURVES
# Pofit
def plot_profit(profit,avg):
    host = host_subplot(111)  
    plt.subplots_adjust(right=0.8) 

    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total profit")
       
    # plot curves
    p1, = host.plot(range(len(profit)), profit, label="Total Profit")
    p2, = host.plot(range(len(avg)), avg, label="Running Average Total Profit")
    host.legend(loc=1)
 
    # set color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
   
    host.set_xlim([0, NUM_EPISODES])
    host.set_ylim([0,500])

 
    plt.draw()
    plt.show()
    
    
def plot_tasks(tasks,avg1):
    host = host_subplot(111)  
    plt.subplots_adjust(right=0.8)  

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total accepted tasks")
 

    # plot curves
    p1, = host.plot(range(len(tasks)), tasks, label="Total Accepted Tasks")
    p2, = host.plot(range(len(avg1)), avg1, label="Running Average Total Accepted Tasks")
 

    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0,NUM_EPISODES])
    host.set_ylim([0,MAX_TASKS])
 
    plt.draw()
    plt.show()
    
def plot_ap(ap,avg2):
    host = host_subplot(111) 
    plt.subplots_adjust(right=0.8)  

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Average Profit")
 
    # plot curves
    p1, = host.plot(range(len(ap)), ap, label="Average Profit")
    p2, = host.plot(range(len(avg2)), avg2, label="Running Average Profit")
 
    # set location 
    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    host.set_xlim([0,NUM_EPISODES])
    host.set_ylim([0,9])
 
    plt.draw()
    plt.show()

#DQN framework , in this framework I change the network parameters

#define class to store memory


class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  
        self.memory = []  
        self.index = 0  

    def push(self, state, action, state_next, reward):

        if len(self.memory) < self.capacity:
            self.memory.append(None)  
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  

    def sample(self, batch_size):
        
        return random.sample(self.memory, batch_size)

    def __len__(self):
     
        return len(self.memory)


#  50_888 BATCH_SIZE = 80  200
BATCH_SIZE = 250
CAPACITY = 1000


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  #get the action numbers of the environment 2 (0,1)

        # create object to store memory
        self.memory = ReplayMemory(CAPACITY)

        # build network
        self.model = nn.Sequential()
        self.model.add_module('fc1',nn.Linear(num_states,60))
        self.model.add_module('relu1',nn.ReLU())
        self.model.add_module('fc2',nn.Linear(60,60))
        self.model.add_module('relu2',nn.ReLU())
        self.model.add_module('fc3',nn.Linear(60, num_actions))

        print(self.model)  

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.00068)#change the learning rate
         
        
    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        

        batch = Transition(*zip(*transitions))
        

        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
    

        self.model.eval()

        state_action_values = self.model(state_batch).gather(1, action_batch)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,batch.next_state))) 
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = reward_batch + GAMMA * next_state_values
        
    
        self.model.train()
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()  
        self.optimizer.step()  
        
    def decide_action(self,state,episode):
        epslion = 0.5*(1/(episode+1))
        if epslion<= np.random.uniform(0,1):
            self.model.eval()
            with torch.no_grad():
                action=self.model(state).max(1)[1].view(1,1)
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])
        return action

#
class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)


# Execute the environment


class Environment_d:

    def __init__(self):
        self.env = Environment()  # connect the satellite problem
        self.num_states = 4  # set the number of state and action
        self.num_actions = 2 
        # build agent
        self.agent = Agent(self.num_states,self.num_actions)

    def run(self):
        
        T = 0
        for episode in range(NUM_EPISODES):
            
            start = time.time()
            total_reward = 0
            accepted_tasks=0
            #reset or initilize the environment
            self.env.reset()
            observation = self.env.observe()  

            state = observation  
            state = torch.from_numpy(state).type(
                torch.FloatTensor)  
            state = torch.unsqueeze(state, 0)   
            for step in range(MAX_TASKS-1):
                action = self.agent.get_action(state, episode)  
                
                #interact with the environment
                observation_next, reward, done = self.env.update_env(
                    action.item()) 

                if done: 
                    state_next = None
                    break
                else: 
                    reward_t = torch.FloatTensor([reward]) 
                    
                    state_next = observation_next 
                    
                    state_next = torch.from_numpy(state_next).type(
                        torch.FloatTensor)  
                    state_next = torch.unsqueeze(state_next, 0)  
                    
                self.agent.memorize(state, action, state_next, reward_t)

                # Update
                self.agent.update_q_function()
                state = state_next
            end = time.time()
            times = end - start
            T = T + times
            print(" ")
            print(self.env.task)
            print(self.env.time_windows)
            state_next = None
            #record data
            total_reward = self.env.total_profit
            accepted_tasks= self.env.atasks
            eval_profit_list.append(total_reward)
            avg = sum(eval_profit_list)/len(eval_profit_list)
            avg_profit_list.append(avg)        
            eval_tasks_list.append(accepted_tasks)
            avg_1 = sum(eval_tasks_list)/len(eval_tasks_list)
            avg_tasks_list.append(avg_1)
            avg_profit = total_reward/accepted_tasks
            eval_ap_list.append(avg_profit)
            avg_2 = sum(eval_ap_list)/len(eval_ap_list)
            avg_ap_list.append(avg_2)
            print('%d Episode: Accepted tasks numbers：%d Total reward: %d'%(episode,accepted_tasks,total_reward))
        #print and draw
        print('Aveage Total Profit', avg)
        print('Aveage accepted tasks',avg_1)
        print('Aveage Profit', avg_2)
        print('Average Response Time', T / episode)
        plot_profit(eval_profit_list,avg_profit_list)
        plot_tasks(eval_tasks_list,avg_tasks_list)
        plot_ap(eval_ap_list,avg_ap_list)
                    
cartpole_env = Environment_d()
cartpole_env.run()