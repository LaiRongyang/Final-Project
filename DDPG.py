# -*- coding: utf-8 -*-


import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from DRL_environment import Environment
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import time
#hyper parameters

MAX_EPISODES = 2000
MAX_TASKS = 100
LR_A = 0.001                # learning rate for actor
LR_C = 0.002                # learning rate for critic
GAMMA = 0.9                 # reward discount
TAU = 0.3                   # soft replacement
MEMORY_CAPACITY = 10000     # MEMORY_CAPACITY
BATCH_SIZE = 32             # BATCH_SIZE

RENDER = False


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
    host.set_xlim([0, MAX_EPISODES])
    host.set_ylim([0,500])

 
    plt.draw()
    plt.show()
    
#Tasks
def plot_tasks(tasks,avg1):
    host = host_subplot(111)  
    plt.subplots_adjust(right=0.8)

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total accepted tasks")
 
    # plot curves
    p1, = host.plot(range(len(tasks)), tasks, label="Total Accepetd Tasks")
    p2, = host.plot(range(len(avg1)), avg1, label="Running Average Total Accepted Tasks")
 

    host.legend(loc=1)
 
    # set color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    host.set_xlim([0,MAX_EPISODES])
    host.set_ylim([0,MAX_TASKS])
 
    plt.draw()
    plt.show()
    
# Average profit
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
 
    # set color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    host.set_xlim([0,MAX_EPISODES])
    host.set_ylim([0,9])
 
    plt.draw()
    plt.show()
    
#DDPG framework
def convert_eval_to_target(e, t):
    for x in t.state_dict().keys():
        eval('t.' + x + '.data.mul_((1-TAU))')
        eval('t.' + x + '.data.add_(TAU*e.' + x + '.data)')

class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = torch.nn.Linear(30, a_dim)
        self.fc1.weight.data.normal_(0, 0.1)

    def forward(self, state_input):
        self.net = F.relu(self.fc1(state_input))
        self.a = F.tanh(self.fc2(self.net))
        return self.a

class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.w1_s = nn.Linear(s_dim, 30)
        self.w1_s.weight.data.normal_(0, 0.1)
        self.w1_a = nn.Linear(a_dim, 30)
        self.w1_a.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30,1)
        self.out.weight.data.normal_(0, 0.1)  # initialization
        
    def forward(self, s, a):
        net = F.relu((self.w1_s(s) + self.w1_a(a)))
        return self.out(net)

class DDPG(object):
    def __init__(self, a_dim, s_dim):
        self.memory = torch.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=torch.float32)
        self.pointer = 0
        self.a_dim, self.s_dim= a_dim, s_dim
        
        self.actor_eval = Actor()
        self.actor_target = Actor()
        
        self.critic_eval = Critic()
        self.critic_target = Critic()
        
        self.ae_optimizer = torch.optim.Adam(params=self.actor_eval.parameters(), lr=0.001)
        self.ce_optimizer = torch.optim.Adam(params=self.critic_eval.parameters(), lr=0.001)
        
        self.mse = nn.MSELoss()
        
    def return_c_loss(self, S, a, R, S_):
        a_ = self.actor_target(S_).detach()
        q = self.critic_eval(S, a)
        q_ = self.critic_target(S_, a_).detach()
        q_target = R + GAMMA * q_
        td_error = self.mse(q_target, q)
        return td_error
        
    def return_a_loss(self, S):
        a = self.actor_eval(S)
        q = self.critic_eval(S, a)
        a_loss = q.mean()
        return a_loss
        
    def choose_action(self, s):
        return self.actor_eval(s[np.newaxis, :])[0]

    def learn(self):
        # soft target replacement
        convert_eval_to_target(self.actor_eval, self.actor_target)
        convert_eval_to_target(self.critic_eval, self.critic_target)
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        a_loss = self.return_a_loss(bs)
        c_loss = self.return_c_loss(bs, ba, br, bs_)
        
        self.ae_optimizer.zero_grad()
        a_loss.backward()
        self.ae_optimizer.step()
        
        self.ce_optimizer.zero_grad()
        c_loss.backward()
        self.ce_optimizer.step()

    def store_transition(self, s, a, r, s_):
        transition = torch.FloatTensor(np.hstack((s, a, [r], s_)))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

# training
#Combine with the online scheduling of image satellite  problem
env = Environment()
# IN this project the state is 4 dimension and the action 1 dimension
s_dim = 4
a_dim = 1


ddpg = DDPG(a_dim, s_dim)
print(s_dim)
print(a_dim)

var = 3  # control exploration
T = 0
for episode in range(MAX_EPISODES):
    s = env.reset()
    start = time.time()
    for j in range(MAX_TASKS):

        # get the first state and add exploration
        s = env.observe()
        a = ddpg.choose_action(torch.FloatTensor(s))

        a = np.clip(np.random.normal(a.detach().numpy(), var), -1, 1)
        
        a=int((a+2)/4+0.5)
        #update from the environment
        s_, r, done = env.update_env(a)
        ddpg.store_transition(s, a, r, s_)
        # learn
        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        
        if j == MAX_TASKS-2:
            print('Episode:', episode, 'Reward:', env.total_profit, 'Accepted Tasks:' ,env.atasks )
            break
    end = time.time()
    times = end - start
    T = T + times
    total_reward = env.total_profit
    accepted_tasks= env.atasks
    #record data
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
#Print the results
print('Average Total Profit', avg)
print('Average accepted tasks',avg_1)
print('Average Profit', avg_2)
print('Average Time',T/MAX_EPISODES)
print('Average Response Time', T / episode)
plot_profit(eval_profit_list,avg_profit_list)
plot_tasks(eval_tasks_list,avg_tasks_list)
plot_ap(eval_ap_list,avg_ap_list)

