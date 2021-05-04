# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:05:33 2021

@author: 93585
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import random
import time



eval_profit_list = []
avg_profit_list=[]
eval_tasks_list=[]
avg_tasks_list=[]
eval_ap_list=[]
avg_ap_list=[]


task=[]
arrival=[]
test=[]
tasks=[]
code=[]
total_tasks = 100
Steps=100
AVG=0
atasks = 0

def plot_profit(profit,avg):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total reward")
 
    plt.title('Random Pick')
  

    
    # plot curves
    p1, = host.plot(range(len(profit)), profit, label="Total Profit")
    p2, = host.plot(range(len(avg)), avg, label="Average profit")
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0, 100])
    host.set_ylim([0,400])
    # par1.set_ylim([-0.1, 1.1])
 
    plt.draw()
    plt.show()
    
    
def plot_tasks(tasks,avg1):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total accepted tasks")
 
    plt.title('Random Pick')
    # plot curves
    p1, = host.plot(range(len(tasks)), tasks, label="Total Accepetd Tasks")
    p2, = host.plot(range(len(avg1)), avg1, label="Average Accepetd tasks")
 
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0,100])
    host.set_ylim([0,60])
    # par1.set_ylim([-0.1, 1.1])
 
    plt.draw()
    plt.show()
    
def plot_ap(ap,avg2):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Average Profit")
 
    plt.title('Random Pick')
    # plot curves
    p1, = host.plot(range(len(ap)), ap, label="Average Profit")
    p2, = host.plot(range(len(avg2)), avg2, label="Running average profit")
 
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0,100])
    host.set_ylim([0,9])
    # par1.set_ylim([-0.1, 1.1])
 
    plt.draw()
    plt.show()

#produce tasks
def produce_tasks():
    task.clear()
    arrival.clear()
    for i in range(total_tasks):
        #a= random.randint(0,888)
        a= random.randint(0,1200)
        arrival.append(a)
    arrival.sort()
    for i in range(total_tasks):
        task.append([])
    storage = 5
    for i in range(total_tasks):
        task[i].append(arrival[i])
        et= random.randint(20,40)
        task[i].append(et)
        task[i].append(storage)
        profit = random.randint(1,10)
        task[i].append(profit)
        task[i].append(arrival[i])
        task[i].append(arrival[i]+et)
    return task
#set the probability to select the task
def Random_Pick():
    value_list = [0,1]
    probability = [0.5,0.5]
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for action, action_probability in zip(value_list, probability):
        cumulative_probability += action_probability
        if x < cumulative_probability:
            break
    return action

def RP():
   
    time_windows=[]
    max_time = 1200
    max_storage = 200
    total_tasks = 100
    T = 0
    
    
    for i in range(Steps):
        start = time.time()
        counts = 0
        first = 0
        total_profit = 0
        total_storage = 0
        time_windows.clear()
        t = produce_tasks()
        print(t)
        print(t[1][5])

        none1 = True
        while none1:
            action = Random_Pick()
            if(action==1):
                counts= counts+1
            else:
                if(first==0):        
                    time_windows = time_windows +[counts]
                    t[counts][4]=t[counts][0]
                    t[counts][5]=t[counts][0]+t[counts][1]
                    total_storage = total_storage+t[counts][2]
                    total_profit = total_profit+t[counts][3]
                    first=first +1
                else:
                    n=len(time_windows)
                    a=time_windows[n-1]
                    #print('a', a)
                    #print('counts',counts)
                
                    
                    q = t[a][5]+10
                    if(q< t[counts][0] and t[counts][5]<max_time):
                        time_windows = time_windows +[counts]
                        t[counts][4]=t[counts][0]
                        t[counts][5]=t[counts][0]+t[counts][1]
                        total_storage = total_storage+t[counts][2]
                        total_profit = total_profit+t[counts][3]
                        if(total_storage>max_storage):
                            none1=False
                        counts = counts+1
                    else:
                        counts = counts+1

            if(counts>total_tasks-1):
                none1=False
        end = time.time()
        times = end -start
        T = T+times
        print(time_windows)
        print('epsoide:',i)
        total_reward = total_profit
        atasks = len(time_windows)
        accepted_tasks=atasks
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
    print('Average Total Profit',avg)
    print('Average Count', avg_1)
    print('Average Profit',avg_2)
    print('Average Response Time', T/Steps)
    plot_profit(eval_profit_list,avg_profit_list)
    plot_tasks(eval_tasks_list,avg_tasks_list)
    plot_ap(eval_ap_list,avg_ap_list)
        
    
RP()
    
    
