# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 21:31:03 2020

@author: 93585
"""

import random
task=[]
arrival=[]
bad_list=[]
total_tasks = 100
Steps=100
AVG=0
start_time=35
end_time=40
def produce_tasks():
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
        #change the domain of execution time
        et= random.randint(start_time,end_time)
        task[i].append(et)
        task[i].append(storage)
        profit = random.randint(1,10)
        task[i].append(profit)
        task[i].append(arrival[i])
        task[i].append(arrival[i]+et)
    print(task)
    print('Total tasks number:', len(task))



def attack_ratio():
    count=0
    #print(len(task))
    #print(len(bad_list))
    #print(total_tasks-1)
    for i in range(total_tasks-1):
        t=len(bad_list)
        if(task[i][5]>task[i+1][4]):
            if(count>0 and t>=1):
                if(task[i]==bad_list[t-1]):
                    bad_list.append(task[i+1])
            else:
                bad_list.append(task[i])
                bad_list.append(task[i+1])
        count=count+1
    #print('Attacked tasks',bad_list)
    print('Ratio:',len(bad_list)/total_tasks)
    return len(bad_list)/total_tasks



for m in range(Steps):
    task.clear()
    arrival.clear()
    bad_list.clear()
    produce_tasks()
    q =attack_ratio()
    AVG=AVG+q
    print('AVG',AVG/Steps)