# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import random
import time

#Draw
def plot_profit(profit,avg):
    host = host_subplot(111) 
    plt.subplots_adjust(right=0.8)  

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total reward")
 
    plt.title('GA')
  

    
    # plot curves
    p1, = host.plot(range(len(profit)), profit, label="Total Profit")
    p2, = host.plot(range(len(avg)), avg, label="Average profit")

    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    host.set_xlim([0, 100])
    host.set_ylim([0,400])
 
    plt.draw()
    plt.show()
    
    
def plot_tasks(tasks,avg1):
    host = host_subplot(111) 
    plt.subplots_adjust(right=0.8)  

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total accepted tasks")
 
    plt.title('GA')
    # plot curves
    p1, = host.plot(range(len(tasks)), tasks, label="Total Accepted Tasks")
    p2, = host.plot(range(len(avg1)), avg1, label="Average accepted tasks")
 
    host.legend(loc=1)
 
    # set color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    host.set_xlim([0,100])
    host.set_ylim([0,60])
 
    plt.draw()
    plt.show()
    
def plot_ap(ap,avg2):
    host = host_subplot(111) 
    plt.subplots_adjust(right=0.8)  

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Average Profit")
 
    plt.title('GA')
    # plot curves
    p1, = host.plot(range(len(ap)), ap, label="Average Profit")
    p2, = host.plot(range(len(avg2)), avg2, label="Running average profit")
 
    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    host.set_xlim([0,100])
    host.set_ylim([0,9])
 
    plt.draw()
    plt.show()


task=[]
arrival=[]
code=[]
total_tasks = 100


def produce_tasks():
    task.clear()
    arrival.clear()
    for i in range(total_tasks):
        a= random.randint(0,1200)
        #a= random.randint(0,888)
        arrival.append(a)
    arrival.sort()
    for i in range(total_tasks):
        task.append([])
    storage = 5
    for i in range(total_tasks):
        task[i].append(arrival[i])
        et= random.randint(10,15)
        task[i].append(et)
        task[i].append(storage)
        profit = random.randint(1,10)
        task[i].append(profit)
        task[i].append(arrival[i])
        task[i].append(arrival[i]+et)
    return task


def init(popsize, n):
    population = []
    for i in range(popsize):
        pop = ''
        for j in range(n):
            pop = pop + str(np.random.randint(0, 2))
        population.append(pop)
    return population

#changes has made according to this satellite problem
def decode1(x, n, w, W, tasks, T):
    tasks_list = []  
    TS = 5
    storage = 0
    value = 0
    count = 0
    #add constraints according to the specific problem
    for i in range(n):

        if(x[i] == '1'):
            if (count == 0):
                tasks_list = tasks_list + [i]
                storage  = storage + w
                value = value + tasks[i][3]
                count = count+1
            else:
                length = len(tasks_list)     
                j = tasks_list[length-1]
                y = tasks[j][5]+2*TS
                if(tasks[i][0]>y):
                    if(storage+w>W or tasks[i][5]>T):
                        break
                    else:
                        storage  = storage + w
                        tasks_list = tasks_list + [i]
                        value = value + tasks[i][3]
    return value,tasks_list

                


def fitnessfun1(population, n, w, W, tasks, T):
    value = [] 
    lists = [] 
    for i in range(len(population)):
        [f, s] = decode1(population[i], n, w, W,tasks, T)
        value.append(f)
        lists.append(s)
    return value, lists



def roulettewheel(population, value, pop_num):
    fitness_sum = []

    value_sum = sum(value)
    fitness = [i / value_sum for i in value]
    for i in range(len(population)):  
        if i == 0:
            fitness_sum.append(fitness[i])
        else:
            fitness_sum.append(fitness_sum[i - 1] + fitness[i])
    population_new = []
    for j in range(pop_num):  
        r = np.random.uniform(0, 1)
        for i in range(len(fitness_sum)):  
            if i == 0:
                if 0 <= r <= fitness_sum[i]:
                    population_new.append(population[i])
            else:
                if fitness_sum[i - 1] <= r <= fitness_sum[i]:
                    population_new.append(population[i])
    return population_new


def crossover(population_new, pc, ncross):
    a = int(len(population_new) / 2)
    parents_one = population_new[:a]
    parents_two = population_new[a:]
    np.random.shuffle(parents_one)
    np.random.shuffle(parents_two)
    offspring = []
    for i in range(a):
        r = np.random.uniform(0, 1)
        if r <= pc:
            point1 = np.random.randint(0, (len(parents_one[i]) - 1))
            point2 = np.random.randint(point1, len(parents_one[i]))
            off_one = parents_one[i][:point1] + parents_two[i][point1:point2] + parents_one[i][point2:]
            off_two = parents_two[i][:point1] + parents_one[i][point1:point2] + parents_two[i][point2:]
            ncross = ncross + 1
        else:
            off_one = parents_one[i]
            off_two = parents_two[i]
        offspring.append(off_one)
        offspring.append(off_two)
    return offspring



def mutation1(offspring, pm, nmut):
    for i in range(len(offspring)):
        r = np.random.uniform(0, 1)
        if r <= pm:
            point = np.random.randint(0, len(offspring[i]))
            if point == 0:
                if offspring[i][point] == '1':
                    offspring[i] = '0' + offspring[i][1:]
                else:
                    offspring[i] = '1' + offspring[i][1:]
            else:
                if offspring[i][point] == '1':
                    offspring[i] = offspring[i][:(point - 1)] + '0' + offspring[i][point:]
                else:
                    offspring[i] = offspring[i][:(point - 1)] + '1' + offspring[i][point:]
            nmut = nmut + 1
    return offspring


def mutation2(offspring, pm, nmut):
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            r = np.random.uniform(0, 1)
            if r <= pm:
                if j == 0:
                    if offspring[i][j] == '1':
                        offspring[i] = '0' + offspring[i][1:]
                    else:
                        offspring[i] = '1' + offspring[i][1:]
                else:
                    if offspring[i][j] == '1':
                        offspring[i] = offspring[i][:(j - 1)] + '0' + offspring[i][j:]
                    else:
                        offspring[i] = offspring[i][:(j - 1)] + '1' + offspring[i][j:]
                nmut = nmut + 1 
    return offspring


def produce_value(task):
    value = []
    t = len(task)
    for i in range(t):
        value.append(task[i][3])
    return value
def produce_time(task):
    E_t = []
    t = len(task)
    for i in range(t):
        E_t.append(task[i][1])
    return E_t


def GA():
    gen = 200  
    pc = 0.25  
    pm = 0.02  
    popsize = 10  
    n = 100  
    tasks = produce_tasks()
    w = 5  
    W = 250  
    T = 1200  
    fun = 1  
    
    number = 0
    value = 0
    population = init(popsize, n)
    if fun == 1:
        value, s = fitnessfun1(population, n, w,W,tasks,T)
    ncross = 0
    nmut = 0
    t = []
    best_ind = []
    for i in range(gen):

        # cross
        offspring_c = crossover(population, pc, ncross)
        # mutate
        offspring_m = mutation2(offspring_c, pm, nmut)
        mixpopulation = population + offspring_m
    
        if fun == 1:
            value, s = fitnessfun1(mixpopulation, n, w, W, tasks, T)
        # select
        population = roulettewheel(mixpopulation, value, popsize)
        # store best solution
        result = []
        if i == gen - 1:
            if fun == 1:
                value1, s1 = fitnessfun1(population, n, w, W, tasks, T)
                result = value1
        else:
            if fun == 1:
                value1, s1 = fitnessfun1(population, n, w, W, tasks, T)
                result = value1
        maxre = max(result)
        h = result.index(max(result))
        t.append(maxre)
        best_ind.append(population[h])
    
    if fun == 1:
        
        hh = t.index(max(t))
        print(best_ind[hh])
        FCB = 0
        f2, s2 = decode1(best_ind[hh], n, w, W, tasks, T)
        print("Accepted Tasks:", s2)
    
        number = len(s2)
        for i in range(number):
        
            FCB= FCB+task[s2[i]][3]
        print(task)
        print("Total accepted tasks",len(s2))
        print("Total Reward",f2)
        print(FCB)
        value = f2

    return number, value

episode = 100
eval_profit_list = []
avg_profit_list=[]
eval_tasks_list=[]
avg_tasks_list=[]
eval_ap_list=[]
avg_ap_list=[]
avg =0
avg_1 = 0
avg_2 = 0
jy = 1
T = 0
for i in range(episode):
    start = time.time()
    print('episode:',jy)
    jy= jy+1
    t_n,profit=GA()
    eval_profit_list.append(profit)
    avg = sum(eval_profit_list)/len(eval_profit_list)
    avg_profit_list.append(avg)        
    eval_tasks_list.append(t_n)
    avg_1 = sum(eval_tasks_list)/len(eval_tasks_list)
    avg_tasks_list.append(avg_1)
    avg_profit = profit/t_n
    eval_ap_list.append(avg_profit)
    avg_2 = sum(eval_ap_list)/len(eval_ap_list)
    avg_ap_list.append(avg_2)
    end = time.time()
    times = end -start
    T = T+times
print('Aveage Total Profit', avg)
print('Aveage accepted tasks',avg_1)
print('Aveage Profit', avg_2)
print('Average Response Time', T/episode)
plot_profit(eval_profit_list,avg_profit_list)
plot_tasks(eval_tasks_list,avg_tasks_list)
plot_ap(eval_ap_list,avg_ap_list)


