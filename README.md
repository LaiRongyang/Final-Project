# Final-Project
The code about this final project (Online scheduling of image satellites based on deep reinforcement learning)
There are 10 py files of this project:
In the all code of the project, for A3C, DDPG, DQN and GA, I use a framework then make some changes according to the environment and problem I study,other files are wirtten by myself.
5 files for Deep reinforcement learning algorithms, 3 for traditional scheduling algorithms,  1 for environment file and 1 for user to understand the task produce process.Next there are the specific description of each file:
DRL_environment.py : this is the environment file of the project's problem which can be used to train A3C, DQN, and DDPG, but when you test, you need to put these files, in same directory.There is one important mention, because the study need test different environment, so when you want to do it, first you need to change the task number in this file, then change task number in realted DRL algorithms file to make it same.If you want to make comparison between DRL algorithms with traditional scheduling algorithms, you also need to make it same.
DRL files:
DQN.py: which is a framework of DQN then combine this specific satellite problem, interact with environment.
A3C.py  utils.py shared_adam.py :These three files are build A3C algorthms, same as DQN, I do changes to let it satisfy the environment.
DDPG.py: same as A3C,DQN
Traditional scheduling algorithms files:
GA.py: same as DRL, I do some changes in the decode function and main function according to the problem.
FCFS.py
RP.py
produce_tasks_test.py: This file can be used to check the producetion of tasks and check the task conflict rate.
If you want to  test the adaptability of DDPG, you need to use two different environment files firstly,then do some changes in DDPG.py.

