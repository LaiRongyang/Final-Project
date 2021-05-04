import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import time
import os
from DRL_environment import Environment

os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use('TKAgg')

eval_profit_list = []
avg_profit_list = []
eval_tasks_list = []
avg_tasks_list = []
eval_ap_list = []
avg_ap_list = []

# hyper parameters
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 1000
MAX_TASKS = 50
N_S = 4
N_A = 2


# DRAW CURVES
# Pofit
def plot_profit(profit, avg):
    host = host_subplot(111)
    plt.subplots_adjust(right=0.8)

    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total profit")

    # plot curves
    p1, = host.plot(range(len(profit)), profit, label="Total Profit")
    p2, = host.plot(range(len(avg)), avg, label="Running Average Total Profit")
    # set location of the legend
    host.legend(loc=1)

    # set color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())

    host.set_xlim([0, MAX_EP])
    host.set_ylim([0, 300])

    plt.draw()
    plt.show()

#Tasks
def plot_tasks(tasks, avg1):
    host = host_subplot(111)
    plt.subplots_adjust(right=0.8)

    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total accepted tasks")

    # plot curves
    p1, = host.plot(range(len(tasks)), tasks, label="Total Accepetd Tasks")
    p2, = host.plot(range(len(avg1)), avg1, label="Running Average Total Accepted Tasks")

    # set location of the legend,
    host.legend(loc=1)

    # set color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    host.set_xlim([0, MAX_EP])
    host.set_ylim([0, MAX_TASKS])

    plt.draw()
    plt.show()

# Average profit
def plot_ap(ap, avg2):
    host = host_subplot(111)
    plt.subplots_adjust(right=0.8)

    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Average Profit")

    # plot curves
    p1, = host.plot(range(len(ap)), ap, label="Average Profit")
    p2, = host.plot(range(len(avg2)), avg2, label="Running Average Profit")

    # set location of the legend,
    host.legend(loc=1)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0, MAX_EP])
    host.set_ylim([0, 9])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
    plt.show()

# A3C framework, for this framework  I changed the network structure and other network parameters
class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)

        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.relu(self.pi1(x))
        logits = self.pi2(pi1)
        logits1 = torch.relu(logits)
        v1 = torch.relu(self.pi1(x))
        values = self.v2(v1)
        return logits1, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)  # local network
        # add environment
        self.env = Environment()

    def run(self):
        T = 0
        total_step = 1
        episode = 1
        while episode < MAX_EP:
            start = time.time()
            #reset the new environment
            self.env.reset()
            s = self.env.observe()
            #print the list of tasks
            print(self.env.task)
            total_tasks = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:

                a = self.lnet.choose_action(v_wrap(s[None, :]))
                # interact with the environment
                s_, r, done = self.env.update_env(a)

                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)
                # update the value of network each 5 steps
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        break
                s = s_
                total_step = total_step + 1
                if (self.env.counts >= MAX_TASKS - 1):
                    break
                if (len(self.env.time_windows) > 25):
                    break
            # print(self.env.time_windows)
            end = time.time()
            times = end - start
            T = T + times
            print(self.env.time_windows)
            #record the data
            total_reward = self.env.total_profit
            accepted_tasks = self.env.atasks
            eval_profit_list.append(total_reward)
            avg = sum(eval_profit_list) / len(eval_profit_list)
            avg_profit_list.append(avg)
            eval_tasks_list.append(accepted_tasks)
            avg_1 = sum(eval_tasks_list) / len(eval_tasks_list)
            avg_tasks_list.append(avg_1)

            avg_profit = total_reward / accepted_tasks
            eval_ap_list.append(avg_profit)
            avg_2 = sum(eval_ap_list) / len(eval_ap_list)
            avg_ap_list.append(avg_2)
            print('%d Episode: Accepted tasks numbers：%d Total reward: %d' % (
                episode, accepted_tasks, total_reward))
            episode = episode + 1

        #print and draw
        print('Aveage Total Profit', avg)
        print('Aveage accepted tasks', avg_1)
        print('Aveage Profit', avg_2)
        print('Average Response Time', T / episode)
        # print('T',T)
        print('episode', episode)
        self.res_queue.put(None)
        plot_profit(eval_profit_list, avg_profit_list)
        plot_tasks(eval_tasks_list, avg_tasks_list)
        plot_ap(eval_ap_list, avg_ap_list)


if __name__ == "__main__":
    gnet = Net(N_S, N_A)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    # opt = SharedAdam(gnet.parameters(), lr=0.0068, betas=(0.92, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    # workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(1)]
    print(len(workers))
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
