import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from ppgen import *
import matplotlib.pyplot as plt
import random

###########################################################################
# load data

T_max = 20
num_seq = 1000

intensity = IntensityHawkesPlusPoly(mu=1, alpha=0, beta=1,
                                    segs=[0, T_max/4., T_max*2./4., T_max*3./4., T_max],
                                    b=0, A=[1, -1, 1, -1])
expert_time = generate_sample(intensity, T=T_max, n=num_seq)


expert_len = []
for s in expert_time:
    expert_len.append(len(s))
max_expert_len = max(expert_len)

seq_len = max_expert_len + 10

ee = T_max * np.ones((num_seq, seq_len))
for i in range(num_seq):
    ee[i, 0:expert_len[i]] = np.array(expert_time[i]) # pad sequence with T_max


print(np.shape(ee))


###########################################################################
# generator
# LSTM type of architecture to generate time

class PointProcessGenerator(nn.Module):


    def __init__(self, kernel_bandwidth=1, input_size=1, hidden_size=5, seq_len=10, batch_size=2, T_max=5):
        super(PointProcessGenerator, self).__init__()
        self.input_size = input_size
        self.kernel_bandwidth = kernel_bandwidth

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.T_max = T_max

        # Define parameters to learn
        # Define the LSTM layer
        self.lstm_cell = nn.LSTMCell(self.input_size, self.hidden_size, bias=True)
        # Define feedforward layer
        self.V = nn.Linear(self.hidden_size, 1, bias=True)

        # Define nonlinear activation functions
        self.elu = nn.ELU()

###########################################################################


    def generate_time(self, rand_uniform_pool):
        # rand_uniform_pool: batch_size * seq_len
        # initialize hidden_state

        hx = torch.zeros(self.batch_size, self.hidden_size).float()
        cx = torch.zeros(self.batch_size, self.hidden_size).float()

        # initialize outputs
        cum_output = torch.zeros(self.batch_size, 1).float()
        cum_output_array = []

        for i in range(self.seq_len):

            # sigma = self.elu(self.V(hidden_state_1)) + 1
            sigma = self.elu(self.V(hx)) + 1 # apply a nonlinear function from hidden_state to sigma
            # sequentially generate time-interval

            output = (- torch.log(torch.reshape(torch.Tensor(rand_uniform_pool[:, i]), (self.batch_size,1)))) / sigma  # reparametrization trick
            cum_output = cum_output + output
            cum_output_array.append(cum_output)
            # apply LSTM cell for each step
            hx, cx = self.lstm_cell(cum_output, (hx, cx))

        learner_time_mat = torch.cat(cum_output_array, dim=1)
        return learner_time_mat

###########################################################################
    def reward_kernel_matrix(self, expert_time, learner_time):
        # Note: for reparametrization trick, learner_time contains the gradient
        # input: learner_time: tensor (batch_size, seq_len)
        # input: expert_time: numpy array

        learner_time_flattern = learner_time.view([1, self.batch_size*self.seq_len])
        expert_time_flattern = torch.Tensor(expert_time).view([1, self.batch_size*self.seq_len])


        expert_expert_mat = expert_time_flattern - expert_time_flattern.view([self.batch_size*self.seq_len,1])
        expert_expert_kernel = torch.exp(- expert_expert_mat ** 2 /self.kernel_bandwidth)

        learner_expert_mat = learner_time_flattern - expert_time_flattern.view([self.batch_size*self.seq_len,1])
        learner_expert_kernel = torch.exp(-learner_expert_mat ** 2 / self.kernel_bandwidth)

        learner_learner_mat = learner_time_flattern - learner_time_flattern.view([self.batch_size * self.seq_len, 1])
        learner_learner_kernel = torch.exp(-learner_learner_mat ** 2 / self.kernel_bandwidth)

        learner_time_mask = (learner_time_flattern < self.T_max)*(learner_time_flattern > 0) # [1, batch_size * seq_len]
        expert_time_mask = (expert_time_flattern < self.T_max)*(expert_time_flattern > 0)# [1, batch_size * seq_len]


        learner_time_mask = learner_time_mask.type(torch.float)
        expert_time_mask = expert_time_mask.type(torch.float)


        expert_expert_kernel_sum = torch.mm(torch.mm(expert_time_mask, expert_expert_kernel),
                                            expert_time_mask.view([self.batch_size*self.seq_len,1]))
        learner_learner_kernel_sum = torch.mm(torch.mm(learner_time_mask, learner_learner_kernel),
                                            learner_time_mask.view([self.batch_size*self.seq_len,1]))

        learner_expert_kernel_sum = torch.mm(torch.mm(expert_time_mask, learner_expert_kernel),
                                            learner_time_mask.view([self.batch_size*self.seq_len,1]))
        loss = expert_expert_kernel_sum + learner_learner_kernel_sum - 2 * learner_expert_kernel_sum

        return loss

###########################################################################

kernel_bandwidth = 1
hidden_size = 64
batch_size = 20
input_size = 1
iter = 4000
num_batches = int(num_seq / batch_size)


PointProcessGenerator = PointProcessGenerator(kernel_bandwidth, input_size, hidden_size, seq_len, batch_size, T_max)
optimizer = Adam(PointProcessGenerator.parameters(), betas=(0.6, 0.6), lr=0.001, weight_decay=0.1)

# training
expert_time = ee



fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()


for i in range(iter):

    for b in range(num_batches):

                cur_rand_uniform_pool = np.random.rand(batch_size, seq_len).astype(np.float32)
                learner_time = PointProcessGenerator.generate_time(
                        cur_rand_uniform_pool)

                batch_idx = np.arange(batch_size * b, batch_size * (b + 1))
                expert_time_batch = expert_time[batch_idx]

                loss = PointProcessGenerator.reward_kernel_matrix(expert_time_batch, learner_time)
                PointProcessGenerator.zero_grad()
                loss.backward() # computes gradient
                optimizer.step() # update the parameters using the gradients



                # visualize results

                ax.clear()
                test_point_flat = learner_time.flatten()
                test_intensity_cum = []
                for grid in np.arange(0, T_max, 0.5):
                    idx = (test_point_flat < grid)
                    event_count_cum = len(test_point_flat[idx])
                    test_intensity_cum = np.append(test_intensity_cum, event_count_cum)

                test_intensity = np.append(test_intensity_cum[0], np.diff(test_intensity_cum)) / batch_size
                plt.plot(np.arange(0, T_max, 0.5), test_intensity)

                train_point_flat = expert_time_batch.flatten()
                train_intensity_cum = []
                for grid in np.arange(0, T_max, 0.5):
                    idx = (train_point_flat < grid)
                    event_count_cum = len(train_point_flat[idx])
                    train_intensity_cum = np.append(train_intensity_cum, event_count_cum)

                train_intensity = np.append(train_intensity_cum[0],
                                            np.diff(train_intensity_cum)) / batch_size
                plt.plot(np.arange(0, T_max, 0.5), train_intensity)
                plt.pause(0.02)

                print('loss', loss)









