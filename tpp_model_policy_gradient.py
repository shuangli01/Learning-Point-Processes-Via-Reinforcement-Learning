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

        # initialize output
        cum_output = torch.zeros(self.batch_size, 1).float()

        output_array = []
        cum_output_array = []
        sigma_array = []


        for i in range(self.seq_len):

            # sigma = self.elu(self.V(hidden_state_1)) + 1
            sigma = self.elu(self.V(hx)) + 1 # apply a nonlinear function from hidden_state to sigma
            sigma_array.append(sigma)

            # sequentially generate time-interval

            output = (- torch.log(torch.reshape(torch.Tensor(rand_uniform_pool[:, i]), (self.batch_size,1)))) / sigma  # batch_size * 1
            output_array.append(output)

            cum_output = cum_output + output
            cum_output_array.append(cum_output)
            # apply LSTM cell for each step
            hx, cx = self.lstm_cell(cum_output, (hx, cx))

         # detach the gradients for the generated samples

        learner_time_list = [learner_time_batch.detach().numpy() for learner_time_batch in
                             cum_output_array]  # detach gradient
        learner_time_mat = np.concatenate(learner_time_list, axis=1)  # ( batch_size,  seq_len )
        learner_time_interval_list = [learner_time_interval_batch.detach() for learner_time_interval_batch in
                             output_array]  # detach gradient
        learner_time_interval = torch.cat(learner_time_interval_list, dim=1)
        sigma = torch.cat(sigma_array, dim=1)

        return learner_time_mat, learner_time_interval, sigma


###########################################################################
    def reward_kernel_matrix(self, expert_time, learner_time):
        # Note: for policy gradient, when we compute the reward, we detach the gradient
        # So we first change learner_time form a list of tensors to numpy array
        # input: learner_time: list( tensor )

        learner_time_flattern = np.reshape(learner_time, newshape=[1, self.batch_size*self.seq_len])
        expert_time_flattern = np.reshape(expert_time, newshape=[1, self.batch_size*self.seq_len])

        learner_learner_mat = np.exp(- np.square(learner_time_flattern - np.transpose(learner_time_flattern)) / self.kernel_bandwidth)
        learner_expert_mat = np.exp(- np.square(learner_time_flattern - np.transpose(expert_time_flattern))/self.kernel_bandwidth)

        learner_time_mask = np.multiply(learner_time_flattern < self.T_max, learner_time_flattern > 0).astype(float) # [1, batch_size * seq_len]
        expert_time_mask = np.multiply(expert_time_flattern < self.T_max, expert_time_flattern > 0).astype(float)   # [1, batch_size * seq_len]

        learner_learner_mat_mask = np.matmul(learner_time_mask, learner_learner_mat)
        learner_expert_mat_mask = np.matmul(np.multiply(learner_time_mask, expert_time_mask), learner_expert_mat)


        reward_array = np.sum(learner_expert_mat_mask, axis=0) - np.sum(learner_learner_mat_mask, axis=0)


        return reward_array, learner_time_mask.squeeze()


###########################################################################
    def compute_policy_gradient(self, reward_array, learner_time_mask, learner_time_interval, sigma):
        # policy gradient
        # sigma is a list of tensors
        # only sigma contains unknown parameters
        # use sigma to compute the log-likelihood

        # detach the gradient for learner_time_interval

        loglik_interval = torch.log(sigma) - sigma * learner_time_interval # with gradient

        # compute reward to go
        reward_with_mask = np.reshape(reward_array * learner_time_mask, [self.batch_size, self.seq_len])
        reward_to_go_inverse = np.cumsum(reward_with_mask, axis=1)
        reward_total = np.reshape(np.sum(reward_with_mask, axis=1), [batch_size,1])
        a = np.tile(reward_total, reps=self.seq_len) - reward_to_go_inverse
        reward_to_go = np.concatenate((reward_total, a[:, 1:]), axis=1)
        reward_to_go = torch.tensor(reward_to_go).float()

        # compute loss
        loss = - torch.sum(loglik_interval * reward_to_go)

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
        learner_time, learner_time_interval, sigma = PointProcessGenerator.generate_time(
            cur_rand_uniform_pool)

        batch_idx = np.arange(batch_size * b, batch_size * (b + 1))
        expert_time_batch = expert_time[batch_idx]
        reward_array, learner_time_mask = PointProcessGenerator.reward_kernel_matrix(expert_time_batch, learner_time)
        PointProcessGenerator.zero_grad()
        loss = PointProcessGenerator.compute_policy_gradient(reward_array, learner_time_mask, learner_time_interval, sigma)

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









