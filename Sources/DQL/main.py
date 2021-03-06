import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt 

import experience_replay, image_preprocessing

class CNN(nn.Module):

    def __init__(self, action_space_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        self.fc1 = nn.Linear(in_features = self.count_neurons((3, 400, 600)), out_features = 40)
        self.fc2 = nn.Linear(in_features = 40, out_features = action_space_size)

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Making the body

class SoftmaxBody(nn.Module):
    
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)   
        actions = probs.multinomial(2)
        return actions

# Making the AI

class AI:

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)).cuda())
        output = self.brain(input)
        actions = self.body(output)
        return actions.cpu().data.numpy()

if __name__ == "__main__":

    env = gym.make('LunarLander-v2').unwrapped
    number_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = CNN(number_actions).to(device)
    softmax_body = SoftmaxBody(T = 1.0).to(device)
    ai = AI(brain = cnn, body = softmax_body)

    # Setting up Experience Replay
    n_steps = experience_replay.NStepProgress(env = env, ai = ai, n_step = 10)
    memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)
        
    # Implementing Eligibility Trace
    def eligibility_trace(batch):
        gamma = 0.99
        inputs = []
        targets = []
        for series in batch:
            input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)).cuda())
            output = cnn(input)
            cumul_reward = 0.0 if series[-1].done else output[1].data.max()
            for step in reversed(series[:-1]):
                cumul_reward = step.reward + gamma * cumul_reward
            state = series[0].state
            target = output[0].data
            target[series[0].action] = cumul_reward
            inputs.append(state)
            targets.append(target)
        return torch.from_numpy(np.array(inputs, dtype = np.float32)).cuda(), torch.stack(targets)

    # Making the moving average on 100 steps
    class MA:
        def __init__(self, size):
            self.list_of_rewards = []
            self.size = size
        def add(self, rewards):
            if isinstance(rewards, list):
                self.list_of_rewards += rewards
            else:
                self.list_of_rewards.append(rewards)
            while len(self.list_of_rewards) > self.size:
                del self.list_of_rewards[0]
        def average(self):
            return np.mean(self.list_of_rewards)
    ma = MA(100)
    r = []
    # Training the AI
    loss = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
    nb_epochs = 100
    for epoch in range(1, nb_epochs + 1):
        memory.run_steps(200)
        for batch in memory.sample_batch(128):
            inputs, targets = eligibility_trace(batch)
            inputs, targets = Variable(inputs), Variable(targets)
            predictions = cnn(inputs)
            loss_error = loss(predictions, targets)
            optimizer.zero_grad()
            loss_error.backward()
            optimizer.step()
        rewards_steps = n_steps.rewards_steps()
        ma.add(rewards_steps)
        avg_reward = ma.average()
        r.append(avg_reward)
        print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))

    plt.plot(range(1, nb_epochs+1),r )
    plt.title('Average reward for each epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Reward on the last 100 actions')
    plt.show()
