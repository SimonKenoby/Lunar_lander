import torch
import torch.nn.functional as F 
import gym
from model import AC_netwok
from torch.autograd import Variable


def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank)
    env = gym.make(params.env_name)
    env.seed(params.seed + rank)

    model = AC_netwok(params.s_size, params.a_size)

    env.reset()
    state = env.render(mode='rgb_array')
    state = torch.from_numpy(state)

    done = True
    episode_length = 0

    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        if done: 
            cx = Varialbe(torch.zeros((1, 256)))
            hx = Varialbe(torch.zeros((1, 256)))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_proba = []
        rewards = []
        entropies = []

        for step in range(params.num_steps):
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            proba = F.softmax(action_values)
            log = F.log_softmax(action_values)
            entropy = -(log * proba).sum(1)
            entropies.append(entropy)
            action = proba.multinomial().data
            log = log.gather(1, Variable(action))
            values.append(value)
            log_proba.append(log)
            state, reward, done, _ = env.step(action.numpy())
            state = env.render(mode='rgb_array')
            done = (done or episode_length >= params.max_episode_length)
            #reward = max(min(reward, 1), -1)
            if done:
                episode_length = 0
                env.reset()
                state = env.render(mode='rgb_array')
            state = torch.from_numpy(state)
            rewards.append(reward)
            if done:
                break

        R = torch.zeros(1,1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx))))
            R = value.data
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            TD = rewards[i] + params.gamma * values[i+1].data - values[i].data
            gae = gae * params.gamma * params.tau + TD
            policy_loss = policy_loss - log_proba[i] * Variable(gae) - 0.01 * entropies[i]
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        optimizer.step()