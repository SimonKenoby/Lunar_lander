import gym
import cv2
import numpy as np

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    env.reset()
    for _ in range(10):
        #img is a matrix of size (400, 600, 3)
        img = env.render(mode='rgb_array')
        print(img.shape)
        '''
        # Can reshape resize image if needed.
        img = cv2.resize(img, dsize=(80,80))
        print(img.shape)
        '''
        observation, reward, done, info = env.step(env.action_space.sample()) # take a random action