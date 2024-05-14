from formation_flight_env import FormationFlightEnv
import pygame
import sys
import time

EPISODE = 100
STEP = 1500

env = FormationFlightEnv()

for i in range(EPISODE):
    env.reset()
    total_episode_reward = 0

    for i in range(STEP):
        env.render()

        action = [0, 1, 0, 1, 0, 1]
        action2 = [1, 0, 1, 0, 1, 0]
        if i >= 500:
            next_obs, reward, done, _ = env.step(action2)
        else:
            next_obs, reward, done, _ = env.step(action)
        total_episode_reward += reward
        print(reward)
        if done:
            print("SUCCESS", total_episode_reward)
            time.sleep(5)
            break
