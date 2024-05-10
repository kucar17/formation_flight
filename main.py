from formation_flight_env import FormationFlightEnv
import pygame
import sys
import time
import os
from ray.rllib.algorithms.sac import SACConfig

EPISODE = 100
STEP = 1000

env = FormationFlightEnv()

for i in range(EPISODE):
    env.reset()
    total_episode_reward = 0

    for i in range(STEP):
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         sys.exit()
        pygame.display.quit()
        pygame.quit()

        action = [1, 1, 0, 1, 1, 0]  # TODO: RL AGENT WILL BE ADDED HERE

        next_obs, reward, done, _ = env.step(action)
        total_episode_reward += reward
        print(reward)
        if done:
            print("SUCCESS", total_episode_reward)
            time.sleep(5)
            break
