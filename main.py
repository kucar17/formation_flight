from formation_flight_env import FormationFlightEnv
import pygame
import sys
import time

EPISODE = 100
STEP = 1000

env = FormationFlightEnv()

i = 0

for i in range(EPISODE):
    env.reset()
    total_episode_reward = 0

    for i in range(STEP):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action = 0  # TODO: RL AGENT WILL BE ADDED HERE
        
        reward, done = env.step(action)
        total_episode_reward += reward

        if done:
            print("SUCCESS", total_episode_reward)
            time.sleep(5)
            break
