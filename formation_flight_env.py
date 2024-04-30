import gym
from GAME_4 import *
import numpy as np

class FormationFlightEnv(gym.Env):
    def __init__(self):
        self.game = FormationFlightGame()

    def _get_obs(self):
        pos = np.array(self.game.ALLY_POS).flatten()
        bools = np.array([self.game.tb2_in_radar, self.game.akinci_in_radar, self.game.kizilelma_in_radar, self.game.tb2_mountain_collide, self.game.akinci_mountain_collide, self.game.kizilelma_mountain_collide], dtype=np.float64).flatten()

        obs = np.append(pos, bools)

        return obs

    def reset(self):
        self.game.init()

    def step(self, action):
        self.game.game_step(action)
        obs = self._get_obs()

        reward, done = self.reward(obs)

        return reward, done

    def reward(self, obs):
        done = False

        tb2_pos_x = obs[0]
        tb2_pos_y = obs[1]
        akn_pos_x = obs[3]
        akn_pos_y = obs[4]
        kzl_pos_x = obs[6]
        kzl_pos_y = obs[7]

        collision = obs[9] + obs[10] + obs[11] + obs[12] + obs[13] + obs[14]

        dif_tb2_akn = - ((tb2_pos_x - akn_pos_x) ** 2 + (tb2_pos_y - akn_pos_y) ** 2) ** 0.5
        dif_kzl_akn = - ((kzl_pos_x - akn_pos_x) ** 2 + (kzl_pos_y - akn_pos_y) ** 2) ** 0.5
        dif_tb2_kzl = - ((tb2_pos_x - kzl_pos_x) ** 2 + (tb2_pos_y - kzl_pos_y) ** 2) ** 0.5

        distance_to_target = - (((tb2_pos_x + akn_pos_x + kzl_pos_x)/3) ** 2 + ((tb2_pos_y + akn_pos_y + kzl_pos_y)/3) ** 2) ** 0.5

        reward = (distance_to_target + (dif_tb2_akn + dif_kzl_akn + dif_tb2_kzl)) / 1e3

        if collision:
            reward -= (1 * collision)

        if abs(distance_to_target) < 100:
            done = True
            reward = 1000

        return reward, done

    def render(self):
        pass
