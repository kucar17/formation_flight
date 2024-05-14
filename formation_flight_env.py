import gym
from formation_flight_game import *
import numpy as np
from gym import spaces


class FormationFlightEnv(gym.Env):
    def __init__(self):
        self.game = FormationFlightGame()
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2, 2, 2])

    def _get_obs(self):
        pos = (
            np.array(
                [
                    self.game.TB2_POS[0],
                    self.game.TB2_POS[1],
                    self.game.AKINCI_POS[0],
                    self.game.AKINCI_POS[1],
                    self.game.KIZILELMA_POS[0],
                    self.game.KIZILELMA_POS[1],
                ]
            ).flatten()
            / 700
        )

        bools = np.array(
            [
                self.game.tb2_in_radar,
                self.game.akinci_in_radar,
                self.game.kizilelma_in_radar,
            ],
            dtype=np.float64,
        ).flatten()

        obs = np.append(pos, bools)

        return obs

    def reset(self):
        self.game.init()

        obs = self._get_obs()

        return obs

    def step(self, action):
        self.game.game_step(action)
        next_obs = self._get_obs()

        reward, done = self.reward(next_obs)

        info = None

        return next_obs, reward, done, info

    def reward(self, obs):
        done = False

        # Extract positions
        tb2_pos_x, tb2_pos_y = obs[0] * 700, obs[1] * 700
        akn_pos_x, akn_pos_y = obs[2] * 700, obs[3] * 700
        kzl_pos_x, kzl_pos_y = obs[4] * 700, obs[5] * 700

        # Check collisions
        collision_penalty = sum(obs[6:9])

        # Calculate distances between drones
        dif_tb2_akn = np.sqrt(
            (tb2_pos_x - akn_pos_x) ** 2 + (tb2_pos_y - akn_pos_y) ** 2
        )
        dif_kzl_akn = np.sqrt(
            (kzl_pos_x - akn_pos_x) ** 2 + (kzl_pos_y - akn_pos_y) ** 2
        )
        dif_tb2_kzl = np.sqrt(
            (tb2_pos_x - kzl_pos_x) ** 2 + (tb2_pos_y - kzl_pos_y) ** 2
        )

        # Calculate the centroid of the formation
        centroid_x = (tb2_pos_x + akn_pos_x + kzl_pos_x) / 3
        centroid_y = (tb2_pos_y + akn_pos_y + kzl_pos_y) / 3
        distance_to_target = np.sqrt(centroid_x**2 + centroid_y**2)

        # Formulate the reward
        reward = (
            -(dif_tb2_akn + dif_kzl_akn + dif_tb2_kzl) / 1000
        )  # Normalizing distance factor
        reward -= distance_to_target / 1000  # Incentive to stay close to the origin
        reward -= collision_penalty * 50  # Significant penalty for collisions

        if collision_penalty > 0:
            done = True  # End episode on collision

        if distance_to_target < 100:
            reward += 1000  # Big reward for getting to the target
            done = True

        return reward, done

    def render(self):
        self.game.render = True
