import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer import ReplayBuffer
import glob
from utils import save, collect_random
import random
from agent import SAC
from formation_flight_env import FormationFlightEnv

def get_config():
    env = FormationFlightEnv()
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument("--run_name", type=str, default="SAC", help="Run name, default: SAC")
    parser.add_argument("--episodes", type=int, default=1_000_000_000, help="Number of episodes, default: 100")
    parser.add_argument("--steps", type=int, default=1_000, help="Number of steps, default: 1000")
    parser.add_argument("--exploration", type=int, default=10000, help="Number of exploration episodes, default: 1000")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=25, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size, default: 256")

    args = parser.parse_args()
    return args

def train(config):
    # Creating env
    env = FormationFlightEnv()

    # Seeds
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    env.action_space.seed(config.seed)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Steps
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0

    with wandb.init(project="SAC_Discrete", name=config.run_name, config=config):

        # Agent
        agent = SAC(state_size=9, action_size=env.action_space.shape[0], device=device)

        # Wandb
        wandb.watch(agent, log="gradients", log_freq=10)

        # Replay buffer
        buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)

        # Exploration
        collect_random(env=env, dataset=buffer, num_samples=config.exploration)

        # Training loops
        for i in range(1, config.episodes + 1):
            state = env.reset()
            episode_steps = 0
            rewards = 0
            for j in range(config.steps):
                action = agent.get_action(state)
                steps += 1
                next_state, reward, done, _ = env.step(action)
                buffer.add(state, action, reward, next_state, done)
                (policy_loss, bellmann_error1, bellmann_error2) = agent.learn(steps, buffer.sample(), gamma=0.99)
                state = next_state
                rewards += reward
                episode_steps += 1
                if done:
                    break

            # Reward calculation
            average10.append(rewards)
            total_steps += episode_steps
            print("Episode: {} | Reward: {} | Policy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps))

            # Wandb log file
            wandb.log({
                    "Reward": rewards,
                    "Average10": np.mean(average10),
                    "Steps": total_steps,
                    "Policy Loss": policy_loss,
                    "Bellmann error 1": bellmann_error1,
                    "Bellmann error 2": bellmann_error2,
                    "Steps": steps,
                    "Episode": i,
                    "Buffer size": buffer.__len__()})

            # Saving agent
            if i % config.save_every == 0:
                save(config, save_name="SAC_discrete", model=agent.actor_local, wandb=wandb, ep=0)

if __name__ == "__main__":
    config = get_config()
    train(config)
