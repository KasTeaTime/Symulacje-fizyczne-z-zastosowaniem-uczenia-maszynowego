import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import pygame
from gym.wrappers import TimeLimit

def control_2d():   #Sterowanie obiektem 2D
    env = gym.make("Reacher-v4", render_mode="human",  max_episode_steps=1000) #Ręka robota
    obs, _ = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((100, 100))
    clock = pygame.time.Clock()

    while True:
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        action = np.zeros(2)
        if keys[pygame.K_LEFT]:
            action[0] = -1.0
        if keys[pygame.K_RIGHT]:
            action[0] = 1.0
        if keys[pygame.K_DOWN]:
            action[1] = -1.0
        if keys[pygame.K_UP]:
            action[1] = 1.0
            
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

        clock.tick(60)
    pygame.quit()
    env.close()

def control_3d(): #Sterowanie obiektem 3D
    env = gym.make("HalfCheetah-v4", render_mode="human")
    env = TimeLimit(env.env, max_episode_steps=1000)
    obs, _ = env.reset()
    
    pygame.init()
    pygame.display.set_mode((100, 100))
    clock = pygame.time.Clock()

    while True:
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        action = np.zeros(6)
        if keys[pygame.K_q]: action[0] = 1.0
        if keys[pygame.K_a]: action[0] = -1.0

        if keys[pygame.K_w]: action[1] = 1.0
        if keys[pygame.K_s]: action[1] = -1.0

        if keys[pygame.K_e]: action[2] = 1.0
        if keys[pygame.K_d]: action[2] = -1.0

        if keys[pygame.K_r]: action[3] = 1.0
        if keys[pygame.K_f]: action[3] = -1.0

        if keys[pygame.K_t]: action[4] = 1.0
        if keys[pygame.K_g]: action[4] = -1.0

        if keys[pygame.K_y]: action[5] = 1.0
        if keys[pygame.K_h]: action[5] = -1.0
        
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        clock.tick(60)

    env.close()
    pygame.quit()

def display_training(file, env_name):
    env = gym.make(env_name, render_mode="human", max_episode_steps=2000)   # środowisko z renderowaniem
    model = PPO.load(file, device="cpu")   # wczytanie modelu
    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

def teach_2d(): #Uczenie w środowisku 2D
    env = gym.make("Reacher-v5")
    model = PPO("MlpPolicy", env, verbose=0, n_steps=2048, batch_size=64, learning_rate=3e-4, device="cpu",)
    model.learn(total_timesteps=300_000)
    model.save("ppo_reacher")
    display_training("ppo_reacher", "Reacher-v5")

def teach_3d(): #Uczenie w środowisku 3D
    env = gym.make("Walker2d-v5")
    model = PPO("MlpPolicy", env, verbose=0, n_steps=2048, batch_size=64, learning_rate=3e-4, device="cpu",)
    model.learn(total_timesteps=300_000)
    model.save("ppo_reacher3d")
    display_training("ppo_reacher3d", "Walker2d-v5")


# control_2d()
# control_3d()

# teach_2d()
# display_training("ppo_reacher", "Reacher-v5")

# teach_3d()
display_training("ppo_reacher3d", "Walker2d-v5")