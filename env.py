import gym
import time
env = gym.make('Pong-v0')
print(env.action_space)
print(env.observation_space)
for i_episode in range(2):
    observation = env.reset()
    for t in range(1000):
        time.sleep(0.05)
        env.render()
        if t%4 ==0:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action) # 0-不动, 1-不动， 2-向上，3-向下，4-向上，5-向下
            print(reward)
        else:
            observation, reward, done, info = env.step(1)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()