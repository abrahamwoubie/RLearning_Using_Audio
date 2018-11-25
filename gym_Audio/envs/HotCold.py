import gym
env=gym.make('HotterColder-v0')
s=env.reset()
number_of_observations=env.observation_space.n
counter = 0
done=False
while done != True:
    state, reward, done, info = env.step(env.action_space.sample())
    counter += 1
    print(env.action_space.sample,state,reward,done)
print(counter)