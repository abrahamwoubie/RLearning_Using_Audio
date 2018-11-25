import gym
import gym_Audio
import numpy as np

env=gym.make('Audio-v0')

# Parameters for Q-Learning
discount_factor=0.95
learning_rate=0.8

number_of_episodes=2000

Q_Table=np.zeros([env.observation_space.n,env.action_space.n])

rewardList=[]

for i in range(0,number_of_episodes):
    state=env.reset()
    iteration=0
    reward_per_episode=0
    while iteration<100:
        action=np.argmax(Q_Table[state,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        new_state,reward,done,_=env.step(action)
        Q_Table[state,action]=Q_Table[state,action] + \
            learning_rate*(reward + discount_factor*np.max(Q_Table[new_state,:]) - Q_Table[state,action])
        state=new_state
        if done==True:
            reward_per_episode+=reward
            break
        iteration=iteration+1
    rewardList.append(reward_per_episode)

percentage_of_sucessful_episodes=(sum(rewardList)/number_of_episodes)*100
print("Reward List",rewardList)
print("Percentage of successful episodes", percentage_of_sucessful_episodes)
