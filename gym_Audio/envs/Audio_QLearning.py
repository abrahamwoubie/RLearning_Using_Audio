import gym
import gym_Audio
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('Audio-v0')

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])

# Set learning parameters
lr = .8
y = .95
num_episodes = 2000

number_of_iterations_per_episode=[]
number_of_episodes=[]

#create lists to contain total rewards and steps per episode
#jList = []

rList = []
for i in range(num_episodes):
    #print("*******************************************************************")
     #Reset environment and get first new observation
    s,g = env.reset()
    print("Start state at Iteration {} is {}".format(i,s))
    print("Goal state at Iteration {} is {}".format(i, g))
    env.render()
    rAll = 0
    d = False
    j = 0

    number_of_episodes.append(i)
    #The Q-Table learning algorithm
    while j < 100:
        env.render()
        j+=1
        #Choose an action by greedily picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            #print("Iteration Number {} has finished with {} number of timestamps".format(i,j-1))
            break
    #jList.append(j)
    number_of_iterations_per_episode.append(j - 1)
    rList.append(rAll)
percentage_of_successful_episodes=(sum(rList)/num_episodes)*100
print("Reward List",rList)
print ("Percent of successful episodes: ",percentage_of_successful_episodes, "%")
plt.xlabel("Episode")
plt.ylabel("Number of Iterations")
plt.plot(number_of_episodes, number_of_iterations_per_episode, 'ro')
plt.grid(True)
#plt.savefig("test.png")
plt.show()


