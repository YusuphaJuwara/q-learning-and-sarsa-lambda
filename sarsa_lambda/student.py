import numpy as np
import random

from tqdm import tqdm

def greedy_policy(Qtable, state):
  # Exploitation: take the action with the highest state, action value
  action = np.argmax(Qtable[state][:])

  return action

def epsilon_greedy_action(env, Q, state, epsilon):
    # TODO choose the action with epsilon-greedy strategy
    
    # Randomly generate a number between 0 and 1
    # random_num = random.uniform(0,1)
    random_num = random.random()
    
    # if random_num > epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        action = greedy_policy(Q, state)
        
    # else --> exploration
    else:
        # Randomly sample an action from the action space
        action = env.action_space.sample()
        
    return action

def sarsa_lambda(env, alpha=0.2, gamma=0.99, lambda_= 0.9, initial_epsilon=1.0, n_episodes=20_000 ):

    ####### Hyperparameters
    # alpha = learning rate
    # gamma = discount factor
    # lambda_ = elegibility trace decay
    # initial_epsilon = initial epsilon value
    # n_episodes = number of episodes

    ############# define Q table and initialize to zero
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    # E = np.zeros((env.observation_space.n, env.action_space.n))
    print("TRAINING STARTED")
    print("...")
    # init epsilon
    epsilon = initial_epsilon

    received_first_reward = False

    for ep in tqdm(range(n_episodes)):
        
        # zeros the Eligibility traces
        E = np.zeros((env.observation_space.n, env.action_space.n))
        
        ep_len = 0
        state, _ = env.reset()
        action = epsilon_greedy_action(env, Q, state, epsilon)
        done = False
        while not done:
            ############## simulate the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_len += 1
            # env.render()
            next_action = epsilon_greedy_action(env, Q, next_state, epsilon)

            # TODO update q table and eligibility
            
            # Check the slides about how to update these!
            if done:
                delta = reward - Q[state, action]
            else:
                delta = reward + gamma*Q[next_state, next_action] - Q[state, action]
            
            # This corresponds to the part I(S_t=s) in the formulae. 
            # So, only the current state and action gets 1
            E[state, action] += 1
            
            # Update the Q table and Eligibility Traces for all states and actions
            Q = Q + alpha * delta * E
            E = gamma * lambda_ * E

            # Really slow; takes about 10 minutes. So, I use the faster, vectorized version above
            # for s in range(env.observation_space.n):
            #     for a in range(env.action_space.n):
            #         Q[s,a] = Q[s,a] + alpha*delta*E[s,a] 
            #         E[s,a] = gamma*lambda_ * E[s,a]

            if not received_first_reward and reward > 0:
                received_first_reward = True
                print("Received first reward at episode ", ep)
            # update current state
            state = next_state
            action = next_action
        
        # print(f"Episode {ep} finished after {ep_len} steps.")

        # update current epsilon
        if received_first_reward:
            epsilon = 0.99 * epsilon
    print("TRAINING FINISHED")
    return Q