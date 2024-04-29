import random
import numpy as np
# import gymnasium as gym
# import time
# from gymnasium import spaces
# import os
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
import pickle


class VanillaFeatureEncoder:
    def __init__(self, env):
        self.env = env
        
    def encode(self, state):
        return state
    
    @property
    def size(self): 
        return self.env.observation_space.shape[0]

class RBFFeatureEncoder:
    def __init__(self, env, n_components=1000, n_samples=10000, random_state=1): # modify
        self.env = env
        self.n_components = n_components
        self.n_samples = n_samples
        
        # TODO init rbf encoder
        samples = np.array([self.env.observation_space.sample() for i in range(self.n_samples)])
        print(f"Samples' shape: {samples.shape}")
        
        ################################
        # Standardize the data => (sample -mean)/std
        self.s_mean = samples.mean(axis=0, keepdims=True)
        self.s_std = samples.std(axis=0, keepdims=True)
        self.transformed_states = (samples - self.s_mean)/self.s_std
        #################################
        
        # self.scaler = sklearn.preprocessing.StandardScaler()
        # self.scaler.fit(samples)
        # self.transformed_states = self.scaler.transform(samples)
        
        num_rbfs = 10
        self.rbf_state_featurizer = sklearn.pipeline.FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=self.n_components//num_rbfs, random_state=random_state)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=self.n_components//num_rbfs, random_state=random_state)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=self.n_components//num_rbfs, random_state=random_state)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=self.n_components//num_rbfs, random_state=random_state)),
                ("rbf5", RBFSampler(gamma=4.0, n_components=self.n_components//num_rbfs, random_state=random_state)),
                ("rbf6", RBFSampler(gamma=6.0, n_components=self.n_components//num_rbfs, random_state=random_state)),
                ("rbf7", RBFSampler(gamma=7.0, n_components=self.n_components//num_rbfs, random_state=random_state)),
                ("rbf8", RBFSampler(gamma=8.0, n_components=self.n_components//num_rbfs, random_state=random_state)),
                ("rbf9", RBFSampler(gamma=9.0, n_components=self.n_components//num_rbfs, random_state=random_state)),
                ("rbf10", RBFSampler(gamma=3.0, n_components=self.n_components//num_rbfs, random_state=random_state))
                ])

        self.rbf_state_featurizer.fit(self.transformed_states)
        
    def encode(self, state): # modify
        # TODO use the rbf encoder to return the features
        state = state[np.newaxis, :] # Add new dim to have a shape: (1,2) from shape: (2,)
        
        #scaled = self.scaler.transform(state)
        scaled = (state - self.s_mean)/self.s_std
        state_features = self.rbf_state_featurizer.transform(scaled) # shape (1, n_components)
        
        return state_features[0] # shape (n_components,)

    @property
    def size(self): # modify
        # TODO return the number of features
        return self.n_components

class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder, alpha=0.01, alpha_decay=1, 
                 gamma=0.9999, epsilon=0.3, epsilon_decay=0.995, final_epsilon=0.2, lambda_=0.9): # modify if you want (e.g. for forward view)
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (self.env.action_space.n, self.feature_encoder.size)
        self.weights = np.random.random(self.shape)
        self.traces = np.zeros(self.shape)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.lambda_ = lambda_
        
    def Q(self, feats):
        feats = feats.reshape(-1,1)
        return self.weights@feats
    
    def update_transition(self, s, action, s_prime, reward, done): # modify
        s_feats = self.feature_encoder.encode(s)
        s_prime_feats = self.feature_encoder.encode(s_prime)
        
        # TODO update the weights
        
        self.traces = self.gamma * self.lambda_ * self.traces
        self.traces[action] += s_feats
        
        if done:
            delta = reward - self.Q(s_feats)[action]
        else:
            # Notice the max(Q(s,a)) since this is using the Q-Learning TD(lambda) instead of the SARSA variant
            delta = reward + self.gamma * np.max(self.Q(s_prime_feats)) - self.Q(s_feats)[action]
        
        # Update the weights w as w_{t+1} = w_t + alpha*delta*e_t
        self.weights[action] = self.weights[action] + self.alpha*delta*self.traces[action]
        
        
    def update_alpha_epsilon(self): # do not touch
        self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)
        self.alpha = self.alpha*self.alpha_decay
        
    def policy(self, state): # do not touch
        state_feats = self.feature_encoder.encode(state)
        return self.Q(state_feats).argmax()
    
    def epsilon_greedy(self, state, epsilon=None): # do not touch
        if epsilon is None: epsilon = self.epsilon
        if random.random()<epsilon:
            return self.env.action_space.sample()
        return self.policy(state)
       
        
    def train(self, n_episodes=200, max_steps_per_episode=200): # do not touch
        print(f'ep | eval | epsilon | alpha')
        for episode in range(n_episodes):
            done = False
            s, _ = self.env.reset()
            self.traces = np.zeros(self.shape)
            for i in range(max_steps_per_episode):
                
                action = self.epsilon_greedy(s)
                s_prime, reward, done, _, _ = self.env.step(action)
                self.update_transition(s, action, s_prime, reward, done)
                
                s = s_prime
                
                if done: break
                
            self.update_alpha_epsilon()

            if episode % 20 == 0:
                print(episode, self.evaluate(), self.epsilon, self.alpha)
                                
    def evaluate(self, env=None, n_episodes=10, max_steps_per_episode=200): # do not touch
        if env is None:
            env = self.env
            
        rewards = []
        for episode in range(n_episodes):
            total_reward = 0
            done = False
            s, _ = env.reset()
            for i in range(max_steps_per_episode):
                action = self.policy(s)
                
                s_prime, reward, done, _, _ = env.step(action)
                
                total_reward += reward
                s = s_prime
                if done: break
            
            rewards.append(total_reward)
            
        return np.mean(rewards)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        return pickle.load(open(fname,'rb'))
