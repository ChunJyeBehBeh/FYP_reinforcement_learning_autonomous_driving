import sys
import gym
import random
import numpy as np
import cv2
import skimage as skimage
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import normal, identity
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow.keras import backend as K

import gym
import gym_donkeycar
import gym_donkeycar.envs
import my_cv
import matplotlib.pyplot as plt

from utils.utils import load_vae

EPISODES = 10000
img_rows , img_cols = 80, 80
# Convert image into Black and white
img_channels = 4 # We stack 4 frames

class DQNAgent:

    def __init__(self, state_size, action_size):
        self.t = 0
        self.max_Q = 0
        self.train = True
        self.lane_detection = False # Set to True to train on images with segmented lane lines
        
        
        # Get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        
        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        if (self.train):
            self.epsilon = 1.0
            self.initial_epsilon = 1.0
        else:
            self.epsilon = 1e-6
            self.initial_epsilon = 1e-6
        self.epsilon_min = 0.02
        self.batch_size = 64
        self.train_start = 100
        self.explore = 10000

        # Create replay memory using deque
        self.memory = deque(maxlen=100000)

        self.vae_used = False

        if self.vae_used:
            vae_path = 'vae-level-0-dim-32.pkl'
            # assert vae_path == None , "Missing weight file for vae"
            self.vae = load_vae(vae_path)
        # Create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()

    def build_model(self):
        if not self.vae_used:
            print("Now we build the model for image raw")
            model = Sequential()
            model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same",input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
            model.add(Activation('relu'))
            model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
            model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation('relu'))
    
            # 15 categorical bins for Steering angles
            model.add(Dense(15, activation="linear")) 
    
            adam = Adam(lr=self.learning_rate)
            model.compile(loss='mse',optimizer=adam)
            print("We finished building the model")
        else:
            print("Now we build the model for VAE")
            model = Sequential()
            model.add(Dense(64, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(15,activation="linear"))
            adam = Adam(lr=self.learning_rate)
            model.compile(loss='mse',optimizer=adam)
            print("We finished building the model")
        return model

    def process_image(self, obs):

        if not self.lane_detection:
            if self.vae_used:
                return obs
            obs = skimage.color.rgb2gray(obs)
            obs = skimage.transform.resize(obs, (img_rows, img_cols))
            return obs
        else:
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = cv2.resize(obs, (img_rows, img_cols))
            edges = my_cv.detect_edges(obs, low_threshold=50, high_threshold=150)

            rho = 0.8
            theta = np.pi/180
            threshold = 25
            min_line_len = 5
            max_line_gap = 10

            hough_lines = my_cv.hough_lines(edges, rho, theta, threshold, min_line_len, max_line_gap)

            left_lines, right_lines = my_cv.separate_lines(hough_lines)

            filtered_right, filtered_left = [],[]
            if len(left_lines):
                filtered_left = my_cv.reject_outliers(left_lines, cutoff=(-30.0, -0.1), lane='left')
            if len(right_lines):
                filtered_right = my_cv.reject_outliers(right_lines,  cutoff=(0.1, 30.0), lane='right')

            lines = []
            if len(filtered_left) and len(filtered_right):
                lines = np.expand_dims(np.vstack((np.array(filtered_left),np.array(filtered_right))),axis=0).tolist()
            elif len(filtered_left):
                lines = np.expand_dims(np.expand_dims(np.array(filtered_left),axis=0),axis=0).tolist()
            elif len(filtered_right):
                lines = np.expand_dims(np.expand_dims(np.array(filtered_right),axis=0),axis=0).tolist()

            ret_img = np.zeros((80,80))

            if len(lines):
                try:
                    my_cv.draw_lines(ret_img, lines, thickness=1)
                except:
                    pass

            return ret_img
        

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, s_t):
        if np.random.rand() <= self.epsilon:
            #print("Return Random Value")
            #return random.randrange(self.action_size)
            return np.random.uniform(-1,1)
        else:
            #print("Return Max Q Prediction")
            q_value = self.model.predict(s_t)
            # Convert q array to steering value
            return linear_unbin(q_value[0])

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            #self.epsilon *= self.epsilon_decay
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore


    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)
        targets = self.model.predict(state_t)
        self.max_Q = np.max(targets[0])
        target_val = self.model.predict(state_t1)
        target_val_ = self.target_model.predict(state_t1)
        for i in range(batch_size):
            if terminal[i]:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = np.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])

        self.model.train_on_batch(state_t, targets)

    def load_model(self, name):
        self.model.load_weights(name)

    # Save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)

## Utils Functions ##

def linear_bin(a):
    """
    Convert a value to a categorical array.

    Parameters
    ----------
    a : int or float
        A value between -1 and 1

    Returns
    -------
    list of int
        A list of length 15 with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a + 1
    b = round(a / (2 / 14))
    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr):
    """
    Convert a categorical array to value.

    See Also
    --------
    linear_bin
    """
    if not len(arr) == 15:
        raise ValueError('Illegal array length, must be 15')
    b = np.argmax(arr)
    a = b * (2 / 14) - 1
    return a


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    env = gym.make("donkey-generated-roads-v0")

    # Get size of state and action from environment
    state_size = (img_rows, img_cols, img_channels)
    action_size = env.action_space.shape[0] # Steering and Throttle

    agent = DQNAgent(state_size, action_size)

    throttle = 0.4 # Set throttle as constant value

    episodes = []

    if not agent.train:
        print("Now we load the saved model")
        agent.load_model("./save_model/save_model.h5")

    episode_reward_store = np.array([])
    steering_different_store = np.array([])
    step_store = np.array([])

    for e in range(EPISODES):

        print("Episode: ", e)

        done = False
        
        obs = env.reset()
        obs = env.reset()

        episode_len = 0
        
        episode_reward = 0.0
        steering_diff = 0.0
        last_steering = 0.0

        if not agent.vae_used:
            x_t = agent.process_image(obs)
            s_t = np.stack((x_t,x_t,x_t,x_t),axis=2)
            # In Keras, need to reshape
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2]) #1*80*80*4       
        else:
            obs = obs[40:,:]
            s_t = agent.vae.encode(obs)

        while not done:

            # Get action for the current state and go one step in environment
            steering = agent.get_action(s_t)
            action = [steering, throttle]
            next_obs, reward, done, info = env.step(action)

            steering_diff = steering_diff + abs(last_steering-steering)
            last_steering =steering

            if not agent.vae_used:
                x_t1 = agent.process_image(next_obs)

                x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
                s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) #1x80x80x4
            else:
                next_obs = next_obs[40:,:]   
                s_t1 = agent.vae.encode(next_obs)

            episode_reward +=reward    

            # Save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(s_t, np.argmax(linear_bin(steering)), reward, s_t1, done)

            s_t = s_t1
            agent.t = agent.t + 1
            episode_len = episode_len + 1
            if agent.t % 30 == 0:
                print("EPISODE",  e, "TIMESTEP", agent.t,"/ ACTION", action, "/ REWARD", reward, "/ EPISODE LENGTH", episode_len, "/ Q_MAX " , agent.max_Q)

            if done:
                
                if agent.train:
                    print("Training")
                    agent.train_replay()

                # Every episode update the target model to be same with model
                agent.update_target_model()

                episodes.append(e)
                steering_different_store = np.append(steering_different_store,steering_diff/episode_len)
                episode_reward_store = np.append(episode_reward_store,episode_reward)
                step_store = np.append(step_store,episode_len)
                
                plt.subplot(121)
                plt.plot(steering_different_store,label='steering')
                plt.legend()
                plt.subplot(122)
                plt.plot(episode_reward_store,label='episode')
                plt.legend()
                plt.tight_layout()
                plt.show()
                plt.close()

                val_to_save= {'steering_different_store':steering_different_store,
                'episode_reward_store':episode_reward_store,
                'step_store':step_store}
                np.savez("training_result.npz",**val_to_save)
                
                # Save model for each episode
                if agent.train:
                    agent.save_model("./save_model/save_model.h5")

                print("episode:", e, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon, " episode length:", episode_len)

