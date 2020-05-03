import time
from collections import deque

import numpy as np
from stable_baselines import SAC
from stable_baselines import logger
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.common import TensorboardWriter
import cv2
import matplotlib.pyplot as plt

class SACWithVAE(SAC):
    """
    Custom version of Soft Actor-Critic (SAC) to use it with donkey car env.
    It is adapted from the stable-baselines version.

    Notable changes:
    - optimization is done after each episode and not at every step
    - this version is integrated with teleoperation

    """
    def optimize(self, step, writer, current_lr):
        """
        Do several optimization steps to update the different networks.

        :param step: (int) current timestep
        :param writer: (TensorboardWriter object)
        :param current_lr: (float) Current learning rate
        :return: ([np.ndarray]) values used for monitoring
        """
        train_start = time.time()
        mb_infos_vals = []
        for grad_step in range(self.gradient_steps):
            if step < self.batch_size or step < self.learning_starts:
                break
            self.n_updates += 1
            # Update policy and critics (q functions)
            mb_infos_vals.append(self._train_step(step, writer, current_lr))

            if (step + grad_step) % self.target_update_interval == 0:
                # Update target network
                self.sess.run(self.target_update_op)
        if self.n_updates > 0:
            print("SAC training duration: {:.2f}s".format(time.time() - train_start))
        return mb_infos_vals

    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=1, tb_log_name="SAC", print_freq=100, reset_num_timesteps=True):
        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            if self.action_noise is not None:
                self.action_noise.reset()
            is_teleop_env = hasattr(self.env, "wait_for_teleop_reset")
            # TeleopEnv
            if is_teleop_env:
                print("Waiting for teleop")
                obs = self.env.wait_for_teleop_reset()
            else:
                obs = self.env.reset()

            self.episode_reward = np.zeros((1,))
            self.total_episode_reward = np.array([])

            self.throttle_episode_store = np.array([])
            self.steering_episode_store = np.array([])

            self.step_episode_store = np.array([])
            self.throttle_min_max = np.array([])
            self.throttle_mean = np.array([])
            self.steering_diff = np.array([])

            ep_info_buf = deque(maxlen=100)
            ep_len = 0
            self.n_updates = 0
            infos_values = []
            mb_infos_vals = []

            for step in range(total_timesteps):
                # Compute current learning_rate
                frac = 1.0 - step / total_timesteps
                current_lr = self.learning_rate(frac)

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if (step < self.learning_starts
                    or np.random.rand() < self.random_exploration):
                    # No need to rescale when sampling random action
                    rescaled_action = action = self.env.action_space.sample()
                else:
                    action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # Add noise to the action (improve exploration,
                    # not needed in general)
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(rescaled_action)
                ep_len += 1

                if print_freq > 0 and ep_len % print_freq == 0 and ep_len > 0:
                    print("{} steps".format(ep_len))

                # Debug Purpose: Check the input shape that store into replay_buffer
                # print(obs.shape)
                # cv2.imwrite("TEST/{}.jpg".format(step),obs)
                   
                self.steering_episode_store = np.append(self.steering_episode_store,action[0])
                self.throttle_episode_store = np.append(self.throttle_episode_store,0.4)

                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, reward, new_obs, float(done))
                obs = new_obs

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, step)

                if ep_len > self.train_freq:
                    print("Additional training")
                    self.env.reset()
                    mb_infos_vals = self.optimize(step, writer, current_lr)
                    done = True

                episode_rewards[-1] += reward
                if done:
                    if self.action_noise is not None:
                        self.action_noise.reset()
                    if not (isinstance(self.env, VecEnv) or is_teleop_env):
                        obs = self.env.reset()

                    self.throttle_min_max = np.append(self.throttle_min_max,[np.amin(self.throttle_episode_store),np.amax(self.throttle_episode_store)])
                    self.throttle_mean = np.append(self.throttle_mean,np.sum(self.throttle_episode_store)/len(self.throttle_episode_store))
                    self.throttle_episode_store = 0.0
                    
                    # print("---",len(self.steering_episode_store),ep_len)
                    self.steering_diff = np.append(self.steering_diff,np.sum(abs(np.diff(self.steering_episode_store,axis=0)))/len(self.steering_episode_store))
                    self.steering_episode_store = 0.0

                    self.step_episode_store = np.append(self.step_episode_store,ep_len)
                    
                    
                    print("Episode finished. Reward: {:.2f} {} Steps".format(episode_rewards[-1], ep_len))
                    self.total_episode_reward = np.append(self.total_episode_reward,episode_rewards[-1])
                    episode_rewards.append(0.0)
                    ep_len = 0
                    
                    plt.subplot(221)
                    plt.plot(self.total_episode_reward,label='episode')
                    plt.legend()

                    plt.subplot(222)
                    plt.plot(self.throttle_mean,label='throttle')
                    plt.legend()

                    plt.subplot(223)
                    plt.plot(self.steering_diff,label='steering')
                    plt.legend()

                    plt.subplot(224)
                    plt.plot(self.step_episode_store,label='step')   
                    plt.legend()
                    plt.show()
                    plt.close()
                    
                    mb_infos_vals = self.optimize(step, writer, current_lr)
                    
                
                    # Refresh obs when using TeleopEnv
                    if is_teleop_env:
                        print("Waiting for teleop")
                        obs = self.env.wait_for_teleop_reset()
                    obs = self.env.reset()
                    obs = self.env.reset()

                # Log losses and entropy, useful for monitor training
                if len(mb_infos_vals) > 0:
                    infos_values = np.mean(mb_infos_vals, axis=0)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards) - 1
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", self.n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', "{:.2f}".format(time.time() - start_time))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", step)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
            if is_teleop_env:
                self.env.is_training = False
            # Use last batch
            print("Final optimization before saving")
            self.env.reset()
            mb_infos_vals = self.optimize(step, writer, current_lr)
        return self
