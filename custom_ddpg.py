# Original author: Roma Sokolkov
# Edited by Antonin Raffin
import time

import numpy as np
from mpi4py import MPI

from stable_baselines import logger
from stable_baselines.ddpg.ddpg import DDPG
from stable_baselines.common import TensorboardWriter
import cv2
from config import Debug_RL_Input
import matplotlib.pyplot as plt

class DDPGWithVAE(DDPG):
    """
    Custom DDPG version in order to work with donkey car env.
    It is adapted from the stable-baselines version.

    Changes:
    - optimization is done after each episode
    - more verbosity.
    """
    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=1, tb_log_name="DDPG", print_freq=100):
        with TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:

            rank = MPI.COMM_WORLD.Get_rank()
            # we assume symmetric actions.
            assert np.all(np.abs(self.env.action_space.low) == self.env.action_space.high)

            self.total_episode_reward = np.zeros((1,))
            self.throttle_min_max = np.array([])
            self.throttle_mean = np.array([])
            self.steering_diff = np.array([])

            self.throttle_episode_store = np.array([])
            self.steering_episode_store = np.array([])

            self.step_episode_store = np.array([])

            with self.sess.as_default(), self.graph.as_default():
                # Prepare everything.
                self._reset()
                episode_reward = 0.
                episode_step = 0
                episodes = 0
                step = 0
                total_steps = 0

                start_time = time.time()

                actor_losses = []
                critic_losses = []
                should_return = False

                while True:
                    obs = self.env.reset()
                    # Rollout one episode.
                    while True:
                        if total_steps >= total_timesteps:
                            if should_return:
                                return self
                            should_return = True
                            break

                        # Predict next action.
                        action, q_value = self._policy(obs, apply_noise=True, compute_q=True)
                        if self.verbose >= 2:
                            print(action)
                        assert action.shape == self.env.action_space.shape

                        # Execute next action.
                        new_obs, reward, done, info = self.env.step(action * np.abs(self.action_space.low))

                        #TODO: Writer->total_episode_reward_logger

                        step += 1
                        total_steps += 1
                        if rank == 0 and self.render:
                            self.env.render()
                        episode_reward += reward
                        episode_step += 1
                        
                        if print_freq > 0 and episode_step % print_freq == 0 and episode_step > 0:
                            print("{} steps".format(episode_step))

                        # Book-keeping.
                        if Debug_RL_Input:
                            cv2.imwrite("TEST\{}.jpg".format(step),obs)
                        
                        if(action.shape[0]==2):
                            self.steering_episode_store = np.append(self.steering_episode_store,action[0])
                            self.throttle_episode_store = np.append(self.throttle_episode_store,action[1])
                        else:
                            self.steering_episode_store = np.append(self.steering_episode_store,action)
                            self.throttle_episode_store = np.append(self.throttle_episode_store,0.4)
                        self._store_transition(obs, action, reward, new_obs, done)

                        obs = new_obs
                        if callback is not None:
                            # Only stop training if return value is False, not when it is None. This is for backwards
                            # compatibility with callbacks that have no return statement.
                            if callback(locals(), globals()) is False:
                                return self

                        if done:
                            print("Episode finished. Reward: {:.2f} {} Steps".format(episode_reward, episode_step))

                            self.total_episode_reward = np.append(self.total_episode_reward,episode_reward)

                            self.throttle_min_max = np.append(self.throttle_min_max,[np.amin(self.throttle_episode_store),np.amax(self.throttle_episode_store)])                  
                            self.throttle_mean = np.append(self.throttle_mean,np.sum(self.throttle_episode_store)/len(self.throttle_episode_store))
                            self.throttle_episode_store = 0.0

                            self.steering_diff = np.append(self.steering_diff,np.sum(abs(np.diff(self.steering_episode_store,axis=0)))/len(self.steering_episode_store))
                            self.steering_episode_store = 0.0

                            self.step_episode_store = np.append(self.step_episode_store,episode_step)
                            
                            plt.subplot(221)
                            plt.plot(self.total_episode_reward,label='reward')
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
                            
                            # Episode done.
                            episode_reward = 0.
                            episode_step = 0
                            episodes += 1

                            self._reset()
                            # self.env.reset()
                            # self.env.reset()
                            obs = self.env.reset()
                            obs = self.env.reset()
                            # Finish rollout on episode finish.
                            break

                    print("Rollout finished.")

                    # Train DDPG.
                    actor_losses = []
                    critic_losses = []
                    train_start = time.time()
                    for t_train in range(self.nb_train_steps):
                        critic_loss, actor_loss = self._train_step(step, writer, log=t_train == 0)
                        # critic_loss, actor_loss = self._train_step(0, None, log=t_train == 0)
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)
                        self._update_target_net()
                    print("DDPG training duration: {:.2f}s".format(time.time() - train_start))

                    mpi_size = MPI.COMM_WORLD.Get_size()
                    # Log stats.
                    # XXX shouldn't call np.mean on variable length lists
                    duration = time.time() - start_time
                    stats = self._get_stats()
                    combined_stats = stats.copy()
                    combined_stats['train/loss_actor'] = np.mean(actor_losses)
                    combined_stats['train/loss_critic'] = np.mean(critic_losses)
                    combined_stats['total/duration'] = duration
                    combined_stats['total/steps_per_second'] = float(step) / float(duration)
                    combined_stats['total/episodes'] = episodes

                    combined_stats_sums = MPI.COMM_WORLD.allreduce(
                        np.array([as_scalar(x) for x in combined_stats.values()]))
                    combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

                    # Total statistics.
                    combined_stats['total/steps'] = step

                    for key in sorted(combined_stats.keys()):
                        logger.record_tabular(key, combined_stats[key])
                    logger.dump_tabular()
                    logger.info('')


def as_scalar(scalar):
    """
    check and return the input if it is a scalar, otherwise raise ValueError

    :param scalar: (Any) the object to check
    :return: (Number) the scalar if x is a scalar
    """
    if isinstance(scalar, np.ndarray):
        assert scalar.size == 1
        return scalar[0]
    elif np.isscalar(scalar):
        return scalar
    else:
        raise ValueError('expected scalar, got %s' % scalar)
