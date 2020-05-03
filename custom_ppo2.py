import time
from collections import deque

import gym
import numpy as np
from stable_baselines import logger, PPO2
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import explained_variance, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.ppo2.ppo2 import get_schedule_fn, safe_mean, swap_and_flatten

import matplotlib.pyplot as plt

episode_reward_store = np.array([])

class PPO2WithVAE(PPO2):
    """
    Custom PPO2 version.

    Notable changes:
        - optimization is done after each episode and not after n steps
    """
    def learn(self, total_timesteps, callback=None, seed=None, log_interval=1,
              tb_log_name="PPO2", reset_num_timesteps=True):
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)


        with TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) as writer:
            # self._setup_learn(seed)

            runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)
            self.episode_reward = np.zeros((self.n_envs,))
            self.total_episode_reward = np.zeros((1,))

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()
            n_timesteps = 0
            # nupdates = total_timesteps // self.n_batch
            for timestep in range(1, total_timesteps + 1):
                assert self.n_batch % self.nminibatches == 0
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - timestep / total_timesteps
                lr_now = self.learning_rate(frac)
                cliprangenow = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)
                # true_reward is the reward without discount
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()
                n_timesteps += len(obs)
                ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, batch_size):
                            end = start + batch_size
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_loss_vals.append(self._train_step(lr_now, cliprangenow, *slices, writer=writer,
                                                                 update=n_timesteps, cliprange_vf=cliprange_vf_now))
                else:  # recurrent version
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for stan_timestepsrt in range(0, self.n_envs, envs_per_batch):
                            # timestep = ((update * self.noptepochs * self.n_envs + epoch_num * self.n_envs + start) //
                            #             envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprangenow, *slices, update=n_timesteps,
                                                                 writer=writer, states=mb_states))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                # if writer is not None:
                #     self.episode_reward = total_episode_reward_logger(self.episode_reward,
                #                                                       true_reward.reshape((self.n_envs, self.n_steps)),
                #                                                       masks.reshape((self.n_envs, self.n_steps)),
                #                                                       writer, n_timesteps)

                if self.verbose >= 1 and (timestep % log_interval == 0 or timestep == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("total_timesteps", n_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()
                    self.total_episode_reward = runner.total_episode_reward

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break
                if n_timesteps > total_timesteps:
                    break

            return self


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma
        self.total_episode_reward = np.zeros((1,))


    def _run(self):
        global episode_reward_store
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        

        self.throttle_episode_store = np.array([])
        self.steering_episode_store = np.array([])

        self.throttle_min_max = np.array([])
        self.throttle_mean = np.array([])
        self.steering_diff = np.array([])

        while True:
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)
            
            # self.steering_episode_store = np.append(self.steering_episode_store,clipped_actions[0][0])
            # self.throttle_episode_store = np.append(self.throttle_episode_store,clipped_actions[0][1])
            
            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
                    
            mb_rewards.append(rewards)
            
            if self.dones:
                print("Episode finished. Reward: {:.2f} {} Steps".format(np.sum(mb_rewards), len(mb_rewards)))
                self.total_episode_reward = np.append(self.total_episode_reward,mb_rewards)
                
                print("Length of episode: {}".format(len(mb_rewards)))

                episode_reward_store = np.append(episode_reward_store,np.sum(mb_rewards))
                value_to_save={'episode_reward_store':episode_reward_store}
                np.savez("ppo_episode_reward",**value_to_save)
                
                plt.plot(episode_reward_store)
                plt.show()
                plt.close()

                # self.throttle_min_max = np.append(self.throttle_min_max,[np.amin(self.throttle_episode_store),np.amax(self.throttle_episode_store)])
                # self.throttle_mean = np.append(self.throttle_mean,np.sum(self.throttle_episode_store)/len(self.throttle_episode_store))
                # self.throttle_episode_store = 0.0
                
                # self.steering_diff = np.append(self.steering_diff,np.sum(abs(np.diff(self.steering_episode_store,axis=0)))/len(self.steering_episode_store))
                # self.steering_episode_store = 0.0
                
                # plt.subplot(221)
                # plt.plot(self.total_episode_reward,label='episode')
                # plt.legend()

                # plt.subplot(222)
                # plt.plot(self.throttle_mean,label='throttle')
                # plt.legend()

                # plt.subplot(223)
                # plt.plot(self.steering_diff,label='steering')
                # plt.legend()

                if len(mb_rewards) >= self.n_steps:
                    break

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward
