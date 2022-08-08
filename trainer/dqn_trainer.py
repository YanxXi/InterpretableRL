from itertools import count
from collections import namedtuple
import torch
import numpy as np
from .base_trainer import BaseTrainer
from .buffer import PrioritizedExperienceReplayBuffer, ExperienceReplayBuffer
import ray


def check(input):
    '''
    check whether the type of input is Tensor
    '''

    input = torch.from_numpy(input) if type(input) == np.ndarray else input
    return input


def _t2n(x):
    '''
    change the tensor to numpy
    '''

    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    return x


class DQNTrainer(BaseTrainer):
    def __init__(self, env, policy, args, device=torch.device("cpu")):
        super(DQNTrainer, self).__init__(args)
        self.policy = policy
        self.env = env
        self.env_name = args.env
        self.env_act_size = self.policy.act_size
        self.env_obs_size = self.policy.obs_size
        self.use_per = args.use_per
        if self.use_per:
            self.buffer = PrioritizedExperienceReplayBuffer(args)
            self.sample_generator = self.buffer.prioritized_generator
        else:
            self.buffer = ExperienceReplayBuffer(args)
            self.sample_generator = self.buffer.feedforward_generator

        self.dqn_epoch = args.dqn_epoch
        self.Transition = namedtuple(
            'transition', ['state', 'action', 'reward', 'done', 'next_state'])
        self.tpdv = dict(dtype=torch.float32, device=device)

    @torch.no_grad()
    def collect(self, state):
        '''
        collect date and insert the data into the buffer for training
        '''

        self.prep_rollout()
        state = torch.FloatTensor(state).unsqueeze(0)
        # obtain all data in one transition
        action = self.policy.get_actions(state)

        action = np.array(_t2n(action))

        return action

    def update(self, samples):
        '''
        update policy based on samples        
        '''
        if self.use_per:
            states, actions, rewards, dones, next_states, is_weights = samples
            is_weights = check(is_weights).to(**self.tpdv).reshape(-1, 1)
        else:
            states, actions, rewards, dones, next_states = samples

        states = check(states).to(**self.tpdv).reshape(-1, self.env_obs_size)
        actions = check(actions).to(**self.tpdv).reshape(-1, self.env_act_size)
        rewards = check(rewards).to(**self.tpdv).reshape(-1, 1)
        dones = check(dones).to(**self.tpdv).reshape(-1, 1)
        next_states = check(next_states).to(
            **self.tpdv).reshape(-1, self.env_obs_size)

        # compute the predicted value
        q_values_pred = self.policy.get_values(states, actions)

        # compute the target value
        with torch.no_grad():
            # max Q value of next state 
            next_max_q = self.policy.q_net_target(next_states).max(
                dim=-1)[0].unsqueeze(-1)
            q_values_tgt = rewards + (1 - dones) * self.gamma * next_max_q

        # update Q network
        td_errors = q_values_tgt - q_values_pred
        if self.use_per:
            loss = (td_errors.pow(2) * is_weights).mean()
        else:
            loss = td_errors.pow(2).mean()

        self.policy.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(
        #     self.policy.critic.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        return loss, td_errors

    def train(self):
        '''
        train the policy based on all data in the buffer, in training mode, the policy 
        will choose an action randomly.
        '''

        self.prep_train()

        # average loss in training process
        train_info = {}
        train_info['loss'] = 0

        # number of update
        update_step = 0

        # satrt to update the policy
        for _ in range(self.dqn_epoch):
            num_batch = 1
            for batch_data in self.sample_generator(num_batch=num_batch,
                                                    batch_size=self.batch_size):

                # different generator generate diffenent data
                if self.use_per:
                    batch_transitions, batch_tree_idx,  batch_IS_weight = batch_data
                else:
                    batch_transitions, _ = batch_data

                # obtain states, values, actions, rewards in transitions and reshape for nn input
                batch_states = np.array(
                    [t.state for t in batch_transitions], dtype=float)
                batch_actions = np.array(
                    [t.action for t in batch_transitions], dtype=float)
                batch_rewards = np.array(
                    [t.reward for t in batch_transitions], dtype=float)
                batch_dones = np.array(
                    [t.done for t in batch_transitions], dtype=bool)
                batch_next_states = np.array(
                    [t.next_state for t in batch_transitions], dtype=float)

                # gather samples to update
                if self.use_per:
                    samples = batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states, batch_IS_weight
                else:
                    samples = batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states

                # update the policy
                loss, td_error = self.update(samples)
                update_step += 1

                self.policy.soft_update()

                if self.use_per:
                    self.buffer.update_sum_tree(
                        batch_tree_idx, _t2n(td_error.squeeze()))

                # store train information
                train_info['loss'] += loss.item()

        # average all loss information
        for k in train_info.keys():
            train_info[k] /= update_step

        return train_info

    def run(self):

        print('\n-- Start Running --\n')

        if self.eval_in_paral:
            ray.init()

        for i in range(self.num_episode):
            episode_reward = 0
            state = self.env.reset()
            # run one episode
            for t in count():
                # interact with the environment to collent data
                action = self.collect(state)
                next_state, reward, done, _ = self.env.step(action.item())
                episode_reward += reward
                # store one step transition into buffer
                transition = self.Transition(
                    state, action, reward, done, next_state)
                self.buffer.insert(transition)
                # for next step
                state = next_state

                # at each time step, update the network by sampling a minibatch in the buffer
                if self.buffer.current_size >= self.batch_size:
                    train_info = self.train()

                if done:
                    print(
                        '-- episode: {}, timestep: {}, episode reward: {} --'.format(i+1, t+1, episode_reward))
                    break

            # log infomation
            if self.log_info:
                self.log(i+1, t+1, episode_reward)

            # evaluate the policy and check whether the problem is sloved
            if (i+1) % self.eval_interval == 0 and i+1 >= self.start_eval:
                eval_average_episode_reward = self.eval()
                if eval_average_episode_reward >= self.solved_reward:
                    print("-- Problem Solved --\n")
                    break

        if self.save_model:
            print("-- Save Model --\n")
            self.save()
        else:
            print("-- Not Save Model --\n")

        print('-- End Running --\n')

    @torch.no_grad()
    def eval(self):
        '''
        evaluate current policy, in evaluation mode, the policy will choose an action 
        according to maximum probability.
        '''

        self.prep_rollout()

        if self.eval_in_paral:
            obj_ref_list = [self.async_eval_one_episode.remote(
                self.env, self.policy) for _ in range(self.eval_episodes)]
            eval_episode_rewards = ray.get(obj_ref_list)
        else:
            eval_episode_rewards = []
            for _ in range(self.eval_episodes):
                eval_episode_reward = self.eval_one_episode(
                    self.env, self.policy)
                eval_episode_rewards.append(eval_episode_reward)

        # calculate average reward
        eval_average_episode_reward = np.array(eval_episode_rewards).mean()
        print("\n-- Average Evaluation Episode Reward: {} --\n".format(
            eval_average_episode_reward))

        return eval_average_episode_reward

    @staticmethod
    def eval_one_episode(env, policy):
        '''
        eval one episode and return episode reward
        '''
        with torch.no_grad():
            eval_state = env.reset()
            eval_episode_reward = 0
            for t in count():
                eval_state = torch.FloatTensor(eval_state).unsqueeze(0)
                eval_action = policy.act(eval_state)
                eval_action = _t2n(eval_action)

                eval_next_state, eval_reward, eval_done, _ = env.step(
                    eval_action.item())
                eval_episode_reward += eval_reward
                eval_state = eval_next_state
                if eval_done:
                    env.close()
                    break

        return eval_episode_reward

    @staticmethod
    @ray.remote(num_returns=1)
    def async_eval_one_episode(env, policy):
        '''
        eval one episode asynchronously and return episode reward
        '''
        with torch.no_grad():
            eval_state = env.reset()
            eval_episode_reward = 0
            for t in count():
                eval_state = torch.FloatTensor(eval_state).unsqueeze(0)
                eval_action = policy.act(eval_state)
                eval_action = _t2n(eval_action)

                eval_next_state, eval_reward, eval_done, _ = env.step(
                    eval_action)
                eval_episode_reward += eval_reward
                eval_state = eval_next_state
                if eval_done:
                    env.close()
                    break

        return eval_episode_reward

    def prep_rollout(self):
        '''
        turn to eval mode
        '''

        self.policy.q_net.eval()

    def prep_train(self):
        '''
        turn to train mode
        '''

        self.policy.q_net.train()
