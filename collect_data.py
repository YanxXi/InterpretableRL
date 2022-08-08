from re import T
import torch
from itertools import count
import numpy as np
import argparse
from config import *
import gym
import os
import time


def _t2n(x):
    '''
    change the tensor to numpy
    '''

    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    return x


@torch.no_grad()
def collect_ppo_data(env, policy, target_label, bucket_size):
    '''
    generate data in given environment according to given policy

    Args:
        env: environment
        policy: any policy
        bucket_size: the number of data in data set
        running_step: if running time step in envirionment exceed this number,
                      then the environment will stop.

    Returns:
        data_set(numpy): data set with size (bucket_size, env.obs.shape + env.action.shape)
    '''

    policy.actor.eval()
    policy.critic.eval()
    obs_bucket = []
    target_bucket = []
    for i in count():
        state = env.reset()
        rnn_states_actor = np.zeros((1, policy.args.recurrent_hidden_layers, policy.args.recurrent_hidden_size),
                                    dtype=np.float32)
        rnn_states_critic = np.zeros((1, policy.args.recurrent_hidden_layers, policy.args.recurrent_hidden_size),
                                     dtype=np.float32)
        for t in count():
            obs_bucket.append(state)
            state = torch.FloatTensor(state).unsqueeze(0)
            action, rnn_states_actor = policy.act(state, rnn_states_actor)
            action = np.array(_t2n(action))
            if target_label == 'action':
                target_bucket.append(action.reshape(-1))
            elif target_label == 'value':
                value, rnn_states_critic = policy.get_values(
                    state, rnn_states_critic)
                value = np.array(_t2n(value))
                target_bucket.append(value.reshape(-1))
            else:
                print('Wrong Target Type')

            # Obser reward and next obs
            next_state, reward, done, _ = env.step(action.item())

            if len(obs_bucket) >= bucket_size:
                env.close()
                obs_set = np.array(obs_bucket)
                target_set = np.array(target_bucket)
                data_set = np.concatenate([obs_set, target_set], axis=-1)
                return data_set

            state = next_state

            if done:
                break


@torch.no_grad()
def collect_ddpg_data(env, policy, target_label, bucket_size):
    '''
    generate data in given environment according to given policy

    Args:
        env: environment
        policy: any policy
        bucket_size: the number of data in data set
        running_step: if running time step in envirionment exceed this number,
                      then the environment will stop.

    Returns:
        data_set(numpy): data set with size (bucket_size, env.obs.shape + env.action.shape)
    '''

    policy.actor.eval()
    policy.critic.eval()
    obs_bucket = []
    target_bucket = []
    for i in count():
        state = env.reset()
        for t in count():
            obs_bucket.append(state)
            state = torch.FloatTensor(state).unsqueeze(0)
            action, rnn_states_actor = policy.act(state)
            action = np.array(_t2n(action))
            if target_label == 'action':
                target_bucket.append(action.reshape(-1))
            else:
                print('Wrong Target Type')

            # Obser reward and next obs
            next_state, reward, done, _ = env.step(action)

            if len(obs_bucket) >= bucket_size:
                env.close()
                obs_set = np.array(obs_bucket)
                target_set = np.array(target_bucket)
                data_set = np.concatenate([obs_set, target_set], axis=-1)
                return data_set

            state = next_state

            if done:
                break


@torch.no_grad()
def collect_dqn_data(env, policy, target_label, bucket_size):
    '''
    generate data in given environment according to given policy

    Args:
        env: environment
        policy: any policy
        bucket_size: the number of data in data set
        running_step: if running time step in envirionment exceed this number,
                      then the environment will stop.

    Returns:
        data_set(numpy): data set with size (bucket_size, env.obs.shape + env.action.shape)
    '''

    policy.q_net.eval()
    obs_bucket = []
    target_bucket = []
    for i in count():
        state = env.reset()
        for t in count():
            obs_bucket.append(state)
            state = torch.FloatTensor(state).unsqueeze(0)
            action = policy.act(state)
            action = _t2n(action)
            if target_label == 'action':
                target_bucket.append(action.reshape(-1))
            elif target_label == 'value':
                q_value = policy.get_values(state, action.unsqueeze(0))
                obs_bucket.append(np.concatenate([_t2n(state).reshape(-1), action.reshape(-1)], axis=-1))
                target_bucket.append(_t2n(q_value).reshape(-1))
            else:
                print('Wrong Target Type')

            # Obser reward and next obs
            next_state, reward, done, _ = env.step(action.item())

            if len(obs_bucket) >= bucket_size:
                env.close()
                feature_set = np.array(obs_bucket)
                target_set = np.array(target_bucket)
                data_set = np.concatenate([feature_set, target_set], axis=-1)
                return data_set

            state = next_state

            if done:
                break

        
if __name__ == '__main__':
    # get train config
    ppo_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
    ppo_parser = get_ppo_config(ppo_parser)

    ddpg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
    ddpg_parser = get_ddpg_config(ddpg_parser)

    dqn_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
    dqn_parser = get_dqn_config(dqn_parser)

    parser = argparse.ArgumentParser(
        description='ALL Congfig For Collecting Data')
    parser.add_argument('--env', type=str, required=True,
                        help='environment used for testing')
    parser.add_argument('--save-time', type=str,
                        required=True, help='time to save the model')
    parser.add_argument('--target-label', type=str,
                        required=True, help='target is action or value')
    parser.add_argument('--bucket-size', type=int, default=1e4,
                        help='size of the data bucket')

    subparsers = parser.add_subparsers(
        help='choose one algorithm to start corresponding testing function', dest='algorithm')

    subparsers.add_parser('ppo', help='ppo algorithm', parents=[ppo_parser])
    ppo_parser.set_defaults(func=collect_ppo_data)

    subparsers.add_parser('ddpg', help='ddpg algorithm', parents=[ddpg_parser])
    ddpg_parser.set_defaults(func=collect_ddpg_data)

    subparsers.add_parser('dqn', help='dqn algorithm', parents=[dqn_parser])
    dqn_parser.set_defaults(func=collect_dqn_data)


    # args = parser.parse_args(
        # '--env LunarLander-v2 --target-label action --save-time 01_21_21_12 ppo --use-recurrent-layer'.split(' '))

    # args = parser.parse_args(
    #     '--env CartPole-v0 --target-label value --save-time 02_12_17_10 dqn'.split(' '))

    args = parser.parse_args()

    env = env = gym.make(args.env)
    obs_space = env.observation_space
    act_sapce = env.action_space

    # train model
    if args.algorithm == 'ppo':
        from algorithm.ppo import PPOPolicy
        policy = PPOPolicy(args, obs_space, act_sapce)
        policy.load_param(args.env, args.save_time)
        data_set = collect_ppo_data(
            env, policy, args.target_label, args.bucket_size)
    elif args.algorithm == 'ddpg':
        from algorithm.ddpg import DDPGPolicy
        policy = DDPGPolicy(args, obs_space, act_sapce)
        policy.load_param(args.env, args.save_time)
        data_set = collect_ddpg_data(
            env, policy, args.target_label, args.bucket_size)
    elif args.algorithm == 'dqn':
        from algorithm.dqn import DQNPolicy
        policy = DQNPolicy(args, obs_space, act_sapce)
        policy.load_param(args.env, args.save_time)
        data_set = collect_dqn_data(
            env, policy, args.target_label, args.bucket_size)
    else:
        print('Not Exit This Algorithm!')

    save_dir = './data/' + str(args.env) + '/' + str(args.algorithm) + '/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_time = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))

    save_path = save_dir + str(args.target_label) + '_' + save_time + '.npy'

    np.save(save_path, data_set)
