from ast import arg
import torch
from torch.autograd.grad_mode import F
import numpy as np
import os
import csv


class BaseTrainer():
    def __init__(self, args, device=torch.device("cpu")):
        self.args = args
        self.num_episode = int(args.num_episode)
        self.gamma = args.gamma
        self.device = device
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.eval_interval = args.eval_interval
        self.eval_episodes = args.eval_episodes
        self.buffer = None
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.max_grad_norm = args.max_grad_norm
        self.eval_in_paral = args.eval_in_paral
        self.log_info = args.log_info
        self.save_model = args.save_model
        self.solved_reward = args.solved_reward
        self.start_eval = args.start_eval

    def update(self):
        '''
        update policy based on samples
        '''

        pass

    def train(self):
        '''
        train the policy based on all data in the buffer, in training mode, the policy 
        will choose an action randomly.
        '''

        pass

    def run(self):
        pass

    @torch.no_grad()
    def eval(self):
        '''
        evaluate current policy, in evaluation mode, the policy will choose an action 
        according to maximum probability.
        '''

        pass

    def save(self):
        '''
        save parameter in policy
        '''
        save_path = './model_save/' + \
            str(self.policy.name) + '/' + str(self.env_name) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            # os.makedirs('./model_save/ppo/' + self.env_name + '/actor')
            # os.makedirs('./model_save/ppo/' + self.env_name + 'critic')
        self.policy.save_param(save_path)

    def log(self, episode, time_step, episode_reward):
        '''
        log time step and episode reward
        '''
        log_dir = 'rl_results/' + str(self.policy.name) + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = log_dir + self.env_name + '.cvs'
        if episode == 1:
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time_step', 'episode_reward'])

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time_step, episode_reward])
