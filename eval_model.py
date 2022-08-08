from re import T
import torch
from itertools import count
import numpy as np
import pandas as pd


def _t2n(x):
    '''
    change the tensor to numpy
    '''

    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    return x


@torch.no_grad()
def eval_im(env, model, label=None):
    '''
    evaluate interpretable model in given environment

    Args:
        env: environment
        policy: any policy
        bucket_size: the number of data in data set
        running_step: if running time step in envirionment exceed this number,
                      then the environment will stop.

    Returns:
        data_set(numpy): data set with size (bucket_size, env.obs.shape + env.action.shape)
    '''

    print("-- Start Evaluate Interpretable Model --\n")
    accumulated_reward = []
    for i in range(100):
        print('\r -- Episode {} --'.format(i+1), end='', flush=True)
        episode_reward = 0
        state = env.reset()
        for t in count():
            state = np.array(state).reshape(1, -1)
            state = pd.DataFrame(state, columns=label[:-1])

            # if model.name == 'CDT':
            #     state = pd.DataFrame(state, columns=label[:-1])
            #     action = model.tree_classify(state)

            # elif model.name == 'SDT':
            #     action = model.tree_classify(state)

            # elif model.name == 'RF':
            #     state = pd.DataFrame(state, columns=label[:-1])
            #     action = model.forest_classify(state)
            
            # elif model.name == 'RDT':
            #     q_values = []
            #     for act in range(2):
            #         state_act = np.concatenate([state, np.array(act).reshape(1, -1)], axis=-1)
            #         state_act = pd.DataFrame(state_act, columns=label[:-1])
            #         q_value = model.tree_predict(state_act)
            #         q_values.append(q_value)
            #     q_values = np.array(q_values)
            #     action = np.argmax(q_values, axis=0)

            # else:
            #     # print('Wrong Decision Tree Name')

            action = model.predict(state)

            action = np.array(float(str(action, 'utf-8')))

            # Obser reward and next obs
            next_state, reward, done, _ = env.step(int(action.item()))

            episode_reward += reward

            state = next_state

            if done:
                break

        accumulated_reward.append(episode_reward)

    average_reward = np.array(accumulated_reward).mean()

    print('')
    print("\n-- Average Test Episode Reward: {} --\n".format(average_reward))
    print('-- End Testing --\n')

    return average_reward


