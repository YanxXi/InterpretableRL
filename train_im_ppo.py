from cgi import test
from copy import deepcopy
from matplotlib import projections
from matplotlib.pyplot import pink
import numpy as np
from itertools import count
import pandas as pd
import torch
from utils.tool import train_test_split
from eval_model import eval_im
from collect_data import collect_ppo_data
import joblib
import matplotlib.pyplot as plt


def _t2n(x):
    '''
    change the tensor to numpy
    '''

    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    return x


def collect_cdt_data(env, model, label, bucket_size):
    '''
    collect classification tree data

    Args:
        env: environment
        policy: any policy
        bucket_size: the number of data in data set
        running_step: if running time step in envirionment exceed this number,
                      then the environment will stop.

    Returns:
        data_set(numpy): data set with size (bucket_size, env.obs.shape + env.action.shape)
    '''

    feature_bucket = []
    target_bucket = []
    for i in count():
        state = env.reset()
        for t in count():
            feature_bucket.append(state)

            state = np.array(state).reshape(1, -1)
            state = pd.DataFrame(state, columns=label[:-1])
            action = model.predict(state)

            action = np.array(float(str(action, 'utf-8')))

            target_bucket.append(action.reshape(-1))

            # Obser reward and next obs
            next_state, reward, done, _ = env.step(int(action.item()))

            if len(feature_bucket) >= bucket_size:
                env.close()
                feature_set = np.array(feature_bucket)
                target_set = np.array(target_bucket)
                data_set = np.concatenate([feature_set, target_set], axis=-1)
                return data_set

            state = next_state

            if done:
                break


def label_data_set(rl_policy, data_set):
    obs_bucket = []
    action_bucket = []
    different_action = 0
    for *state, ml_action in data_set[:]:
        obs_bucket.append(state)
        rnn_states_actor = np.zeros((1, rl_policy.args.recurrent_hidden_layers, rl_policy.args.recurrent_hidden_size),
                                    dtype=np.float32)
        state = torch.FloatTensor(state).unsqueeze(0)
        rl_action, _ = rl_policy.act(state, rnn_states_actor)
        rl_action = _t2n(rl_action).reshape(-1)
        action_bucket.append(rl_action)

    new_data_set = np.concatenate(
        (np.array(obs_bucket), np.array(action_bucket)), axis=-1)

    return new_data_set


def merge_data_set(data_set1, data_set2):
    return np.concatenate((data_set1, data_set2), axis=0)


if __name__ == "__main__":
    import gym
    from config import *
    # from classification_tree import ClassificationDecisionTree
    from sklearn import tree
    env = gym.make('LunarLander-v2')
    obs_space = env.observation_space
    act_sapce = env.action_space

    # classification_decision_tree.load('03_28_10_55')
    labels = ['x_position', 'y_position', 'x_speed', 'speed',
              'angle', 'angular speed', 'leg1', 'leg2', 'action']

    MODE = "dagger"
    # MODE = "bc"

    M = 4000
    N = 10

    # get train config
    ppo_parser = argparse.ArgumentParser()
    parser = get_ppo_config(ppo_parser)

    args = parser.parse_args('--use-recurrent-layer'.split(' '))

    clf_list = []
    avg_reward_list = []

    # train model
    from algorithm.ppo import PPOPolicy
    ppo_policy = PPOPolicy(args, obs_space, act_sapce)
    ppo_policy.load_param('LunarLander-v2', '01_21_21_12')

    # print("\n-- Start Train Interpretable Model --\n")

    print("-- Collect Data By Reforcement Learnning Algorigtm --\n")

    data_set = collect_ppo_data(
        env, ppo_policy, 'action', M)

    np.random.shuffle(data_set)
    train_data_set = pd.DataFrame(data_set, columns=labels)

    y = list(train_data_set['action'])
    y = [str(target).encode('utf-8') for target in y]
    features = list(train_data_set.columns)

    features.remove('action')
    X = train_data_set.loc[:, features]

    # train explainable model
    clf = tree.DecisionTreeClassifier().fit(X, y)

    clf_list.append(deepcopy(clf))

    # classification_decision_tree.fit(train_data_set, test_data_set, False)

    avg_reward = eval_im(env, clf, labels)
    avg_reward_list.append(avg_reward)

    retrain_step = 1

    while True:

        retrain_step += 1

        print("-- ReTrainning Round: {} --\n".format(retrain_step))

        old_data_set = data_set

        if MODE == "bc":

            print("-- Collect Data By Reinforcement Learning algorithm --\n")
            new_data_set = collect_ppo_data(env, ppo_policy, 'action', M)

        if MODE == "dagger":

            print("-- Collect Data By Classification Decision Tree --\n")
            new_data_set = collect_cdt_data(env, clf, labels, M)
            print("-- Label Data By Reforcement Learnning Algorithm --\n")
            new_data_set = label_data_set(ppo_policy, new_data_set)

        print("-- Merge Old and New Data Set --\n")

        data_set = merge_data_set(old_data_set, new_data_set)

        np.random.shuffle(data_set)

        train_data_set = pd.DataFrame(data_set, columns=labels)

        y = list(train_data_set['action'])
        y = [str(target).encode('utf-8') for target in y]

        features = list(train_data_set.columns)

        features.remove('action')
        X = train_data_set.loc[:, features]

        print("-- Train Decision Tree --\n")
        clf = tree.DecisionTreeClassifier().fit(X, y)
        # classification_decision_tree = ClassificationDecisionTree(cdt_args)
        # classification_decision_tree.fit(train_data_set, test_data_set, False)

        avg_reward = eval_im(env, clf, labels)
        clf_list.append(deepcopy(clf))
        avg_reward_list.append(avg_reward)

        if retrain_step >= N:
            break

    print('-- Train Done --\n')

    avg_reward_np = np.array(avg_reward_list)

    best_clf_idx = np.argmax(avg_reward_np)
    best_clf_performance = avg_reward_list[best_clf_idx]

    best_clf = clf_list[best_clf_idx]

    train_data_set, test_data_set = train_test_split(
        data_set, train_size=0.5)
    train_data_set = pd.DataFrame(train_data_set, columns=labels)
    test_data_set = pd.DataFrame(test_data_set, columns=labels)

    features = list(train_data_set.columns)
    features.remove('action')
    targets = test_data_set['action'].to_numpy()
    X = test_data_set.loc[:, features]

    predictions = best_clf.predict(X)

    predictions = np.array([float(str(prediction, 'utf-8')) for prediction in predictions])

    acc = np.equal(predictions, targets).sum() / len(targets)

    print("-- Accuracy of best model is: {}, with performance {} --".format(acc, best_clf_performance))

    if MODE == "bc":
        np.save('cart_lunarlander_avg2.npy', avg_reward_np)

    if MODE == "dagger":
        np.save('cart_lunarlander_avg1.npy', avg_reward_np)

        joblib.dump(best_clf, 'cart_lunarlander.pkl')

        import pybaobabdt
        ax = pybaobabdt.drawTree(best_clf, features=labels[:-1])

        plt.show()

