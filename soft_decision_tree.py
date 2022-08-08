import os
import time
import pickle
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import T
import torch.optim as optim
import numpy as np
from trainer.ppo_trainer import _t2n
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import copy

def check(input):
    input = torch.from_numpy(input) if type(input) == np.ndarray else input
    return input


class MLPLayer(nn.Module):
    def __init__(self, input_dim:int, hidden_size:str):
        super(MLPLayer, self).__init__()

        fc = []
        self.act_func = nn.ReLU()
        self._hidden_size = [input_dim] + list(map(int, hidden_size.split(' ')))
        for i in range(len(self._hidden_size)-1):
            fc.extend([nn.Linear(self._hidden_size[i], self._hidden_size[i+1]), self.act_func])
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        return self.fc(x)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def output_dim(self):
        return self._hidden_size[-1]     


class InnerNode():

    def __init__(self, depth, args, input_dim):
        self.args = args
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 1)
        beta = torch.randn(1)
        self.beta = nn.Parameter(beta)
        self.leaf = False
        self.prob = None
        self.path_prob = None
        self.lmbda = self.args.lmbda * 2 ** (-depth)
        self.depth = depth
        self.build_child(depth)


    def reset(self):
        self.prob = None
        self.path_prob = None
        self.left.reset()
        self.right.reset()


    def build_child(self, depth):
        if depth < self.args.max_depth:
            self.left = InnerNode(depth+1, self.args, self.input_dim)
            self.right = InnerNode(depth+1, self.args, self.input_dim)
        else :
            self.left = LeafNode(self.args)
            self.right = LeafNode(self.args)


    def forward(self, x):

        return torch.sigmoid(self.beta * self.fc(x))



    def cal_prob(self, x, path_prob):
        '''
        obtain the probability reaching the leaf node and the distribution of leaf node under current node
        '''

        # it will contain all leaf nodes, and each leaf node list will contain all batch samples
        leaf_accumulator = []
        # probability of selecting right node
        prob = self.forward(x)
        # store probability and path probability of this inner node
        self.prob = prob
        self.path_prob = path_prob

        left_leaf_accumulator = self.left.cal_prob(x, path_prob * (1 - prob))
        right_leaf_accumulator = self.right.cal_prob(x, path_prob * prob)
        leaf_accumulator.extend(left_leaf_accumulator)
        leaf_accumulator.extend(right_leaf_accumulator)
        return leaf_accumulator


    def get_penalty(self):
        '''
        obtain penalties of all inner node below current node and
        the penalty of current node 
        '''
        penalty_list = []

        if not self.left.leaf or not self.right.leaf:
            left_penalty = self.left.get_penalty()
            right_penalty = self.right.get_penalty()
            penalty_list.extend(left_penalty)
            penalty_list.extend(right_penalty)
        
        penalty = (torch.sum(self.prob * self.path_prob) / torch.sum(self.path_prob), self.lmbda)
        penalty_list.append(penalty)

        return penalty_list



class LeafNode():
    def __init__(self, args):
        self.args = args
        self.param = torch.randn(self.args.output_dim)
        self.param = nn.Parameter(self.param)
        self.leaf = True
        self.softmax = nn.Softmax(dim=1)


    def forward(self):
        return self.softmax(self.param.view(1,-1))


    def reset(self):
        pass


    def cal_prob(self, x, path_prob):
        '''
        obtain the probability reaching this leaf node and the distribution in this leaf node 
        '''

        Q = self.forward()
        Q = Q.expand((path_prob.size()[0], self.args.output_dim))
        return [[path_prob, Q]]



class SoftDecisionTree(nn.Module):

    def __init__(self, args, device=torch.device("cpu")):
        super(SoftDecisionTree, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.output_dim = args.output_dim 
        self.epoch = args.epoch
        self.max_depth = args.max_depth
        self.num_leaf = 2 ** self.max_depth
        self.acc_threshold = 0.96
        self.use_mlp_layer = False
        self._name = 'SDT'
       
        # create mlp layer to extract features
        if self.args.mlp_hidden_size:
            self.use_mlp_layer = True
            self.mlp = MLPLayer(args.input_dim, args.mlp_hidden_size)
            input_dim = self.mlp.output_dim
          
        else:
            input_dim = args.input_dim

        # create the empty tree
        self.root = InnerNode(1, self.args, input_dim)
        self.register_parameters()

        self.optimizer = optim.SGD(self.parameters(), lr=args.lr, momentum=args.momentum)
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.to(device)

        # for param in self.parameters():
        #     print(type(param.data), param.size())


    @property
    def name(self):
        return self._name
        

    def cal_loss(self, batch_samples, target_dist):
        '''
        calculate loss for updating
        '''

        batch_size = batch_samples.shape[0]
        
        path_prob_init = torch.ones((batch_size, 1)).to(**self.tpdv)

        if self.use_mlp_layer:
            batch_samples = self.mlp(batch_samples)

        # the first list contains all leaf nodes, the second list contains the path probability and distribution of all batch samples
        leaf_accumulator = self.root.cal_prob(batch_samples, path_prob_init)
        loss = 0.
        
        # enumerate all leaf nodes 
        for (path_prob, Q) in leaf_accumulator:
            TQ = -torch.bmm(target_dist.view(batch_size, 1, self.args.output_dim), torch.log(Q).view(batch_size, self.args.output_dim, 1)).view(-1,1)
            loss += path_prob * TQ

        loss = loss.mean()
        node_penalties = self.root.get_penalty()
        C = 0.
        for (penalty, lmbda) in node_penalties:
            C += -lmbda * 0.5 * (torch.log(penalty) + torch.log(1-penalty))

        # reset all nodes
        self.root.reset() 

        return loss + C


    def register_parameters(self):
        '''
        register all parameters in the soft decision tree
        '''

        nodes = [self.root]
        self.module_list = nn.ModuleList()
        self.param_list = nn.ParameterList()
        while nodes:
            node = nodes.pop(0)
            if node.leaf:
                param = node.param
                self.param_list.append(param)
            else:
                fc = node.fc
                beta = node.beta
                nodes.append(node.right)
                nodes.append(node.left)
                self.param_list.append(beta)
                self.module_list.append(fc)


    def train_(self, data_set):
        '''
        train the soft decision tree
        '''

        self.train()
        iteration = 0
        for batch_data in BatchSampler(SubsetRandomSampler(data_set), batch_size=self.batch_size, drop_last=True):
            batch_data = np.array(batch_data)
            batch_size = batch_data.shape[0]
            data = batch_data[..., :-1]
            target = batch_data[..., -1]

        # for data, target in data_set:
        #     batch_size = data.shape[0]
            
            # conver to tensor
            samples = check(data).to(**self.tpdv).view(batch_size, -1)
            target_ = check(target).to(dtype=torch.int64).view(-1, 1)

            # convert int target to one-hot vector
            target_dist = torch.zeros((batch_size, self.output_dim)).to(**self.tpdv)
            target_dist.scatter_(1, target_, 1)

            loss = self.cal_loss(samples, target_dist)

            # update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            iteration += 1

            # log information
            if iteration % self.args.log_interval == 0:

                output = self.tree_classify(_t2n(samples), normalize=False)
                accuracy = self.cal_accuracy(_t2n(output), target)

                print('-- Iteration: {} [{}/{}]\tLoss: {:.4f}, Accuracy: {:.2f}% --\n'.format(
                    iteration,
                    iteration * len(samples), len(data_set),
                    loss.item(),
                    100 * accuracy))


    @torch.no_grad()
    def test_(self, data_set):
        '''
        test the soft decision tree
        '''

        self.eval()
        num_correct = 0
        # for data, target in data_set:
        #     batch_size = data.shape[0]

        for batch_data in BatchSampler(SubsetRandomSampler(data_set), batch_size=self.batch_size, drop_last=False):
            batch_data = np.array(batch_data)
            batch_size = batch_data.shape[0]
            data = batch_data[..., :-1]
            target = batch_data[..., -1]

            output = self.tree_classify(data)

            assert len(output) == batch_size

            num_correct += self.cal_accuracy(output, target) * batch_size
        accuracy = num_correct / len(data_set)
        print('-- Test Set: Accuracy: {:.2f}% --\n'.format(100 * accuracy))
        
        return accuracy


    @torch.no_grad()
    def classify(self, tree, x, path_prob):
        '''
        classify sample according to maximum probability
        ''' 

        path_probs = []
        output_dist = []
        if tree.leaf:
            dist = tree.forward()
            output_dist.append(np.array(_t2n(dist)))
            path_probs.append(_t2n(path_prob))

        elif not tree.leaf:
            prob = tree.forward(x)
            left_dist, left_probs = self.classify(tree.left, x, path_prob * (1 - prob))
            right_dist, right_probs = self.classify(tree.right, x, path_prob * prob)
            output_dist.extend(left_dist)
            output_dist.extend(right_dist)
            path_probs.extend(left_probs)
            path_probs.extend(right_probs)
            
        return output_dist, path_probs
     

    @torch.no_grad()
    def tree_classify(self, data_set:np.ndarray, normalize=True):
        '''
        classify the data set based on soft decision tree

        Args:
            normalize (bool): whether to normalize the data set (in trainning mode, the data has been normalized, so set this flag to False)
        '''

        assert isinstance(data_set, np.ndarray)

        self.eval()
        num_data = len(data_set)

        # normalize the data
        if normalize:
            data_set = (data_set - self.mu) / self.std

        data_set = check(data_set).to(**self.tpdv).view(num_data, -1)
        
        if self.use_mlp_layer:
            data_set = self.mlp(data_set)

        # outputs = [self.classify(self.root, vec, 1) for vec in data_set]
        path_prob_init = torch.ones((num_data, 1)).to(**self.tpdv)
        leaf_dist, path_probs = self.classify(self.root, data_set, path_prob_init)
        # distributions of all leaf nods  (number of leaf nodes, output_dim)
        leaf_dist = np.array(leaf_dist).reshape(self.num_leaf, self.output_dim)
        # path probability of reaching all leaf nodes (number of leaf nodes, number of data)
        path_probs =  np.array(path_probs).reshape(self.num_leaf, num_data).T
        # index of the maximum probability of reaching the leaf node 
        max_index = np.argmax(path_probs, axis=1)
        # distributions with maximum path probabilities
        dist_output = leaf_dist[max_index]
        # the output class of the tree
        tree_output = np.argmax(dist_output, axis=1)

        return tree_output



    def cal_accuracy(self, tree_output, target):
        '''
        calculate the classification accuracy
        '''

        assert len(tree_output) == len(target)
        tree_output = np.array(tree_output) if type(tree_output) == np.ndarray else tree_output
        target = np.array(target) if type(target) == np.ndarray else target
        accuracy = np.sum(np.equal(tree_output, target)) / len(tree_output)
        return accuracy

    
    def save(self):
        '''
        save the parameters in soft decision tree
        '''

        if not os.path.exists('./model_save/soft_decision_tree'):
            os.makedirs('model_save/soft_decision_tree')

        parameters = {'net':self.state_dict(), 'mu': self.mu, 'std': self.std}

        torch.save(parameters, './model_save/soft_decision_tree/' + time.strftime('%m_%d_%H_%M', time.localtime(time.time())) + '.pkl')


    def load(self, save_time):
        '''
        load the parameters to the soft decision tree
        '''

        parameters = torch.load('./model_save/soft_decision_tree/' + str(save_time) + '.pkl')
        
        state_dict = parameters['net']
        self.load_state_dict(state_dict)
        self.mu = parameters['mu']
        self.std = parameters['std'] 


    def fit(self, train_data_set:np.ndarray, test_data_set:np.ndarray, save=False):
        '''
        fit the soft decision tree based on train_data_set, test the tree based on test data set.

        Args:
            train_data_set: data set used for trainning the soft decision tree
            test_data_set: data set used for testing the soft decision tree
            save (bool): whether to save the parameters in soft decision tree
        '''

        old_test_accuracy = 0
        print('\n-- Start Fitting Soft Decision Tree --\n')

        for i in range(self.epoch):
            print('-- Train Epoch: {} --\n'.format(i+1))
            self.train_(train_data_set)
            test_accuracy = self.test_(test_data_set)
            # if test accuracy reach the given threshold then stop fitting
            if test_accuracy > self.acc_threshold:
                print('-- Test Set Accuracy Exceeds Threshold --')
                break
            # if test accuracy no longer improve then stop fitting
            elif (test_accuracy - old_test_accuracy) <= 0:
                print('-- Test Set Accuracy Stop Improving --')
                break

            old_test_accuracy = test_accuracy
        
        if save:
            print('-- Save Model --\n')
            self.save()
        else:
            print('-- Not Save Model --\n')

        print('-- END --\n')


    def pre_normalize(self, data_set):
        '''
        normalize the data set and return normalized data set, mu and std
        '''

        self.mu = np.mean(data_set[..., :-1], axis=0)
        self.std = np.std(data_set[..., :-1], axis=0)
        data_set[..., :-1] = (data_set[..., :-1] - self.mu) / self.std

        return data_set

    




if __name__ == "__main__":
    from torchvision import datasets, transforms
    from config import get_sdt_config
    parser = get_sdt_config()
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    kwargs = {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    
    sdt_tree = SoftDecisionTree(args)
    sdt_tree.fit(train_loader, test_loader)


