# from itertools import _Step
from logging import error
import numpy as np
import os
import time
import pandas as pd
from utils.tool import *
from copy import deepcopy
from regression_tree import RegressionDecisionTree


class GradientBoostingDecisionTree():
    def __init__(self, args):

        self.args = args
        self._name = 'GBDT'
        self.tree_list = []
        self.num_trees = args.num_trees
        self.post_pruning = args.post_pruning
        self.use_independent_ts = args.use_independent_ts
        self.min_loss = args.min_loss

        if self.post_pruning and self.use_independent_ts:
            self.prune_method = 'REP'
        elif self.post_pruning and not self.use_independent_ts:
            self.prune_method = 'PEP'
        else:
            self.prune_method = ''


    @property
    def name(self):
        return self._name


    def gbdt_predict(self, data_set):
        outpout = np.zeros(data_set.shape[0])
        for tree in self.tree_list:
            outpout += tree_predict_(data_set, tree)
        
        return outpout
            
               

    def fit(self, train_data_set:pd.DataFrame, test_data_set=None, save=False):

        '''
        fit the boosting model based on gradient descent and regression decision tree

        Args:
            data_set: for training the tree (feature sample + targets)
            test_data_set: data set used for post pruning and testing the tree (Optional)
            save: save the tree if this variable is True (Optional)
        '''

        print('\n-- Start Fitting Gradient Boosting Decision Tree --\n')

        if self.prune_method == 'PEP':
            print('-- Pessimistic Error Pruning --\n ')
        elif self.prune_method == 'REP':
            print('-- Reduce Error Pruning -- \n')
        else:
            print('-- Not Pruning --\n')
        
        targets_train = deepcopy(train_data_set.iloc[:, -1])
        
        for i in range(self.num_trees):
            decision_tree = RegressionDecisionTree(self.args)
            decision_tree.total_num_data = train_data_set.shape[0]

            tree = decision_tree._create_tree(train_data_set, 0)
            if self.prune_method == 'REP':
                assert not test_data_set is None, print('Pruning Need Independent Test Set!')
                tree = decision_tree.reduce_error_pruning(tree, test_data_set)
          
            self.tree_list.append(tree)

            gbdt_outputs = self.gbdt_predict(train_data_set.iloc[:, :-1])
            residual = targets_train - gbdt_outputs
            loss = 1 / 2 * np.power(residual, 2).sum()
            train_data_set.iloc[:, -1] = residual

            print('-- Constructing Tree: {}, Loss: {} --\n'.format(i+1, loss))

            if loss < self.min_loss:
                print('-- Loss < {}, Stop Fitting --\n'.format(self.min_loss))
                break

        if not test_data_set is None:
            assert isinstance(test_data_set, pd.DataFrame)
            gbdt_outputs_test = self.gbdt_predict(test_data_set.iloc[:, :-1])
            targets_test = test_data_set.iloc[:, -1]
            error_test = mean_squared_error(gbdt_outputs_test, targets_test)
            print('-- Test Set Mean Squared Error : {} --\n'.format(error_test))
            r2_test = r2_score(gbdt_outputs_test, targets_test)
            print('-- Test Set R^2 Score: {} --\n'.format(r2_test))

        if save:
            print('-- Save Model --\n')
            self.save()
        else:
            print('-- Not Save Model --\n')

        print('-- END --\n')


    def save(self):
        '''
        save the tree
        '''

        if not os.path.exists('./model_save/gradient_boosting_decision_tree'):
            os.makedirs('./model_save/gradient_boosting_decision_tree')
        np.save('./model_save/gradient_boosting_decision_tree/'  + time.strftime('%m_%d_%H_%M', time.localtime(time.time())) + '.npy', self._tree)


    def load(self, save_time):
        self._tree = np.load('./model_save/gradient_boosting_decision_tree/' + str(self._name) + '_' + str(save_time) + '.npy', allow_pickle=True).item()

                  

if __name__ == "__main__":

    from config import get_gbdt_config

    gbdt_parser = get_gbdt_config()
    gbdt_args  = gbdt_parser.parse_args()

    gbdt = GradientBoostingDecisionTree(gbdt_args)

    data_set = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'housing/housing.data',
                header=None,
                sep='\s+').values

    train_data_set, test_data_set = train_test_split(data_set, train_size=0.75)
    labels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    train_data_set = pd.DataFrame(train_data_set, columns=labels)
    test_data_set = pd.DataFrame(test_data_set, columns=labels)
    gbdt.fit(train_data_set, test_data_set)




