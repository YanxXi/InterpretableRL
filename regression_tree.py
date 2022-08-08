import numpy as np
import os
import time
from numpy.core.fromnumeric import mean
from plot import _plot_tree
import pandas as pd
from utils.tool import *
from decision_tree import BaseDecisionTree

# TODO: seperate the input data set to "discreate" and "continuous" type
class RegressionDecisionTree(BaseDecisionTree):
    def __init__(self, args):
        super(RegressionDecisionTree, self).__init__(args)

        self.lower_error = args.lower_error
        self._name = 'RDT'
        self._total_num_data = None


    @property
    def name(self):
        return self._name

    @property
    def total_num_data(self):
        return self._total_num_data
    @total_num_data.setter
    def total_num_data(self, value):
        self._total_num_data = value


    def cal_leaf_value(self, targets):
        '''
        calculate the predicted value of the leaf node, we take mean value of the targets in leaf node
        '''
        
        leaf_value = targets.mean()

        return leaf_value


    def cal_leaf_error(self, data_set, leaf_value):
        '''
        the prediction error of a leaf node given the data set on leaf node on given data set
        '''

        assert isinstance(data_set, pd.DataFrame)

        targets = data_set.iloc[:, -1]
    
        leaf_error = mean_squared_error(leaf_value, targets)

        return leaf_error


    def choose_best_feature(self, data_set):
        '''
        choose the best feature in the data set
        '''
       
        
        labels = data_set.columns.tolist()
        num_data = data_set.shape[0]

        min_subtree_error = float('inf')
        best_feature = None
        best_split_value = None
        # the final lable is the class, so put it away
          
        # calculate the impurity of each feature
        for feature in labels[:-1]:
            feature_vec = data_set[feature]
            feature_vec = feature_vec.sort_values().unique()
            # contral the the number of split position not too much
            if len(feature_vec) > 100:
                split_position = [np.percentile(feature_vec, x) for x in np.linspace(1, 99, 99)]
            else:
                split_position = (feature_vec[:-1] + feature_vec[1:]) / 2
            # calculat the impurity of each split position
            for value in split_position:
                subtree_error = 0
                for splitted_dataset in split_dataset(data_set, feature, value):
                    subtree_error += (splitted_dataset.shape[0] / num_data) * np.var(splitted_dataset.iloc[:, -1])

                if subtree_error < min_subtree_error:
                    min_subtree_error = subtree_error
                    best_split_value = value
                    best_feature = feature

        return best_feature, best_split_value, min_subtree_error


    def _create_tree(self, data_set, tree_depth:int):
        '''
        recursive-partitioning regression
        '''       

        assert isinstance(data_set, pd.DataFrame), print('The data set used for creating the tree must be [pd.DataFrame]')

        tree_depth += 1
        num_data = data_set.shape[0]
         
        # prediction of current node if the leaf node
        targets = data_set.iloc[:, -1]
        leaf_value = self.cal_leaf_value(targets)

        # (1)
        if  tree_depth >= self.max_depth:
            return leaf_value
        
        # (2)
        if  num_data <= self.min_samples_split:
            return leaf_value
        
        # choose best feature
        best_feature, best_split_value, error_after_split = self.choose_best_feature(data_set)

        # seperate the data set into sub data set
        left_data_set, right_data_set = split_dataset(data_set, best_feature, best_split_value)
        
        # (3)
        if self.lower_error:
            # calculate the error before split
            error_before_split = self.cal_leaf_error(data_set, leaf_value)
            if error_before_split <= error_after_split:
                return leaf_value

        # keep splitting the tree
        Tree = {best_feature: {}}

        Tree[best_feature]['<=' + str(best_split_value)] = self._create_tree(left_data_set, tree_depth)
        Tree[best_feature]['>' + str(best_split_value)] = self._create_tree(right_data_set, tree_depth)
        
        # Pessimistic Error Pruning
        if self.prune_method == 'PEP':
            tree_outputs = tree_predict_(data_set, Tree)
            targets_test = data_set.iloc[:, -1]

            # prediction error of the sub tree
            e_tree = mean_squared_error(tree_outputs, targets_test) + cal_num_leaf(Tree) * self.alpha

            # prediction error of the current node 
            e_leaf = self.cal_leaf_error(data_set, leaf_value) + self.alpha
            
            # prunning the tree according to One-SE rule 
            if e_tree >= e_leaf:
                print('-- Pruning feature {}, Leaf Node Error: {}, Sub Tree Error: {} -- \n'.format(best_feature, e_leaf, e_tree))
                Tree = leaf_value

        return Tree

    
    def get_mean(self, tree):
        '''
        get the mean value of sub trees (In processing)
        '''

        leaf_accumulator = get_leaf(tree)

        return mean(leaf_accumulator)


    def reduce_error_pruning(self, tree, data_set):
        '''
        Reduce Error Pruning based on given data set
        '''
   
        # if the test data set is empty, then do not prune
        if data_set.shape[0] == 0:
            return tree
        
        else:
            # recursive-partitioning the data set and tree untill reaching leaf node
            if isinstance(tree, dict):
                split_feature = list(tree.keys())[0]
                split_dict = tree[split_feature]
                split_value = float(list(split_dict.keys())[0][2:])
                left_data_set, right_data_set = split_dataset(data_set, split_feature, split_value)
                left_tree = split_dict['<=' + str(split_value)]
                right_tree = split_dict['>' + str(split_value)]
                tree[split_feature]['<=' + str(split_value)] = self.reduce_error_pruning(left_tree, left_data_set)
                tree[split_feature]['>' + str(split_value)] = self.reduce_error_pruning(right_tree, right_data_set)

                tree_outputs = tree_predict_(data_set, tree)
                targets_test = data_set.iloc[:, -1]
                leaf_value = self.get_mean(tree)

                # prediction error of the sub tree
                e_tree = mean_squared_error(tree_outputs, targets_test)

                # prediction error of the current node 
                e_leaf = self.cal_leaf_error(data_set, leaf_value)
                
                # prunning the tree according to One-SE rule 
                if e_tree >= e_leaf:
                    print('-- Pruning feature {}, Leaf Node Error: {}, Sub Tree Error: {} -- \n'.format(split_feature, e_leaf, e_tree))
                    tree = leaf_value

            return tree


    def tree_predict(self, data_set):
        '''
        predict the value of the data set based on own decision tree

        Args:
            tree: DecisionTreeClassifier

        Returns:
            tree_output: result of classification
        '''
        
        tree_output = [predict(vec, self._tree) for (idx, vec) in data_set.iterrows()]


        return np.array(tree_output)

               

    def fit(self, train_data_set:pd.DataFrame, test_data_set=None, save=False):

        '''
        fit the regression tree based on given data_set

        Pre pruning if one of the following condition is satisfied:  
            (1) if current deepth exceeds the given max depth (Optional) \n
            (2) if the number of samples in the node reach the given threshold \n
            (3) if the prediction error before split is lower than after split (Optional)

        Args:
            data_set: for training the tree (feature sample + targets)
            test_data_set: data set used for post pruning and testing the tree (Optional)
            save: save the tree if this variable is True (Optional)
        '''

        self._total_num_data = train_data_set.shape[0]

        print('\n-- Start Fitting Regression Decision Tree --\n')

        if self.prune_method == 'PEP':
            print('-- Pessimistic Error Pruning --\n ')
        elif self.prune_method == 'REP':
            print('-- Reduce Error Pruning -- \n')
        else:
            print('-- Not Pruning --\n')

        tree = self._create_tree(train_data_set, 0)

        if self.prune_method == 'REP':
            assert not test_data_set is None, print('Pruning Need Independent Test Set!')
            tree = self.reduce_error_pruning(tree, test_data_set)  

        self._tree = tree

        tree_outputs_train = self.tree_predict(train_data_set.iloc[:, :-1])
        targets_train = train_data_set.iloc[:, -1]
        error_train = mean_squared_error(tree_outputs_train, targets_train)
        print('-- Train Set Mean Squared Error: {} --\n'.format(error_train))

        if not test_data_set is None:
            assert isinstance(test_data_set, pd.DataFrame)
            tree_outputs_test = self.tree_predict(test_data_set.iloc[:, :-1])
            targets_test = test_data_set.iloc[:, -1]
            error_test = mean_squared_error(tree_outputs_test, targets_test)
            print('-- Test Set Mean Squared Error: {} --\n'.format(error_test))
            r2_test = r2_score(tree_outputs_test, targets_test)
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

        if not os.path.exists('./model_save/decision_tree'):
            os.makedirs('./model_save/decision_tree')
        np.save('./model_save/decision_tree/' + str(self._name) + '_' + time.strftime('%m_%d_%H_%M', time.localtime(time.time())) + '.npy', self._tree)


    def load(self, save_time):
        self._tree = np.load('./model_save/decision_tree/' + str(self._name) + '_' + str(save_time) + '.npy', allow_pickle=True).item()


    def print_tree(self):
        print(self._tree)


    def plot_tree(self):
        '''
        plot the decision tree
        '''
        
        _plot_tree(self._tree, str(self.tree_type) + "_tree")
                  


if __name__ == "__main__":

    from config import get_rdt_config

    dt_parser = get_rdt_config()
    dt_args  = dt_parser.parse_args(['--post-pruning', '--use-independent-ts'])

    decision_tree = RegressionDecisionTree(dt_args)
    
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
    decision_tree.fit(train_data_set, test_data_set)
    # decision_tree.print_tree()
    num_leaf = cal_num_leaf(decision_tree.tree)
    print('-- Leaf Number: {} --\n'.format(num_leaf))






