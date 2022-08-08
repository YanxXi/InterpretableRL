from logging import root
import math
import numpy as np
from numpy.random.mtrand import sample
import pandas as pd
import os
import time
from classification_tree import ClassificationDecisionTree
from utils.tool import *
import ray


class RandomForest():
    def __init__(self, args):
        self.args = args
        self.args.post_pruning = False
        self.args.tree_type = 'CART'
        self.args.higher_accuracy = False
        self.num_trees = args.num_trees
        self.max_features = args.max_features
        self._name = 'RF'
        self._forest = []
        self._oob_data = []


    @property
    def forest(self):
        return self._forest

    @property
    def oob_data(self):
        return self._oob_data
    
    @property
    def name(self):
        return self._name


    def async_fit(self, train_data_set:pd.DataFrame, test_data_set=None, save=False):
        '''
        fit the random forest asynchronously based on ray

        Attention: 
            The tree type in random forest is restricted to [CART] and post-pruning is retricted to [False]

        Args:
            train_data_set: the data set used for fitting the tree
            test_data_set: data set uesd for testing the forest (Optional)
        '''

        assert isinstance(train_data_set, pd.DataFrame)

        print('\n-- Start Fitting Random Forest Asynchronously --\n')

        ray.init()

        labels = train_data_set.columns.tolist()
        num_features = len(labels) -1
        gini_decreases = np.zeros(num_features)
        
        # decide the size of sub features
        if type(self.max_features).__name__ == 'str':
            if self.max_features == 'sqrt':
                sub_feature_size = math.floor(math.sqrt(num_features))
            elif self.max_features == 'log':
                sub_feature_size = math.floor(math.log(num_features))
            else:
                print('Only [sqrt] and [log] are provided in this model')
        elif type(self.max_features).__name__ == float:
            sub_feature_size = math.floor(num_features * self.max_features)
        else:
            print('The max_features must be [str] or [float] type')
    
        # create the random forest

        obj_ref_list = [self.async_create_tree.remote(self.args, train_data_set, sub_feature_size) for _ in range(self.num_trees)]
        forest_ref = [obj_ref[0] for obj_ref in obj_ref_list]
        oob_data_ref = [obj_ref[1] for obj_ref in obj_ref_list]
        gini_decrease_ref = [obj_ref[2] for obj_ref in obj_ref_list]
        self._forest = ray.get(forest_ref)
        self._oob_data = ray.get(oob_data_ref)
        gini_decrease_list = ray.get(gini_decrease_ref)
        for gini_decrease_dict in gini_decrease_list:
            gini_decreases += np.array(list(gini_decrease_dict.values()))

        # average gini decreases over all trees in the forest
        gini_decreases = gini_decreases / self.num_trees
        # rank variable according to their average gini decrease
        self.gini_importance = sorted(dict(zip(labels[:-1], gini_decreases)).items(), key=lambda x: x[1], reverse=True)

        # calculate the out of bag error
        oob_error = self.cal_oob_error(train_data_set)
        print('-- The Out of Bag (OOB) Error: {} --\n'.format(oob_error))

        targets_train = train_data_set.iloc[:, -1]
        forest_output_train = self.forest_classify(train_data_set)
        acc_train = cal_accuracy(forest_output_train, targets_train)
        print('-- Train Set Accuracy: {} --\n'.format(acc_train))

        # if test data set is given, then test the forest based on this data set
        if not test_data_set is None:
            assert isinstance(test_data_set, pd.DataFrame)
            targets_test = test_data_set.iloc[:, -1]
            forest_output_test = self.forest_classify(test_data_set)
            acc_test = cal_accuracy(forest_output_test, targets_test)
            print('-- Test Set Accuracy: {} --\n'.format(acc_test))
   
        # save the forest
        if save:
            print('-- Save Model --\n')
            self.save()
        else:
            print('-- Not Save Model --\n')

        print('-- END --\n')

    @staticmethod
    @ray.remote(num_returns=3)
    def async_create_tree(args, train_data_set, sub_feature_size):
        num_data = train_data_set.shape[0]
        labels = train_data_set.columns.tolist()
        data_idx = np.arange(0, num_data)
        # draw a boostrap sample from trainning set
        bootstrap_sample = train_data_set.sample(frac=1, replace=True, axis=0).reset_index()
        # Each subset selected to make each individual bth tree grow usually contains 2/3 of the calibration dataset.
        bag_data_idx = bootstrap_sample['index'].unique()
        # The samples which are not present in the calibration subset are included as part of another subset called out-of-bag (oob)
        oob_data_idx = np.delete(data_idx, bag_data_idx)
        # the first column is index 
        del bootstrap_sample['index']

        decision_tree = ClassificationDecisionTree(args, labels)
        decision_tree.sub_feature_size = sub_feature_size
        tree_dict = decision_tree._create_tree(bootstrap_sample, 0)

        return tree_dict, oob_data_idx, decision_tree.gini_decrease


    def fit(self, train_data_set:pd.DataFrame, test_data_set=None, save=False):
        '''
        fit the random forest, we create the tree by ramdonmly select samples (2/3) and we choose a random set of the features

        Attention: 
            The tree type in random forest is restricted to [CART] and post-pruning is retricted to [False]

        Args:
            train_data_set: the data set used for fitting the tree
            test_data_set: data set uesd for testing the forest (Optional)
        '''

        assert isinstance(train_data_set, pd.DataFrame)

        print('\n-- Start Fitting Random Forest --\n')

        labels = train_data_set.columns.tolist()
        num_data = train_data_set.shape[0]
        num_features = len(labels) -1
        gini_decreases = np.zeros(num_features)

        # decide the size of sub features
        if type(self.max_features).__name__ == 'str':
            if self.max_features == 'sqrt':
                sub_feature_size = math.floor(math.sqrt(num_features))
            elif self.max_features == 'log':
                sub_feature_size = math.floor(math.log(num_features))
            else:
                print('Only [sqrt] and [log] are provided in this model')
        elif type(self.max_features).__name__ == float:
            sub_feature_size = math.floor(num_features * self.max_features)
        else:
            print('The max_features must be [str] or [float] type')
    
        # create the random forest
        for i in range(self.num_trees):
            data_idx = np.arange(0, num_data)
            # draw a boostrap sample from trainning set
            bootstrap_sample = train_data_set.sample(frac=1, replace=True, axis=0).reset_index()
            # Each subset selected to make each individual bth tree grow usually contains 2/3 of the calibration dataset.
            bag_data_idx = bootstrap_sample['index'].unique()
            # The samples which are not present in the calibration subset are included as part of another subset called out-of-bag (oob)
            oob_data_idx = np.delete(data_idx, bag_data_idx)
            self._oob_data.append(oob_data_idx)
            # the first column is index 
            del bootstrap_sample['index']

            print('-- Constructing Tree {} --\n'.format(i+1))
            decision_tree  = ClassificationDecisionTree(self.args, labels)
            decision_tree.sub_feature_size = sub_feature_size
            tree_dict = decision_tree._create_tree(bootstrap_sample, 0)
            self._forest.append(tree_dict)

            # add gini impurity decrease of each tree
            gini_decreases += np.array(list(decision_tree.gini_decrease.values()))
        
        # average gini decreases over all trees in the forest
        gini_decreases = gini_decreases / self.num_trees
        # rank variable according to their average gini decrease
        self.gini_importance = sorted(dict(zip(labels[:-1], gini_decreases)).items(), key=lambda x: x[1], reverse=True)

        # calculate the out of bag error
        oob_error = self.cal_oob_error(train_data_set)
        print('-- The Out of Bag (OOB) Error: {} --\n'.format(oob_error))

        targets_train = train_data_set.iloc[:, -1]
        forest_output_train = self.forest_classify(train_data_set)
        acc_train = cal_accuracy(forest_output_train, targets_train)
        print('-- Train Set Accuracy: {} --\n'.format(acc_train))

        # if test data set is given, then test the forest based on this data set
        if not test_data_set is None:
            assert isinstance(test_data_set, pd.DataFrame)
            targets_test = test_data_set.iloc[:, -1]
            forest_output_test = self.forest_classify(test_data_set)
            acc_test = cal_accuracy(forest_output_test, targets_test)
            print('-- Test Set Accuracy: {} --\n'.format(acc_test))
   
        # save the forest
        if save:
            print('-- Save Model --\n')
            self.save()
        else:
            print('-- Not Save Model --\n')

        print('-- END --\n')


    def forest_classify(self, data_set):
        '''
        classify the data set based each tree in forest
        '''

        assert isinstance(data_set, pd.DataFrame)

        forest_output = np.zeros((self.num_trees, data_set.shape[0]))
        for i, tree in enumerate(self.forest):
            tree_output = tree_classify_(data_set, tree)
            forest_output[i] = tree_output
        forest_output = pd.DataFrame(forest_output)
        forest_output = forest_output.apply(major_vote).values

        return forest_output


    def cal_oob_error(self, data_set):
        '''
        calculate the out of bag error.

        The proportion between the misclassifications and the total number of oob elements 
        contributes an unbiased estimation of the generalization error
        '''

        samples = data_set.iloc[:, :-1]
        targets = data_set.iloc[:, -1]
        error_count = 0

        # each element of the oob subset has been classified on average by a third of the overall number of the trees
        for (idx, vec) in samples.iterrows():
            output_list = []
            for tree, oob_data_idx in zip(self.forest, self.oob_data):
                if idx in oob_data_idx:
                    output = classify(vec, tree)
                    output_list.append(output)
            
            # if this vector is used in every tree
            if output_list:
                oob_output = major_vote(output_list)
            else:
                oob_output = 0

            if oob_output != targets[idx]:
                error_count += 1

        oob_error = error_count / data_set.shape[0]

        return oob_error


    def print_forst(self):

        for tree in self.forest:
            print(tree)


    def print_var_importance(self):
        '''
        print variable importance
        '''

        for rank, (key, value) in enumerate(self.gini_importance):
            print('-- Variable {} rank {} with average gini index decrease {} --\n'.format(key, rank+1, value))


    def save(self):
        '''
        save the forest
        '''

        if not os.path.exists('./model_save/random_forest/'):
            os.makedirs('./model_save/random_forest/')

        np.save('./model_save/random_forest/' + time.strftime('%m_%d_%H_%M', time.localtime(time.time())) + '.npy', self.forest)

    
    def load(self, save_time):
        '''
        load the forest
        '''

        self._forest = np.load('./model_save/random_forest/' + str(save_time) + '.npy', allow_pickle=True).tolist()


            


if __name__ == "__main__":

    from config import get_rf_config

    rf_parser = get_rf_config()
    rf_args = rf_parser.parse_args()

    random_forest = RandomForest(rf_args)

    dataSet = [[1, 0, 125, 0],
               [0, 1, 100, 0],
               [0, 0, 70, 0],
               [1, 1, 120, 0],
               [0, 2, 95, 1],
               [0, 1, 60, 0],
               [1, 2, 220, 0],
               [0, 0, 85, 1],
               [0, 1, 75, 0],
               [0, 0, 90, 1]]

    labels = ['是否有房', '婚姻状况', '年收入(k)', '拖欠贷款']  # 三个特征

    data_set = pd.DataFrame(dataSet, columns=labels)

    random_forest.asy_fit(data_set)

    random_forest.print_forst()
