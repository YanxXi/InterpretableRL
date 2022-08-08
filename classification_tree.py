import numpy as np
from collections import Counter
import os
import time
from plot import _plot_tree
import pandas as pd
from utils.tool import *
from decision_tree import BaseDecisionTree


# TODO: seperate the input data set to "discreate" and "continuous" type
class ClassificationDecisionTree(BaseDecisionTree):
    def __init__(self, args, labels=None):
        '''
        labels are set in random forest
        '''
        super(ClassificationDecisionTree, self).__init__(args)

        self.tree_type = args.tree_type
        self.higher_accuracy = args.higher_accuracy
        self.min_impurity_decrease = args.min_impurity_decrease
        self._name = 'CDT'
        self._sub_feature_size = None
        self.labels = labels
        self._gini_decrease = dict(zip(labels[:-1], [0 for _ in range(len(labels))])) if labels else None



    @property
    def name(self):
        return self._name


    @property
    def sub_feature_size(self):
        return self._sub_feature_size
    @sub_feature_size.setter
    def sub_feature_size(self, value):
        self._sub_feature_size = value

    @property 
    def gini_decrease(self):
        return self._gini_decrease



    def cal_entropy(self, data_set):
        '''
        calculate entropy
        '''

        num_data = data_set.shape[0]
        targets = data_set.iloc[:, -1]
        label_count = Counter(targets)

        entroy = 0

        probs = np.array(list(label_count.values())) / num_data
        entroy += (-probs * np.log2(probs)).sum()

        return entroy


    def cal_gini(self, data_set):
        '''
        calculate gini diversity index
        '''
        num_data = data_set.shape[0]
        targets = data_set.iloc[:, -1]
        label_count = Counter(targets)

        gini = 1
        probs = np.array(list(label_count.values())) / num_data
        gini -= np.power(probs, 2).sum()

        return gini


    def choose_best_feature(self, data_set, labels):
        '''
        choose the best feature in the data set
        '''
        
        num_data = data_set.shape[0]
        # labels = data_set.columns.tolist()
        best_split_goodness = float('-inf')
        best_feature = None
        best_split_value = None
        
        # ID3 calculate impurity decrease according to entropy gain
        if self.tree_type == 'ID3':
            # the base impurity of current node
            base_entropy = self.cal_entropy(data_set)
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
                    feature_entropy = 0
                    for splitted_dataset in split_dataset(data_set, feature, value):
                        feature_entropy += (splitted_dataset.shape[0] / num_data) * self.cal_entropy(splitted_dataset)
                    split_goodness = base_entropy - feature_entropy

                    # choose the best best feature and split position with largest impurity decrease
                    if split_goodness > best_split_goodness:
                        best_split_goodness = split_goodness
                        best_split_value = value
                        best_feature = feature

        # ID3 calculate impurity decrease according to entropy gain ratio
        elif self.tree_type == 'C45':
            # the base impurity of current node
            base_entropy = self.cal_entropy(data_set)
            # calculate the impurity of each feature
            for feature in labels[:-1]:
                feature_vec = data_set[feature]
                feature_vec = feature_vec.sort_values().unique()
                # contral the the number of split position not too much
                if len(feature_vec) > 100:
                    split_position = [np.percentile(feature_vec, x) for x in np.linspace(1, 99, 99)]
                else:
                    split_position = (feature_vec[:-1] + feature_vec[1:]) / 2
                for value in split_position:
                    feature_entropy = 0
                    IV = 0
                    # calculat the impurity of each split position
                    for splitted_dataset in split_dataset(data_set, feature, value):
                        prob = splitted_dataset.shape[0] / num_data
                        feature_entropy += prob * self.cal_entropy(splitted_dataset)
                        IV  -= prob * np.log2(prob)
                    split_goodness = (base_entropy - feature_entropy) / IV
                     
                    # choose the best best feature and split position with largest impurity decrease
                    if  split_goodness > best_split_goodness:
                        best_split_goodness = split_goodness
                        best_split_value = value
                        best_feature = feature

        # CART calculate impurity decrease according to gini index
        elif self.tree_type == 'CART':
            # the base impurity of current node
            base_gini = self.cal_gini(data_set)
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
                    feature_gini = 0
                    for splitted_dataset in split_dataset(data_set, feature, value):
                        feature_gini += (splitted_dataset.shape[0] / num_data) * self.cal_gini(splitted_dataset)
                    split_goodness = base_gini - feature_gini

                    # choose the best best feature and split position with largest impurity decrease
                    if split_goodness > best_split_goodness:
                        best_split_goodness = split_goodness
                        best_split_value = value
                        best_feature = feature

        else:
            print('WRONG TYPE (only ID3 C45 CARF are valid) !')

        # if best_split_value:
        #     print('{}: the best feature is {} with split goodness {} and value {} \n'
        #     .format(str(self.tree_type), best_feature, best_split_goodness, best_split_value))
        # else:
        #     print('can not find a good feature')

        return best_feature, best_split_value, best_split_goodness


    def _create_tree(self, data_set, tree_depth:int):
        '''
        Construct tree recursively.
        '''

        tree_depth += 1
        num_data = data_set.shape[0]
         
        # classification of current node if the leaf node
        targets = data_set.iloc[:, -1]
        leaf_output = major_vote(targets)

        # (1)
        if  tree_depth >= self.max_depth:
            return leaf_output
        
        # (2)
        if num_data <= self.min_samples_split:
            return leaf_output

       # if all class in the data set is same, return this class
        if targets.unique().shape[0] == 1:
            return leaf_output

        # choose best feature   
        if self._sub_feature_size is None:
            # use original data set 
            labels = data_set.columns.tolist()
            best_feature, best_split_value, impurity_decrease = self.choose_best_feature(data_set, labels)
        else:
            # at each node, using only randomly selected m variables to determine the best split at that node
            sub_labels = np.random.choice(self.labels[:-1], self._sub_feature_size, replace=False).tolist() + [self.labels[-1]]
            sub_data_set = data_set.loc[:, sub_labels]
            best_feature, best_split_value, impurity_decrease = self.choose_best_feature(sub_data_set, sub_labels)

        # if the best feature can't be decided or (3)
        if not best_feature:
            return leaf_output
        # record gini decrease of each feature
        elif self._gini_decrease:
            self._gini_decrease[best_feature] += impurity_decrease
        else:
            pass
        
        # (3)
        if impurity_decrease < self.min_impurity_decrease:
            return leaf_output
    
        # seperate the data set into sub trees
        sub_data_sets = split_dataset(data_set, best_feature, best_split_value)
        [left_data_set, right_data_set] = sub_data_sets

        # calculate the accuracy before split
        acc_before_split = self.cal_leaf_acc(data_set, leaf_output)
        # (5)
        if self.higher_accuracy:
            acc_after_split = 0
            for sub_data_set in sub_data_sets:
                sub_leaf_output = major_vote(sub_data_set.iloc[:, -1])
                acc_after_split += self.cal_leaf_acc(sub_data_set, sub_leaf_output) * sub_data_set.shape[0] / num_data

            if acc_before_split >= acc_after_split:
                return leaf_output

        # keep splitting the tree
        Tree = {best_feature: {}}

        Tree[best_feature]['<=' + str(best_split_value)] = self._create_tree(left_data_set, tree_depth)
        Tree[best_feature]['>' + str(best_split_value)] = self._create_tree(right_data_set, tree_depth)

        # Pessimistic Error Pruning 
        if self.prune_method == 'PEP':
            tree_outputs = tree_classify_(data_set, Tree)
            targets_test = data_set.iloc[:, -1]
            # misclassification rate of the sub tree
            e_tree = 1 - cal_accuracy(tree_outputs, targets_test) + cal_num_leaf(Tree) * self.alpha / num_data
            # standard error of the misclassification number of the sub tree
            std = np.sqrt(e_tree * (1 - e_tree) / num_data)
            # misclassification rate of the current node 
            e_leaf = 1 - acc_before_split + self.alpha / num_data

            if e_tree + std >= e_leaf:
                print('-- Pruning feature {}, Leaf Node Error: {}, Sub Tree Error: {} -- \n'.format(best_feature, e_leaf, e_tree))
                Tree = leaf_output

        return Tree


    def tree_classify(self, data_set):
        '''
        classify the data set based on the given decision tree

        Args:
            tree: DecisionTreeClassifier

        Returns:
            tree_output: result of classification
            
        '''

    
        tree_output = [classify(vec, self._tree) for (idx, vec) in data_set.iterrows()]


        return np.array(tree_output)


    def cal_leaf_acc(self, data_set, leaf_output):
        '''
        calculate the accuracy of the classification of a leaf node on given data set
        '''
        
        targets = data_set.iloc[:, -1]
        leaf_outputs = [leaf_output] * data_set.shape[0]
        leaf_acc = cal_accuracy(leaf_outputs, targets)

        return leaf_acc


    def get_most(self, tree):
        '''
        get the most output of sub trees (In processing)
        '''

        leaf_accumulator = get_leaf(tree)
        leaf_counter = Counter(leaf_accumulator)

        return leaf_counter.most_common(1)[0][0]


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

                tree[split_feature]['<=' + str(split_value)] = self.reduce_error_pruning(left_tree, left_data_set)
                
                right_tree = split_dict['>' + str(split_value)]
                tree[split_feature]['>' + str(split_value)] = self.reduce_error_pruning(right_tree, right_data_set)

                tree_outputs = tree_classify_(data_set, tree)
                targets_test = data_set.iloc[:, -1]
                leaf_output = self.get_most(tree)

                # misclassification rate of the sub tree
                e_tree = 1 - cal_accuracy(tree_outputs, targets_test) 
                # misclassification rate of the current node 
                e_leaf = 1 - self.cal_leaf_acc(data_set, leaf_output)

                if e_tree >= e_leaf:
                    print('-- Pruning feature {}, Leaf Node Error: {}, Sub Tree Error: {} -- \n'.format(split_feature, e_leaf, e_tree))
                    tree = leaf_output

            return tree


    def fit(self, train_data_set:pd.DataFrame, test_data_set=None, save=False):
        '''
        fit the calssification decision tree based on given data_set

        Pre pruning if one of the following condition is satisfied:  
            (1) if current deepth exceeds the given max depth (Optional) \n
            (2) if the number of samples in the node reach the given threshold \n
            (3) if impurity decrease smaller than given threshold (Default 0) \n
            (4) if the accuracy before split is higher than after split (Optional)

        Args:
            data_set: for training the tree (feature sample + targets)
            test_data_set: data set used for testing the tree (Optional)
            save: save the tree if this variable is True (Optional)
        '''

        print('\n-- Start Fitting {} Classification Decision Tree --\n'.format(self.tree_type))

        
        if self.prune_method == 'PEP':
            print('-- Pessimistic Error Pruning --\n ')
        elif self.prune_method == 'REP':
            print('-- Reduce Error Pruning -- \n')
        else:
            print('-- Not Post Pruning --\n')

        tree = self._create_tree(train_data_set, 0)


        if self.prune_method == 'REP':
            assert not test_data_set is None, print('Pruning Need Independent Test Set!')
            tree = self.reduce_error_pruning(tree, test_data_set)
        
        self._tree = tree

        tree_outputs_train = self.tree_classify(train_data_set)
        targets_train = train_data_set.iloc[:, -1]
        accuracy_train = cal_accuracy(tree_outputs_train, targets_train)
        print('-- Train Set Accuracy: {} --\n'.format(accuracy_train))

        if not test_data_set is None:
            assert isinstance(test_data_set, pd.DataFrame)
            tree_outputs_test = self.tree_classify(test_data_set)
            targets_test = test_data_set.iloc[:, -1]
            accuracy_test = cal_accuracy(tree_outputs_test, targets_test)
            print('-- Test Set Accuracy: {} --\n'.format(accuracy_test))

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

        save_dir = './model_save/classification_decision_tree/' + str(self.tree_type) + '/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(save_dir + time.strftime('%m_%d_%H_%M', time.localtime(time.time())) + '.npy', self._tree)


    def load(self, save_time):
        load_path = './model_save/classification_decision_tree/' + str(self.tree_type) + '/' + str(save_time) + '.npy'
        self._tree = np.load(load_path, allow_pickle=True).item()


    def print_tree(self):
        print(self._tree)


    def plot_tree(self):
        '''
        plot the decision tree
        '''
        
        _plot_tree(self._tree, str(self.tree_type) + "_tree")
                  


if __name__ == "__main__":

    from config import get_cdt_config

    dt_parser = get_cdt_config()
    dt_args  = dt_parser.parse_args(['--post-pruning', '--tree-type', 'CART', '--use-independent-ts'])

    id3_decision_tree = ClassificationDecisionTree(dt_args)
    # c45_decision_tree = DecisionTreeClassifier('C45', post_pruning=True)
    # cart_decision_tree = DecisionTreeClassifier('CART', post_pruning=True, max_depth=3, higher_accuracy=True)
    
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

    id3_decision_tree.fit(data_set, data_set)

    id3_decision_tree.print_tree()

    # c45_tree = c45_decision_tree.create_tree(dataSet, labels.copy())
    # print(c45_tree)

    # cart_tree = cart_decision_tree.create_tree(dataSet, labels.copy())
    # print(cart_tree)

    # tree_output = cart_decision_tree.tree_classify(cart_tree, dataSet, labels)
    # target = [vec[-1] for vec in dataSet]
    # accuracy = cart_decision_tree.cal_accuracy(tree_output, target)
    # num_leaf = cart_decision_tree.cal_num_leaf(cart_tree)





