import graphviz
from graphviz import Digraph
from numpy import size
import pybaobabdt
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# 获取所有节点中最多子节点的叶节点


def getMaxLeafs(myTree):
    numLeaf = len(myTree.keys())
    for key, value in myTree.items():
        if isinstance(value, dict):
            sum_numLeaf = getMaxLeafs(value)
            if sum_numLeaf > numLeaf:
                numLeaf = sum_numLeaf
    return numLeaf


def _plot_tree(tree, file_name):

    if not os.path.exists('./model_graph'):
        os.makedirs('./model_graph')
    g = Digraph("G", filename=file_name, directory='./model_graph',
                format='png', strict=False)
    g.edge_attr.update()
    first_label = list(tree.keys())[0]
    g.node("0", first_label)
    _sub_plot(g, tree, "0", 1)
    # leafs = str(getMaxLeafs(tree) // 10)
    # g.attr(rankdir='TB', rankstep=leafs)
    g.render(file_name, view=True)


node_id = "0"


def _sub_plot(g, tree, root_id, depth):
    global node_id

    if depth >= 4:
        return

    feature_label = list(tree.keys())[0]
    split_dict = tree[feature_label]
    for split_label in split_dict.keys():
        sub_tree = split_dict[split_label]
        if isinstance(sub_tree, dict):
            node_id = str(int(node_id) + 1)
            label = str(split_label)[:7] if len(str(split_label)) > 7 else str(split_label)

            g.node(node_id, list(sub_tree.keys())[0])
            g.edge(root_id, node_id, label)
            _sub_plot(g, sub_tree, node_id, depth+1)
        else:
            node_id = str(int(node_id) + 1)
            g.node(node_id, str(sub_tree))
            g.edge(root_id, node_id, str(split_label))


if __name__ == "__main__":
    from config import *
    from classification_tree import ClassificationDecisionTree
    from regression_tree import RegressionDecisionTree
    from soft_decision_tree import SoftDecisionTree
    from random_forest import RandomForest
    import joblib
    import numpy as np

    # cdt_parser = get_cdt_config()
    # cdt_args = cdt_parser.parse_args(['--post-pruning', '--use-independent-ts'])
    # classification_decision_tree = ClassificationDecisionTree(cdt_args)

    # rdt_parser = get_rdt_config()
    # rdt_args = rdt_parser.parse_args(['--post-pruning'])
    # regression_decision_tree = RegressionDecisionTree(rdt_args)

    # sdt_parser = get_sdt_config()
    # sdt_args = sdt_parser.parse_args()

    # soft_decision_tree = SoftDecisionTree(sdt_args)

    # rf_parser = get_rf_config()
    # rf_args = rf_parser.parse_args(['--num-trees', '20'])
    # random_forest = RandomForest(rf_args)
    
    # clf = joblib.load('cart_lunarlander.pkl')

    # label = ['x_position', 'y_position', 'x_speed', 'speed',
    #           'angle', 'angular speed', 'leg1', 'leg2', 'action']
    
    # # clf = joblib.load('cart_cartpole.pkl')
    # # label = ['cart position', 'cart velocity',
    # #             'pole angle', 'pole velocity', 'action']
  
    # pybaobabdt.drawTree(clf, size=10, dpi=72, features=label[:-1], colormap='Spectral', maxdepth=4)

    # plt.show()
    

    
    plt.figure(figsize=(16, 6), dpi=100)

    y1 = np.load('cart_cartpole_avg1.npy')
    y2 = np.load('cart_cartpole_avg2.npy')

    x = [2000 * (i+1) for i in range(len(y1))]


    plt.plot(x, y1, label='DTDA', color='red', linestyle='-', marker='.', markersize=10)
    plt.plot(x, y2, label='BC', color='black',linestyle='--', marker='.', markersize=10)

    

    _xtick_labels = ['{}'.format(i) for i in x]
    plt.xticks(x, _xtick_labels, rotation=0, size=16)
    plt.yticks(size=16)
    
    plt.grid(alpha=0.8)

    
    plt.xlabel('Number of Samples', fontsize=20)
    plt.ylabel('Average Episode Reward', fontsize=20)
    # plt.title('Performance of Explainable Model', fontsize=20)
    
    plt.legend(loc='lower right', fontsize=20)

    
    plt.show()


    print('done')

    # regression_decision_tree.load('02_13_11_29')

    # soft_decision_tree.load('12_07_14_59')

    # random_forest.load('01_25_23_26')

    # _plot_tree(regression_decision_tree.tree, 'RDT')

    # tree = {
    #     "tearRate": {
    #         "reduced": "no lenses",
    #         "normal": {
    #             "astigmatic": {
    #                 "yes": {
    #                     "prescript": {
    #                         "myope": "hard",
    #                         "hyper": {
    #                             "age": {
    #                                 "young": "hard",
    #                                 "presbyopic": "no lenses",
    #                                 "pre": "no lenses"
    #                             }
    #                         }
    #                     }
    #                 },
    #                 "no": {
    #                     "age": {
    #                         "young": "soft",
    #                         "presbyopic": {
    #                             "prescript": {
    #                                 "myope": "no lenses",
    #                                 "hyper": "soft"
    #                             }
    #                         },
    #                         "pre": "soft"
    #                     }
    #                 }
    #             }
    #         }
    #     }
    # }

    # _plot_tree(tree, "tree.gv")

