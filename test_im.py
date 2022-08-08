import gym
from eval_model import eval_im
from config import *
from classification_tree import ClassificationDecisionTree
from regression_tree import RegressionDecisionTree
from soft_decision_tree import SoftDecisionTree
from random_forest import RandomForest

cdt_parser = get_cdt_config()
cdt_args = cdt_parser.parse_args(['--post-pruning', '--use-independent-ts'])
classification_decision_tree = ClassificationDecisionTree(cdt_args)

rdt_parser = get_rdt_config()
rdt_args = rdt_parser.parse_args(['--post-pruning'])
regression_decision_tree = RegressionDecisionTree(rdt_args)

sdt_parser = get_sdt_config()
sdt_args = sdt_parser.parse_args()

soft_decision_tree = SoftDecisionTree(sdt_args)

rf_parser = get_rf_config()
rf_args = rf_parser.parse_args(['--num-trees', '20'])
random_forest = RandomForest(rf_args)


classification_decision_tree.load('01_26_11_17')

soft_decision_tree.load('12_07_14_59')

random_forest.load('01_25_23_26')


# env = gym.make('CartPole-v0')
# labels = ['cart position', 'cart velocity',
#           'pole angle', 'pole velocity', 'action']


env = gym.make('LunarLander-v2')
labels = ['horizontal position', 'vertical position', 'horizontal speed', 'vertical speed',
          'angle', 'angular speed', 'first leg', 'second leg', 'action']

# eval_im(env, random_forest, labels)  # step 4: test mode

from plot import _plot_tree

_plot_tree(classification_decision_tree.tree, 'CART')
