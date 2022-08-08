import argparse
from pickle import FALSE


def get_train_ppo_config(parser: argparse.ArgumentParser):
    # get algorithm config
    parser = get_ppo_config(parser)
    parser = get_base_trainer_config(parser)
    parser = get_ppo_trainer_config(parser)

    return parser



def get_train_dqn_config(parser: argparse.ArgumentParser):
    # get algorithm config
    parser = get_dqn_config(parser)
    parser = get_base_trainer_config(parser)
    parser = get_dqn_trainer_config(parser)

    return parser


def get_ppo_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='PPO Alogorithm Config')
    group.add_argument('--lr', type=float, default=5e-4,
                       help='learning rate')
    group.add_argument('--mlp-hidden-size', type=str, default='128',
                       help='hidden size of mlp layer to extract observation feature')
    group.add_argument('--act-hidden-size', type=str,
                       default='128', help='hidden size of mlp layer in action module')
    group.add_argument('--value-hidden-size', type=str,
                       default='128', help='hidden size of mlp layer in value module')
    group.add_argument('--use-recurrent-layer', action='store_true',
                       default=False, help='whether to use recurrent layer')
    group.add_argument('--recurrent-hidden-size',
                       type=int, default=128, help='hidden size of GRU')
    group.add_argument('--recurrent-hidden-layers', type=int,
                       default=1, help='number of hidden layers of GRU')

    return parser


def get_dqn_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='DQN Alogorithm Config')
    group.add_argument('--lr', type=float, default=1e-3,
                       help='learning rate of Q network (default 1e-e)')
    group.add_argument('--tau', type=float, default=5e-3,
                       help='parameter used for updating target network (default 5e-3)')
    group.add_argument('--epsilon', type=float, default=0.9,
                       help='explore noise of action (default 1.0)')
    group.add_argument('--hidden-size', type=str, default='100 100',
                       help='hidden size in Q network (default 100 100)')
    group.add_argument("--use-batch-normalization", action='store_true', default=False,
                       help="Whether to apply Batch Normalization to the feature extraction inputs")
    return parser


def get_base_trainer_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='Base Trainer Config')
    group.add_argument('--gamma', type=float, default=0.99,
                       help='discount factor')
    group.add_argument('--num-episode', type=int, default=1e3,
                       help='number of episodes')
    group.add_argument("--max-grad-norm", type=float, default=4,
                       help='restrict the grad norm to max_grad_norm to prevent gradient explosion')
    group.add_argument("--eval-interval", type=float, default=10,
                       help='evaluate policy each eval_interval episode')
    group.add_argument("--eval-episodes", type=float, default=100,
                       help='episodes for evaluation')
    group.add_argument("--solved-reward", type=int, default=200,
                       help='in evaluation mode, if average reward exceeds this threshold, the the problem is sloved')
    group.add_argument("--start-eval", type=int, default=0,
                       help='only enter evaluation mode if current episode excce this number (default 0)')
    group.add_argument("--eval-in-paral", action='store_true', default=False,
                       help='evaluate the policy in parallel')
    group.add_argument("--log-info", action='store_true', default=False,
                       help='log trainning infomation')
    group.add_argument("--save-model", action='store_true', default=False,
                       help='save model in the trainning')

    return parser


def get_ppo_trainer_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='PPO Trainer Config')

    group.add_argument("--clip-param", type=float, default=0.2,
                       help='ppo clip parameter (default: 0.2)')
    group.add_argument("--value-loss-coef", type=float, default=1,
                       help='ppo value loss coefficient (default: 1)')
    group.add_argument("--entropy-coef", type=float, default=0.01,
                       help='entropy term coefficient (default: 0.01)')
    group.add_argument("--ppo-epoch", type=int, default=10,
                       help='the number of process of training all samples')
    group.add_argument("--buffer-size", type=int, default=1024,
                       help='minimum buffer size to complete one training process')
    group.add_argument("--batch-size", type=int, default=64,
                       help='the number of samples to complete one update')
    group.add_argument("--chunk-length", type=int, default=10,
                       help='the length of chunk data')
    return parser


def get_dqn_trainer_config(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(description='DQN Trainer Config')

    group.add_argument("--use-per", action='store_true', default=False,
                       help='whether to use prioritized experience replay buffer')
    group.add_argument("--dqn-epoch", type=int, default=1,
                       help='the number of process of training all samples')
    group.add_argument("--buffer-size", type=int, default=2 ** 20,
                       help='minimum buffer size to complete one training process')
    group.add_argument("--batch-size", type=int, default=2 ** 6,
                       help='the number of samples to complete one update')
    return parser


def get_dt_config():

    dt_parser = argparse.ArgumentParser(
        description='Base Decision Tree Config')
    dt_parser.add_argument('--max-depth', type=int, default=-1,
                           help='max depth of the tree')
    dt_parser.add_argument('--min-samples-split', type=int, default=5,
                           help='if the number of samples in the node less than this threshold, stop splitting')
    dt_parser.add_argument('--min-prop-split', type=int, default=0.01,
                           help='if the proportion of samples in the node less than this threshold, stop splitting')
    dt_parser.add_argument('--post-pruning', action='store_true', default=False,
                           help='whether to use post pruning')
    dt_parser.add_argument('--alpha', type=float,
                           default=0.1, help='complexity cost factor')
    dt_parser.add_argument('--use-independent-ts', action='store_true', default=False,
                           help='whether to use independent test set')

    return dt_parser


def get_cdt_config():

    cdt_parser = argparse.ArgumentParser(
        description='Classification Decision Tree Config')
    cdt_parser = get_dt_config()
    cdt_parser.add_argument('--tree-type', type=str, default='CART',
                            help='three types of tree are provided: ID3, C45 or CART')
    cdt_parser.add_argument('--min-impurity-decrease', type=int, default=0,
                            help='if impurity decrease in the node less than this threshold, step splitting')
    cdt_parser.add_argument('--higher-accuracy', action='store_true', default=False,
                            help='if this is set then only split when accuracy is higher after splitting')

    return cdt_parser


def get_rdt_config():

    rdt_parser = argparse.ArgumentParser(
        description='Regression Decision Tree Config')
    rdt_parser = get_dt_config()
    rdt_parser.add_argument('--lower-error', action='store_true', default=False,
                            help='if this is set then only split when mes is lower after splitting')

    return rdt_parser


def get_sdt_config():

    sdt_parser = argparse.ArgumentParser(description='Soft Decision Tree')
    sdt_parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
    sdt_parser.add_argument('--input-dim', type=int, default=4, metavar='N',
                            help='input dimension size(default: 4)')
    sdt_parser.add_argument('--output-dim', type=int, default=2, metavar='N',
                            help='output dimension size(default: 2)')
    sdt_parser.add_argument('--max-depth', type=int, default=8, metavar='N',
                            help='maximum depth of tree(default: 8)')
    sdt_parser.add_argument('--epoch', type=int, default=20, metavar='N',
                            help='number of epochs to train (default: 20)')
    sdt_parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.01)')
    sdt_parser.add_argument('--lmbda', type=float, default=0.1, metavar='LR',
                            help='temperature rate (default: 0.1)')
    sdt_parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                            help='SGD momentum (default: 0.5)')
    sdt_parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
    sdt_parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
    sdt_parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
    sdt_parser.add_argument('--device', type=str, default='cpu',
                            help='device used for trainning (default: cpu)')
    sdt_parser.add_argument('--mlp-hidden-size', type=str, default='',
                            help='hidden size of mlp layer to extract observation feature (default: ' ')')

    return sdt_parser


def get_rf_config():

    rf_parser = argparse.ArgumentParser(description='Random Forest Config')
    rf_parser = get_cdt_config()
    rf_parser.add_argument('--max-features', default='sqrt',
                           help='the number of random feature set used for constructing the tree')
    rf_parser.add_argument('--num-trees', type=int, default=8,
                           help='number of trees in the forest')

    return rf_parser


def get_gbdt_config():

    gbdt_parser = argparse.ArgumentParser(
        description='Gradient Boosting Decision Tree Config')
    gbdt_parser = get_rdt_config()
    gbdt_parser.add_argument('--num-trees', type=int, default=8,
                             help='number of trees in GBDT')
    gbdt_parser.add_argument('--min-loss', type=float, default=0.1,
                             help='if the loss lower than this threshold, stop creating trees')

    return gbdt_parser
