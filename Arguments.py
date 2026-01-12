import argparse
from typing import List

def parse_weights(raw_weights: str):
    raw_weight_list = raw_weights.split(',')
    weight_list = [float(weight) for weight in raw_weight_list]
    weight_sum = sum(weight_list)
    weight_normalized = [weight / weight_sum for weight in weight_list]
    return weight_normalized

parser = argparse.ArgumentParser()
parser.add_argument('--object', type=str, default='Armadillo')
parser.add_argument('--depth', type=int, default=5)
parser.add_argument('--f_dim', type=int, default=32)
parser.add_argument('--config', type=str, default='2')
parser.add_argument('--port', type=int, default=29500)
parser.add_argument('--use_tqdm', action='store_true')
parser.add_argument('--num_epochs', type=int, default=500)
# parser.add_argument('--warm_up', type=int, default=5)
# parser.add_argument('--batch_size', type=int, default=262144)
parser.add_argument('--rendering_batch_size', type=int, default=16384)
parser.add_argument('--n_iters', type=int, default=3)
# parser.add_argument('--loss_weights', type=parse_weights, default=[0.2, 0.3, 0.5])
args = parser.parse_args()

object_name : str = args.object
depth : int = args.depth
f_dim : int = args.f_dim
config_name : str = f'L_depth{depth}_fdim{f_dim}_{args.config}'
port : int = args.port
use_tqdm : bool = args.use_tqdm
num_epochs : int = args.num_epochs
# warm_up : int = args.warm_up
# batch_size : int = args.batch_size
rendering_batch_size : int = args.rendering_batch_size
n_iters : int = args.n_iters
# loss_weights : List[float] = args.loss_weights
# assert n_iters == len(loss_weights)
