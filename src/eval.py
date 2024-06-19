import os
import yaml
import wandb
import torch
import argparse

from config import *
from dynamics.mds import MDs
from dynamics import dynamics
from flow import FlowNetAgent
from utils.logging import Logger

args = parse_eval_args()

if __name__ == '__main__':
    args = config_init(args)
    torch.manual_seed(args.seed)

    md = getattr(dynamics, args.molecule.title())(args, args.start_state)
    agent = FlowNetAgent(args, md)
    logger = Logger(args, md)

    logger.info(f"Initialize {args.num_samples} MDs starting at {args.start_state}")
    mds = MDs(args)

    model_path = args.model_path if args.model_path else os.path.join(args.save_dir, args.project, args.date, 'train', str(args.seed), 'policy.pt')
    agent.policy.load_state_dict(torch.load(model_path))
    
    logger.info(f"Start Evaulation")
    log = agent.sample(args, mds, args.temperature)
    logger.log(None, agent.policy, 0, **log)
    logger.info(f"Finish Evaluation")