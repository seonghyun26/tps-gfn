import torch

from config import *
from tqdm import tqdm
from dynamics.mds import MDs
from dynamics import dynamics
from flow import FlowNetAgent
from utils.logging import Logger

args = parse_train_args()

if __name__ == '__main__':
    args = config_init(args)
    torch.manual_seed(args.seed)

    md = getattr(dynamics, args.molecule.title())(args, args.start_state)
    agent = FlowNetAgent(args, md)
    logger = Logger(args, md)
    mds = MDs(args, logger)

    logger.info("Start training\n")
    for rollout in range(args.num_rollouts):
        logger.info(f'Rollout: {rollout}')

        log = agent.sample(args, mds, args.train_temperature)
        loss = 0
        
        for _ in tqdm(range(args.trains_per_rollout), desc='Training'):
            loss += agent.train(args)
        loss = loss / args.trains_per_rollout
        
        agent.scheduler_update(loss)

        logger.log(loss, agent.policy, rollout, **log)
    logger.info("Finish training")
    
    wandb.finish()