import torch
import proxy
import openmm.unit as unit

from tqdm import tqdm
from utils.utils import pairwise_dist 
from utils.optim import set_optimizer, set_scheduler

class FlowNetAgent:
    def __init__(self, args, md):
        self.args = args
        self.num_particles = md.num_particles
        self.v_scale = torch.tensor(md.v_scale, dtype=torch.float, device=args.device)
        self.f_scale = torch.tensor(md.f_scale.value_in_unit(unit.femtosecond), dtype=torch.float, device=args.device)
        self.std = torch.tensor(md.std.value_in_unit(unit.nanometer/unit.femtosecond), dtype=torch.float, device=args.device)
        self.masses = torch.tensor(md.masses.value_in_unit(md.masses.unit), dtype=torch.float, device=args.device).unsqueeze(-1)
        
        self.policy = getattr(proxy, args.molecule.title())(args, md)
        if args.type == 'train':
            self.replay = ReplayBuffer(args, md)
        
        # Set optimizer and scheduler for log z, mlp
        self.log_z_optimizer = set_optimizer(args.log_z_optimizer, [self.policy.log_z], args.log_z_lr)    
        self.log_z_scheduler = set_scheduler(args.log_z_scheduler, self.log_z_optimizer, lr=args.log_z_lr, args=args)
        self.mlp_optimizer = set_optimizer(args.mlp_optimizer, self.policy.mlp.parameters(), args.mlp_lr) 
        self.mlp_scheduler = set_scheduler(args.mlp_scheduler, self.mlp_optimizer, lr=args.mlp_lr, args=args)
        self.log_z_lr = self.log_z_optimizer.param_groups[0]['lr'] if self.log_z_scheduler is not None else args.log_z_lr
        self.mlp_lr = self.mlp_optimizer.param_groups[0]['lr'] if self.mlp_scheduler is not None else args.mlp_lr
        
        self.log_z_2_optimizer = set_optimizer(args.log_z_optimizer, [self.policy.log_z_2], args.log_z_lr)    
        self.log_z_2_scheduler = set_scheduler(args.log_z_scheduler, self.log_z_2_optimizer, lr=args.log_z_lr, args=args)
        self.log_z_2_lr = self.log_z_2_optimizer.param_groups[0]['lr'] if self.log_z_2_scheduler is not None else args.log_z_lr

    def sample(self, args, mds, temperature):
        positions = torch.zeros((args.num_samples, args.num_steps+1, self.num_particles, 3), device=args.device)
        actions = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        biases = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        potentials = torch.zeros(args.num_samples, args.num_steps+1, device=args.device)
        
        position, _, _, potential = mds.report()
        
        positions[:, 0] = position
        potentials[:, 0] = potential

        mds.set_temperature(temperature)
        for s in tqdm(range(args.num_steps), desc='Sampling'):
            # bias = args.bias_scale * self.policy(position.detach()).squeeze().detach()
            position_n_target = torch.cat([position, mds.target_positions.squeeze()], dim=0)
            bias = args.bias_scale * self.policy(position_n_target.detach()).squeeze().detach()
            mds.step(bias)
            
            next_position, velocity, force, potential = mds.report()

            # extract noise
            noise = (next_position - position) / args.timestep - (self.v_scale * velocity + self.f_scale * force / self.masses)

            positions[:, s+1] = next_position
            potentials[:, s+1] = potential - (bias*next_position).sum((1, 2)) # Subtract bias potential to get true potential

            position = next_position
            bias = 1e-6 * bias # kJ/(mol*nm) -> (da*nm)/fs**2
            action = self.f_scale * bias / self.masses + noise
            
            actions[:, s] = action
            biases[:, s] = bias
        mds.reset()

        log_md_reward = -0.5 * torch.square(actions/self.std).mean((1, 2, 3))
        
        # NOTE: Calculate log reward
        target_pds = pairwise_dist(mds.target_positions)
        target_pds = torch.stack([pairwise_dist(pos) for pos in mds.target_positions]).squeeze()
        
        
        log_target_reward = torch.zeros(args.num_samples, args.num_steps+1, device=args.device)
        for i in range(args.num_samples):
            pd = pairwise_dist(positions[i])
            log_target_reward[i] = - torch.square((pd-target_pds[i])/args.sigma).mean((1, 2))
            # log_target_reward[i] = - torch.square((pd-target_pds)/args.sigma).mean((1, 2))
        log_target_reward, last_idx = log_target_reward.max(1)
        
        # NOTE: Another method to calculate log reward
        # log_target_reward = torch.zeros(args.num_samples, args.num_steps, device=args.device)
        # for i in range(args.num_samples) :    
        #     aligned_target_position, rmsd = kabsch(mds.target_position, positions[i][1:])
        #     target_velocity = (aligned_target_position - positions[ill:-1]) / args.timestep
        #     log_target_reward[i] = -0.5 * torch.square((target_velocity-velocities[i][1:])/self.std).mean((1, 2))
        # # print (log_target_reward)
        # log_target_reward, last_idx = log_target_reward.max
        
        log_reward = log_md_reward + log_target_reward
        log_likelihood = (-1/2) * torch.square(noise).mean((0, 1, 2))

        if args.type == 'train':
            self.replay.add((positions, actions, log_reward, mds.target_positions))
            # TODO: HER add transition with additional goals
        
        log = {
            'actions': actions,
            'last_idx': last_idx,
            'positions': positions, 
            'potentials': potentials,
            'target_position': mds.target_positions,
            'last_position': positions[torch.arange(args.num_samples), last_idx],
            'log_z_lr': self.log_z_lr,
            'mlp_lr': self.mlp_lr
        }
        return log

    def train(self, args):
        positions, actions, log_reward, target_positions = self.replay.sample()

        biases = args.bias_scale * self.policy(positions[:, :-1])
        biases = 1e-6 * biases # kJ/(mol*nm) -> (da*nm)/fs**2
        biases = self.f_scale * biases / self.masses
        
        # start_n_goal = torch.stack((positions[:, 0].reshape(args.num_samples, -1), target_positions))
        # start_n_goal = torch.cat((positions[:, 0].reshape(args.num_samples, -1), target_positions.squeeze().reshape(args.num_samples, -1)), dim=1)
        log_z = self.policy.log_z + self.policy.log_z_2
        log_forward = -0.5 * torch.square((biases-actions)/self.std).mean((1, 2, 3))
        loss = (log_z+log_forward-log_reward).square().mean() 
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_([self.policy.log_z], args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_([self.policy.log_z_2], args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.policy.mlp.parameters(), args.max_grad_norm)
        
        self.mlp_optimizer.step()
        self.log_z_optimizer.step()
        self.mlp_optimizer.zero_grad()
        self.log_z_optimizer.zero_grad()
        
        self.log_z_2_optimizer.step()
        self.log_z_2_optimizer.zero_grad()
        
        return loss.item()
    
    def scheduler_update(self, loss=0):
        if self.log_z_scheduler is not None:
            if self.args.log_z_scheduler == "plateau":
                self.log_z_scheduler.step(loss)
            else:
                self.log_z_scheduler.step()
            self.log_z_lr = self.log_z_optimizer.param_groups[0]['lr']
        if self.mlp_scheduler is not None:
            if self.args.mlp_scheduler == "plateau":
                self.mlp_scheduler.step(loss)
            else:
                self.mlp_scheduler.step()
            self.mlp_lr = self.mlp_optimizer.param_groups[0]['lr']
            
    def target_positions2keys(self, target_positions):
        print(target_positions.shape)
        return 1

class ReplayBuffer:
    def __init__(self, args, md):
        self.positions = torch.zeros((args.buffer_size, args.num_steps+1, md.num_particles, 3), device=args.device)
        self.actions = torch.zeros((args.buffer_size, args.num_steps, md.num_particles, 3), device=args.device)
        self.log_reward = torch.zeros(args.buffer_size, device=args.device)
        self.target_positions = torch.zeros((args.buffer_size, 1, md.num_particles, 3), device=args.device)

        self.idx = 0
        self.buffer_size = args.buffer_size
        self.num_samples = args.num_samples

    def add(self, data):
        indices = torch.arange(self.idx, self.idx+self.num_samples) % self.buffer_size
        self.idx += self.num_samples

        self.positions[indices], self.actions[indices], self.log_reward[indices], self.target_positions[indices] = data
        
    def sample(self):
        indices = torch.randperm(min(self.idx, self.buffer_size))[:self.num_samples]
        return self.positions[indices], self.actions[indices], self.log_reward[indices], self.target_positions[indices]