import torch
from tqdm import tqdm
from dynamics import dynamics

class MDs:
    def __init__(self, args):
        self.device = args.device
        self.molecule = args.molecule
        self.end_state = args.end_state
        self.num_samples = args.num_samples
        self.start_state = args.start_state

        # self._init_values_(args)
        self.mds = self._init_mds(args)
        self.target_positions = self._init_target_positions(args)

    def _init_values_(self, args):
        logger.info(f"Initializing MDs")
        
        # Initialize MDs
        mds = []
        for _ in tqdm(range(self.num_samples)):
            md = getattr(dynamics, self.molecule.title())(args, self.start_state)
            mds.append(md)
        self.mds = mds
        
        # Initialize target positions
        target_positions = getattr(dynamics, self.molecule.title())(args, self.end_state).position
        target_positions = torch.tensor(target_positions, dtype=torch.float, device=self.device).unsqueeze(0)
        self.target_positions = torch.atack([target_positions] * self.num_samples)
    
    def _init_mds(self, args):
        mds = []
        for _ in tqdm(range(self.num_samples)):
            md = getattr(dynamics, self.molecule.title())(args, self.start_state)
            mds.append(md)
        return mds

    def _init_target_positions(self, args):
        print(f"Get position of {self.end_state} of {self.molecule}")

        target_positions = getattr(dynamics, self.molecule.title())(args, self.end_state).position
        target_positions = torch.tensor(target_positions, dtype=torch.float, device=self.device).unsqueeze(0)
        target_positions = torch.stack([target_positions] * self.num_samples)
        return target_positions

    def step(self, force):
        force = force.detach().cpu().numpy()
        for i in range(self.num_samples):
            self.mds[i].step(force[i])

    def report(self):
        positions, velocities, forces, potentials = [], [], [], []
        for i in range(self.num_samples):
            position, velocity, force, potential = self.mds[i].report()
            positions.append(position); velocities.append(velocity); forces.append(force); potentials.append(potential)
            
        positions = torch.tensor(positions, dtype=torch.float, device=self.device)
        velocities = torch.tensor(velocities, dtype=torch.float, device=self.device)
        forces = torch.tensor(forces, dtype=torch.float, device=self.device)
        potentials = torch.tensor(potentials, dtype=torch.float, device=self.device)
        return positions, velocities, forces, potentials
    
    def reset(self):
        for i in range(self.num_samples):
            self.mds[i].reset()

    def set_temperature(self, temperature):
        for i in range(self.num_samples):
            self.mds[i].set_temperature(temperature)