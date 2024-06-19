import torch
from tqdm import tqdm
from dynamics import dynamics

class MDs:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = args.device
        self.molecule = args.molecule
        self.end_state = args.end_state
        self.num_samples = args.num_samples
        self.start_state = args.start_state

        print(f"Initializing MDs")
        self.mds = []
        self.target_positions = []
        
        # Initialize MDs
        if hasattr(args, "direction") and args.direction == "two-way":
            for i in tqdm(range(self.num_samples)):
                if i % 2 == 0:
                    md = getattr(dynamics, self.molecule.title())(args, self.start_state)
                else:
                    md = getattr(dynamics, self.molecule.title())(args, self.end_state)
                self.mds.append(md)
        else:
            for _ in tqdm(range(self.num_samples)):
                md = getattr(dynamics, self.molecule.title())(args, self.start_state)
                self.mds.append(md)
            
        # Initialize target positions
        if hasattr(args, "direction") and args.direction == "two-way":
            for i in tqdm(range(self.num_samples)):
                if i % 2 == 0:
                    target_position = getattr(dynamics, self.molecule.title())(args, self.end_state).position
                else:
                    target_position = getattr(dynamics, self.molecule.title())(args, self.start_state).position
                target_position = torch.tensor(target_position, dtype=torch.float, device=self.device).unsqueeze(0)
                self.target_positions.append(target_position)
            self.target_positions = torch.stack(self.target_positions)
        else:
            target_positions = getattr(dynamics, self.molecule.title())(args, self.end_state).position
            target_positions = torch.tensor(target_positions, dtype=torch.float, device=self.device).unsqueeze(0)
            self.target_positions = torch.stack([target_positions] * self.num_samples)

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