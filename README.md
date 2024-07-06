# GFlowNet for Sampling Transition Paths for Large Molecules.

## Installation

1. First, create a new Conda environment:
    ```
    conda create -n gflow_tps python=3.9
    ```

2. Activate the newly created environment:
    ```
    conda activate gflow_tps
    ```

3. Install pytorch.

4. Install the openmmtools for Molecular Dynamics (MD) simulation using the following commands:
    ```
    conda install -c conda-forge openmmtools
    ```

5. Install the openmmforcefields for forcefields of large proteins using the following commands:
    ```
    git clone https://github.com/openmm/openmmforcefields.git
    ```
    ```
    cd openmmforcefields
    ```
    ```
    pip install -e .
    ```
    ```
    cd ..
    ```
6. Install another packages using the following commands:
    ```
    pip install tqdm wandb mdtraj matplotlib
    ```

## Usage

- **Training**: Run the following command to start training:
    ```
    bash scripts/train_alanine.sh
    ```

- **Evaluation**: Run the following command to perform evaluation:
    ```
    bash scripts/eval_alanine.sh
    ```

## Reproduce

- **Table**: Run the following command to reproduce the results of alanine dipeptide with bias potential and reward for fixed length:
    ```
    bash scripts/reproduce_table.sh
    ```

- **Visualization**: Run the following command to reproduce the figure which shows transition paths of multiple reaction channels:
    ```
    bash scripts/reproduce_figure.sh
    ```

## Results

**Alanine Dipeptide**
![3D trajectory of Alanine Dipeptide](pages/alanine.gif)

**Polyproline**
![3D trajectory of Polyproline](pages/poly.gif)

**Chignolin**
![3D trajectory of Chignolin](pages/chignolin.gif)