# Multi-Agent Resource Allocation with Genetic Algorithm

This project simulates multiple agents in a 2D environment using a genetic algorithm (GA). The agents are trained to navigate the environment, avoid obstacles, grab resources, and deliver them to a specific location using neural networks.

## Features

- **Neural Network-Controlled Agents**: Each agent has a neural network that controls its movements.
- **Genetic Algorithm**: The population of agents evolves over time through selection, crossover, and mutation.
- **Pygame Visualization**: The environment and agents are visualized using Pygame, allowing you to see the simulation in action.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Ismail-El-Moufakir/Multi_Agent_Ressource_Gathering.git
    cd Multi_Agent_Ressource_Gathering
    ```

2. **Install Dependencies**:
    - Ensure you have Python installed (>= 3.7).
    - Install the required Python packages:
        -pygame
        -torch and numpy

3. **Run the Simulation**:
    ```bash
    python sample.py
    ```

## How It Works

### Map and Environment
- The environment is a grid with cells representing either empty space, obstacles, or a store area.
- Resources are randomly placed in the environment for agents to find and deliver.

### Agents
- Agents are controlled by a neural network with inputs based on the environment.
- They can move, detect nearby walls, and interact with resources.

### Genetic Algorithm
- **Population**: A population of agents is maintained.
- **Fitness Function**: Agents are scored based on resources grabbed, obstacles hit, and resources delivered.
- **Selection**: The best-performing agents are selected to create the next generation.
- **Crossover and Mutation**: Selected agents produce offspring with combined neural network weights, with some random mutations introduced.

## Customization

### Parameters
- **Population Size**: Adjust the number of agents in the population.
- **Tournament Size**: Change the selection pressure by altering the tournament size.
- **Crossover Rate**: Modify the rate at which crossover occurs between agents.
- **Mutation Rate**: Set the mutation rate to introduce randomness into the offspring.

### Map Configuration
- Modify the `InitMap_1()` method in the `map` class to create custom environments.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
