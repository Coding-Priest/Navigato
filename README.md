# Reinforcement Learning for Robot Navigation

Custom integration of OpenAI Gym and Gazebo environment! This module allows seamless implementation of gym functions on a custom Gazebo environment, enabling you to train reinforcement learning agents for robot navigation tasks.

## Features

- Complete integration of OpenAI Gym and Gazebo environment
- Support for higher-level RL libraries for robot navigation tasks (Such as PTAN, Tensorflow-RL)
- Customizable Gazebo-X-Gym code for easy adaptation to your specific needs

## Prerequisites

- ROS (Robot Operating System)
- Gazebo
- OpenAI Gym
- Python 3.x

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/rl-robot-navigation.git
   cd rl-robot-navigation
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Simulation

1. Start the Gazebo environment:
   ```
   rosrun rl_robot_navigation GazeboEnv-V8.py
   ```

2. Open your Jupyter Notebook:
   ```
   jupyter notebook
   ```

3. Navigate to your notebook file and open it.

4. Execute the cells in your notebook to run your reinforcement learning code.

That's it! Your reinforcement learning agent should now be interacting with the Gazebo environment.

## Customization

You can easily adapt the Gazebo-X-Gym code to fit your specific needs. Refer to the `GazeboEnv-V8.py` file for details on the environment setup and modify as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
