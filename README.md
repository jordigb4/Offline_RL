# Offline Reinforcement Learning Algorithm Implementation (BCQ & CQL)

## Project Description
This project focuses on Offline Reinforcement Learning (Offline RL), also known as Batch RL. It implements two prominent algorithms designed to learn effective policies purely from static datasets without further environment interaction:
- Behavior Cloning Q-Learning (BCQ)
- Conservative Q-Learning (CQL)
The objective is to reproduce the key findings and demonstrate the capabilities of these algorithms in leveraging pre-collected data.

## Papers Replicated
- BCQ: "Off-Policy Deep Reinforcement Learning without Exploration" (Fujimoto et al., 2019)
- CQL: "Conservative Q-Learning for Offline Reinforcement Learning" (Kumar et al., 2020)

## Algorithms Overview

### BCQ (Behavior Cloning Q-Learning)
- **Type**: Offline RL, Actor-Critic / Value-Based
- **Key Features**:
  - Aims to mitigate extrapolation error by constraining actions to be close to the behavior policy distribution.
  - Often uses a generative model (like a VAE) to model the behavior policy's action distribution.
  - Selects actions by sampling from the generative model and perturbing them based on Q-value estimates.
  - Learns Q-functions using standard off-policy methods (like TD3 or DDPG) but with action constraints.

### CQL (Conservative Q-Learning)
- **Type**: Offline RL, Value-Based (can be adapted for Actor-Critic)
- **Key Features**:
  - Addresses Q-value overestimation for out-of-distribution actions via explicit regularization.
  - Modifies the standard Bellman error objective to penalize high Q-values for actions unlikely under the behavior policy.
  - Learns a conservative Q-function that provides a lower bound on the true policy value.
  - Aims for robust performance by avoiding optimistic value estimates outside the data distribution.

## Contributions
Contributions are welcome! Please read the contributing guidelines before submitting a pull request.

## License
MIT License

## References
1. Fujimoto, S., Meger, D., & Precup, D. (2019). Off-Policy Deep Reinforcement Learning without Exploration.
2. Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning.
