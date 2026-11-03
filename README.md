# Implementation of Value Iteration for Optimal Policy Computation using Gymnasium

## Date : 03-11-2026

## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process (MDP) using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman Optimality Equation.

---

## Problem Statement

To find the optimal state-value function and optimal policy for a given FrozenLake environment using the Value Iteration algorithm. The algorithm evaluates possible actions in each state, updates the state values iteratively, and determines the optimal action that maximizes the expected cumulative reward.

---

## Software Requirements

- **Programming Language:** Python
- **Platform:** Google Colab / Jupyter Notebook
- **Libraries:**
  - NumPy
  - Matplotlib
  - Gymnasium

### Installation

```bash
pip install numpy matplotlib gymnasium
```

---

## Environment Description

The experiment uses the custom `FrozenLake-v1` environment provided by Gymnasium.

The environment consists of a 4 × 4 grid containing four types of states:

| Symbol | Description |
|:---:|---|
| S | Starting State |
| F | Frozen State (Safe State) |
| H | Hole State (Terminal State) |
| G | Goal State (Terminal State) |

### Custom Environment

```python
env_desc = [
    "HFFF",
    "FFHS",
    "FFFF",
    "GFFH"
]
```

### Environment Configuration

- Grid Size: 4 × 4
- Total States: 16
- Total Actions: 4
- Slippery Mode: True

When `is_slippery=True`, the agent may move in an unintended direction due to the stochastic nature of the environment.

---

## MDP Representation

The FrozenLake environment is represented as a Markov Decision Process (MDP) defined by the tuple:

**MDP = (S, A, P, R, γ)**

Where:

- **S (States):** Set of 16 possible states in the environment.
- **A (Actions):** Four possible actions: Left, Down, Right, and Up.
- **P (Transition Probability):** Probability of moving from one state to another after performing an action.
- **R (Reward):** Reward received after taking an action and reaching the next state.
- **γ (Discount Factor):** Determines the importance of future rewards.

### Action Representation

| Action Number | Action |
|:---:|---|
| 0 | Left |
| 1 | Down |
| 2 | Right |
| 3 | Up |

---

## Theory

**Value Iteration** is a dynamic programming algorithm used to find the optimal policy of a Markov Decision Process.

It repeatedly updates the state-value function using the Bellman Optimality Equation until the values converge.

### Bellman Optimality Equation

The optimal state-value function is calculated using:

\[
V(s)=\max_{a}\sum_{s'}P(s'|s,a)
[R(s,a,s')+\gamma V(s')]
\]

Where:

- V(s) represents the value of the current state.
- P(s'|s,a) represents the transition probability.
- R(s,a,s') represents the immediate reward.
- γ represents the discount factor.
- max selects the action with the highest expected value.

### Convergence Condition

The algorithm continues updating the value function until the maximum change between successive iterations becomes smaller than the threshold θ.

\[
\Delta < \theta
\]

Once convergence is achieved, the optimal policy is obtained by selecting the action with the maximum expected value in each state.

---

## Algorithm

1. Import the required libraries: Gymnasium, NumPy, and Matplotlib.
2. Create the custom FrozenLake environment with a 4 × 4 grid.
3. Initialize the environment and obtain the number of states and actions.
4. Initialize the state-value function with zeros.
5. Set the discount factor γ and convergence threshold θ.
6. For every state, calculate the expected value of all possible actions using the Bellman Optimality Equation.
7. Update the state-value function using the maximum action value.
8. Determine the corresponding best action using `np.argmax()`.
9. Repeat the updates until the convergence condition is satisfied.
10. Return the optimal state-value function, optimal policy, and number of iterations.
11. Display the final value function and optimal policy.

---

## Python Program

```python
# -------------------------------------------------
# Import Required Libraries
# -------------------------------------------------

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env_desc = [
    "HFFF",
    "FFHS",
    "FFFF",
    "GFFH"
]

env = gym.make(
    "FrozenLake-v1",
    desc=env_desc,
    is_slippery=True
)

env = env.unwrapped


# -------------------------------------------------
# Number of States and Actions
# -------------------------------------------------

n_states = env.observation_space.n
n_actions = env.action_space.n


# -------------------------------------------------
# Parameters
# -------------------------------------------------

gamma = 0.9
theta = 1e-8


# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize Value Function and Policy
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)

    iteration = 0

    # Value Iteration
    while True:

        delta = 0

        new_V = np.zeros(n_states)
        new_policy = np.zeros(n_states, dtype=int)

        for s in range(n_states):

            action_values = np.zeros(n_actions)

            for a in range(n_actions):

                for prob, next_state, reward, done in env.P[s][a]:

                    action_values[a] += prob * (
                        reward + gamma * V[next_state]
                    )

            # Find Maximum Value and Corresponding Action
            new_V[s] = np.max(action_values)
            new_policy[s] = np.argmax(action_values)

            delta = max(
                delta,
                abs(V[s] - new_V[s])
            )

        # Update Value Function and Policy
        V = new_V
        policy = new_policy

        iteration += 1

        # Check Convergence
        if delta < theta:
            break

    return V, policy, iteration


# -------------------------------------------------
# Execute Value Iteration
# -------------------------------------------------

V, policy, iterations = value_iteration(env)


# -------------------------------------------------
# Display Output
# -------------------------------------------------

print("Value Iteration Completed")

print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))


# -------------------------------------------------
# Display Optimal Policy
# -------------------------------------------------

action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()
```

---

## Output

### Number of Iterations

```text
Number of Iterations: 289
```

### Optimal State-Value Function

```text
[[0.     0.7924 0.7581 0.7468]
 [0.8829 0.8506 0.     0.7581]
 [0.942  0.9024 0.8506 0.7924]
 [0.     0.942  0.8829 0.    ]]
```

### Optimal Policy

```text
[['L' 'R' 'U' 'L']
 ['D' 'L' 'L' 'R']
 ['D' 'L' 'D' 'U']
 ['L' 'L' 'L' 'L']]
```

**Action Representation:**

- L – Left
- D – Down
- R – Right
- U – Up

---

## Result

The Value Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The optimal state-value function and optimal policy were computed after **289 iterations**. The algorithm converged when the maximum change in the state-value function became smaller than the specified convergence threshold.

---

## Inference

The experiment demonstrates how Value Iteration can be used to solve a Markov Decision Process and determine an optimal policy.

The state-value function represents the expected cumulative discounted reward from each state, while the optimal policy identifies the best action to perform in each state.

Since the FrozenLake environment uses slippery mode, the agent experiences stochastic state transitions. Value Iteration considers these transition probabilities while calculating the expected action values.

Thus, the experiment demonstrates the application of Dynamic Programming for finding an optimal policy in a stochastic environment.
