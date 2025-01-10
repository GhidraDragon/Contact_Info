#!/usr/bin/env python3
"""
Working code modeling an economic prosperity Markov chain with a 'twitch effect' on society.

DISCLAIMER:
-----------
1. This model is intentionally simplified. Real-world economics is far more complex.
2. The transition matrix and probabilities are hypothetical and do not necessarily reflect real data.
3. Security-wise, if you load transitions or states from external sources, make sure they are trusted 
   to avoid malicious code injection or manipulation. Handling user input safely is essential.

REQUIREMENTS:
-------------
- Python 3.7 or higher (tested up to Python 3.10)
- NumPy (for matrix manipulation and random choice)
"""

import numpy as np

def simulate_markov_chain_with_twitch(
    initial_state_index,
    transition_matrix,
    twitch_prob,
    twitch_transitions,
    num_steps = 30,
    random_seed = None
):
    """
    Simulates an economic Markov chain with a 'twitch effect'.
    
    Parameters
    ----------
    initial_state_index : int
        The index of the initial state in the state list.
    transition_matrix : np.ndarray
        Square transition matrix for the Markov chain. 
        transition_matrix[i, j] is the probability of going from state i to state j.
    twitch_prob : float
        Probability that a random 'twitch' shock happens at each step, overriding normal transitions.
    twitch_transitions : np.ndarray
        Square matrix of 'twitch' transitions. If a twitch occurs, we pick next state from twitch_transitions[i].
    num_steps : int, optional
        Number of steps (iterations) to simulate. Default is 30.
    random_seed : int, optional
        Optional random seed for reproducibility.
    
    Returns
    -------
    states_over_time : list of int
        The sequence of states visited over the simulation.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Number of states is the dimension of the transition matrix
    num_states = transition_matrix.shape[0]
    
    # Validate shapes
    assert transition_matrix.shape == (num_states, num_states), \
        "transition_matrix must be square"
    assert twitch_transitions.shape == (num_states, num_states), \
        "twitch_transitions must be square and match transition matrix size"
    
    # Validate probabilities sum to 1 (within a small tolerance)
    for i in range(num_states):
        if not np.isclose(transition_matrix[i].sum(), 1.0):
            raise ValueError("Row {} of transition_matrix does not sum to 1.".format(i))
        if not np.isclose(twitch_transitions[i].sum(), 1.0):
            raise ValueError("Row {} of twitch_transitions does not sum to 1.".format(i))
    
    current_state = initial_state_index
    states_over_time = [current_state]
    
    for _ in range(num_steps):
        # Roll a random number to see if the 'twitch' occurs
        if np.random.rand() < twitch_prob:
            # Use the twitch transitions for the next state
            next_state = np.random.choice(
                a=num_states,
                p=twitch_transitions[current_state]
            )
        else:
            # Use the normal Markov chain transitions
            next_state = np.random.choice(
                a=num_states,
                p=transition_matrix[current_state]
            )
        current_state = next_state
        states_over_time.append(current_state)
    
    return states_over_time

if __name__ == "__main__":
    # Example economic states:
    economic_states = [
        "Crisis",       # 0
        "Recession",    # 1
        "Stable",       # 2
        "Prosperity"    # 3
    ]
    
    # Standard Markov chain transition matrix (rows sum to 1).
    # This is a purely hypothetical set of transitions.
    #   e.g. from "Crisis" to "Crisis" is 0.60, from "Crisis" to "Recession" is 0.25, etc.
    transition_matrix = np.array([
        [0.60, 0.25, 0.10, 0.05],   # from Crisis
        [0.10, 0.50, 0.30, 0.10],   # from Recession
        [0.05, 0.10, 0.70, 0.15],   # from Stable
        [0.02, 0.08, 0.20, 0.70]    # from Prosperity
    ])
    
    # Twitch transitions represent sudden shocks. For instance, a twitch from "Prosperity" to "Crisis"
    # is given a small but notable probability. The distribution in each row still must sum to 1.
    twitch_transitions = np.array([
        [0.50, 0.20, 0.20, 0.10],   # from Crisis
        [0.05, 0.40, 0.25, 0.30],   # from Recession
        [0.05, 0.15, 0.60, 0.20],   # from Stable
        [0.10, 0.05, 0.25, 0.60]    # from Prosperity
    ])
    
    # Probability of the 'twitch' shock at each step
    twitch_probability = 0.10  # 10% chance of a shock each iteration
    
    # Simulate for 50 steps
    num_simulation_steps = 50
    
    # Choose your initial state, for example "Stable" -> index 2
    initial_state_index = 2
    
    # (Optional) set a random seed for reproducibility
    seed = 42
    
    # Run the simulation
    states_visited = simulate_markov_chain_with_twitch(
        initial_state_index = initial_state_index,
        transition_matrix = transition_matrix,
        twitch_prob = twitch_probability,
        twitch_transitions = twitch_transitions,
        num_steps = num_simulation_steps,
        random_seed = seed
    )
    
    # Print results
    print("Economic states over time:")
    for step, state_idx in enumerate(states_visited):
        print(f"Step {step:2d}: {economic_states[state_idx]}")
