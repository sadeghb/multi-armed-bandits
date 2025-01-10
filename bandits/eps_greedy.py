from typing import Tuple, List
import numpy as np
from .heroes import Heroes
from .helpers import run_trials, save_results_plots

def eps_greedy(
    heroes: Heroes, 
    eps: float, 
    init_value: float = .0
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform epsilon-greedy action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param eps: The epsilon value for exploration (random actions) vs. exploitation (greedy actions).
    :param init_value: Initial estimation of each hero's value.
    :return: 
        - rew_record: Rewards obtained at each timestep.
        - avg_ret_record: Average rewards obtained up to each timestep.
        - tot_reg_record: Total regret accumulated up to each timestep.
        - opt_action_record: Percentage of optimal actions selected.
    """
    
    num_heroes = len(heroes.heroes)       # Number of actions (heroes)
    values = [init_value] * num_heroes    # Initialize action values
    rew_record = []                       # Rewards at each timestep
    avg_ret_record = []                   # Average reward up to each timestep
    tot_reg_record = []                   # Total regret up to each timestep
    opt_action_record = []                # Percentage of optimal actions selected
    
    total_rewards = 0                     # Total rewards accumulated
    total_regret = 0                      # Total regret accumulated

    # Retrieve true success probabilities and determine optimal heroes
    true_succes_probs = [heroes.heroes[i]['true_success_probability'] for i in range(num_heroes)]
    optimal_reward = max(true_succes_probs)    # Optimal reward
    optimal_hero_index = {i for i in range(num_heroes)
                          if true_succes_probs[i] == optimal_reward}  # Indices of optimal heroes
    total_optimal_actions_taken = 0            # Total optimal actions taken
    
    for t in range(heroes.total_quests):
        # Identify the greedy actions (actions with the highest estimated values)
        max_value = max(values)
        greedy_actions = [i for i in range(num_heroes) if values[i] == max_value]
        
        # Choose an action using epsilon-greedy strategy
        greedy_action = np.random.choice(greedy_actions)  # Select one greedy action randomly
        random_action = np.random.randint(0, num_heroes)  # Select a random action
        action = greedy_action if np.random.rand() < 1 - eps else random_action
        
        # Attempt a quest with the chosen action and get the reward
        reward = heroes.attempt_quest(action)
        
        # Update the estimated value for the selected action
        values[action] += (reward - values[action]) / (1 + heroes.heroes[action]['n_quests'])

        # Update optimal actions counter if the chosen action was optimal
        if action in optimal_hero_index:
            total_optimal_actions_taken += 1

        # Update performance metrics
        total_rewards += reward
        total_regret += optimal_reward - reward

        # Record metrics for this timestep
        rew_record.append(reward)
        avg_ret_record.append(total_rewards / (t + 1))
        tot_reg_record.append(total_regret)
        opt_action_record.append(total_optimal_actions_taken / (t + 1))
    
    return rew_record, avg_ret_record, tot_reg_record, opt_action_record


if __name__ == "__main__":
    # Define the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])

    # Test various epsilon values
    np.random.seed(42)  # Set random seed for reproducibility
    eps_values = [0.2, 0.1, 0.01, 0.]  # Epsilon values for testing
    results_list = []
    for eps in eps_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(
            2000, 
            heroes=heroes, 
            bandit_method=eps_greedy, 
            eps=eps, 
            init_value=0.0
        )
        
        results_list.append({
            'exp_name': f'eps={eps}',
            'reward_rec': rew_rec,
            'average_rew_rec': avg_ret_rec,
            'tot_reg_rec': tot_reg_rec,
            'opt_action_rec': opt_act_rec
        })

    # Save results for different epsilon values
    save_results_plots(results_list, 
                       plot_title='Epsilon-Greedy Experiment Results On Various Epsilons', 
                       results_folder='results', 
                       pdf_name='epsilon_greedy_various_epsilons.pdf')

    # Test various initial value settings with eps=0.0
    np.random.seed(42)  # Reset random seed for reproducibility
    init_values = [0.0, 0.5, 1]  # Initial values for testing
    results_list = []
    for init_val in init_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(
            2000, 
            heroes=heroes, 
            bandit_method=eps_greedy, 
            eps=0.0, 
            init_value=init_val
        )
        
        results_list.append({
            'exp_name': f'init_val={init_val}',
            'reward_rec': rew_rec,
            'average_rew_rec': avg_ret_rec,
            'tot_reg_rec': tot_reg_rec,
            'opt_action_rec': opt_act_rec
        })
    
    # Save results for different initial values
    save_results_plots(results_list, 
                       plot_title='Epsilon-Greedy Experiment Results On Various Initial Values',
                       results_folder='results', 
                       pdf_name='epsilon_greedy_various_init_values.pdf')
