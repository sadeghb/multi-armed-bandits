from typing import Tuple, List
import numpy as np
from .heroes import Heroes
from .helpers import run_trials, save_results_plots

def boltzmann_policy(x, tau):
    """
    Returns an index sampled from the softmax probabilities with temperature tau.
    
    :param x: 1-dimensional array of action values.
    :param tau: Temperature parameter controlling exploration vs exploitation.
    :return: idx -- chosen index based on softmax probabilities.
    """
    
    # Compute softmax probabilities with temperature tau
    policy = np.exp(np.array(x) / tau) / (np.exp(np.array(x) / tau)).sum()
    
    # Generate a random number and sample an action based on the probabilities
    random_num = np.random.rand()
    index = 0
    while random_num > policy[index]:
        random_num -= policy[index]
        index += 1

    return index


def boltzmann(
    heroes: Heroes, 
    tau: float = 0.1, 
    init_value: float = .0
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform Boltzmann action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param tau: Temperature parameter for the Boltzmann policy.
    :param init_value: Initial estimation of each hero's value.
    :return: 
        - rew_record: Rewards obtained at each timestep.
        - avg_ret_record: Average reward obtained up to each timestep.
        - tot_reg_record: Total regret accumulated up to each timestep.
        - opt_action_record: Percentage of optimal actions selected.
    """

    # Number of actions (heroes)
    num_heroes = len(heroes.heroes)
    
    # Initialize action values and records
    values = [init_value] * num_heroes
    rew_record = []                       # Record of rewards
    avg_ret_record = []                   # Record of average rewards
    tot_reg_record = []                   # Record of total regret
    opt_action_record = []                # Record of optimal actions taken
    
    # Initialize performance metrics
    total_rewards = 0
    total_regret = 0

    # Extract true success probabilities and identify the optimal heroes
    true_succes_probs = [heroes.heroes[i]['true_success_probability'] for i in range(num_heroes)]
    optimal_reward = max(true_succes_probs)
    optimal_hero_index = {i for i in range(num_heroes)
                          if true_succes_probs[i] == optimal_reward}
    total_optimal_actions_taken = 0
    
    # Iterate through the total number of quests
    for t in range(heroes.total_quests):
        # Select an action using the Boltzmann policy
        action = boltzmann_policy(values, tau)
        
        # Attempt the quest and observe the reward
        reward = heroes.attempt_quest(action)
        
        # Update the estimated value for the selected action
        values[action] += (reward - values[action]) / (1 + heroes.heroes[action]['n_quests'])
        
        # Track if the chosen action was optimal
        if action in optimal_hero_index:
            total_optimal_actions_taken += 1
        
        # Update performance metrics
        total_rewards += reward
        total_regret += optimal_reward - reward
        
        # Record metrics
        rew_record.append(reward)
        avg_ret_record.append(total_rewards / (t + 1))
        tot_reg_record.append(total_regret)
        opt_action_record.append(total_optimal_actions_taken / (t + 1))
    
    # Return all recorded metrics
    return rew_record, avg_ret_record, tot_reg_record, opt_action_record



if __name__ == "__main__":
    # Define the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])

    # Test various tau values
    np.random.seed(42) # Random seed for reproducibility
    tau_values = [0.01, 0.1, 1, 10]
    results_list = []
    for tau in tau_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(2000,
                                                                    heroes=heroes, bandit_method=boltzmann,
                                                                    tau=tau, init_value=0)
        
        results_list.append({
            "exp_name": f"tau={tau}",
            "reward_rec": rew_rec,
            "average_rew_rec": avg_ret_rec,
            "tot_reg_rec": tot_reg_rec,
            "opt_action_rec": opt_act_rec
        })

    save_results_plots(results_list, plot_title="Boltzmann Experiment Results On Various Tau Values",
                       results_folder='results', pdf_name='boltzmann_various_tau_values.pdf')
