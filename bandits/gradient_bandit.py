from typing import Tuple, List
import numpy as np
from .heroes import Heroes
from .helpers import run_trials, save_results_plots

def softmax(x, tau=1):
    """
    Compute softmax probabilities with temperature scaling.

    :param x: A 1-dimensional array of input values (logits).
    :param tau: Temperature parameter controlling exploration (higher tau -> more exploration).
    :return: A probability distribution over actions.
    """
    e_x = np.exp(np.array(x) / tau)
    return e_x / e_x.sum(axis=0)


def gradient_bandit(
    heroes: Heroes, 
    alpha: float, 
    use_baseline: bool = True
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform Gradient Bandit action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param alpha: The learning rate for updating preferences.
    :param use_baseline: Whether to use an average reward as a baseline.
    :return: 
        - rew_record: Rewards obtained at each timestep.
        - avg_ret_record: Average rewards up to each timestep.
        - tot_reg_record: Total regret accumulated up to each timestep.
        - opt_action_record: Percentage of optimal actions selected.
    """
    
    num_heroes = len(heroes.heroes)       # Number of actions (heroes)
    h = np.array([0] * num_heroes, dtype=float)  # Initialize preferences (logits) for each action
    rew_record = []                       # Rewards at each timestep
    avg_ret_record = []                   # Average reward up to each timestep
    tot_reg_record = []                   # Total regret up to each timestep
    opt_action_record = []                # Percentage of optimal actions selected
    
    reward_bar = 0                        # Average reward baseline
    total_rewards = 0                     # Total rewards accumulated
    total_regret = 0                      # Total regret accumulated

    # Determine optimal heroes and their true success probabilities
    true_succes_probs = [heroes.heroes[i]['true_success_probability'] for i in range(num_heroes)]
    optimal_reward = max(true_succes_probs)    # Maximum reward achievable
    optimal_hero_index = {i for i in range(num_heroes)
                          if true_succes_probs[i] == optimal_reward}  # Indices of optimal heroes
    total_optimal_actions_taken = 0            # Counter for optimal actions

    for t in range(heroes.total_quests):
        # Determine the baseline (average reward) if enabled
        baseline = reward_bar if use_baseline else 0

        # Compute the policy (probabilities of selecting each action)
        policy = softmax(h, tau=1)
        
        # Sample an action based on the policy
        random_num = np.random.rand()
        action = 0
        while random_num > policy[action]:
            random_num -= policy[action]
            action += 1

        # Perform the selected action and obtain the reward
        reward = heroes.attempt_quest(action)
        
        # Update the reward baseline
        reward_bar += (reward - reward_bar) / (t + 1)

        # Update preferences (h) using the gradient bandit algorithm
        h -= alpha * (reward - baseline) * policy  # Penalize all actions
        h[action] += alpha * (reward - baseline)  # Reward the selected action

        # Update performance metrics
        if action in optimal_hero_index:
            total_optimal_actions_taken += 1
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

    # Test various alpha values with baseline
    np.random.seed(42)  # Set random seed for reproducibility
    alpha_values = [0.05, 0.1, 2]  # Learning rate values to test
    results_list = []
    for alpha in alpha_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(
            2000, 
            heroes=heroes, 
            bandit_method=gradient_bandit, 
            alpha=alpha, 
            use_baseline=True
        )
        results_list.append({
            "exp_name": f"alpha={alpha}",
            "reward_rec": rew_rec,
            "average_rew_rec": avg_ret_rec,
            "tot_reg_rec": tot_reg_rec,
            "opt_action_rec": opt_act_rec
        })
    
    # Save results for alpha values with baseline
    save_results_plots(results_list, 
                       plot_title="Gradient Bandits (with Baseline) Experiment Results On Various Alpha Values",
                       results_folder='results', 
                       pdf_name='gradient_bandit_various_alpha_values_with_baseline.pdf')

    # Test various alpha values without baseline
    np.random.seed(42)  # Reset random seed for reproducibility
    alpha_values = [0.05, 0.1, 2]  # Learning rate values to test
    results_list = []
    for alpha in alpha_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(
            2000, 
            heroes=heroes, 
            bandit_method=gradient_bandit, 
            alpha=alpha, 
            use_baseline=False
        )
        results_list.append({
            "exp_name": f"alpha={alpha}",
            "reward_rec": rew_rec,
            "average_rew_rec": avg_ret_rec,
            "tot_reg_rec": tot_reg_rec,
            "opt_action_rec": opt_act_rec
        })

    # Save results for alpha values without baseline
    save_results_plots(results_list, 
                       plot_title="Gradient Bandits (without Baseline) Experiment Results On Various Alpha Values",
                       results_folder='results', 
                       pdf_name='gradient_bandit_various_alpha_values_without_baseline.pdf')
