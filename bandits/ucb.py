from typing import Tuple, List
import numpy as np
from .heroes import Heroes
from .helpers import run_trials, save_results_plots

def ucb(
    heroes: Heroes, 
    c: float, 
    init_value: float = 0.0
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform Upper Confidence Bound (UCB) action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param c: Exploration coefficient balancing exploration and exploitation.
    :param init_value: Initial estimated value for each hero.
    :return: 
        - rew_record: Rewards recorded at each timestep.
        - avg_ret_record: Average rewards up to step t.
        - tot_reg_record: Total regret up to step t.
        - opt_action_record: Percentage of optimal actions selected.
    """

    num_heroes = len(heroes.heroes)  # Number of available heroes
    values = [init_value] * num_heroes  # Initial action values
    rew_record = []  # Rewards at each timestep
    avg_ret_record = []  # Average rewards up to each timestep
    tot_reg_record = []  # Total regret up to each timestep
    opt_action_record = []  # Percentage of optimal actions selected

    total_rewards = 0  # Cumulative rewards
    total_regret = 0  # Cumulative regret

    # True success probabilities and optimal actions
    true_succes_probs = [hero['true_success_probability'] for hero in heroes.heroes]
    optimal_reward = max(true_succes_probs)
    optimal_hero_index = {i for i, prob in enumerate(true_succes_probs) if prob == optimal_reward}
    total_optimal_actions_taken = 0

    for t in range(heroes.total_quests):
        # Compute UCB values for all heroes
        ucb_values = [
            float('inf') if t == 0 or hero['n_quests'] == 0
            else values[i] + c * np.sqrt(np.log(t) / hero['n_quests'])
            for i, hero in enumerate(heroes.heroes)
        ]

        # Select action with maximum UCB value
        max_ucb_value = max(ucb_values)
        ucb_actions = [i for i, ucb_val in enumerate(ucb_values) if ucb_val == max_ucb_value]
        action = np.random.choice(ucb_actions)

        # Attempt quest and update metrics
        reward = heroes.attempt_quest(action)
        values[action] += (reward - values[action]) / heroes.heroes[action]['n_quests']

        if action in optimal_hero_index:
            total_optimal_actions_taken += 1
        total_rewards += reward
        total_regret += optimal_reward - reward

        # Record metrics
        rew_record.append(reward)
        avg_ret_record.append(total_rewards / (t + 1))
        tot_reg_record.append(total_regret)
        opt_action_record.append(total_optimal_actions_taken / (t + 1))

    return rew_record, avg_ret_record, tot_reg_record, opt_action_record

if __name__ == "__main__":
    # Initialize the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])

    # Test UCB with various exploration coefficients (c values)
    np.random.seed(42)  # Ensure reproducibility
    c_values = [0.0, 0.5, 2.0]
    results_list = []

    for c in c_values:
        # Run trials for each value of c
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(
            number_of_trials=2000,
            heroes=heroes,
            bandit_method=ucb,
            c=c,
            init_value=0.0
        )

        results_list.append({
            'exp_name': f'c={c}',
            'reward_rec': rew_rec,
            'average_rew_rec': avg_ret_rec,
            'tot_reg_rec': tot_reg_rec,
            'opt_action_rec': opt_act_rec
        })

    # Save and plot results
    save_results_plots(
        results_list,
        plot_title='UCB Experiment Results on Various C Values',
        results_folder='results',
        pdf_name='ucb_various_c_values.pdf'
    )
