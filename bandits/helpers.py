import matplotlib.pyplot as plt
import os
import numpy as np


def run_trials(number_of_trials, heroes, bandit_method, **kwargs):
    """
    Run a bandit method multiple times and average the results.

    :param number_of_trials: Number of independent trials to run.
    :param heroes: Instance of the Heroes class, representing the bandit problem.
    :param bandit_method: The bandit method function to test (e.g., eps_greedy, gradient_bandit).
    :param kwargs: Additional arguments required by the bandit method.
    :return: 
        - rew_rec: Average rewards at each timestep over all trials.
        - avg_ret_rec: Average of cumulative returns at each timestep over all trials.
        - tot_reg_rec: Average total regret at each timestep over all trials.
        - opt_act_rec: Average percentage of optimal actions selected at each timestep over all trials.
    """
    # Initialize arrays to accumulate results
    rew_rec = np.zeros(heroes.total_quests)
    avg_ret_rec = np.zeros(heroes.total_quests)
    tot_reg_rec = np.zeros(heroes.total_quests)
    opt_act_rec = np.zeros(heroes.total_quests)

    for _ in range(number_of_trials):
        # Reinitialize heroes for a fresh trial
        heroes.init_heroes()

        # Run the bandit method and collect metrics
        cur_rew_rec, cur_avg_ret_rec, cur_tot_reg_rec, cur_opt_act_rec = bandit_method(heroes=heroes, **kwargs)

        # Accumulate results
        rew_rec += np.array(cur_rew_rec)
        avg_ret_rec += np.array(cur_avg_ret_rec)
        tot_reg_rec += np.array(cur_tot_reg_rec)
        opt_act_rec += np.array(cur_opt_act_rec)

    # Compute averages across trials
    rew_rec /= number_of_trials
    avg_ret_rec /= number_of_trials
    tot_reg_rec /= number_of_trials
    opt_act_rec /= number_of_trials

    return rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec


def save_results_plots(experiments, plot_title='Experiment Results', results_folder='results', pdf_name='experiment_results.pdf'):
    """
    Generate and save a 2x2 plot summarizing multiple experiments.

    :param experiments: List of experiments, each as a dictionary with keys:
                        'exp_name', 'reward_rec', 'average_rew_rec',
                        'tot_reg_rec', and 'opt_action_rec'.
    :param plot_title: Title for the plot.
    :param results_folder: Directory where the PDF will be saved (created if it doesn't exist).
    :param pdf_name: Name of the output PDF file.
    """
    # Ensure the results folder exists
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Set up the figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(plot_title, fontsize=16)

    # Define a color scheme
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))

    # Iterate over experiments and plot their results
    for exp, color in zip(experiments, colors):
        exp_name = exp['exp_name']
        reward_rec = exp['reward_rec']
        average_rew_rec = exp['average_rew_rec']
        tot_reg_rec = exp['tot_reg_rec']
        opt_action_rec = exp['opt_action_rec']

        # Plot reward over time
        axs[0, 0].plot(reward_rec, label=exp_name, color=color)
        axs[0, 0].set_title('Reward Over Time')
        axs[0, 0].set_xlabel('Number of Attempts')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].legend()

        # Plot average reward over time
        axs[0, 1].plot(average_rew_rec, label=exp_name, color=color)
        axs[0, 1].set_title('Average Reward Over Time')
        axs[0, 1].set_xlabel('Number of Attempts')
        axs[0, 1].set_ylabel('Average Reward')
        axs[0, 1].legend()

        # Plot total regret over time
        axs[1, 0].plot(tot_reg_rec, label=exp_name, color=color)
        axs[1, 0].set_title('Total Regret Over Time')
        axs[1, 0].set_xlabel('Number of Attempts')
        axs[1, 0].set_ylabel('Total Regret')
        axs[1, 0].legend()

        # Plot percentage of optimal actions over time
        axs[1, 1].plot(opt_action_rec, label=exp_name, color=color)
        axs[1, 1].set_title('Percentage of Optimal Actions Over Time')
        axs[1, 1].set_xlabel('Number of Attempts')
        axs[1, 1].set_ylabel('Optimal Actions (%)')
        axs[1, 1].legend()

    # Save the figure as a PDF
    pdf_path = os.path.join(results_folder, pdf_name)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for the title
    plt.savefig(pdf_path, format='pdf')
    plt.close()

    print(f"Results saved to {pdf_path}")
