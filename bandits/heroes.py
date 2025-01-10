import numpy as np  # Numpy used for generating random numbers

class Heroes:
    """
    The Heroes class simulates a multi-armed bandit problem, where each hero represents an arm with a specific 
    probability of success. The class tracks the performance of each hero over multiple quests.
    """
    def __init__(self,
                 total_quests: int = 2000,
                 true_probability_list: list = [0.4, 0.6]):
        """
        Initialize the Heroes class with a list of true success probabilities and the total number of quests.
        
        :param total_quests: Total number of quests to be performed.
        :param true_probability_list: List of true success probabilities for each hero.
        """
        self.heroes = [{
            'name': f"Hero_{i+1}",                 # Name of the hero
            'true_success_probability': p,        # Hero's true probability of quest success
            'successes': 0,                       # Number of successful quests
            'n_quests': 0                         # Total number of quests attempted
        } for i, p in enumerate(true_probability_list)]
        self.total_quests = total_quests

    def init_heroes(self):
        """
        Reset the heroes' performance metrics for a new simulation. 
        This method should be called before starting a new set of experiments.
        """
        for hero in self.heroes:
            hero['successes'] = 0
            hero['n_quests'] = 0

    def attempt_quest(self, hero_index: int):
        """
        Simulate a hero attempting a single quest and update their performance metrics.

        :param hero_index: Index of the hero in the `self.heroes` list.
        :return: Reward for the quest (1 for success, 0 for failure).
        :raises IndexError: If the hero_index is out of bounds.
        """
        # Validate hero index
        if hero_index < 0 or hero_index >= len(self.heroes):
            raise IndexError("Hero index out of range.")

        # Retrieve the hero and update quest count
        hero = self.heroes[hero_index]
        hero['n_quests'] += 1  # Increment the number of quests attempted

        # Generate success or failure based on the hero's true success probability
        pr_success = hero['true_success_probability']
        if np.random.rand() < pr_success:  # Success case
            reward = 1
            hero['successes'] += 1  # Increment the success count
        else:  # Failure case
            reward = 0
        
        return reward
