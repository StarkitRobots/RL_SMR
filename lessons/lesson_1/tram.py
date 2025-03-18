import random
import numpy as np
import copy

class TramMDP:
    def __init__(self, fail_prob, length=10, init_state=1, gamma=1):
        self.length = length
        self.gamma = gamma
        assert self.length > 1, "length should be greater than 1"
        self.fail_prob = fail_prob
        self.state = init_state
        self.actions = ("walk", "tram")
        self.utility = 0
    
    def get_actions(self):
        pass

    def get_state(self):
        return self.state
    
    def get_info(self, state, action):
        if action == "walk":
            return [(min(state+1, self.length), 1)]
        elif action == "tram":
            return [(min(2*state, self.length), 1 - self.fail_prob), (state, self.fail_prob)]

    def is_terminal(self):
        return self.state == self.length
     
    def step(self, action):
        if action == "walk":
            self.state += 1
        elif action == "tram":
            random_num = random.random()
            if random_num >= self.fail_prob:
                self.state = min(2 * self.state, self.length)
        else:
            raise ValueError("Action must be 'walk' or 'tram'")
        reward = -1
        self.utility += reward
        return self.state, reward
    
    def reset(self):
        self.utility = 0
        self.state = 1

class SimplestPolicy():
    def __init__(self, action):
        self.action = action
    
    def act(self, state):
        return self.action

class PolicyEvaluation:
    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy
        self.value = np.zeros(mdp.length)
    
    def play_game(self):
        while not self.mdp.is_terminal():
            action = self.policy.act(self.mdp.get_state())
            state, _ = self.mdp.step(action)
    
    def evaluate(self, num_iter):
        for iter in range(num_iter):
            new_values = np.zeros(self.mdp.length)
            # print(f"old values {self.value}")
            for s in range(1, self.mdp.length):
                action = self.policy.act(s)
                info = self.mdp.get_info(s, action)
                for var in info:
                    next_s = var[0]
                    # print(next_s)
                    prob = var[1]
                    if prob > 0:
                        new_values[s-1] += prob * (-1 + self.value[next_s-1])
                    # print(new_values)
            # print(f'iter {iter}')
            # print(new_values)
            self.value = copy.deepcopy(new_values)
        print(self.value)


class DeterministicPolicy:
    def __init__(self, actions_list):
        self.actions_list = actions_list
    
    def act(self, state):
        return self.actions_list[state-1]

def play_game(mdp:TramMDP, policy):
    print(f"State = {mdp.get_state()}")
    while not mdp.is_terminal():
        action = policy.act(mdp.get_state())
        print(action)
        state, _ = mdp.step(action)
        print(f"State = {mdp.get_state()}")
    print(f"Terminal state. Utility = {mdp.utility}")


def main(args=None):
    mdp = TramMDP(0.3)
    # policy = SimplestPolicy("walk")
    policy = DeterministicPolicy(["walk"] + ["tram"] * 9)
    # play_game(mdp, policy)
    evaluator = PolicyEvaluation(mdp, policy)
    evaluator.evaluate(1000)
    
if __name__ == '__main__':
    main()
    

