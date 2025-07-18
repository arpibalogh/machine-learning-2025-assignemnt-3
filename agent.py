import random
from collections import defaultdict

class MonteCarloAgent:
    def __init__(self, actions, epsilon=0.1, gamma=1.0):
        self.Q = defaultdict(lambda: {a: 0.0 for a in actions})
        self.returns = defaultdict(list)
        self.epsilon = epsilon
        self.gamma = gamma
        self.actions = actions

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = self.Q[state]
        max_q = max(q_values.values())
        return random.choice([a for a, q in q_values.items() if q == max_q])

    def update(self, episode):
        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward

            if (state, action) not in visited:
                self.returns[(state, action)].append(G)
                self.Q[state][action] = sum(self.returns[(state, action)]) / len(self.returns[(state, action)])
                visited.add((state, action))
