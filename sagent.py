import numpy as np
import random
import os
import pickle

class QLearningAgent:
    def __init__(self, id, alpha=0.05, alpha_decay=1.0, alpha_min=0.01, gamma=0.95, epsilon=1.0, epsilon_min=0.1):
        self.id = id
        self.q_table = {}
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min

    def get_state_key(self, robot, packages):
        pos = (robot[0], robot[1])
        carrying = robot[2]

        if carrying == 0 and packages:
            nearest = min(packages, key=lambda p: abs(p[1]-pos[0]) + abs(p[2]-pos[1]))
            target = (nearest[1], nearest[2])
        elif carrying != 0:
            for p in packages:
                if p[0] == carrying:
                    target = (p[3], p[4])
                    break
            else:
                target = (-1, -1)
        else:
            target = (-1, -1)

        return (pos, carrying, target)

    def possible_actions(self):
        moves = ['S', 'L', 'R', 'U', 'D']
        pkg_acts = ['0', '1', '2']
        return [(m, p) for m in moves for p in pkg_acts]

    def choose_action(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.possible_actions()}
        if random.random() < self.epsilon:
            return random.choice(self.possible_actions())
        return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def update(self, prev_state, action, reward, next_state):
        if prev_state not in self.q_table:
            self.q_table[prev_state] = {a: 0.0 for a in self.possible_actions()}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.possible_actions()}
        old_value = self.q_table[prev_state][action]
        best_next = max(self.q_table[next_state].values())

        self.q_table[prev_state][action] = (
            old_value + self.alpha * (reward + self.gamma * best_next - old_value)
        )

    def decay_epsilon(self, decay_rate):
        self.epsilon = max(self.epsilon * decay_rate, self.epsilon_min)

    def decay_alpha(self, decay_rate):
        self.alpha = max(self.alpha * decay_rate, self.alpha_min)

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f'q_table_agent_{self.id}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, folder):
        path = os.path.join(folder, f'q_table_agent_{self.id}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.q_table = pickle.load(f)

class Agents:
    def __init__(self, alpha=0.05, alpha_decay=1.0, alpha_min=0.01, gamma=0.95, epsilon=1.0, epsilon_min=0.1):
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.move_cost = 0
        self.agents = []
        self.prev_states = []
        self.prev_actions = []

    def init_agents(self, state, move_cost, load_folder=None):
        self.move_cost = move_cost
        n = len(state['robots'])
        self.agents = [QLearningAgent(i, alpha=self.alpha, alpha_decay=self.alpha_decay, alpha_min=self.alpha_min, gamma=self.gamma, epsilon=self.epsilon, epsilon_min=self.epsilon_min)
                       for i in range(n)]
        if load_folder:
            for ag in self.agents:
                ag.load(load_folder)
        self.prev_states = [None] * n
        self.prev_actions = [None] * n

    def get_actions(self, state):
        actions = []
        for i, ag in enumerate(self.agents):
            robot = state['robots'][i]
            packages = state['packages']
            s_key = ag.get_state_key(robot, packages)
            a = ag.choose_action(s_key)
            self.prev_states[i] = s_key
            self.prev_actions[i] = a
            actions.append(a)
        return actions

    def update_agents(self, state, actions, reward, next_state, next_actions):
        n = len(self.agents)
        indiv_rewards = [0.0] * n

        # Tính phí di chuyển
        for i, act in enumerate(actions):
            if act[0] in ['L','R','U','D']:
                indiv_rewards[i] += self.move_cost

        # Thưởng agent có hành động đặc biệt act[1] == '2'
        for i, act in enumerate(actions):
            if act[1] == '2':
                indiv_rewards[i] += reward
                break

        # Cập nhật Q-learning cho từng agent
        for i, ag in enumerate(self.agents):
            next_key = ag.get_state_key(next_state['robots'][i], next_state['packages'])
            ag.update(self.prev_states[i], self.prev_actions[i], indiv_rewards[i], next_key)

    def decay_epsilon_all(self, decay_rate):
        for ag in self.agents:
            ag.decay_epsilon(decay_rate)

    def decay_alpha_all(self, decay_rate):
        for ag in self.agents:
            ag.decay_alpha(decay_rate)

    def save_all(self, folder):
        for ag in self.agents:
            ag.save(folder)

    def load_all(self, folder):
        for ag in self.agents:
            ag.load(folder)
