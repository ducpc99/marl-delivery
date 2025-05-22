import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

class ACOAgents:
    def __init__(
        self, n_robots=5, map_size=(10, 10),
        evaporation_rate=0.1, pheromone_init=1.0, pheromone_boost=5.0, alpha=1.0, beta=3.0,
        stuck_limit=6, seed=2025  # Thêm seed
    ):
        self.n_robots = n_robots
        self.map_size = map_size
        self.evaporation_rate = evaporation_rate
        self.pheromone_init = pheromone_init
        self.pheromone_boost = pheromone_boost
        self.alpha = alpha      # Pheromone importance
        self.beta = beta        # Heuristic (distance) importance
        self.pheromones = defaultdict(lambda: self.pheromone_init)
        self.packages_busy = set()
        self.target_package = [None] * n_robots
        self.stuck_limit = stuck_limit
        self.prev_dists = [None] * n_robots
        self.stuck_steps = [0] * n_robots
        self.rng = np.random.RandomState(seed)  # Đảm bảo random deterministic

    def init_agents(self, state):
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.map_size = (len(self.map), len(self.map[0]))
        self.robots = [(r[0] - 1, r[1] - 1, r[2]) for r in state['robots']]
        self.target_package = [None] * self.n_robots
        self.prev_dists = [None] * self.n_robots
        self.stuck_steps = [0] * self.n_robots
        self.packages_all = []
        for p in state['packages']:
            # (id, sx, sy, ex, ey, start, deadline)
            self.packages_all.append((p[0], p[1] - 1, p[2] - 1, p[3] - 1, p[4] - 1, p[5], p[6]))
        self.packages_busy = set()

    def update_state(self, state):
        self.robots = [(r[0] - 1, r[1] - 1, r[2]) for r in state['robots']]
        for p in state['packages']:
            pkg = (p[0], p[1] - 1, p[2] - 1, p[3] - 1, p[4] - 1, p[5], p[6])
            if pkg not in self.packages_all:
                self.packages_all.append(pkg)

    def centralized_assignment(self, now):
        free_robots = [i for i, (_, _, carrying) in enumerate(self.robots) if carrying == 0]
        free_packages = [pkg for pkg in self.packages_all if pkg[0] not in self.packages_busy and now >= pkg[5] and pkg[6] - now > 0]
        if not free_robots or not free_packages:
            return
        cost = np.zeros((len(free_robots), len(free_packages)))
        for i, rid in enumerate(free_robots):
            rx, ry, _ = self.robots[rid]
            for j, pkg in enumerate(free_packages):
                sx, sy = pkg[1], pkg[2]
                cost[i, j] = abs(rx - sx) + abs(ry - sy)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            self.target_package[free_robots[i]] = free_packages[j][0]

    def _get_package_by_id(self, pid):
        for pkg in self.packages_all:
            if pkg[0] == pid:
                return pkg
        return None

    def _select_move(self, cur_pos, goal):
        moves = []
        scores = []
        total = 0.0
        for dx, dy, mv in [(-1,0,'U'), (1,0,'D'), (0,-1,'L'), (0,1,'R')]:
            nx, ny = cur_pos[0] + dx, cur_pos[1] + dy
            if 0 <= nx < self.map_size[0] and 0 <= ny < self.map_size[1]:
                if self.map[nx][ny] == 0:
                    pher = self.pheromones[(cur_pos, (nx,ny))]
                    dist = abs(nx - goal[0]) + abs(ny - goal[1]) + 1e-4
                    heuristic = 1.0 / dist
                    score = (pher ** self.alpha) * (heuristic ** self.beta)
                    moves.append((nx, ny, mv))
                    scores.append(score)
                    total += score
        if not moves:
            return 'S'
        probs = [s / total for s in scores]
        chosen = self.rng.choice(len(moves), p=probs)  # Sử dụng self.rng thay vì np.random
        return moves[chosen][2]

    def get_actions(self, state):
        self.update_state(state)
        now = state['time_step']
        actions = []
        busy_now = set(self.packages_busy)

        # ---- Deadlock/yield: reset target if stuck ----
        for i, (rx, ry, carrying) in enumerate(self.robots):
            if carrying == 0 and self.target_package[i] is not None:
                pkg = self._get_package_by_id(self.target_package[i])
                if pkg is not None:
                    sx, sy = pkg[1], pkg[2]
                    dist = abs(rx - sx) + abs(ry - sy)
                    if self.prev_dists[i] is not None and dist >= self.prev_dists[i]:
                        self.stuck_steps[i] += 1
                    else:
                        self.stuck_steps[i] = 0
                    self.prev_dists[i] = dist
                    if self.stuck_steps[i] >= self.stuck_limit:
                        self.target_package[i] = None
                        self.stuck_steps[i] = 0
                else:
                    self.stuck_steps[i] = 0

        # ---- Centralized assignment (gán target_package cho robot free) ----
        self.centralized_assignment(now)

        for i, (rx, ry, carrying) in enumerate(self.robots):
            if carrying != 0:
                pkg = self._get_package_by_id(carrying)
                ex, ey = pkg[3], pkg[4]
                if (rx, ry) == (ex, ey):
                    actions.append(('S', '2'))   # drop
                else:
                    move = self._select_move((rx, ry), (ex, ey))
                    actions.append((move, '0'))
            else:
                pid = self.target_package[i]
                pkg = self._get_package_by_id(pid)
                if pkg is None:
                    actions.append(('S', '0'))
                else:
                    sx, sy, stime = pkg[1], pkg[2], pkg[5]
                    if (rx, ry) == (sx, sy) and now >= stime:
                        actions.append(('S', '1'))   # pick up
                        busy_now.add(pid)
                    else:
                        move = self._select_move((rx, ry), (sx, sy))
                        actions.append((move, '0'))
                        busy_now.add(pid)
        self.packages_busy = busy_now
        return actions
