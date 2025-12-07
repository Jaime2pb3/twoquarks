import math
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd

ACTIONS = [0, 1, 2, 3]
DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


@dataclass
class GridCfg:
    H: int = 5
    W: int = 5
    slip: float = 0.15
    start: tuple = (0, 0)
    goal: tuple = (4, 4)
    max_steps: int = 200


CFG = GridCfg()
rng_global = np.random.default_rng(2027)


def in_bounds(r, c):
    return 0 <= r < CFG.H and 0 <= c < CFG.W


def idx(s):
    return s[0] * CFG.W + s[1]


def unidx(i):
    return (i // CFG.W, i % CFG.W)


def reset_env():
    return idx(CFG.start)


def step_env(s, a):
    global rng_global
    r, c = unidx(s)
    if rng_global.random() < CFG.slip:
        a = int(rng_global.integers(len(ACTIONS)))
    dr, dc = DELTA[a]
    r2, c2 = r + dr, c + dc
    if not in_bounds(r2, c2):
        r2, c2 = r, c
    s2 = idx((r2, c2))
    done = False
    success = False
    if (r2, c2) == CFG.goal:
        rwd = 1.0
        done = True
        success = True
    else:
        rwd = -0.01
    return s2, rwd, done, success


def softmax(z, tau):
    z = np.asarray(z, float)
    z = z / max(tau, 1e-8)
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    return p


def norm_entropy(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    h = -np.sum(p * np.log(p))
    return float(h / math.log(len(p)))


def kl_norm(p, q):
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    q = np.clip(np.asarray(q, float), 1e-12, 1.0)
    kl = np.sum(p * (np.log(p) - np.log(q)))
    return float(abs(kl) / math.log(len(p)))


def cosine(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def schedule_at(t, a_delta=1.0, T=500.0):
    return float(a_delta * math.exp(-t / T))


class AdamWUpdater:
    def __init__(self, n_params, lr=1e-2, beta1=0.9, beta2=0.999, wd=1e-2, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.wd = wd
        self.eps = eps
        self.m = np.zeros(n_params, dtype=float)
        self.v = np.zeros(n_params, dtype=float)
        self.t = 0

    def step(self, theta, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        update = m_hat / (np.sqrt(v_hat) + self.eps)
        theta = theta - self.lr * (update + self.wd * theta)
        return theta


class LionUpdater:
    def __init__(self, n_params, lr=1e-2, beta1=0.9, beta2=0.99, wd=1e-2):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.wd = wd
        self.m = np.zeros(n_params, dtype=float)

    def step(self, theta, grad):
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        update = np.sign(self.m)
        theta = theta - self.lr * (update + self.wd * theta)
        return theta


class BellmanShiftAgent:
    def __init__(self, optimizer="adamw", lr=1e-2):
        self.n_states = CFG.H * CFG.W
        self.n_actions = len(ACTIONS)
        self.n_params = self.n_states * self.n_actions
        self.theta = np.zeros(self.n_params, dtype=float)
        if optimizer == "adamw":
            self.opt = AdamWUpdater(self.n_params, lr=lr)
        elif optimizer == "lion":
            self.opt = LionUpdater(self.n_params, lr=lr)
        else:
            raise ValueError("unknown optimizer")
        self.prev_policy = None
        self.global_t = 0

    def q_values(self, s):
        base = s * self.n_actions
        return self.theta[base : base + self.n_actions]

    def policy(self, s, tau=0.6):
        q = self.q_values(s)
        return softmax(q, tau)

    def act(self, s, tau=0.6, rng=None):
        p = self.policy(s, tau)
        if rng is None:
            a = int(np.random.choice(len(ACTIONS), p=p))
        else:
            a = int(rng.choice(len(ACTIONS), p=p))
        return a, p

    def td_update(self, s, a, r, s2, gamma, at_scale):
        base = s * self.n_actions
        q = self.theta[base : base + self.n_actions]
        q_next = self.q_values(s2)
        target = r + gamma * float(q_next.max())
        td = target - float(q[a])
        grad = np.zeros_like(self.theta)
        grad[base + a] = -td * at_scale
        theta_old = self.theta.copy()
        self.theta = self.opt.step(self.theta, grad)
        delta_q = float(np.mean(np.abs(self.theta - theta_old)))
        return td, delta_q

    def policy_shift(self, p):
        if self.prev_policy is None:
            self.prev_policy = p
            return 0.0
        shift = kl_norm(p, self.prev_policy)
        self.prev_policy = p
        return shift


def neighborhood_variability(td_batch, theta, n_actions):
    if len(td_batch) == 0:
        return 0.0
    state_map = {}
    for s, td in td_batch:
        if s not in state_map:
            state_map[s] = []
        state_map[s].append(td)
    keys = list(state_map.keys())
    if len(keys) == 1:
        vals = state_map[keys[0]]
        return float(np.var(np.asarray(vals, float)))
    q_vectors = []
    for s in keys:
        base = s * n_actions
        q_vectors.append(theta[base : base + n_actions])
    groups = []
    used = set()
    for i, s in enumerate(keys):
        if s in used:
            continue
        group = [s]
        used.add(s)
        for j, s2 in enumerate(keys):
            if s2 in used:
                continue
            c = cosine(q_vectors[i], q_vectors[j])
            if c >= 0.9:
                group.append(s2)
                used.add(s2)
        groups.append(group)
    vars_group = []
    for g in groups:
        vals = []
        for s in g:
            vals.extend(state_map[s])
        if len(vals) > 1:
            vars_group.append(float(np.var(np.asarray(vals, float))))
    if len(vars_group) == 0:
        return 0.0
    return float(np.mean(vars_group))


def run_shift_experiment(optimizer="adamw", lr=1e-2, episodes=200, gamma=0.99, batch_size=64, a_delta=1.0, T=500.0, seed=42):
    rng = np.random.default_rng(seed)
    agent = BellmanShiftAgent(optimizer=optimizer, lr=lr)
    success_hist = []
    entropy_hist = []
    jerk_hist = []
    deltaq_hist = []
    var_hist = []
    td_buffer = []
    for ep in range(episodes):
        s = reset_env()
        done = False
        steps = 0
        ep_success = False
        while not done and steps < CFG.max_steps:
            at_scale = schedule_at(agent.global_t, a_delta=a_delta, T=T)
            a, p = agent.act(s, tau=0.6, rng=rng)
            s2, r, done, succ = step_env(s, a)
            td, d_q = agent.td_update(s, a, r, s2, gamma, at_scale)
            shift = agent.policy_shift(p)
            td_buffer.append((s, td))
            success_hist.append(1 if succ else 0)
            entropy_hist.append(norm_entropy(p))
            jerk_hist.append(shift)
            deltaq_hist.append(d_q)
            agent.global_t += 1
            s = s2
            ep_success = ep_success or succ
            steps += 1
            if len(td_buffer) >= batch_size:
                v = neighborhood_variability(td_buffer, agent.theta, agent.n_actions)
                var_hist.append(v)
                td_buffer = []
    return dict(
        optimizer=optimizer,
        success=np.asarray(success_hist, float),
        entropy=np.asarray(entropy_hist, float),
        jerk=np.asarray(jerk_hist, float),
        delta_q=np.asarray(deltaq_hist, float),
        var_inst=np.asarray(var_hist, float),
    )


def aggregate_shift_results(episodes=2000):
    res_adamw = run_shift_experiment(optimizer="adamw", lr=5e-3, episodes=episodes, a_delta=1.0, T=500.0, seed=1)
    res_lion = run_shift_experiment(optimizer="lion", lr=5e-3, episodes=episodes, a_delta=1.0, T=500.0, seed=2)
    t_eval = 1500
    a_t = schedule_at(t_eval, a_delta=1.0, T=500.0)
    out = {}
    for res in [res_adamw, res_lion]:
        name = res["optimizer"]
        succ_mean = float(res["success"].mean())
        H_mean = float(res["entropy"].mean())
        jerk_mean = float(res["jerk"].mean())
        dq_mean = float(res["delta_q"].mean())
        var_mean = float(res["var_inst"].mean()) if res["var_inst"].size > 0 else 0.0
        out[name] = dict(
            success=succ_mean,
            entropy=H_mean,
            jerk=jerk_mean,
            delta_q=dq_mean,
            var_inst=var_mean,
            a_t_1500=a_t,
        )
    df = pd.DataFrame(out).T
    return df


if __name__ == "__main__":
    df = aggregate_shift_results(episodes=2000)
    print(df)
