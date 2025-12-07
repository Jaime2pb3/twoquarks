import math
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd

ACTIONS = [0, 1, 2, 3]
DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


@dataclass
class GWCfg:
    H: int = 5
    W: int = 5
    slip: float = 0.15
    start: tuple = (0, 0)
    goal: tuple = (4, 4)
    max_steps: int = 200


CFG = GWCfg()
rng_global = np.random.default_rng(1234)


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


def ece_brier(conf, labels, n_bins=10):
    conf = np.asarray(conf, float)
    labels = np.asarray(labels, int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.sum() == 0:
            continue
        acc = labels[m].mean()
        c_bar = conf[m].mean()
        ece += (m.sum() / len(conf)) * abs(acc - c_bar)
    brier = np.mean((conf - labels) ** 2)
    return float(ece), float(brier)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) / n) + (z ** 2 / (4 * n ** 2))) / denom
    return float(center - half), float(center + half)


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    from math import erf, sqrt
    pval = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return float(z), float(pval)


def run_eps_episode(Q, rng, eps=0.05, alpha=0.2, gamma=0.99):
    s = reset_env()
    total = 0
    success = False
    prev_p = None
    ent_list = []
    jerk_list = []
    confs = []
    labels = []
    for _ in range(CFG.max_steps):
        if rng.random() < eps:
            a = int(rng.integers(len(ACTIONS)))
            p = np.ones(len(ACTIONS)) / len(ACTIONS)
        else:
            qs = np.array([Q[s][aa] for aa in ACTIONS], float)
            a = int(np.argmax(qs))
            p = np.zeros(len(ACTIONS))
            p[a] = 1.0
        if prev_p is not None:
            jerk_list.append(kl_norm(p, prev_p))
        prev_p = p
        ent_list.append(norm_entropy(p))
        confs.append(float(p.max()))
        s2, r, done, success2 = step_env(s, a)
        gr, gc = CFG.goal
        r0, c0 = unidx(s)
        r1, c1 = unidx(s2)
        labels.append(int(abs(gr - r1) + abs(gc - c1) < abs(gr - r0) + abs(gc - c0)))
        Q[s][a] += alpha * (r + gamma * max(Q[s2][aa] for aa in ACTIONS) - Q[s][a])
        s = s2
        success = success2
        total += 1
        if done:
            break
    ece, brier = ece_brier(np.array(confs), np.array(labels))
    return dict(success=success, steps=total, H=np.mean(ent_list), jerk=np.mean(jerk_list), ece=ece, brier=brier)


def run_soft_episode(Q, rng, tau=0.5, alpha=0.2, gamma=0.99):
    s = reset_env()
    total = 0
    success = False
    prev_p = None
    ent_list = []
    jerk_list = []
    confs = []
    labels = []
    for _ in range(CFG.max_steps):
        qs = np.array([Q[s][aa] for aa in ACTIONS], float)
        p = softmax(qs, tau)
        a = int(rng.choice(len(ACTIONS), p=p))
        if prev_p is not None:
            jerk_list.append(kl_norm(p, prev_p))
        prev_p = p
        ent_list.append(norm_entropy(p))
        confs.append(float(p.max()))
        s2, r, done, success2 = step_env(s, a)
        gr, gc = CFG.goal
        r0, c0 = unidx(s)
        r1, c1 = unidx(s2)
        labels.append(int(abs(gr - r1) + abs(gc - c1) < abs(gr - r0) + abs(gc - c0)))
        Q[s][a] += alpha * (r + gamma * max(Q[s2][aa] for aa in ACTIONS) - Q[s][a])
        s = s2
        success = success2
        total += 1
        if done:
            break
    ece, brier = ece_brier(np.array(confs), np.array(labels))
    return dict(success=success, steps=total, H=np.mean(ent_list), jerk=np.mean(jerk_list), ece=ece, brier=brier)


def modulation_params(p_probe, td_abs, cfg):
    conf = 1.0 - norm_entropy(p_probe)
    tau = np.clip(cfg["tau0"] * np.exp(-cfg["k_conf"] * conf) + cfg["k_td"] * td_abs, cfg["tau_min"], cfg["tau_max"])
    wE = np.clip(cfg["w0"] + cfg["c_td"] * td_abs, cfg["w_min"], cfg["w_max"])
    alpha = 0.8 * (1.0 / (1.0 + np.exp(-(cfg["a"] * conf - cfg["b"] * td_abs))))
    kappa = np.clip(cfg["kappa0"] * (1.0 - 0.5 * conf + 0.5 * td_abs), 0.1, 2.0)
    return tau, wE, alpha, kappa


def lowpass(prev_p, p, kappa):
    beta = np.clip(1.5 - (kappa - 0.1) / (2.0 - 0.1), 0.05, 0.6)
    if prev_p is None:
        return p
    q = (1.0 - beta) * p + beta * prev_p
    q = np.clip(q, 1e-12, None)
    q = q / q.sum()
    return q


def run_levo_episode(Q, rng, gamma=0.99):
    s = reset_env()
    total = 0
    success = False
    prev_p = None
    prev_tau = None
    prev_alpha = None
    ent_list = []
    jerk_list = []
    confs = []
    labels = []
    cfg_mod = dict(
        tau0=0.6,
        tau_min=0.2,
        tau_max=1.2,
        k_conf=1.2,
        k_td=0.4,
        w0=0.3,
        w_min=0.1,
        w_max=1.0,
        c_td=0.5,
        a=4.5,
        b=2.0,
        kappa0=1.0,
        eta=0.15,
        zeta=0.1,
        lam=0.35,
        mu=0.08,
    )
    N_sa = defaultdict(lambda: {a: 0 for a in ACTIONS})
    for _ in range(CFG.max_steps):
        qs = np.array([Q[s][aa] for aa in ACTIONS], float)
        p_probe = softmax(qs, tau=0.6)
        td_abs = float(abs(max(qs) - np.mean(qs)))
        tau, wE, alpha, kappa = modulation_params(p_probe, td_abs, cfg_mod)
        p_raw = softmax(qs, tau)
        p = lowpass(prev_p, p_raw, kappa)
        a = int(rng.choice(len(ACTIONS), p=p))
        if prev_p is not None:
            jerk_list.append(kl_norm(p, prev_p))
        prev_p = p
        ent_list.append(norm_entropy(p))
        confs.append(float(p.max()))
        s2, r, done, success2 = step_env(s, a)
        gr, gc = CFG.goal
        r0, c0 = unidx(s)
        r1, c1 = unidx(s2)
        labels.append(int(abs(gr - r1) + abs(gc - c1) < abs(gr - r0) + abs(gc - c0)))
        r_comp = r
        d_tau = 0.0 if prev_tau is None else abs(tau - prev_tau)
        d_alpha = 0.0 if prev_alpha is None else abs(alpha - prev_alpha)
        r_est = -(d_tau + d_alpha)
        cost_conf = max(0.0, alpha - p.max())
        cost_jerk = 0.0 if len(jerk_list) == 0 else jerk_list[-1]
        r_shaped = (
            r_comp
            + cfg_mod["eta"] * r_comp
            + cfg_mod["zeta"] * r_est
            - cfg_mod["lam"] * cost_conf
            - cfg_mod["mu"] * cost_jerk
        )
        alpha_lr = 0.2
        Q[s][a] += alpha_lr * (r_shaped + gamma * max(Q[s2][aa] for aa in ACTIONS) - Q[s][a])
        N_sa[s][a] += 1
        prev_tau, prev_alpha = tau, alpha
        s = s2
        success = success2
        total += 1
        if done:
            break
    ece, brier = ece_brier(np.array(confs), np.array(labels))
    return dict(success=success, steps=total, H=np.mean(ent_list), jerk=np.mean(jerk_list), ece=ece, brier=brier)


def run_many(agent_fn, label, reps=10, episodes=500, **kwargs):
    stats = []
    bundle = []
    for rep in range(reps):
        Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
        rng = np.random.default_rng(2025 + rep)
        for _ in range(episodes):
            out = agent_fn(Q, rng, **kwargs)
            stats.append(dict(algorithm=label, **out))
            bundle.append(out)
    df = pd.DataFrame(stats)
    summary = dict(
        algorithm=label,
        success_rate=float(df["success"].mean()),
        avg_steps=float(df["steps"].mean()),
        H_entropy=float(df["H"].mean()),
        jerk_mean=float(df["jerk"].mean()),
        ece=float(df["ece"].mean()),
        brier=float(df["brier"].mean()),
        bundle=bundle,
    )
    return summary, df


def run_all(reps=50, episodes=1000):
    results = []
    s_eps, df_eps = run_many(run_eps_episode, "Bellman-eps", reps=reps, episodes=episodes, eps=0.05, alpha=0.2, gamma=0.99)
    results.append(s_eps)
    s_soft, df_soft = run_many(
        run_soft_episode,
        "Bellman-Reflex-Q",
        reps=reps,
        episodes=episodes,
        tau=0.5,
        alpha=0.2,
        gamma=0.99,
    )
    results.append(s_soft)
    s_levo, df_levo = run_many(run_levo_episode, "LevoBell", reps=reps, episodes=episodes, gamma=0.99)
    results.append(s_levo)
    df_sum = pd.DataFrame([{k: v for k, v in r.items() if k != "bundle"} for r in results])
    bundles = dict(eps=df_eps, soft=df_soft, levo=df_levo)
    return df_sum, results, bundles
