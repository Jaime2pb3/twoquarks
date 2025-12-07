"""
policies.py
------------
Policies for Levo-family.

This module defines tabular policies used by:
- HF-Levo (individual agent, base quark)
- LevoThinking ensemble (multi-head mode / CHARM)

For STRANGE:
- hf_levo_policy_tabular is used directly by each agent
in the swarm.

- Strangeness and confinement do NOT reside here, but in
LevoSwarmStrange (swarm level).

""

import math
from typing import Optional

import numpy as np

# -------------------------
# Helpers
# -------------------------

def softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:

""
Numerically stable softmax.

tau: temperature 
tau < 1 -> sharper distribution 
tau > 1 -> flatter distribution 
""" 
x = np.asarray(x, dtype=float) 

if tau <= 0.0: 
raise ValueError("tau must be positive") 

x = x / tau 
x = x - np.max(x) 
ex = np.exp(x) 
s = ex.sum() 

if s <= 0.0: 
# ultra defensive fallback 
return np.ones_like(ex) / len(ex) 

return ex /s


# -------------------------
# HF–LEVO (single quark)
# -------------------------

def hf_levo_policy_tabular( 
Q_s: np.ndarray, 
t: int, 
A: float = 0.5, 
omega: float = 0.1, 
phi: float = 0.0,
ent_weight: float = 0.0,

tau: float = 1.0,

prior: Optional[np.ndarray] = None,
):

""
Tabular HF-Levo policy.

score(a) =

Q(s,a)

+ A * cos(omega * t + phase_a + phi)

Then it is normalized with softmax( / tau ).

Parameters:

- Q_s : vector Q(s,·) of size (n_actions,)

- t : global time step

- A, omega, phi : HF oscillation parameters

- ent_weight : mixing weight with 'prior'

- tau : softmax temperature

- prior : optional prior mixing policy

In STRANGE:

- we usually use ent_weight = 0.

- confinement is applied outside,

in the LevoSwarmStrange module.

""" 

q_s = np.asarray(Q_s, dtype=float) 
n_actions = q_s.shape[0] 

# fixed phases per action 
phases = np.linspace(0.0, 2.0 * math.pi, n_actions, endpoint=False) 

osc = A * np.cos(omega * float(t) + phases + phi) 

score = q_s + osc 

p = softmax(score, tau=tau) 

if prior is None or ent_weight <= 0.0: 
return p 

# Mix with prior (more "coherent" mode, not typical of STRANGE) 
prior = np.asarray(prior, dtype=float) 
prior = prior / max(prior.sum(), 1e-8) 

w = max(0.0, min(1.0, float(ent_weight))) 

p_mix = (1.0 - w) * p + w * prior 
p_mix = np.clip(p_mix, 1e-8, 1.0) 
p_mix /= p_mix.sum() 

return p_mix


# -------------------------
# ENSEMBLE LEVO THINKING
# (multi-head / CHARM support)
# -------------------------

def ensemble_levo_thinking_policy( 
Q_ensemble: np.ndarray, 
state: int, 
t: int, 
A: float = 0.5, 
omega: float = 0.1, 
phi: float = 0.0, 
lambda_var: float = 0.5, 
tau: float = 1.0, 
prior: Optional[np.ndarray] = None,
): 
""" 
Ensemble-based policy (multi-head). 

Q_ensemble has the form:

(n_heads, n_states, n_actions)

For a state s:

mu(a) = mean_h Q_h(s, a)

var(a) = var_h Q_h(s, a)

base_score(a) = mu(a) - lambda_var * var(a)

score(a) = base_score(a) + A * cos(omega * t + phase_a + phi)

Then softmax(score / tau) is applied.

Conceptually, it is a "sister" to STRANGE:

- The variance measures disagreement between heads,

just as STRANGE measures disagreement between agents.

""" 

q_heads = np.asarray(Q_ensemble[:, state, :], dtype=float) 

mu = q_heads.mean(axis=0) 
var = q_heads.var(axis=0) 

n_actions = mu.shape[0] 
phases = np.linspace(0.0, 2.0 * math.pi, n_actions, endpoint=False) 

osc = A * np.cos(omega * float(t) + phases + phi) 

base_score = mu - lambda_var * var 
score = base_score + osc 

p = softmax(score, tau=tau) 

if prior is None: 
return p 

# Light mix with prior 
prior = np.asarray(prior, dtype=float) 
prior = prior / max(prior.sum(), 1e-8) 

p_mix = 0.8 * p + 0.2 * prior 
p_mix = np.clip(p_mix, 1e-8, 1.0) 
p_mix /= p_mix.sum() 

return p_mix