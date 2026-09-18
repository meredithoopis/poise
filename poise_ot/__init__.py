"""POISE-OT: mechanics-induced OT alignment + two-critic SAC (Stage 1/2 testbed).

Distinct from the sibling ``physics_ot`` package (the already-validated
Physics-OT Experiment-0 pipeline): the token, ground cost, and RL algorithm
here follow the revised POISE-OT paper (5-dim mechanical token, primal/dual
kinetic-energy ground cost, two-critic SAC), not the original 3-channel
Euclidean-cost/PPO design. Reuses ``physics_ot.dynamics.knob``,
``physics_ot.ot.sinkhorn``, and ``physics_ot.references`` where the math is
unchanged; everything token/cost/RL-specific is new.
"""
