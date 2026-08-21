# TE_AN — Transformer-Enhanced Hierarchical DRL for Autonomous Navigation
### Phase 2 — Path-Following (PF) Controller: LSTM-TD3 Baseline

This repository contains the  Level-2 path-following
controller baseline for a study on transformer-enhanced deep RL autonomous driving.
The controller is trained end-to-end in CARLA 0.9.15 (Town10HD_Opt) and follows the
map centerline; it is the control group against which later planner/architecture
variants are compared.

> Scope: This is a single Level-2 lane-following controller, not a 2-level
> hierarchy. The "plan" it follows is the CARLA centerline (`get_waypoint(project_to_road=True)`).
> The Local Path Planner (LPP) and transformer backbone are future phases.

## Method
- Algorithm: TD3 (twin critics, delayed policy, target-policy smoothing).
- Backbone: recurrent — split-branch causal twin critics with LSTM memory (`hidden=256`).
- Perception: single 3×120×288 RGB frame → CNN feature extractor, fused with a
  7-dim kinematic vector. No frame-stacking; temporal reasoning is delegated to the LSTM.
- Action: 2-D continuous `[steer, throttle/brake]` with in-env EMA smoothing.
- Reward: lane-centering × heading-alignment × speed shaping + potential-based
  progress − steering-comfort penalty.
- Sim: synchronous 20 Hz (`fixed_delta=0.05`), Buildings/Particles unloaded for VRAM.

## Ablation roadmap (3 independent from-scratch runs, TD3 arm fixed)
| Arm | Backbone | Planner | 

| 1. PF-LSTM-TD3 (this repo) | LSTM | none (centerline) |
| 2. LPP+PF-LSTM-TD3 | LSTM | Dual-Transformer |
| 3. LPP+PF-GTrXL-TD3 | GTrXL | Dual-Transformer | 

The algorithm is fixed to TD3 across all arms → architecture is the sole manipulated variable.


