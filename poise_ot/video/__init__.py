"""Stage 2 -- perception pipeline (real video, no RL).

video_loader -> smoothing -> inverse_dynamics -> confidence_gate -> the SAME
``poise_ot.token_builder`` used by Stage 1, unchanged.

Requires an actual HOI4D download to produce meaningful results; see
``video_loader.py``'s module docstring for the external-data prerequisite.
"""
