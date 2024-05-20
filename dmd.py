import numpy as np
from pydmd import BOPDMD

def compute_velocity_fields(density_fields, delta_t):
    velocity_fields = []
    for i in range(1, len(density_fields)):
        velocity_fields.append((density_fields[i] - density_fields[i - 1]) / delta_t)
    return velocity_fields

def compute_dmd_matrix(velocity_fields):
    V = np.column_stack([field.flatten() for field in velocity_fields[:-1]])
    V_prime = np.column_stack([v.flatten() for v in velocity_fields[1:]])
    U, S, VT = np.linalg.svd(V, full_matrices = False)
    A = V_prime @ VT.T @ np.linalg.inv(np.diag(S)) @ U.T
    return A

def perform_dmd(velocity_fields):
    dmd = BOPDMD(
        svd_rank = 25,
        num_trials = 200,
        trial_size = 0.8,
        eig_constraints = {"imag", "conjugate_pairs"},
        varpro_opts_dict = {"tol": 0.2, "verbose": True}
    )
    t = np.arange(0, 100)

    dmd.fit(np.column_stack([field.flatten() for field in velocity_fields]), t)

    new_t = np.arange(0, 110)
    forecast = dmd.forecast(new_t)[0]

    print(forecast.shape)

def predict_next_density_field(p_last, p_prev, A, delta_t):
    v_last = (p_last - p_prev).flatten()
    v_next = (A @ v_last).reshape(p_last.shape)
    p_next = p_last + delta_t * v_next
    return p_next

density_fields = np.load('./_data/density0.npy')
delta_t = 1.0

velocity_fields = compute_velocity_fields(density_fields, delta_t)

A = perform_dmd(velocity_fields)
