"""Verify that causal ARIMA residuals do not use future observations."""
import numpy as np

from iot_revision import causal_residuals


def main():
    rng = np.random.default_rng(7)
    n = 80
    order = (1, 0, 1)
    t = 45
    base = np.cumsum(rng.normal(0, 0.2, n)) + 10.0
    perturbed = base.copy()
    perturbed[t + 1:] += 100.0

    r_base = causal_residuals(base, order, warmup=20, refit_every=10)
    r_pert = causal_residuals(perturbed, order, warmup=20, refit_every=10)

    if not np.array_equal(r_base[:t + 1], r_pert[:t + 1]):
        raise AssertionError('residuals through t changed after perturbing future observations')
    if np.allclose(r_base[t + 1:], r_pert[t + 1:]):
        raise AssertionError('future perturbation did not change later residuals')

    print('PASS: residuals through t are unchanged by future perturbations')


if __name__ == '__main__':
    main()
