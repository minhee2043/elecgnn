import numpy as np

from elecgnn.raman import COMPONENTS, _gauss, fit_components, subtract_solvent


def test_fit_recovers_coordinated_fraction():
    lo, hi, comps = COMPONENTS["FSI"]
    x = np.arange(lo, hi + 1, 1.0)
    y = _gauss(x, 1.0, 718, 10) + _gauss(x, 0.4, 731, 10) + _gauss(x, 0.2, 746, 12)
    truth = (0.4 * 10 + 0.2 * 12) / (1.0 * 10 + 0.4 * 10 + 0.2 * 12)
    fit = fit_components(x, y, comps)
    assert abs(fit["f_bound"] - truth) < 0.03 and fit["fit_R2"] > 0.99


def test_solvent_subtraction_leaves_the_anion_band():
    nu = np.arange(600, 900, 1.0)
    solvent = _gauss(nu, 5.0, 650, 15) + _gauss(nu, 3.0, 850, 15)
    anion = _gauss(nu, 1.0, 740, 10)
    resid, scale = subtract_solvent(nu, 0.8 * solvent + anion + 0.1, solvent, 700, 780)
    win = (nu >= 700) & (nu <= 780)
    assert abs(scale - 0.8) < 0.02 and abs(resid[win].max() - 1.0) < 0.05
