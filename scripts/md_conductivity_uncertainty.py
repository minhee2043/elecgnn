"""Statistical uncertainty of an Einstein conductivity from block averaging of one MD trajectory.

The collective charge displacement M(t) = sum_i q_i r_i(t) of the ions (unwrapped
coordinates, charges as in the topology, i.e. already scaled) gives

    sigma = lim  < |M(t + tau) - M(t)|^2 > / (6 V k_B T tau),

which includes all cation-anion cross-correlations. The analysed part of the trajectory is
split into equal blocks; sigma is computed in each block from a linear fit of the collective
MSD over a lag window, and the standard error over blocks is reported.

    pip install MDAnalysis
    python scripts/md_conductivity_uncertainty.py --tpr prod.tpr --xtc prod.xtc \\
        --ions "resname LI" "resname FSI" --temperature 363 --start-ns 10 --blocks 5

Run it for a few representative compositions (a fast carbonate, a glyme, a viscous sulfone and a
fluorinated diluent) and report sigma +/- standard error, together with the relative error
in log10 sigma, in the SI.
"""

import argparse

import numpy as np

E = 1.602176634e-19  # C
KB = 1.380649e-23  # J/K


def collective_msd(m: np.ndarray, max_lag: int) -> np.ndarray:
    """MSD of a (frames, 3) vector over all time origins, lags 0..max_lag-1 (FFT based)."""
    n = len(m)
    out = np.zeros(max_lag)
    for d in range(3):
        x = m[:, d]
        f = np.fft.rfft(x, 2 * n)
        acf = np.fft.irfft(f * np.conj(f))[:n] / np.arange(n, 0, -1)
        sq = x**2
        s1 = (
            2 * sq.sum()
            - np.concatenate([[0], np.cumsum(sq[:-1])])
            - np.concatenate([[0], np.cumsum(sq[::-1][:-1])])
        )
        msd = s1 / np.arange(n, 0, -1) - 2 * acf
        out += msd[:max_lag]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tpr", required=True)
    ap.add_argument("--xtc", required=True)
    ap.add_argument("--ions", nargs="+", required=True, help="MDAnalysis selections of the charged species")
    ap.add_argument("--temperature", type=float, default=363.0)
    ap.add_argument("--start-ns", type=float, default=0.0, help="analyse frames after this time")
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--fit", type=float, nargs=2, default=[0.1, 0.4],
                    help="lag window for the linear fit, as fractions of the block length")  # fmt: skip
    a = ap.parse_args()

    import MDAnalysis as mda
    from MDAnalysis import transformations as trans

    u = mda.Universe(a.tpr, a.xtc)
    ions = sum((u.select_atoms(s) for s in a.ions[1:]), u.select_atoms(a.ions[0]))
    u.trajectory.add_transformations(trans.unwrap(ions))
    q = ions.charges
    times, m, vol = [], [], []
    for ts in u.trajectory:
        if ts.time / 1000.0 < a.start_ns:
            continue
        times.append(ts.time)
        m.append((q[:, None] * ions.positions).sum(axis=0) / 10.0)  # e nm
        vol.append(ts.volume / 1000.0)  # nm^3
    m, times = np.asarray(m), np.asarray(times)
    dt = float(np.median(np.diff(times)))  # ps
    v = float(np.mean(vol))
    n_block = len(m) // a.blocks
    lo, hi = (max(1, int(f * n_block)) for f in a.fit)
    sig = []
    for b in range(a.blocks):
        msd = collective_msd(m[b * n_block : (b + 1) * n_block], hi + 1)
        lags = np.arange(lo, hi + 1) * dt
        slope = np.polyfit(lags, msd[lo : hi + 1], 1)[0]  # e^2 nm^2 / ps
        s = E**2 * slope * 1e-18 / 1e-12 / (6 * v * 1e-27 * KB * a.temperature)  # S/m
        sig.append(10 * s)  # mS/cm
    sig = np.asarray(sig)
    mean, sem = sig.mean(), sig.std(ddof=1) / np.sqrt(len(sig))
    print(f"blocks: {np.round(sig, 3)} mS/cm")
    print(f"sigma = {mean:.3f} +/- {sem:.3f} mS/cm (standard error over {a.blocks} blocks)")
    print(f"relative error {sem / mean:.1%}; error in log10 sigma about {sem / mean / np.log(10):.3f}")


if __name__ == "__main__":
    main()
