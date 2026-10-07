# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "emcee", "zeus-mcmc"]
# ///
"""Evaluations per effective draw: norn's ensemble sampler, zeus, emcee.

Runs ensemble_gate.exe for norn's chains, then zeus and emcee on the same
targets with the same walkers, burn-in and steps. Effective draws are steps
times walkers over the largest integrated autocorrelation time of the
coordinates (emcee's estimator, for all three). norn counts every row its
density is called on.

  uv run packages/norn/bench/ensemble_gate.py [DIR]

With DIR, norn's chains are read from ensemble_gate.exe's earlier output
there, its counts in DIR/counts.txt, instead of running it.
"""

import os
import subprocess
import sys
import tempfile

import emcee
import numpy as np
import zeus

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
EXE = os.path.join(ROOT, "_build/default/packages/norn/bench/ensemble_gate.exe")
BURN = 500
THIN = 10


def correlated(d):
    s = 0.9 ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    p = np.linalg.inv(s)
    return lambda x: -0.5 * x @ p @ x


def rosenbrock(x):
    return -(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2) / 20.0


def isotropic(x):
    return -0.5 * x @ x


TARGETS = [("correlated10", 10, 10000, correlated(10)),
           ("rosenbrock2", 2, 40000, rosenbrock),
           ("isotropic50", 50, 40000, isotropic)]


def walkers(d):
    return max(32, 4 * d)


def ess(chain):
    """chain: every THIN-th step, [steps / THIN, walkers, d]; thinning divides
    the autocorrelation time as it divides the steps. A '?' marks an estimate from a chain
    shorter than 50 autocorrelation times, emcee's condition for trusting it."""
    tau = np.max(emcee.autocorr.integrated_time(chain, quiet=True))
    flag = "?" if chain.shape[0] < 50 * tau else " "
    return chain.shape[0] * chain.shape[1] / tau, flag


def run_emcee(d, steps, lp, start):
    s = emcee.EnsembleSampler(walkers(d), d, lp)
    s.run_mcmc(start, BURN + steps)
    return s.get_chain(discard=BURN, thin=THIN), walkers(d) * (BURN + steps)


def run_zeus(d, steps, lp, start):
    s = zeus.EnsembleSampler(walkers(d), d, lp, verbose=False)
    s.run_mcmc(start, BURN + steps, progress=False)
    return s.get_chain(discard=BURN, thin=THIN), s.ncall


def norn_chains(directory, out):
    counts = {name: float(r) for name, r in
              (line.split() for line in out.splitlines())}
    chains = {name: np.load(os.path.join(directory, name + ".npy"))
              for name, _, _, _ in TARGETS}
    return counts, chains


def main():
    if len(sys.argv) > 1:
        with open(os.path.join(sys.argv[1], "counts.txt")) as f:
            counts, chains = norn_chains(sys.argv[1], f.read())
    else:
        subprocess.run(["dune", "build",
                        "packages/norn/bench/ensemble_gate.exe"],
                       cwd=ROOT, check=True)
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run([EXE, tmp], check=True, capture_output=True,
                                 text=True).stdout
            counts, chains = norn_chains(tmp, out)
    rng = np.random.default_rng(1)
    print(f"{'target':<14}{'norn':>9}{'zeus':>9}{'emcee':>9}"
          "   (evaluations per effective draw)")
    for name, d, steps, lp in TARGETS:
        start = rng.standard_normal((walkers(d), d))
        e, f = ess(chains[name])
        rows = counts[name]
        zc, zn = run_zeus(d, steps, lp, start)
        ec, en = run_emcee(d, steps, lp, start)
        (ze, zf), (ee, ef) = ess(zc), ess(ec)
        print(f"{name:<14}{rows / e:>8.1f}{f}"
              f"{zn / ze:>8.1f}{zf}{en / ee:>8.1f}{ef}", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
