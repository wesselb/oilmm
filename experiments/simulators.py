import argparse

import lab.torch as B
import numpy as np
import torch
import wbml.plot
from matrix import Dense, Diagonal, Kronecker
from oilmm import OILMM, Normaliser
from stheno.torch import Matern52 as Mat52
from varz import Vars
from varz.torch import minimise_l_bfgs_b
from wbml.data.cmip5 import load
from wbml.experiment import WorkingDirectory

if __name__ == "__main__":

    # Parse script arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", type=int, default=1_000, help="Number of optimisation iterations."
    )
    parser.add_argument("-n", type=int, default=10_000, help="Number of time points.")
    parser.add_argument(
        "-mr", type=int, default=10, help="Number of latent processes for space."
    )
    parser.add_argument(
        "-ms", type=int, default=5, help="Number of latent processes for simulators."
    )
    parser.add_argument(
        "--separable", action="store_true", help="Use a separable model."
    )
    args = parser.parse_args()

    # Determine suffix.
    if args.separable:
        suffix = "_separable"
    else:
        suffix = ""

    B.epsilon = 1e-8
    wbml.out.report_time = True
    wd = WorkingDirectory("_experiments", "simulators", log=f"log{suffix}.txt")

    # Load data.
    loc, temp, sims = load()
    sims = {k: v for k, v in list(sims.items())}
    x_data = np.array([(day - temp.index[0]).days for day in temp.index[: args.n]])
    y_data = np.concatenate([sim.to_numpy()[: args.n] for sim in sims.values()], axis=1)
    wbml.out.out("Data loaded")

    # Normalise training data.
    normaliser = Normaliser(y_data)
    y_data = normaliser.normalise(y_data)

    # Determine initialisation of spatial length scales.
    scales_init = 0.5 * np.array(loc.max() - loc.min())

    # Convert to PyTorch.
    loc = torch.tensor(np.array(loc), dtype=torch.float64)
    x_data = torch.tensor(x_data, dtype=torch.float64)
    y_data = torch.tensor(y_data, dtype=torch.float64)

    # Determine number of latent processes.
    m_r = args.mr
    m_s = args.ms
    m = m_r * m_s

    # Determine number of outputs.
    p_s = len(sims)
    p_r = loc.shape[0]
    p = p_s * p_r

    # Compute inducing point locations, assuming that inputs are time.
    n_ind = int(args.n / 60)  # One inducing point per two months.
    x_ind_init = B.linspace(x_data.min(), x_data.max(), n_ind)

    # Determine initialisation for covariance between sims.
    rho = 0.5
    u, s, _ = B.svd((1 - rho) * B.eye(p_s) + rho * B.ones(p_s, p_s))
    u_s_init = u[:, :m_s]
    s_sqrt_s_init = B.sqrt(s[:m_s])

    vs = Vars(torch.float64)

    def construct_model(vs):
        if args.separable:
            # Copy same kernel `m` times.
            kernel = [Mat52().stretch(vs.bnd(6 * 30, lower=60, name="k_scale"))]
            kernels = kernel * m
        else:
            # Parametrise different kernels.
            kernels = [
                Mat52().stretch(vs.bnd(6 * 30, lower=60, name=f"{i}/k_scale"))
                for i in range(m)
            ]
        noise = vs.bnd(1e-2, name="noise")
        latent_noises = vs.bnd(1e-2 * B.ones(m), name="latent_noises")

        # Construct component of the mixing matrix over simulators.
        u = vs.orth(init=u_s_init, shape=(p_s, m_s), name="sims/u")
        s_sqrt = vs.bnd(init=s_sqrt_s_init, shape=(m_s,), name="sims/s_sqrt")

        u_s = Dense(u)
        s_sqrt_s = Diagonal(s_sqrt)

        # Construct components of the mixing matrix over space from a
        # covariance.
        scales = vs.bnd(init=scales_init, name="space/scales")
        k = Mat52().stretch(scales)

        u, s, _ = B.svd(B.dense(k(loc)))
        u_r = Dense(u[:, :m_r])
        s_sqrt_r = Diagonal(B.sqrt(s[:m_r]))

        # Compose.
        s_sqrt = Kronecker(s_sqrt_s, s_sqrt_r)
        u = Kronecker(u_s, u_r)

        return OILMM(kernels, u, s_sqrt, noise, latent_noises)

    def objective(vs):
        x_ind = vs.unbounded(x_ind_init, name="x_ind")
        return -construct_model(vs).logpdf(x_data, y_data, x_ind=x_ind)

    minimise_l_bfgs_b(objective, vs, trace=True, iters=args.i)

    # Print variables.
    vs.print()

    def cov_to_corr(k):
        std = B.sqrt(B.diag(k))
        return k / std[:, None] / std[None, :]

    # Compute correlations between simulators.
    u = Dense(vs["sims/u"])
    s_sqrt = Diagonal(vs["sims/s_sqrt"])
    k = u @ s_sqrt @ s_sqrt @ u.T
    std = B.sqrt(B.diag(k))
    corr_learned = cov_to_corr(k)

    # Compute empirical correlations.
    all_obs = np.concatenate(
        [sim.to_numpy()[: args.n].reshape(-1, 1) for sim in sims.values()], axis=1
    )
    corr_empirical = cov_to_corr(np.cov(all_obs.T))

    # Compute predictions for latent processes.
    model = construct_model(vs)
    model = model.condition(x_data, y_data, x_ind=vs["x_ind"])
    x_proj, y_proj, _, _ = model.project(x_data, y_data)
    means, lowers, uppers = model.model.predict(x_proj)

    # Save for processing.
    wd.save(
        B.to_numpy(
            {
                "n": args.n,
                "m": m,
                "p": p,
                "m_r": m_r,
                "m_s": m_s,
                "x_proj": x_proj,
                "y_proj": y_proj,
                "means": means,
                "lowers": lowers,
                "uppers": uppers,
                "learned_parameters": {name: vs[name] for name in vs.names},
                "corr_learned": corr_learned,
                "corr_empirical": corr_empirical,
            }
        ),
        f"results_mr{m_r}_ms{m_s}{suffix}.pickle",
    )
