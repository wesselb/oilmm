import argparse

import lab.torch as B
import numpy as np
import torch
import wbml.plot
from matrix import Dense, Diagonal
from oilmm import OILMM, Normaliser
from stheno import Matern52
from varz import Vars
from varz.torch import minimise_l_bfgs_b
from wbml.data.cmip5 import load
from wbml.experiment import WorkingDirectory

if __name__ == "__main__":
    # Parse arguments of script.
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=13 * 19)
    args = parser.parse_args()

    B.epsilon = 1e-6
    wbml.out.report_time = True
    wd = WorkingDirectory("_experiments", f"temperature_{args.m}")

    loc, temp, _ = load()

    # Smooth and subsample temperature data.
    temp = temp.rolling(window=31, center=True, min_periods=1, win_type="hamming")
    temp = temp.mean().iloc[::31, :]

    # Create train and test splits
    x = np.array([(day - temp.index[0]).days for day in temp.index])
    y = np.array(temp)

    # Divide into training and test set.
    x_train = x[:250]
    y_train = y[:250]
    x_test = x[250:350]
    y_test = y[250:350]

    # Perform normalisation.
    normaliser = Normaliser(y_train)
    y_train_norm = normaliser.normalise(y_train)

    # Determine initialisation of spatial length scales.
    scales_init = np.maximum(0.2 * np.array(loc.max() - loc.min()), 1)

    # Convert to PyTorch.
    loc = torch.tensor(np.array(loc))

    p = B.shape(y)[1]
    m = args.m
    vs = Vars(torch.float64)

    def construct_model(vs):
        kernels = [
            vs.pos(0.5, name=f"{i}/k_var")
            * Matern52().stretch(vs.bnd(2 * 30, name=f"{i}/k_scale"))
            + vs.pos(0.5, name=f"{i}/k_per_var")
            * (Matern52().stretch(vs.bnd(1.0, name=f"{i}/k_per_scale")).periodic(365))
            for i in range(m)
        ]
        latent_noises = vs.pos(1e-2 * B.ones(m), name="latent_noises")
        noise = vs.pos(1e-2, name="noise")

        # Construct components of mixing matrix from a covariance over
        # outputs.
        variance = vs.pos(1, name="h/variance")
        scales = vs.pos(init=scales_init, name="h/scales")
        k = variance * Matern52().stretch(scales)
        u, s, _ = B.svd(B.dense(B.reg(k(loc))))
        u = Dense(u[:, :m])
        s_sqrt = Diagonal(B.sqrt(s[:m]))

        return OILMM(kernels, u, s_sqrt, noise, latent_noises)

    def objective(vs):
        return -construct_model(vs).logpdf(
            torch.tensor(x_train), torch.tensor(y_train_norm)
        )

    # Perform optimisation.
    minimise_l_bfgs_b(objective, vs, trace=True, iters=1000)

    # Print variables.
    vs.print()

    # Predict.
    model = construct_model(vs)
    model = model.condition(torch.tensor(x_train), torch.tensor(y_train_norm))
    means, lowers, uppers = B.to_numpy(model.predict(x_test))

    # Compute RMSE.
    wbml.out.kv("RMSE", B.mean((normaliser.unnormalise(means) - y_test) ** 2) ** 0.5)

    # Compute LML.
    lml = -objective(vs) + normaliser.normalise_logdet(y_train)
    wbml.out.kv("LML", lml / B.length(y_train))

    # Compute PPLP.
    logprob = model.logpdf(
        torch.tensor(x_test), torch.tensor(normaliser.normalise(y_test))
    )
    logdet = normaliser.normalise_logdet(y_test)
    pplp = logprob + logdet
    wbml.out.kv("PPLP", pplp / B.length(y_test))
