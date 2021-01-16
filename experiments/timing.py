import time

import lab.torch as B
import numpy as np
import torch
import wbml.plot
from matrix import Dense, Diagonal
from oilmm import OILMM
from stheno import Matern52
from varz import Vars
from wbml.data.cmip5 import load
from wbml.experiment import WorkingDirectory

if __name__ == "__main__":
    B.epsilon = 1e-8
    wd = WorkingDirectory("_experiments", "timing")

    loc, temp, _ = load()

    # Smooth and subsample temperature data.
    temp = temp.rolling(window=31, center=True, min_periods=1, win_type="hamming")
    temp = temp.mean().iloc[::31, :]

    x = np.array([(day - temp.index[0]).days for day in temp.index])
    y = np.array(temp)

    p = B.shape(y)[1]

    def construct_model(vs, m):
        kernels = [
            vs.pos(0.5, name=f"{i}/k_var")
            * Matern52().stretch(vs.bnd(2 * 30, name=f"{i}/k_scale"))
            + vs.pos(0.5, name=f"{i}/k_per_var")
            * (Matern52().stretch(vs.bnd(1.0, name=f"{i}/k_per_scale")).periodic(365))
            for i in range(m)
        ]
        noise = vs.pos(1e-2, name="noise")
        latent_noises = vs.pos(1e-2 * B.ones(m), name="latent_noises")

        # Construct orthogonal matrix and time it.
        time_h_start = time.time()
        u = Dense(vs.orth(shape=(p, m), name="u"))
        s_sqrt = Diagonal(vs.pos(shape=(m,), name="s_sqrt"))
        dur_h = time.time() - time_h_start

        return OILMM(kernels, u, s_sqrt, noise, latent_noises), dur_h

    ns = [100, 200, 300]
    ms = [5, 25, 50, 75, 100, 125, 150, 175, 200, 225, 247]

    for n in ns:
        for m in ms:
            hs = []
            totals = []
            percs = []

            vs = Vars(torch.float64)
            for _ in range(21):
                time_total_start = time.time()
                model, dur_h = construct_model(vs, m)
                model.logpdf(torch.tensor(x[:n]), torch.tensor(y[:n]))
                dur_total = time.time() - time_total_start

                totals.append(dur_total)
                hs.append(dur_h)
                percs.append(dur_h / dur_total * 100)

            # Discard the first run.
            percs = percs[1:]

            wbml.out.kv("n", n)
            wbml.out.kv("m", m)
            with wbml.out.Section("Total"):
                wbml.out.kv("Mean", np.mean(totals))
                wbml.out.kv("Error", 2 * np.std(totals) / np.sqrt(len(totals)))
            with wbml.out.Section("H"):
                wbml.out.kv("Mean", np.mean(hs))
                wbml.out.kv("Error", 2 * np.std(hs) / np.sqrt(len(hs)))
            with wbml.out.Section("Percentage"):
                wbml.out.kv("Mean", np.mean(percs))
                wbml.out.kv("Error", 2 * np.std(percs) / np.sqrt(len(percs)))
