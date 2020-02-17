import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wbml.plot
from matrix import Dense
from stheno import Matern12
from varz import Vars
from varz.torch import minimise_l_bfgs_b
from wbml.data.exchange import load
from wbml.experiment import WorkingDirectory

from oilmm import ILMMPP, Normaliser

if __name__ == '__main__':

    wbml.out.report_time = True
    wd = WorkingDirectory('_experiments', 'exchange_ilmm')

    _, train, test = load()

    x = np.array(train.index)
    y = np.array(train)

    # Normalise data.
    normaliser = Normaliser(y)
    y_norm = normaliser.normalise(y)

    p = B.shape(y)[1]
    m = 3
    vs = Vars(torch.float64)


    def construct_model(vs):
        kernels = [vs.pos(1, name=f'{i}/var') *
                   Matern12().stretch(vs.pos(0.1, name=f'{i}/scale'))
                   for i in range(m)]
        noise = vs.pos(1e-2, name='noise')
        latent_noises = vs.pos(1e-2 * B.ones(m), name='latent_noises')
        h = Dense(vs.get(shape=(p, m), name='h'))

        return ILMMPP(kernels, h, noise, latent_noises)


    def objective(vs):
        return -construct_model(vs).logpdf(torch.tensor(x),
                                           torch.tensor(y_norm))


    minimise_l_bfgs_b(objective, vs, trace=True, iters=1000)

    # Predict.
    model = construct_model(vs)
    model = model.condition(torch.tensor(x), torch.tensor(y_norm))
    means, lowers, uppers = B.to_numpy(model.predict(torch.tensor(x)))

    # Undo normalisation
    means, lowers, uppers = normaliser.unnormalise(means, lowers, uppers)

    # For the purpose of comparison, standardise using the mean of the
    # *training* data. This is not how the SMSE usually is defined!
    pred = pd.DataFrame(means, index=train.index, columns=train.columns)
    smses = ((pred - test) ** 2).mean(axis=0) / \
            ((train.mean(axis=0) - test) ** 2).mean(axis=0)

    # Report and save average SMSE.
    wbml.out.kv('Average SMSE', smses.mean())
    with open(wd.file('average_smse.txt'), 'w') as f:
        f.write(str(smses.mean()))

    # Plot the result.
    plt.figure(figsize=(12, 2))
    wbml.plot.tex()

    for i, name in enumerate(test.columns):
        p = list(train.columns).index(name)  # Index of output.
        plt.subplot(1, 3, i + 1)
        plt.plot(x, means[:, p], c='tab:blue')
        plt.fill_between(x, lowers[:, p], uppers[:, p],
                         facecolor='tab:blue', alpha=.25)
        plt.scatter(x, y[:, p], c='black', marker='o', s=8)
        plt.scatter(test[name].index, test[name],
                    c='tab:orange', marker='x', s=8)
        plt.xlabel('Time (year)')
        plt.ylabel(name)
        wbml.plot.tweak(legend=False)

    plt.tight_layout()
    plt.savefig(wd.file('exchange.pdf'))