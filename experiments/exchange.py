import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wbml.plot
from matrix import Dense, Diagonal
from stheno import Matern12
from varz import Vars
from varz.torch import minimise_l_bfgs_b
from wbml.data.exchange import load
from wbml.experiment import WorkingDirectory

from oilmm import OILMM, Normaliser, ILMMPP

if __name__ == '__main__':

    wbml.out.report_time = True
    wd = WorkingDirectory('_experiments', 'exchange')

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
        u = Dense(vs.orth(shape=(p, m), name='u'))
        s_sqrt = Diagonal(vs.pos(shape=(m,), name='s_sqrt'))

        return OILMM(kernels, u, s_sqrt, noise, latent_noises)


    def construct_model_ilmm_equivalent(vs):
        kernels = [vs.pos(1, name=f'{i}/var') *
                   Matern12().stretch(vs.pos(0.1, name=f'{i}/scale'))
                   for i in range(m)]
        noise = vs.pos(1e-2, name='noise')
        latent_noises = vs.pos(1e-2 * B.ones(m), name='latent_noises')
        u = vs.orth(shape=(p, m), name='u')
        s_sqrt = vs.pos(shape=(m,), name='s_sqrt')
        h = Dense(u * s_sqrt[None, :])

        return ILMMPP(kernels, h, noise, latent_noises)


    def objective(vs):
        return -construct_model(vs).logpdf(torch.tensor(x),
                                           torch.tensor(y_norm))


    minimise_l_bfgs_b(objective, vs, trace=True, iters=1000)

    # Predict.
    model = construct_model(vs)
    model = model.condition(torch.tensor(x), torch.tensor(y_norm))
    means, lowers, uppers = B.to_numpy(model.predict(x))

    # Undo normalisation
    means, lowers, uppers = normaliser.unnormalise(means, lowers, uppers)

    # For the purpose of comparison, standardise using the mean of the
    # *training* data. This is not how the SMSE usually is defined!
    pred = pd.DataFrame(means, index=train.index, columns=train.columns)
    smse = ((pred - test) ** 2).mean(axis=0) / \
           ((train.mean(axis=0) - test) ** 2).mean(axis=0)

    # Report average SMSE.
    wbml.out.kv('SMSEs', smse.dropna())
    wbml.out.kv('Average SMSE', smse.mean())

    # Compute PPLP.
    for name, model in [
        ('OILMM', construct_model(vs)),
        ('ILMM', construct_model_ilmm_equivalent(vs))
    ]:
        with wbml.out.Section(name):
            model = model.condition(torch.tensor(x), torch.tensor(y_norm))
            x_test = np.array(test.index)
            y_test = np.array(test.reindex(train.columns, axis=1))
            logprob = model.logpdf(torch.tensor(x_test),
                                   torch.tensor(normaliser.normalise(y_test)))
            logdet = normaliser.normalise_logdet(y_test)
            pplp = logprob + logdet
            wbml.out.kv('PPLP', pplp / B.length(y_test))

    # Plot the result.
    plt.figure(figsize=(12, 2))
    wbml.plot.tex()

    for i, name in enumerate(test.columns):
        p = list(train.columns).index(name)  # Index of output.
        plt.subplot(1, 3, i + 1)
        plt.plot(x, means[:, p], style='pred')
        plt.fill_between(x, lowers[:, p], uppers[:, p], style='pred')
        plt.scatter(x, y[:, p], style='train')
        plt.scatter(test[name].index, test[name], style='test')
        plt.xlabel('Time (year)')
        plt.ylabel(name)
        wbml.plot.tweak(legend=False)

    plt.tight_layout()
    plt.savefig(wd.file('exchange.pdf'))
