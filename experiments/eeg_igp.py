import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wbml.metric as metric
import wbml.plot
from oilmm import IGP, Normaliser
from stheno import EQ
from varz import Vars
from varz.torch import minimise_l_bfgs_b
from wbml.data.eeg import load
from wbml.experiment import WorkingDirectory

if __name__ == '__main__':
    wbml.out.report_time = True
    wd = WorkingDirectory('_experiments', 'eeg_igp')

    _, train, test = load()

    x = np.array(train.index)
    y = np.array(train)

    # Normalise data.
    normaliser = Normaliser(y)
    y_norm = normaliser.normalise(y)

    p = B.shape(y)[1]
    vs = Vars(torch.float64)


    def construct_model(vs):
        kernels = [vs.pos(1, name=f'{i}/var') *
                   EQ().stretch(vs.pos(0.02, name=f'{i}/scale'))
                   for i in range(p)]
        noises = vs.pos(1e-2 * B.ones(p), name='noises')
        return IGP(kernels, noises)


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

    # Report SMSE.
    pred = pd.DataFrame(means, index=train.index, columns=train.columns)
    smse = metric.smse(pred, test)
    wbml.out.kv('SMSEs', smse.dropna())
    wbml.out.kv('Average SMSEs', smse.mean())

    # Compute PPLP.
    x_test = np.array(test.index)
    y_test = np.array(test.reindex(train.columns, axis=1))
    logprob = model.logpdf(torch.tensor(x_test),
                           torch.tensor(normaliser.normalise(y_test)))
    logdet = normaliser.normalise_logdet(y_test)
    pplp = logprob + logdet
    wbml.out.kv('PPLP', pplp / B.length(y_test))

    # Name of output to plot.
    name = 'F2'

    # Plot the result.
    plt.figure(figsize=(12, 1.75))
    wbml.plot.tex()

    p = list(train.columns).index(name)
    plt.plot(x, means[:, p], style='pred')
    plt.fill_between(x, lowers[:, p], uppers[:, p], style='pred')
    plt.scatter(x, y[:, p], style='train')
    plt.scatter(test[name].index, test[name], style='test')
    plt.xlabel('Time (second)')
    plt.xlim(0.4, 1)
    plt.ylabel(f'{name} (volt)')
    wbml.plot.tweak(legend=False)

    plt.tight_layout()
    plt.savefig(wd.file('eeg.pdf'))
