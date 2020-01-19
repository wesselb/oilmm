import csv
import datetime

import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import torch
from stheno import EQ, Matern52
from varz import Vars
from varz.torch import minimise_l_bfgs_b
import wbml.out

from olmm import IGP, OLMM

B.epsilon = 1e-8


def date_to_decimal_year(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year + 1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length


def safe_inverse_float(x):
    try:
        return 1 / float(x)
    except ValueError:
        return np.nan
    except ZeroDivisionError:
        return np.nan


# Parse the data.
x, y = [], []
with open('experiments/data/exchange.csv') as f:
    reader = csv.reader(f)
    header = next(reader)[3:]  # Skip the first three columns.
    for row in reader:
        dt = datetime.datetime.strptime(row[1], '%Y/%m/%d')
        x.append(date_to_decimal_year(dt))
        y.append([safe_inverse_float(c) for c in row[3:]])

x = np.stack(x, axis=0)
y = np.stack(y, axis=0)

# Reorder the data, putting the to be predicted outputs last.
#   Note: output 2 misses quite a lot of data.
to_predict = [header.index('CAD/USD'),
              header.index('JPY/USD'),
              header.index('AUD/USD')]
others = sorted(set(range(len(header))) - set(to_predict) - {2})
order = others + [2] + to_predict

# Perform reordering of the data
y = y[:, order]
header = [header[i] for i in order]

# Remove regions from training data.
y_all = y.copy()
regions = [('CAD/USD', np.arange(49, 100), header.index('CAD/USD')),
           ('JPY/USD', np.arange(49, 150), header.index('JPY/USD')),
           ('AUD/USD', np.arange(49, 200), header.index('AUD/USD'))]
for _, inds, p in regions:
    y[inds, p] = np.nan

y_mean = np.nanmean(y, keepdims=True, axis=0)
y_scale = np.nanstd(y, keepdims=True, axis=0)
y_norm = (y - y_mean) / y_scale

p = B.shape(y)[1]
m = 3
vs = Vars(torch.float64)


def construct_model(vs):
    igp = IGP(vs,
              lambda vs_: vs_.pos(1) * Matern52().stretch(vs.pos(0.1)),
              vs.pos(1e-2, name='igp/noise'))
    olmm = OLMM(vs,
                igp,
                vs.orth(shape=(p, p), name='u_full')[:, :m],
                vs.pos(shape=(m,), name='s_sqrt'),
                vs.pos(1e-2, name='olmm/noise'))
    return olmm


def objective(vs):
    return -construct_model(vs).logpdf(torch.tensor(x),
                                       torch.tensor(y_norm))


minimise_l_bfgs_b(objective, vs, trace=True, iters=1000)

model = construct_model(vs)
model.condition(torch.tensor(x), torch.tensor(y_norm))
means, vars = B.to_numpy(model.predict(x))
lowers = means - vars ** .5
uppers = means + vars ** .5

means = means * y_scale + y_mean
lowers = lowers * y_scale + y_mean
uppers = uppers * y_scale + y_mean

# # Fit and predict GPAR.
# model = GPARRegressor(scale=0.1,
#                       linear=True, linear_scale=10.,
#                       nonlinear=True, nonlinear_scale=1.,
#                       rq=True,
#                       noise=0.01,
#                       impute=True, replace=False, normalise_y=True)
# model.fit(x, y)
# means, lowers, uppers = \
#     model.predict(x, num_samples=200, credible_bounds=True, latent=False)

# Compute SMSEs.
smses = []
for _, inds, p in regions:
    # For the purpose of comparison, standardise using the mean of the
    # *training* data! This is *not* how the SMSE usually is defined.
    mse_mean = np.nanmean((y_all[inds, p] - np.nanmean(y[:, p])) ** 2)
    mse_gpar = np.nanmean((y_all[inds, p] - means[inds, p]) ** 2)
    smses.append(mse_gpar / mse_mean)
wbml.out.kv('Average SMSE', np.mean(smses))

# Plot the result.
plt.figure(figsize=(12, 3))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

for i, (name, inds, p) in enumerate(regions):
    ax = plt.subplot(1, 3, i + 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.plot(x, means[:, p], c='tab:blue')
    plt.fill_between(x, lowers[:, p], uppers[:, p],
                     facecolor='tab:blue', alpha=.25)
    plt.scatter(x, y[:, p], c='tab:green', marker='x', s=10)
    plt.scatter(x[inds], y_all[inds, p], c='tab:orange', marker='x', s=10)
    plt.xlabel('Time (year)')
    plt.ylabel('Exchange rate')
    plt.title(name)

plt.tight_layout()
plt.savefig('experiments/exchange.pdf')
plt.show()

