from stheno import GP, Exp
from wbml.data.exchange import load

from oilmm.tensorflow import OILMM
from util import *

if __name__ == "__main__":
    wd = WorkingDirectory("_experiments", "exchange")

    _, train, test = load()

    x = np.array(train.index)
    y = np.array(train)

    def build_latent_processes(ps):
        # Return models for latent processes, which are noise-contaminated GPs.
        return [
            (
                p.variance.positive(1) * GP(Exp().stretch(p.scale.positive(0.1))),
                p.noise.positive(1e-2),
            )
            for p, _ in zip(ps, range(3))
        ]

    model = OILMM(tf.float64, build_latent_processes)
    model.fit(x, y, jit=True, trace=True)
    model.vs.print()
    model = model.condition(x, y)

    mean, var = model.predict(x)
    lower = mean - 1.96 * np.sqrt(var)
    upper = mean + 1.96 * np.sqrt(var)

    # Report metrics.
    pred = pd.DataFrame(mean, index=train.index, columns=train.columns)
    smse = metric.smse(pred, test)
    out.kv("SMSEs", smse.dropna())
    out.kv("Average SMSEs", smse.mean())
    x_test, y_test = np.array(test.index), np.array(test.reindex(train.columns, axis=1))
    out.kv("PPLP", model.logpdf(x_test, y_test) / count(y_test))
    out.kv("PPLP (exact)", ilmm(model).logpdf(x_test, y_test) / count(y_test))

    # Plot the result.
    plt.figure(figsize=(12, 2))

    for i, name in enumerate(test.columns):
        p = list(train.columns).index(name)  # Index of output.
        plt.subplot(1, 3, i + 1)
        plt.plot(x, mean[:, p], style="pred")
        plt.fill_between(x, lower[:, p], upper[:, p], style="pred")
        plt.scatter(x, y[:, p], style="train")
        plt.scatter(test[name].index, test[name], style="test")
        plt.xlabel("Time (year)")
        plt.ylabel(name)
        tweak()

    plt.tight_layout()
    plt.savefig(wd.file("exchange.pdf"))
