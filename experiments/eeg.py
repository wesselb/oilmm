from stheno import GP, EQ
from wbml.data.eeg import load

from oilmm.tensorflow import OILMM
from util import *

if __name__ == "__main__":
    wd = WorkingDirectory("_experiments", "eeg")

    _, train, test = load()

    x = np.array(train.index)
    y = np.array(train)

    def build_latent_processes(ps):
        # Return models for latent processes, which are noise-contaminated GPs.
        return [
            (
                p.variance.positive(1) * GP(EQ().stretch(p.scale.positive(0.02))),
                p.noise.positive(1e-2),
            )
            for p, _ in zip(ps, range(3))
        ]

    model = OILMM(tf.float64, build_latent_processes)
    model.fit(x, y, trace=True, jit=True)
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
    name = "F2"  # Name of output to plot
    plt.figure(figsize=(12, 1.75))
    p = list(train.columns).index(name)
    plt.plot(x, mean[:, p], style="pred")
    plt.fill_between(x, lower[:, p], upper[:, p], style="pred")
    plt.scatter(x, y[:, p], style="train")
    plt.scatter(test[name].index, test[name], style="test")
    plt.xlabel("Time (second)")
    plt.xlim(0.4, 1)
    plt.ylabel(f"{name} (volt)")
    tweak()
    plt.savefig(wd.file("eeg.pdf"))
