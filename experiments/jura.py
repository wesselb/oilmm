from stheno import GP, EQ
from wbml.data.jura import load

from oilmm.tensorflow import OILMM
from util import *

if __name__ == "__main__":
    wd = WorkingDirectory("_experiments", "jura")

    train, test = load()

    x = np.array(list(map(list, train.index)))
    y = np.array(train)

    def build_latent_processes(ps):
        # Return models for latent processes, which are noise-contaminated GPs.
        return [
            (
                p.variance.positive(1)
                * GP(EQ().stretch(p.scale.positive(1e-1, shape=(2,)))),
                p.noise.positive(1e-2),
            )
            for p, _ in zip(ps, range(1))
        ]

    model = OILMM(
        tf.float64,
        build_latent_processes,
        data_transform="normalise,positive",
    )
    model.fit(x, y, trace=True, jit=True)
    model.vs.print()
    model = model.condition(x, y)

    mean, var = model.predict(x)
    lower = mean - 1.96 * np.sqrt(var)
    upper = mean + 1.96 * np.sqrt(var)

    # Report metrics.
    pred = pd.DataFrame(mean, index=train.index, columns=train.columns)
    out.kv("SMSEs", metric.smse(pred, test).dropna())
    out.kv("MAEs", metric.mae(pred, test).dropna())
    x_test = np.array(list(map(list, test.index)))
    y_test = np.array(test.reindex(train.columns, axis=1))
    out.kv("PPLP", model.logpdf(x_test, y_test) / count(y_test))
    out.kv("PPLP (exact)", ilmm(model).logpdf(x_test, y_test) / count(y_test))
