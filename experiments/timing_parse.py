from wbml.parser import Parser, Whitespace, Literal, Float, Integer
import wbml.plot
import matplotlib.pyplot as plt
from wbml.experiment import WorkingDirectory
import numpy as np

wd = WorkingDirectory("_experiments", "timing_parse")

parser = Parser("_experiments/timing/log.txt")

# Skip header.
for _ in range(10):
    parser.next_line()

totals = {n: {} for n in [100, 200, 300]}
hs = {n: {} for n in [100, 200, 300]}
percs = {n: {} for n in [100, 200, 300]}

while True:
    try:
        parser.find_line("n:")
        n = parser.parse(Literal("n:"), Whitespace(), Integer())
        parser.find_line("m:")
        m = parser.parse(Literal("m:"), Whitespace(), Integer())

        # Parse total time.
        parser.find_line("Total:")
        parser.find_line("Mean:")
        total_mean = parser.parse(Whitespace(), Literal("Mean:"), Whitespace(), Float())
        parser.find_line("Error:")
        total_error = parser.parse(
            Whitespace(), Literal("Error:"), Whitespace(), Float()
        )
        totals[n][m] = (total_mean, total_error)

        # Parse time for H
        parser.find_line("H:")
        parser.find_line("Mean:")
        total_mean = parser.parse(Whitespace(), Literal("Mean:"), Whitespace(), Float())
        parser.find_line("Error:")
        total_error = parser.parse(
            Whitespace(), Literal("Error:"), Whitespace(), Float()
        )
        hs[n][m] = (total_mean, total_error)

        # Parse percentage.
        parser.find_line("Percentage:")
        parser.find_line("Mean:")
        perc_mean = parser.parse(Whitespace(), Literal("Mean:"), Whitespace(), Float())
        parser.find_line("Error:")
        perc_error = parser.parse(
            Whitespace(), Literal("Error:"), Whitespace(), Float()
        )
        percs[n][m] = (perc_mean, perc_error)
    except RuntimeError as e:
        print(e)
        break


def get(d, i):
    keys = sorted(d.keys())
    return np.array([d[k][i] for k in keys])


wbml.plot.tex()

plt.figure(figsize=(12, 3))

plt.subplot(1, 3, 1)
for n in [100, 200, 300]:
    plt.plot(sorted(totals[n].keys()), get(totals[n], 0), "-o", label=f"$n={n}$")
plt.xlim(0, 250)
plt.xlabel("Number of latent processes $m$")
plt.ylabel("Total time (s)")
wbml.plot.tweak(legend=True, legend_loc="upper left")

plt.subplot(1, 3, 2)
for n in [100, 200, 300]:
    plt.plot(sorted(hs[n].keys()), get(hs[n], 0) * 1e3, "-o", label=f"$n={n}$")
plt.xlim(0, 250)
plt.xlabel("Number of latent processes $m$")
plt.ylabel("Time spent on basis (ms)")
wbml.plot.tweak(legend=True, legend_loc="upper left")

plt.subplot(1, 3, 3)
for n in [100, 200, 300]:
    plt.plot(sorted(percs[n].keys()), get(percs[n], 0), "-o", label=f"$n={n}$")
plt.xlim(0, 250)
plt.xlabel("Number of latent processes $m$")
plt.ylabel("Time spent on basis (\\%)")
wbml.plot.tweak(legend=True, legend_loc="upper left")

plt.savefig(wd.file("timing_h.pdf"))
wbml.plot.pdfcrop(wd.file("timing_h.pdf"))

plt.show()
