from wbml.parser import Parser, Whitespace, Literal, Float, SkipUntil
import wbml.plot
import matplotlib.pyplot as plt
from wbml.experiment import WorkingDirectory


wd = WorkingDirectory("_experiments", "temperature_parse")


def parse(path):
    parser = Parser(path)
    parser.find_line("RMSE")
    rmse = parser.parse(
        SkipUntil("|"), Whitespace(), Literal("RMSE:"), Whitespace(), Float()
    )
    parser.find_line("PPLP")
    pplp = parser.parse(
        SkipUntil("|"), Whitespace(), Literal("PPLP:"), Whitespace(), Float()
    )
    return rmse, pplp


ms = [1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200, 225, 247]

oilmm_rmses, oilmm_pplps = zip(
    *[parse(f"_experiments/temperature_{m}/log.txt") for m in ms]
)
igp_rmse, igp_pplp = parse("_experiments/temperature_igp/log.txt")

wbml.plot.tex()

plt.figure(figsize=(5.5, 3))
plt.axvline(x=247, ymin=0, ymax=1, ls="--", c="black", lw=1)
plt.plot(ms, oilmm_pplps, "o-", lw=1.5, c="tab:blue", label="OILMM")
plt.text(243, -1.25, "$m=p$", horizontalalignment="right", verticalalignment="center")
# plt.gca().set_xscale('log')
plt.plot(
    [0, 300], [igp_pplp, igp_pplp], "-", lw=1.5, c="tab:orange", label="Independent GPs"
)
wbml.plot.tweak(legend=True, legend_loc="center")
plt.xlim(0, 250)
plt.xlabel("Number of latent processes $m$")
plt.ylabel("PPLP of held-out data")
plt.savefig(wd.file("temperature_pplp.pdf"))
wbml.plot.pdfcrop(wd.file("temperature_pplp.pdf"))

plt.figure(figsize=(5.5, 3))
plt.axvline(x=247, ymin=0, ymax=1, ls="--", c="black", lw=1)
plt.plot(ms, oilmm_rmses, "o-", lw=1.5, c="tab:blue", label="OILMM")
plt.text(243, 2.075, "$m=p$", horizontalalignment="right", verticalalignment="center")
# plt.gca().set_xscale('log')
plt.plot(
    [0, 300], [igp_rmse, igp_rmse], "-", lw=1.5, c="tab:orange", label="Independent GPs"
)
wbml.plot.tweak(legend=True, legend_loc="upper right")
plt.xlim(0, 250)
plt.xlabel("Number of latent processes $m$")
plt.ylabel("RMSE of held-out data")
plt.savefig(wd.file("temperature_rmse.pdf"))
wbml.plot.pdfcrop(wd.file("temperature_rmse.pdf"))

plt.show()
