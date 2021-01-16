import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as spc
import wbml.out
import wbml.plot
from matplotlib.pyplot import cm
from wbml.data.cmip5 import load
from wbml.experiment import WorkingDirectory

if __name__ == "__main__":

    # Process settings:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n_plot", type=int, default=1_000, help="Number of time points to plot."
    )
    parser.add_argument(
        "-mr", type=int, default=10, help="Number of latent processes for space."
    )
    parser.add_argument(
        "-ms", type=int, default=5, help="Number of latent processes for simulators."
    )
    parser.add_argument(
        "--separable", action="store_true", help="Use a separable model."
    )
    args = parser.parse_args()

    # Determine paths to write things to.
    if args.separable:
        suffix = "_separable"
    else:
        suffix = ""

    wd = WorkingDirectory(
        "_experiments", "simulators", subtle=True, log=f"log_process{suffix}.txt"
    )

    results = wd.load(f"results_mr{args.mr}_ms{args.ms}{suffix}.pickle")

    # Give overview of things that have been stored.
    wbml.out.kv("Results", ", ".join(results.keys()))
    wbml.out.kv("Parameters", ", ".join(results["learned_parameters"].keys()))

    # Print learned scales.
    scales = results["learned_parameters"]["space/scales"]
    wbml.out.kv("Latitude scale", scales[0])
    wbml.out.kv("Longitude scale", scales[1])

    # Extract everything from the dictionary of results.
    m = results["m"]
    p = results["p"]
    m_s = results["m_s"]
    m_r = results["m_r"]
    x_proj = results["x_proj"]
    y_proj = results["y_proj"].T
    preds = [
        [results[k][:, i] for k in ["means", "lowers", "uppers"]] for i in range(m)
    ]
    learned_parameters = results["learned_parameters"]
    corr_learned = results["corr_learned"]
    corr_empirical = results["corr_empirical"]

    wbml.plot.tex()

    # Plot predictions of latent processes.
    plt.figure(figsize=(6, 4))
    for i_r in range(2):
        for i_s in range(2):
            plt.subplot(2, 2, i_r + i_s * 2 + 1)
            if i_s == 0:
                plt.title(f"$i_r={i_r + 1}$", fontsize=12)
            if i_r == 0:
                plt.ylabel(f"$i_s={i_s + 1}$", fontsize=12)
            if i_s == 2:
                plt.xlabel("Day", fontsize=10)
            i_lat = i_r + i_s * m_r

            # Plot prediction.
            mean, lower, upper = preds[i_lat]
            plt.plot(
                x_proj[-args.n_plot :], y_proj[i_lat][-args.n_plot :], style="train"
            )
            plt.plot(x_proj[-args.n_plot :], mean[-args.n_plot :], style="pred")
            plt.fill_between(
                x_proj[-args.n_plot :],
                lower[-args.n_plot :],
                upper[-args.n_plot :],
                style="pred",
            )
            wbml.plot.tweak(legend=False)

            plt.gca().tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(
        wd.file(f"simulators_latents{suffix}.pdf"), format="pdf", bbox_inches="tight"
    )

    # Plot predictions of all latent processes.
    plt.figure(figsize=(25, 8))
    for i_r in range(10):
        for i_s in range(5):
            plt.subplot(5, 10, i_r + i_s * 10 + 1)
            if i_s == 0:
                plt.title(f"$i_r={i_r + 1}$", fontsize=12)
            if i_r == 0:
                plt.ylabel(f"$i_s={i_s + 1}$", fontsize=12)
            if i_s == 5:
                plt.xlabel("Day", fontsize=10)
            i_lat = i_r + i_s * m_r

            # Plot prediction.
            mean, lower, upper = preds[i_lat]
            plt.plot(
                x_proj[-args.n_plot :], y_proj[i_lat][-args.n_plot :], style="train"
            )
            plt.plot(x_proj[-args.n_plot :], mean[-args.n_plot :], style="pred")
            plt.fill_between(
                x_proj[-args.n_plot :],
                lower[-args.n_plot :],
                upper[-args.n_plot :],
                style="pred",
            )
            wbml.plot.tweak(legend=False)

    plt.tight_layout()
    plt.savefig(
        wd.file(f"simulators_latents_all{suffix}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )

    # Load data.
    loc, temp, sims = load()
    names = list(sims.keys())

    name_corrections = {
        "ACCESS1-0": "ACCESS1.0",
        "ACCESS1-3": "ACCESS1.3",
        "MRI-AGCM3-2H": "MRI-AGCM3.2H",
        "inmcm4": "INMCM4",
        "MRI-AGCM3-2S": "MRI-AGCM3.2S",
        "CSIRO-Mk3-6-0": "CSIRO-Mk3.6.0",
        "bcc-csm1-1-m": "BCC_CSM1.1(m)",
        "bcc-csm1-1": "BCC_CSM1.1",
    }
    for i in range(len(names)):
        try:
            names[i] = name_corrections[names[i]]
        except:
            continue

    # Escape underscores.
    names = [name.replace("_", "\\_") for name in names]

    # Perform clustering w.r.t learned correlations.
    pdist = 1 - np.abs(corr_learned)
    pdist = pdist[np.triu_indices_from(pdist, k=1)]
    linkage = spc.linkage(pdist, method="complete", optimal_ordering=True)
    order = spc.leaves_list(linkage)

    # Show dendogram.
    plt.figure(figsize=(4, 4))
    cmap = cm.rainbow(np.linspace(0, 1, 8))
    colors = [mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap]
    spc.set_link_color_palette(colors)
    d = spc.dendrogram(
        linkage,
        above_threshold_color="k",
        color_threshold=0.4,
        orientation="right",
        leaf_label_func=lambda i: names[i],
        leaf_font_size=10,
        leaf_rotation=0,
    )
    ax = plt.gca()

    num_leaves = len(order)
    link_to_color = {}

    # Apparently the colors are from a depth-first search...

    def traverse(link, i):
        i0 = int(linkage[link, 0])
        i1 = int(linkage[link, 1])

        if i0 >= len(order):
            i = traverse(i0 - num_leaves, i)
        if i1 >= len(order):
            i = traverse(i1 - num_leaves, i)

        c = d["color_list"][i]
        link_to_color[link] = c
        i += 1

        return i

    traverse(num_leaves - 2, 0)

    label_to_color = {}
    for i, link in enumerate(linkage):
        i0 = int(link[0])
        i1 = int(link[1])
        if i0 < len(order):
            leaf0 = list(order).index(i0)
            label_to_color[d["ivl"][leaf0]] = link_to_color[i]
        if i1 < len(order):
            leaf1 = list(order).index(i1)
            label_to_color[d["ivl"][leaf1]] = link_to_color[i]

    for label in plt.gca().get_ymajorticklabels():
        label.set_color(label_to_color[label.get_text()])

    wbml.plot.tweak(grid=False, legend=False)
    plt.savefig(
        wd.file(f"simulators_dendrogram{suffix}.pdf"), format="pdf", bbox_inches="tight"
    )

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    order = np.argsort(names)
    plt.imshow(
        corr_empirical[order, :][:, order], cmap=plt.get_cmap("RdBu"), vmin=-1, vmax=1
    )
    # plt.xticks(np.arange(28), names, rotation='vertical', fontsize=10)
    plt.xticks([])
    plt.yticks(np.arange(28), np.array(names)[order], fontsize=10)
    plt.ylim(27.5, -0.5)
    wbml.plot.tweak(grid=False, legend=False)
    plt.subplot(1, 2, 2)
    plt.imshow(
        corr_learned[order, :][:, order], cmap=plt.get_cmap("RdBu"), vmin=-1, vmax=1
    )
    plt.xticks([])
    plt.yticks([])
    # plt.xticks(np.arange(28), names, rotation='vertical', fontsize=10)
    # plt.yticks(np.arange(28), names, fontsize=10)
    plt.ylim(27.5, -0.5)
    wbml.plot.tweak(grid=False, legend=False)
    plt.tight_layout()
    plt.savefig(
        wd.file(f"simulators_correlations{suffix}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
