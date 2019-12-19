import itertools as itt
from obstools.phot.diagnostics import scatter_density_plot


def corner():
    fig, axes = plt.subplots(dof, dof,
                             figsize=(11.3, 8.75),
                             sharex='col', sharey='row',
                             gridspec_kw=dict(top=0.975,
                                              bottom=0.1,
                                              left=0.08,
                                              right=0.975,
                                              hspace=0.05,
                                              wspace=0.05))
    samples = sampler.chain[:, ::100, :].reshape(-1, dof)

    bins = 50
    min_count_density = 10
    alpha = 0.75
    labels = 'm', 'b', r'$\alpha$', r'$\mu_b$', r'$\sigma_b$'

    for i, j in itt.combinations(range(dof), 2):
        ii, jj = dof - i - 1, dof - j - 1
        pair = samples[:, [jj, ii]]
        ax = axes[ii, jj]
        # plot scatter / density
        hvals, poly_coll, points = scatter_density_plot(
                ax, pair, bins, min_count_density,
                scatter_kws=dict(marker='.',
                                 ms=0.5, alpha=alpha),
                density_kws=dict(cmap='magma',
                                 alpha=alpha,
                                 edgecolors='face'))

        # plot histograms
        if i == 0:
            cmap = poly_coll.get_cmap()
            hvals, hbins, patches = axes[jj, jj].hist(samples[:, jj], bins)

        # Now, we'll loop through our objects and set the color of each accordingly
        for h, p in zip(hvals / hvals.max(), patches):
            p.set_facecolor(cmap(h))

        # make pretty
        ax.tick_params(labelrotation=45)
        if ii == dof - 1:
            ax.set_xlabel(labels[jj])
        if jj == 0:
            ax.set_ylabel(labels[ii])

        # remove axes in upper right triangle
        axes[i, j].remove()
        # unshare diagnonal y axius
        ax.get_shared_y_axes().remove(axes[ii, ii])

