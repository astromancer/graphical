
# third-party
import numpy as np
import matplotlib.pyplot as plt

# relative
from .. import pprint


def plot_chains_ts(samples, names=None, truths=None, truth_color='k',
                   n_max_walkers=50, n_max_points=1e4):
    
    nwalkers, nsamples, npar = samples.shape
    fig, axes = plt.subplots(npar, 1, figsize=(10, 12),
                             gridspec_kw=dict(hspace=0), sharex=True)

    if truths is not None:
        assert len(truths) == npar
    if names is not None:
        assert len(names) == npar

    sstep = 1
    if nsamples > n_max_points:
        sstep = int(nsamples // n_max_points)

    wstep = 1
    if nwalkers > n_max_walkers:
        wstep = int(nwalkers // n_max_walkers)

    for i in range(npar):
        dat = samples[::wstep, ::sstep, i].T
        ax = axes[i]
        ax.plot(dat)
        if truths is not None:
            ax.axhline(truths[i], color=truth_color)
        if names is not None:
            ax.set_ylabel(names[i])
        ax.grid()
        # y limits
        q = np.percentile(dat, (2, 98))
        ylim = q + np.multiply((-1, 1), q.ptp())
        ax.set_ylim(ylim)

    # label bottom axis
    ax.set_xlabel('step')
    fig.tight_layout()
    return fig


def parameter_estimate(samples, percentiles=(16, 50, 84), names=None,
                       echo=True):
    #
    q = np.percentile(samples, percentiles, 0)

    # median, percentile range
    μ = q[1]
    δ = abs(q[[0, -1]] - q[1])

    # TODO: use pprint.uarray
    μs = np.vectorize(pprint.numeric, ['U10'])(μ, switch=3)
    δs = np.vectorize(pprint.numeric, ['U10'])(δ, switch=3)

    s = ['%s^{+%s}_{-%s}' % (v, u, l) for (v, (u, l)) in zip(μs, δs.T)]
    # TODO: sometimes it may be sufficient to do a \pm format display
    if names is None:
        sl = list(map('$%s$'.__mod__, s))
    else:
        sl = list(map('$%s = %s$'.__mod__, zip(names, s)))

    if echo:
        from recipes.shell.interactive import is_interactive
        if is_interactive():
            from IPython.display import Latex, display
            for s in sl:
                display(Latex(s))
        else:
            # TODO
            'TODO'

    return q, sl
