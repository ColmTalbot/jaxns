import pylab as plt
import jax.numpy as jnp
from scipy.stats.kde import gaussian_kde
from jaxns.utils import safe_gaussian_kde

def plot_diagnostics(results, save_name=None):
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
    axs[0].plot(-results.log_X, results.n_per_sample)
    axs[0].set_ylabel(r'$n(X)$')
    axs[1].plot(-results.log_X, jnp.exp(results.log_L_samples))
    axs[1].set_ylabel(r'$L(X)$')
    axs[2].plot(-results.log_X, jnp.exp(results.log_p))
    axs[2].set_ylabel(r'$Z^{-1}L(X) dX$')
    axs[3].plot(-results.log_X, jnp.exp(results.logZ) * jnp.cumsum(jnp.exp(results.log_p)))
    axs[3].set_ylabel(r'$Z(x > X)$')
    axs[3].set_xlabel(r'$-\log X$')
    if save_name is not None:
        fig.savefig(save_name)
    plt.show()

def plot_cornerplot(results, vars=None,save_name=None):
    if vars is None:
        vars = [k for k, v in results.samples.items()]
    ndims = int(sum([jnp.prod(v.shape[1:]) for k, v in results.samples.items() if (k in vars)]))
    fig, axs = plt.subplots(ndims, ndims, figsize=(12, 12))
    weights = jnp.exp(results.log_p)
    nsamples = weights.size
    nbins = int(jnp.sqrt(results.ESS)) + 1
    dim = 0
    for key in sorted(results.samples.keys()):
        for i in range(jnp.prod(results.samples[key].shape[1:])):
            samples1 = results.samples[key].reshape((nsamples, -1))[:, i]
            if jnp.std(samples1) == 0.:
                dim += 1
                continue
            kde1 = gaussian_kde(samples1, weights=weights, bw_method='silverman')
            # kde1 = safe_gaussian_kde(samples1, weights=weights)
            samples1_resampled = kde1.resample(size=int(results.ESS))
            binsx = jnp.linspace(*jnp.percentile(samples1_resampled, [0, 100]), 2*nbins)
            dim2 = 0
            for key2 in sorted(results.samples.keys()):
                for i2 in range(jnp.prod(results.samples[key2].shape[1:])):
                    ax = axs[dim][dim2] if ndims > 1 else axs[0]
                    if dim2 > dim:
                        dim2 += 1
                        ax.set_xticks([])
                        ax.set_xticklabels([])
                        ax.set_yticks([])
                        ax.set_yticklabels([])
                        continue
                    ax.set_title('{}[{}] {}[{}]'.format(key, i, key2, i2))
                    if dim == dim2:
                        ax.plot(binsx, kde1(binsx))
                        sample_mean = jnp.average(samples1, weights=weights)
                        sample_std = jnp.sqrt(jnp.average((samples1 - sample_mean)**2, weights=weights))
                        ax.set_title("{:.2f}:{:.2f}:{:.2f}\n{:.2f}+-{:.2f}".format(*jnp.percentile(samples1_resampled,[5,50,95]), sample_mean, sample_std))
                        ax.vlines(sample_mean, *ax.get_ylim(), linestyles='solid', colors='red')
                        ax.vlines([sample_mean-sample_std, sample_mean + sample_std],
                                  *ax.get_ylim(), linestyles='dotted', colors='red')
                    else:
                        samples2 = results.samples[key2].reshape((nsamples, -1))[:, i2]
                        if jnp.std(samples2) == 0.:
                            dim2 += 1
                            continue
                        kde2 = gaussian_kde(jnp.stack([samples1, samples2], axis=0),
                                            weights=weights,
                                            bw_method='silverman')
                        samples2_resampled = kde2.resample(size=int(results.ESS))
                        ax.scatter(samples2_resampled[0,:], samples2_resampled[1,:], marker='+', c='black',alpha=0.5)
                        binsy = jnp.linspace(*jnp.percentile(samples2_resampled, [0, 100]), 2*nbins)
                        X, Y = jnp.meshgrid(binsx, binsy, indexing='ij')
                        ax.contour(kde2(jnp.stack([X.flatten(), Y.flatten()], axis=0)).reshape((2*nbins,2*nbins)).T,
                                   extent=(binsx.min(), binsx.max(),
                                           binsy.min(), binsy.max()),
                                   origin='lower')
                    if dim == ndims - 1:
                        ax.set_xlabel("{}[{}]".format(key2, i2))
                    if dim2 == 0:
                        ax.set_ylabel("{}[{}]".format(key, i))
                    dim2 += 1
            dim += 1
    if save_name is not None:
        fig.savefig(save_name)
    plt.show()