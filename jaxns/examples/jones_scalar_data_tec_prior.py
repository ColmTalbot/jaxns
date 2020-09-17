from jaxns.nested_sampling import NestedSampler
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.prior_transforms import UniformPrior, PriorChain, LaplacePrior, HalfLaplacePrior

from jax.scipy.linalg import solve_triangular
from jax import jit, vmap, disable_jit
from jax import numpy as jnp, random


def generate_data():
    T = 1
    tec = jnp.cumsum(10. * random.normal(random.PRNGKey(0), shape=(T,)))
    print(tec)
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = jnp.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV  # + 0.2  # + onp.linspace(-onp.pi, onp.pi, T)[:, None]
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=1)
    Y_obs = Y + 0.25 * random.normal(random.PRNGKey(1), shape=Y.shape)
    # Y_obs[500:550:2, :] += 3. * onp.random.normal(size=Y[500:550:2, :].shape_dict)
    Sigma = 0.25 ** 2 * jnp.eye(48)
    amp = jnp.ones_like(phase)
    return Sigma, T, Y_obs, amp, tec, freqs


def main():
    Sigma, T, Y_obs, amp, tec, freqs = generate_data()
    TEC_CONV = -8.4479745e6  # mTECU/Hz

    def log_normal(x, mean, scale):
        dx = (x - mean)/scale
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(scale)) \
               - 0.5 * dx @ dx

    def log_likelihood(tec, uncert, **kwargs):
        # tec = x[0]  # [:, 0]
        # uncert = x[1]  # [:, 1]
        # clock = x[2] * 1e-9
        # uncert = 0.25#x[2]
        phase = tec[:,None] * (TEC_CONV / freqs)  # + clock *(jnp.pi*2)*freqs#+ clock
        Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
        log_prob = jnp.sum(vmap(lambda Y, Y_obs: log_normal(Y, Y_obs, uncert))(Y, Y_obs))
        # print(log_prob)
        return log_prob

    # prior_transform = MVNDiagPrior(prior_mu, jnp.sqrt(jnp.diag(prior_cov)))
    # prior_transform = LaplacePrior(prior_mu, jnp.sqrt(jnp.diag(prior_cov)))
    prior_chain = PriorChain() \
        .push(UniformPrior('tec', [-100.]*T, [100.]*T)) \
        .push(HalfLaplacePrior('uncert', 0.25*jnp.ones(Y_obs.shape[-1])))

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='multi_ellipsoid')
    results = jit(lambda key: ns(key=key,
                      num_live_points=100,
                      max_samples=1e4,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=False,
                 sampler_kwargs=dict(depth=3)))(random.PRNGKey(0))


    ###

    plot_diagnostics(results)
    plot_cornerplot(results)


if __name__ == '__main__':
    main()
