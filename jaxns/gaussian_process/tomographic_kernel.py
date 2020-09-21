from jaxns.gaussian_process.tomographic_kernel_utils import log_tomographic_weight_function_outer
import jax.numpy as jnp
from jaxns.gaussian_process.kernels import Kernel
from jax.scipy.special import logsumexp


class TomographicKernel(Kernel):
    def __init__(self, x0, fed_kernel, S=25):
        self.S = S
        self.x0 = x0
        self.fed_kernel = fed_kernel

    def __call__(self, x1, x2, height, width, l, sigma, *fed_kernel_params):
        a1 = x1[:, 0:3]
        k1 = x1[:, 3:6]
        a2 = x2[:, 0:3]
        k2 = x2[:, 3:6]
        return tomographic_kernel(a1, a2, k1, k2, self.x0, self.fed_kernel, height, width, l, sigma, *fed_kernel_params,
                                  S=self.S)

def log_tomographic_kernel(a1, a2, k1, k2, x0, fed_kernel, height, width, l, sigma, *fed_kernel_params, S=25):
    """
    Computes the tomographic kernel,
        I = int_0^1 int_0^1 K(x1+t1 p1 - x2 - t2 p2) dt1 dt2
    Args:
        key: jnp.ndarray
        a1: [N, 3]
        a2: [M, 3]
        k1: [N, 3]
        k2: [M, 3]
        x0: [3]
        fed_kernel: callable(r2, **params)
        height: ionosphere height, units of a1,a2
        width: ionosphere width, units of a1, a2
        l: FED kernel lengthscale
        **fed_kernel_params: kernel specific parameters

    Returns: [N, M]

    """
    kernel_act = fed_kernel.act
    kernel_inverse_x = fed_kernel.inverse_x
    # N
    s1 = (height - 0.5 * width - (a1[:, 2] - x0[2])) / k1[:, 2]
    # N, 3
    x1 = (a1 + s1[:, None] * k1) / l
    p1 = (k1 * width / k1[:, 2:3]) / l
    # M
    s2 = (height - 0.5 * width - (a2[:, 2] - x0[2])) / k2[:, 2]
    # M, 3
    x2 = (a2 + s2[:, None] * k2) / l
    p2 = (k2 * width / k2[:, 2:3]) / l

    # s = t*b sec phi + s0
    # ds = dt*b*sec phi
    # N,M
    log_conv = 2. * jnp.log(width) - (jnp.log(k1[:, 2:3]) + jnp.log(k2[None, :, 2]))

    # out until r = 5*half_width where the kernel should be really small
    # U
    bins = jnp.linspace(0., kernel_inverse_x(0.01, l=1., sigma=1.), S) ** 2
    u = 0.5 * (bins[:-1] + bins[1:])
    log_du = jnp.log(jnp.diff(bins))

    # N,M,U
    log_w = log_tomographic_weight_function_outer(bins, x1, x2, p1, p2, S=S)

    # N,M,U
    log_K = kernel_act(u, sigma, *fed_kernel_params)
    # N,M
    return logsumexp(log_K + log_w + log_du + log_conv[:, :, None], axis=-1)


def tomographic_kernel(a1, a2, k1, k2, x0, fed_kernel, height, width, l, sigma, *fed_kernel_params, S=25):
    """
    Computes the tomographic kernel,
        I = int_0^1 int_0^1 K(x1+t1 p1 - x2 - t2 p2) dt1 dt2
    Args:
        key: jnp.ndarray
        a1: [N, 3]
        a2: [M, 3]
        k1: [N, 3]
        k2: [M, 3]
        x0: [3]
        fed_kernel: callable(r2, **params)
        height: ionosphere height, units of a1,a2
        width: ionosphere width, units of a1, a2
        l: FED kernel lengthscale
        **fed_kernel_params: kernel specific parameters

    Returns: [N, M]

    """
    return jnp.exp(
        log_tomographic_kernel(a1, a2, k1, k2, x0, fed_kernel, height, width, l, sigma, *fed_kernel_params, S=S))


def test_tomographic_kernel():
    from jax import random
    from jaxns.gaussian_process.kernels import RBF
    import pylab as plt
    n = 300
    a1 = jnp.array([[-1, 0., 0.]])
    k1 = jnp.stack([4. * jnp.pi / 180. * random.uniform(random.PRNGKey(0), shape=(n,), minval=-1, maxval=1),
                    4. * jnp.pi / 180. * random.uniform(random.PRNGKey(1), shape=(n,), minval=-1, maxval=1),
                    jnp.ones(n)], axis=1)
    k1 /= jnp.linalg.norm(k1, axis=-1, keepdims=True)
    n = 1
    a2 = jnp.array([[1., 0., 0.]])
    k2 = jnp.stack([jnp.zeros(n),
                    jnp.zeros(n),
                    jnp.ones(n)], axis=1)
    k2 /= jnp.linalg.norm(k2, axis=-1, keepdims=True)
    x0 = jnp.zeros(3)
    K = tomographic_kernel(a1, a2, k1, k2, x0, RBF(), height=10., width=2., l=1., sigma=1., S=25)
    sc = plt.scatter(k1[:, 0], k1[:, 1], c=K[:, 0])
    plt.colorbar(sc)
    plt.show()


def dtec_tomographic_kernel(key, ref_ant, a1, a2, k1, k2, x0, fed_kernel, height, width, l, S=50, **fed_kernel_params):
    return tomographic_kernel(key, a1, a2, k1, k2, x0, fed_kernel, height, width, l, S=S, **fed_kernel_params) + \
           tomographic_kernel(key, ref_ant[None, :], ref_ant[None, :], k1, k2, x0, fed_kernel, height, width, l, S=S,
                              **fed_kernel_params) - \
           tomographic_kernel(key, ref_ant[None, :], a2, k1, k2, x0, fed_kernel, height, width, l, S=S,
                              **fed_kernel_params) - \
           tomographic_kernel(key, a1, ref_ant[None, :], k1, k2, x0, fed_kernel, height, width, l, S=S,
                              **fed_kernel_params)


def test_dtec_tomographic_kernel():
    from jax import random
    from jaxns.gaussian_process.kernels import m12_act
    import pylab as plt
    n = 300
    a1 = jnp.array([[-1, 0., 0.]])
    k1 = jnp.stack([4. * jnp.pi / 180. * random.uniform(random.PRNGKey(0), shape=(n,), minval=-1, maxval=1),
                    4. * jnp.pi / 180. * random.uniform(random.PRNGKey(1), shape=(n,), minval=-1, maxval=1),
                    jnp.ones(n)], axis=1)
    k1 /= jnp.linalg.norm(k1, axis=-1, keepdims=True)
    n = 1
    a2 = jnp.array([[1., 0., 0.]])
    k2 = jnp.stack([jnp.zeros(n),
                    jnp.zeros(n),
                    jnp.ones(n)], axis=1)
    k2 /= jnp.linalg.norm(k2, axis=-1, keepdims=True)
    x0 = jnp.zeros(3)
    fed_kernel_params = dict(sigma=1.)
    K = dtec_tomographic_kernel(random.PRNGKey(4), x0, a1, a2, k1, k2, x0, m12_act, 10., 2., 1., **fed_kernel_params)
    print(K)
    sc = plt.scatter(k1[:, 0], k1[:, 1], c=K[:, 0])
    plt.colorbar(sc)
    plt.show()


def ddtec_tomographic_kernel(key, ref_dir, ref_ant, a1, a2, k1, k2, x0, fed_kernel, height, width, l, S=50,
                             **fed_kernel_params):
    return dtec_tomographic_kernel(key, ref_ant, a1, a2, k1, k2, x0, fed_kernel, height, width, l, S=S,
                                   **fed_kernel_params) + \
           dtec_tomographic_kernel(key, ref_ant, a1, a2, ref_dir[None, :], ref_dir[None, :], x0, fed_kernel, height,
                                   width, l, S=S, **fed_kernel_params) - \
           dtec_tomographic_kernel(key, ref_ant, a1, a2, ref_dir[None, :], k2, x0, fed_kernel, height, width, l, S=S,
                                   **fed_kernel_params) - \
           dtec_tomographic_kernel(key, ref_ant, a1, a2, k1, ref_dir[None, :], x0, fed_kernel, height, width, l, S=S,
                                   **fed_kernel_params)


def test_ddtec_tomographic_kernel():
    from jax import random
    from jaxns.gaussian_process.kernels import rational_quadratic_act
    import pylab as plt
    n = 300
    a1 = jnp.array([[-1, 0., 0.]])
    k1 = jnp.stack([random.uniform(random.PRNGKey(0), shape=(n,), minval=-0.05, maxval=0.05),
                    random.uniform(random.PRNGKey(1), shape=(n,), minval=-0.05, maxval=0.05),
                    jnp.ones(n)], axis=1)
    k1 /= jnp.linalg.norm(k1, axis=-1, keepdims=True)
    n = 1
    a2 = jnp.array([[1., 0., 0.]])
    k2 = jnp.stack([jnp.zeros(n),
                    jnp.zeros(n),
                    jnp.ones(n)], axis=1)
    k2 /= jnp.linalg.norm(k2, axis=-1, keepdims=True)
    x0 = jnp.zeros(3)
    k0 = jnp.array([0., 0.05, 1.])
    k0 /= jnp.linalg.norm(k0, axis=-1, keepdims=True)
    fed_kernel_params = dict(sigma=1., alpha=2.)
    K = ddtec_tomographic_kernel(random.PRNGKey(4), k0, x0, a1, a2, k1, k2, x0, rational_quadratic_act, 10., 2., 0.36,
                                 S=200, **fed_kernel_params)
    print(K)
    sc = plt.scatter(k1[:, 0], k1[:, 1], c=K[:, 0])
    plt.colorbar(sc)
    plt.show()