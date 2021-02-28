from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import while_loop, dynamic_update_slice, cond, scan
from jax import random, vmap, tree_multimap, tree_map
from jax.scipy.special import logsumexp
from typing import NamedTuple, Dict, Tuple
from collections import namedtuple

from jaxns.prior_transforms import PriorChain
from jaxns.param_tracking import \
    TrackedExpectation
from jaxns.utils import dict_multimap, chunked_pmap
from jaxns.likelihood_samplers import (slice_sampling, init_slice_sampler_state,
                                       multi_ellipsoid_sampler, init_multi_ellipsoid_sampler_state,
                                       sample_discrete_subspace, init_discrete_sampler_state)

import logging

logger = logging.getLogger(__name__)


class NestedSamplerState(NamedTuple):
    key: jnp.ndarray
    done: bool
    i: int
    num_likelihood_evaluations: int  # int, number of times logL evaluated.
    live_points_U: jnp.ndarray  # [N, M] init_U in unit cube of live init_U
    live_points_X: Dict  # [N, M] init_U in constrained space of live init_U in dict structure
    log_L_live: jnp.ndarray  # log likelihood of live init_U
    num_contour_prev: jnp.ndarray  # how many live_points remaining in contour
    last_log_L_contour: jnp.ndarray  # last value of log_L on contour
    reservoir_points_U: jnp.ndarray  # [N, M] init_U in unit cube of live init_U
    reservoir_points_X: Dict  # [N, M] init_U in constrained space of live init_U in dict structure
    log_L_reservoir: jnp.ndarray  # [N] log likelihood of live init_U
    reservoir_point_available: jnp.ndarray  # [R, N] Whether the point in reservoir has been used before.
    dead_points: Dict  # [D, M] dead init_U in dci structure
    log_X: jnp.ndarray  # [D] logX
    log_w: jnp.ndarray  # [D] dX L
    num_dead: int  # int, number of samples (dead init_U) taken so far.
    log_L_dead: jnp.ndarray  # log likelhood of dead init_U
    likelihood_evaluations_per_sample: jnp.ndarray  # array of efficiency per accepted sample
    last_likelihood_evaluations_per_sample: namedtuple  # arbitrary state passed between iterations of the sampling step
    tracked_expectations_state: namedtuple  # marginalised functions states


class NestedSampler(object):
    """
    Implements nested sampling with the following steps:
    . initialise the state
        . Initialise Sampler with Prior:
        . Sampler -> produce pytree of shapes and dytpes of live points U subspaces
        . Sampler -> produce pytree of shapes and dtypes of live points X subspaces (identity transform of U for some)
        . Sampler -> sample initial set of live points in U and X subspaces and log likelihood (maximum number for
                        dynamic types).
        . Collector -> Allocate collector pytree for dead points X subspaces
        . Collector -> Allocate collector pytree for dead points log likelihood
        . Collector -> Allocate collector pytree for number of likelihood evaluations per sample
        . Collector -> Allocate collector pytree for log_X and log_w
    . main loop
        . Select all live points in lowest contour plateau
        . Replenish as many samples as were taken from live points from the reservoir.
            . Check reservoir for how many points can be selected from there.
            . Parallel sample until remaining are filled.
                . When parallel sampling, always replace lowest ones first.
            . Select lowest likelihood points from reservoir which meet the constraint.
        . For each one increment marginalisations, and collect with Collector

    . finalise
    """

    _available_samplers = ['slice', 'multi_ellipsoid']

    def __init__(self,
                 loglikelihood,
                 prior_chain: PriorChain,
                 sampler_name='slice',
                 num_parallel_samplers: int = 1,
                 sampler_kwargs=None,
                 num_live_points=None,
                 max_samples=1e5,
                 collect_samples: bool = True,
                 only_marginalise=False,
                 **marginalised):
        """

        Args:
            loglikelihood: callable(**params, **unused_params)
            prior_chain: PriorChain
            sampler_name:  str, which sampler to use for continuous variables.
            num_live_points: int, number of live points to use in the computation.
                Rule of thumb=(D+1)*(# posterior modes)*O(50), i.e. you want at least D+1 points per mode
                to be able to detect the mode with ellipsoidal clustering, and you need several times more than that
                to be accurate.
            max_samples: int, the maximum number of samples to take.
                Rule of thumb=(information gain)*(num_live_points)*(a few), where information gain can be measured by
                running a low accuracy run with fewer live points. Just make sure this number is big enough.
            collect_samples: bool, whether to keep any tracked variables (tracked=True in the prior transform).
                Consumes much more memory, as the samples need to be allocated space.
            termination_frac: float, the algorthim is terminated when this much of current evidence estimate is greater
                than the amount left in live points.
            only_marginalise: bool, by setting this to true even the diagnostics are not recorded. So plot_diagnostics
                is not possible.
            num_parallel_samplers: int, number of parallel reservoirs to sample from.
            sampler_kwargs: dict of parameters to pass to the likelihood constrained sampler.
            **marginalised: dict of callables(**X, **unused) to marginalise over with the same signature as
                log_likelihood.
        """
        self.sampler_name = sampler_name
        if num_live_points is None:
            raise ValueError("num_live_points must be given.")
        num_live_points = int(num_live_points)
        if num_live_points < 1:
            raise ValueError(f"num_live_points {num_live_points} should be >= 1.")
        self.num_live_points = num_live_points
        max_samples = int(max_samples)
        if max_samples < 1:
            raise ValueError(f"max_samples {max_samples} should be >= 1.")
        self.max_samples = max_samples
        num_parallel_samplers = int(num_parallel_samplers)
        if num_parallel_samplers < 1:
            raise ValueError(f"num_parallel_samplers {num_parallel_samplers} should be >= 1.")
        self.num_parallel_samplers = num_parallel_samplers
        if sampler_kwargs is None:
            sampler_kwargs = dict()
        sampler_kwargs['depth'] = int(sampler_kwargs.get('depth', 3))
        if sampler_kwargs['depth'] < 1:
            raise ValueError(f"depth {sampler_kwargs['depth']} should be >= 1.")
        sampler_kwargs['num_slices'] = int(sampler_kwargs.get('num_slices', 5))
        if sampler_kwargs['num_slices'] < 1:
            raise ValueError(f"num_slices {sampler_kwargs['num_slices']} should be >= 1.")
        self.sampler_kwargs = sampler_kwargs
        self.collect_samples = bool(collect_samples)
        self.only_marginalise = bool(only_marginalise)
        if self.sampler_name not in self._available_samplers:
            raise ValueError("sampler {} should be one of {}.".format(self.sampler_name, self._available_samplers))

        def corrected_likelihood(**x):
            """
            Adds the log-homogeneous measure to the log-likelihood to account for the transform from a PriorBase.

            Args:
                **x: dict of priors in X domain.

            Returns:
                log-likelihood plus the log-homogeneous measure, and -inf if it is a nan.
            """
            log_L = loglikelihood(**x)
            log_homogeneous_measure = prior_chain.log_homogeneous_measure(**x)
            if log_homogeneous_measure is not None:
                log_L += log_homogeneous_measure
            if log_L.shape != ():
                raise ValueError("Shape of likelihood should be scalar, got {}".format(log_L.shape))
            return jnp.where(jnp.isnan(log_L), -jnp.inf, log_L)

        self.loglikelihood = corrected_likelihood
        self.prior_chain = prior_chain

        def loglikelihood_from_U(U_compact):
            return corrected_likelihood(**prior_chain(U_compact))

        self.loglikelihood_from_U = loglikelihood_from_U
        self.marginalised = marginalised if len(marginalised) > 0 else None
        test_input = dict_multimap(lambda shape, dtype: jnp.zeros(shape, dtype=dtype), prior_chain.shapes,
                                   prior_chain.dtypes)
        self.marginalised_shapes = {k: marg(**test_input).shape for k, marg in marginalised.items()} if len(
            marginalised) > 0 else None

    def _filter_prior_chain(self, d):
        """
        Filters a dict's keys to only those where prior variable of same name is tracked.
        Used for removing untracked priors.

        Args:
            d: dict

        Returns:
            dict with only keys that correspond to names being tracked.
        """
        return {name: d[name] for name, prior in self.prior_chain.prior_chain.items() if prior.tracked}

    def initial_state(self, key):
        """
        Initialises the state of samplers.
        """

        def single_sample(key):
            """
            Produces a single sample from the joint-prior and computes the likelihood.

            Args:
                key: PRNG key

            Returns:
                U, X, log_L
            """
            U = self.prior_chain.compactify_U(self.prior_chain.sample_U(key))
            X = self.prior_chain(U)
            log_L = self.loglikelihood(**X)
            return U, X, log_L

        # generate live points and reservoir of points
        key, init_key_live, init_key_reservoir = random.split(key, 3)
        live_points_U, live_points_X, log_L_live = vmap(single_sample)(
            random.split(init_key_live, self.num_live_points))

        reservoir_points_U, reservoir_points_X, log_L_reservoir = vmap(single_sample)(
            random.split(init_key_reservoir, self.num_live_points * self.num_parallel_samplers))
        reservoir_point_available = jnp.ones(log_L_reservoir.shape, dtype=jnp.bool_)

        # establish things to collect
        if not self.only_marginalise:
            # collect tracked dead points and diagnostics
            if self.collect_samples:
                dead_points = dict_multimap(lambda shape, dtype: jnp.zeros((self.max_samples,) + shape, dtype=dtype),
                                            self._filter_prior_chain(self.prior_chain.shapes),
                                            self._filter_prior_chain(self.prior_chain.dtypes))
            else:
                dead_points = None
            log_L_dead = jnp.zeros((self.max_samples,), dtype=jnp.float_)
            likelihood_evaluations_per_sample = jnp.zeros((self.max_samples,), dtype=jnp.float_)
            log_X = -jnp.inf * jnp.ones((self.max_samples,), dtype=jnp.float_)  # [D] logX
            log_w = -jnp.inf * jnp.ones((self.max_samples,), dtype=jnp.float_)  # [D] dX L
        else:
            # collect nothing
            dead_points = None
            log_L_dead = None
            likelihood_evaluations_per_sample = None
            log_X = None
            log_w = None

        # contains the logic for marginalisation
        tracked_expectations = TrackedExpectation(self.marginalised, self.marginalised_shapes)

        state = NestedSamplerState(
            key=key,
            done=jnp.array(False),
            i=jnp.asarray(0),
            num_likelihood_evaluations=self.num_live_points * (self.num_parallel_samplers + 1),
            live_points_X=live_points_X,
            live_points_U=live_points_U,
            log_L_live=log_L_live,
            num_contour_prev=jnp.asarray(0),
            last_log_L_contour=jnp.asarray(-jnp.inf),
            reservoir_points_X=reservoir_points_X,
            reservoir_points_U=reservoir_points_U,
            log_L_reservoir=log_L_reservoir,
            reservoir_point_available=reservoir_point_available,
            dead_points=dead_points,
            log_L_dead=log_L_dead,
            likelihood_evaluations_per_sample=likelihood_evaluations_per_sample,
            num_dead=jnp.asarray(0),
            tracked_expectations_state=tracked_expectations.state,
            log_X=log_X,
            log_w=log_w,
            last_likelihood_evaluations_per_sample=jnp.asarray(1.)
        )

        return state

    def _collect_dead_point(self, state: NestedSamplerState) -> (jnp.ndarray, NestedSamplerState):
        """
        Take one sample, and increase the counter if contour stays the same.

        Args:
            state: NestedSamplerState

        Returns:
            i_min:int
            state:NestedSamplerState
        """
        i_min = jnp.argmin(state.log_L_live)
        x_dead_new = tree_map(lambda x: x[i_min], state.live_points_X)
        log_L_dead_new = state.log_L_live[i_min]

        tracked_expectations = TrackedExpectation(self.marginalised, self.marginalised_shapes,
                                                  state=state.tracked_expectations_state)
        tracked_expectations.update(x_dead_new, self.num_live_points - state.num_contour_prev, log_L_dead_new)

        state = state._replace(num_contour_prev=jnp.where(log_L_dead_new > state.last_log_L_contour,
                                                          jnp.asarray(0),
                                                          state.num_contour_prev + 1),
                               last_log_L_contour=log_L_dead_new,
                               tracked_expectations_state=tracked_expectations.state)

        if not self.only_marginalise:
            log_X = dynamic_update_slice(state.log_X,
                                         tracked_expectations.state.X.log_value[None],
                                         [state.num_dead])
            log_w = dynamic_update_slice(state.log_w,
                                         tracked_expectations.state.dw.log_value[None],
                                         [state.num_dead])
            log_L_dead = dynamic_update_slice(state.log_L_dead,
                                              log_L_dead_new[None],
                                              [state.num_dead])
            likelihood_evaluations_per_sample = dynamic_update_slice(state.likelihood_evaluations_per_sample,
                                                                     state.last_likelihood_evaluations_per_sample[None],
                                                                     [state.num_dead]
                                                                     )
            state = state._replace(log_X=log_X, log_w=log_w, log_L_dead=log_L_dead,
                                   likelihood_evaluations_per_sample=likelihood_evaluations_per_sample)
            if self.collect_samples:
                dead_points = dict_multimap(lambda x, y: dynamic_update_slice(x,
                                                                              y.astype(x.dtype)[None],
                                                                              [state.num_dead] + [0] * len(y.shape)),
                                            state.dead_points, x_dead_new)
                state = state._replace(dead_points=dead_points)
        state = state._replace(num_dead=state.num_dead + 1)
        return i_min, state

    def _refill_reservoirs(self, state: NestedSamplerState) -> NestedSamplerState:
        """
        Refill all the reservoirs points that are not available or less than the current contour level.
        We only need to refill when:
            ~jnp.any(satisfying_reservoir_points)
            = jnp.all(~satisfying_reservoir_points)
            = jnp.all((~state.reservoir_point_available) | (state.log_L_reservoir <= state.last_log_L_contour)
        => all reservoirs need to be completely refilled

        Once are reservoirs are refilled in parallel, we simply merge them into one.

        Args:
            state: NestedSamplerState

        Returns:
            state: NestedSamplerState
        """
        # build sampler initial state from the set of live_points
        key, init_sampler_state_key, sample_key = random.split(state.key, 3)
        state = state._replace(key=key)
        tracked_expectations = TrackedExpectation(self.marginalised, self.marginalised_shapes,
                                                  state=state.tracked_expectations_state)
        # set up subspace samplers
        subspace_sampler_states = []
        for subspace_idx, subspace in enumerate(self.prior_chain.subspaces):
            subspace_type = self.prior_chain.subspace_type(subspace)
            if subspace_type == 'discrete':
                sampler_state = init_discrete_sampler_state(self.prior_chain.prior_chain.num_outcomes)
            elif subspace_type == 'continuous':
                if self.sampler_name == 'slice':
                    sampler_state = init_slice_sampler_state(init_sampler_state_key,
                                                             state.live_points_U[subspace_idx],
                                                             self.sampler_kwargs['depth'],
                                                             tracked_expectations.state.X.log_value,
                                                             self.sampler_kwargs['num_slices'])
                elif self.sampler_name == 'multi_ellipsoid':
                    sampler_state = init_multi_ellipsoid_sampler_state(
                        init_sampler_state_key,
                        state.live_points_U[subspace_idx],
                        self.sampler_kwargs['depth'],
                        tracked_expectations.state.X.log_value)
                else:
                    raise ValueError("Invalid sampler name {}".format(self.sampler_name))
            else:
                raise ValueError("Subspace type {} is invalid".format(subspace_type))
            subspace_sampler_states.append(sampler_state)


        def build_log_likelihood(idx, U):
            def log_likelihood(u_compact_i):
                U_compact = tuple([u_compact_i if i == idx else U[i] for i in range(len(U))])
                return self.loglikelihood_from_U(U_compact)

            return log_likelihood

        def _one_sample(key):
            choice_key, sample_key = random.split(key, 2)
            choice = random.choice(choice_key, self.num_live_points)
            sample_U = list(tree_map(lambda x: x[choice], state.live_points_U))
            sample_log_L = state.last_log_L_contour
            num_likelihood_evaluations = 0
            # TODO: do this loop a configurable number of times for to drop auto-correlation in the chain.
            # Only needed when there are more than one subspace.
            if len(sample_U) > 1:
                logger.warning("In cases where there are more than one subspace, sampling of subspaces occurs via a "
                               "Markov chain, and you should be satisfied with a certain amount of auto-correlation.")
            for subspace_idx, (sampler_state, subspace) in enumerate(
                    zip(subspace_sampler_states, self.prior_chain.subspaces)):
                log_likelihood = build_log_likelihood(subspace_idx, sample_U)
                subspace_type = self.prior_chain.subspace_type(subspace)
                if subspace_type == 'discrete':
                    sampler_results = sample_discrete_subspace(sample_key,
                                                               log_L_constraint=state.last_log_L_contour,
                                                               log_likelihood_from_U=log_likelihood,
                                                               sampler_state=sampler_state)
                    sample_log_L = sampler_results.log_L_new
                    sample_U = tuple(
                        [sampler_results.u_new if i == subspace_idx else sample_U[i] for i in range(len(sample_U))])
                    num_likelihood_evaluations += sampler_results.num_likelihood_evaluations
                    sample_key = sampler_results.key
                elif subspace_type == 'continuous':
                    if self.sampler_name == 'slice':
                        sampler_results = slice_sampling(sample_key,
                                                         log_L_constraint=state.last_log_L_contour,
                                                         init_U=sample_U[subspace_idx],
                                                         num_slices=self.sampler_kwargs['num_slices'],
                                                         log_likelihood_from_U=log_likelihood,
                                                         sampler_state=sampler_state)
                        sample_log_L = sampler_results.log_L_new
                        sample_U = tuple(
                            [sampler_results.u_new if i == subspace_idx else sample_U[i] for i in range(len(sample_U))])
                        num_likelihood_evaluations += sampler_results.num_likelihood_evaluations
                        sample_key = sampler_results.key
                    elif self.sampler_name == 'multi_ellipsoid':
                        sampler_results = multi_ellipsoid_sampler(sample_key,
                                                                  log_L_constraint=state.last_log_L_contour,
                                                                  log_likelihood_from_U=log_likelihood,
                                                                  sampler_state=sampler_state
                                                                  )
                        sample_log_L = sampler_results.log_L_new
                        sample_U = tuple(
                            [sampler_results.u_new if i == subspace_idx else sample_U[i] for i in range(len(sample_U))])
                        num_likelihood_evaluations += sampler_results.num_likelihood_evaluations
                        sample_key = sampler_results.key
                    else:
                        raise ValueError("Invalid sampler name {}".format(self.sampler_name))
                else:
                    raise ValueError("Subspace type {} is invalid".format(subspace_type))
            sample_X = self.prior_chain(sample_U)
            return num_likelihood_evaluations, sample_U, sample_X, sample_log_L

        if self.num_parallel_samplers > 1:
            (num_likelihood_evaluations, reservoir_points_U, reservoir_points_X, log_L_reservoir) = \
                chunked_pmap(_one_sample, random.split(sample_key, state.log_L_reservoir.size),
                             chunksize=self.num_parallel_samplers, use_vmap=True)
        else:
            def body(state, args):
                return state, _one_sample(*args)
            _, (num_likelihood_evaluations, reservoir_points_U, reservoir_points_X, log_L_reservoir) = scan(
                body, (), (random.split(sample_key, state.log_L_reservoir.size),), unroll=1)

        reservoir_point_available = jnp.ones(log_L_reservoir.shape, dtype=jnp.bool_)
        new_likelihood_evaluations_per_sample = jnp.mean(num_likelihood_evaluations)

        state = state._replace(reservoir_points_U=reservoir_points_U,
                               reservoir_points_X=reservoir_points_X,
                               log_L_reservoir=log_L_reservoir,
                               reservoir_point_available=reservoir_point_available,
                               last_likelihood_evaluations_per_sample=new_likelihood_evaluations_per_sample,
                               num_likelihood_evaluations=state.num_likelihood_evaluations + jnp.sum(num_likelihood_evaluations))
        return state

    def _replace_dead_point(self, i_min: int, state: NestedSamplerState) -> NestedSamplerState:
        """
        Replace the dead point with a point from one of the reservoirs.

        Args:
            i_min: int
            state: NestedSamplerState

        Returns:
            state: NestedSamplerState
        """
        satisfying_reservoir_points = state.reservoir_point_available & (
                state.log_L_reservoir > state.last_log_L_contour)
        # key, choice_key = random.split(state.key, 2)
        # state = state._replace(key=key)
        # choice = random.choice(choice_key,
        #               a=jnp.arange(state.log_L_reservoir.size),
        #               p=satisfying_reservoir_points/jnp.sum(satisfying_reservoir_points))
        choice = jnp.argmax(satisfying_reservoir_points)
        live_points_U = tree_multimap(
            lambda x, y: dynamic_update_slice(x, y[None, choice], [i_min] + [0] * len(y.shape[1:])),
            state.live_points_U, state.reservoir_points_U)
        live_points_X = tree_multimap(
            lambda x, y: dynamic_update_slice(x, y[None, choice], [i_min] + [0] * len(y.shape[1:])),
            state.live_points_X, state.reservoir_points_X)
        log_L_live = dynamic_update_slice(state.log_L_live,
                                          state.log_L_reservoir[None, choice],
                                          [i_min])
        reservoir_point_available = dynamic_update_slice(state.reservoir_point_available,
                                                         jnp.zeros((1,), dtype=jnp.bool_),
                                                         [choice])
        state = state._replace(live_points_U=live_points_U,
                               live_points_X=live_points_X,
                               log_L_live=log_L_live,
                               reservoir_point_available=reservoir_point_available)
        return state

    def _one_step(self, state: NestedSamplerState) -> NestedSamplerState:
        """
        Performs one step of the algorithm.

        Args:
            state: NestedSamplerState before iteration
            collect_samples: bool, whether to collect samples
            only_marginalise: bool whether to only marginalise and not collect samples of diagnostics.
            sampler_kwargs: dict of kwargs for the sampler
            num_parallel_samplers: int, how many parallel samplers to run with vmap.

        Returns:
            NestedSamplerState after one iteration.
        """

        # Take one sample, and decrease the counter if contour stays the same.
        i_min, state = self._collect_dead_point(state)

        # refill reservoirs if there are no more satisfying points
        satisfying_reservoir_points = state.reservoir_point_available & (
                state.log_L_reservoir > state.last_log_L_contour)
        do_refill = ~jnp.any(satisfying_reservoir_points)
        state = cond(do_refill,
                     self._refill_reservoirs,
                     lambda state: state,
                     state)

        # Replace dead point with new one.
        state = self._replace_dead_point(i_min, state)

        return state

    def __call__(self, key, termination_frac=0.01):
        """
        Perform nested sampling.

        Args:
            key: PRNG
            termination_frac: float, the algorthim is terminated when this much of current evidence estimate is greater
                than the amount left in live points.

        Returns:
            NestedSamplingResult
        """
        state = self.initial_state(key)

        def body(state: NestedSamplerState):
            # do one sampling step
            state = self._one_step(state)

            tracked_expectations = TrackedExpectation(self.marginalised, self.marginalised_shapes,
                                                      state=state.tracked_expectations_state)
            logZ = tracked_expectations.evidence_mean()
            # Z_live = <L> X_i = exp(logsumexp(log_L_live) - log(N) + log(X))
            logZ_live = logsumexp(state.log_L_live) - jnp.log(state.log_L_live.shape[0]) \
                        + tracked_expectations.state.X.log_value
            # print("i={} | log(Z_live)={} | log(Z)={} | Z_live/Z={}".format(state.i, logZ_live, logZ, jnp.exp(logZ_live-logZ)))
            # Z_live < f * Z => logZ_live < log(f) + logZ
            done = (logZ_live < jnp.log(termination_frac) + logZ) \
                   | ((state.i + 1) >= self.max_samples)
            state = state._replace(done=done,
                                   i=state.i + 1)
            return state

        state = while_loop(lambda state: ~state.done,
                           body,
                           state)
        results = self._finalise_results(state)
        return results

    def _finalise_results(self, state: NestedSamplerState):
        """
        Produces the NestedSamplingResult.
        """
        collect = ['logZ',
                   'logZerr',
                   'ESS',
                   'ESS_err',
                   'H',
                   'H_err',
                   'num_likelihood_evaluations',
                   'efficiency',
                   'marginalised',
                   'marginalised_uncert',
                   'log_L_samples',
                   'n_per_sample',
                   'log_p',
                   'log_X',
                   'sampler_efficiency',
                   'num_samples'
                   ]

        if self.collect_samples and not self.only_marginalise:
            collect.append('samples')

        NestedSamplerResults = namedtuple('NestedSamplerResults', collect)
        tracked_expectations = TrackedExpectation(self.marginalised, self.marginalised_shapes,
                                                  state=state.tracked_expectations_state)
        reservoir_is_satisfying = state.reservoir_point_available & (state.log_L_reservoir > state.last_log_L_contour)
        log_L_reservoir = jnp.where(reservoir_is_satisfying, state.log_L_reservoir, -jnp.inf)
        remaining_points_log_L = jnp.concatenate([log_L_reservoir, state.log_L_live], axis=0)
        remaining_points_X = tree_multimap(lambda x, y: jnp.concatenate([x, y], axis=0),
                                           state.reservoir_points_X, state.live_points_X)
        live_update_results = tracked_expectations.update_from_live_points(remaining_points_X,
                                                                           remaining_points_log_L,
                                                                           log_L_contour=state.last_log_L_contour)
        if self.marginalised is not None:
            marginalised = tracked_expectations.marg_mean()
            marginalised_uncert = None  # tracked_expectations.marg_variance() not stable
        else:
            marginalised = None
            marginalised_uncert = None

        num_live_points = state.log_L_live.shape[0]
        if not self.only_marginalise:
            n_per_sample = jnp.where(jnp.arange(self.max_samples) < state.num_dead, num_live_points, jnp.inf)
            n_per_sample = dynamic_update_slice(n_per_sample,
                                                live_update_results[0], #num_live_points - jnp.arange(num_live_points, dtype=n_per_sample.dtype),
                                                [state.num_dead])
            sampler_efficiency = dynamic_update_slice(1. / state.likelihood_evaluations_per_sample,
                                                      jnp.ones(num_live_points),
                                                      [state.num_dead])
            log_w = dynamic_update_slice(state.log_w,
                                         live_update_results[3],
                                         [state.num_dead])
            log_p = log_w - logsumexp(log_w)
            log_X = dynamic_update_slice(state.log_X,
                                         live_update_results[2],
                                         [state.num_dead])
            log_L_samples = dynamic_update_slice(state.log_L_dead,
                                                 live_update_results[1],
                                                 [state.num_dead])
        else:
            n_per_sample = None
            log_p = None
            log_X = None
            log_L_samples = None
            sampler_efficiency = None

        num_samples = state.num_dead + num_live_points + jnp.sum(reservoir_is_satisfying)

        data = dict(
            logZ=tracked_expectations.evidence_mean(),
            logZerr=jnp.sqrt(tracked_expectations.evidence_variance()),
            ESS=tracked_expectations.effective_sample_size(),
            ESS_err=None,
            H=tracked_expectations.information_gain_mean(),
            H_err=jnp.sqrt(tracked_expectations.information_gain_variance()),
            num_likelihood_evaluations=state.num_likelihood_evaluations,
            efficiency=num_samples / state.num_likelihood_evaluations,
            marginalised=marginalised,
            marginalised_uncert=marginalised_uncert,
            n_per_sample=n_per_sample,
            log_p=log_p,
            log_X=log_X,
            log_L_samples=log_L_samples,
            num_samples=num_samples,
            sampler_efficiency=sampler_efficiency
        )
        if self.collect_samples and not self.only_marginalise:
            ar = jnp.argsort(state.log_L_live)
            samples = dict_multimap(lambda dead_points, live_points:
                                    dynamic_update_slice(dead_points,
                                                         live_points.astype(dead_points.dtype)[ar],
                                                         [state.num_dead] + [0] * (len(dead_points.shape) - 1)),
                                    state.dead_points, state.live_points_X)
            data['samples'] = samples
        return NestedSamplerResults(**data)
