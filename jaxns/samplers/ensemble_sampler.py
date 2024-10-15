from typing import TypeVar, NamedTuple, Tuple

import jax
from jax import numpy as jnp, random, lax

from jaxns.framework.bases import BaseAbstractModel
from jaxns.internals.cumulative_ops import cumulative_op_static
from jaxns.internals.types import PRNGKey, FloatArray, BoolArray, Sample, float_type, int_type, \
    StaticStandardNestedSamplerState, \
    IntArray, UType, StaticStandardSampleCollection
from jaxns.samplers.abc import SamplerState
from jaxns.samplers.bases import SeedPoint, BaseAbstractMarkovSampler


class StaticAcceptanceTrackingNestedSamplerState(NamedTuple):
    key: PRNGKey
    next_sample_idx: IntArray  # the next sample insert index <==> the number of samples
    sample_collection: StaticStandardSampleCollection
    front_idx: IntArray  # the index of the front of the live points within sample collection
    n_accepted: IntArray # the number of accepted points per chain in the last batch
    chain_length: int_type


def _differential_evolution(key: PRNGKey, U0: UType, point1: UType, point2: UType):
    key, prop_key, points_key = random.split(key, 3)
    scale = jax.lax.select(
       random.uniform(prop_key) > 0.5,
        1.0,
        2.38 / (2 * len(U0))**0.5,
    )
    # scale = 2.38 / (2 * len(U0))**0.5
    diff = point1 - point2
    new_point = (U0 + scale * diff) % 1
    assert new_point.shape == U0.shape
    assert diff.shape == U0.shape
    return new_point 


def _new_proposal(key: PRNGKey,
                  seed_point: SeedPoint,
                  log_L_constraint: FloatArray,
                  model: BaseAbstractModel,
                  point1: UType,
                  point2: UType,
) -> Tuple[FloatArray, FloatArray, IntArray]:
    """
    Sample from a slice about a seed point.

    Args:
        key: PRNG key
        seed_point: the seed point to sample from
        log_L_constraint: the constraint to sample within
        model: the model to sample from

    Returns:
        point_U: the new sample
        log_L: the log-likelihood of the new sample
        num_likelihood_evaluations: the number of likelihood evaluations performed
    """

    class Carry(NamedTuple):
        key: PRNGKey
        point_U: UType
        log_L: FloatArray
        num_likelihood_evaluations: IntArray

    def cond(carry: Carry) -> BoolArray:
        # jax.debug.print("{in1}, {in2}", in1=carry.log_L, in2=log_L_constraint)
        satisfaction = carry.log_L > log_L_constraint
        # Allow if on plateau to fly around the plateau for a while
        lesser_satisfaction = jnp.bitwise_and(seed_point.log_L0 == log_L_constraint, carry.log_L == log_L_constraint)
        done = jnp.bitwise_or(satisfaction, lesser_satisfaction)
        return jnp.bitwise_not(done)

    def body(carry: Carry) -> Carry:
        key, t_key, shrink_key = random.split(carry.key, 3)
        # propose point_U from seed_point.U0
        point_U = _differential_evolution(t_key, carry.point_U, point1, point2)
        log_L = model.forward(point_U)
        num_likelihood_evaluations = carry.num_likelihood_evaluations + jnp.ones_like(carry.num_likelihood_evaluations)
        return Carry(
            key=key,
            point_U=point_U,
            log_L=log_L,
            num_likelihood_evaluations=num_likelihood_evaluations,
        )

    key, n_key, t_key = random.split(key, 3)
    num_likelihood_evaluations = jnp.full((), 1, int_type)
    point_U = seed_point.U0
    proposed = _differential_evolution(t_key, seed_point.U0, point1, point2)
    proposed_log_L = model.forward(proposed)
    new_point, new_log_L, accept = jax.lax.cond(
        proposed_log_L > log_L_constraint,
        lambda: (proposed, proposed_log_L, jnp.full((), 1, int_type)),
        lambda: (point_U, seed_point.log_L0, jnp.full((), 0, int_type)),
    )

    # return new_point, new_log_L, accept
    return new_point, new_log_L, num_likelihood_evaluations


class DESampler(BaseAbstractMarkovSampler):
    """
    Slice sampler for a single dimension. Produces correlated samples.
    """

    def __init__(self, model: BaseAbstractModel, num_steps: int, num_phantom_save: int):
        """
        Unidimensional slice sampler.

        Args:
            model: AbstractModel
            num_steps: number of steps for MCMC chain.
            num_phantom_save: number of phantom samples to save. Phantom samples are samples that meeting the constraint
                but are not accepted. They can be used for numerous things, e.g. to estimate the evidence uncertainty.
        """
        super().__init__(model=model)
        if num_steps < 1:
            raise ValueError(f"num_slices should be >= 1, got {num_slices}.")
        if num_phantom_save < 0:
            raise ValueError(f"num_phantom_save should be >= 0, got {num_phantom_save}.")
        if num_phantom_save >= num_steps:
            raise ValueError(f"num_phantom_save should be < num_slices, got {num_phantom_save} >= {num_slices}.")
        self.num_steps = int(num_steps)
        self.num_phantom_save = int(num_phantom_save)

    def num_phantom(self) -> int:
        return self.num_phantom_save

    def pre_process(self, state: StaticStandardNestedSamplerState) -> SamplerState:
        sample_collection = jax.tree.map(lambda x: x[state.front_idx], state.sample_collection)
        return (sample_collection,)

    def post_process(self, sample_collection: StaticStandardSampleCollection,
                     sampler_state: SamplerState) -> SamplerState:
        return (sample_collection,)

    def get_seed_point(self, key: PRNGKey, sampler_state: SamplerState,
                       log_L_constraint: FloatArray) -> SeedPoint:

        sample_collection: StaticStandardSampleCollection
        (sample_collection,) = sampler_state

        select_mask = sample_collection.log_L > log_L_constraint
        # If non satisfied samples, then choose randomly from them.
        any_satisfied = jnp.any(select_mask)
        yes_ = jnp.asarray(0., float_type)
        no_ = jnp.asarray(-jnp.inf, float_type)
        unnorm_select_log_prob = jnp.where(
            any_satisfied,
            jnp.where(select_mask, yes_, no_),
            yes_
        )
        # Choose randomly where mask is True
        g = random.gumbel(key, shape=unnorm_select_log_prob.shape)
        sample_idx = jnp.argmax(g + unnorm_select_log_prob)

        return SeedPoint(
            U0=sample_collection.U_samples[sample_idx],
            log_L0=sample_collection.log_L[sample_idx],
        )

    def get_sample_from_seed(self, key: PRNGKey, seed_point: SeedPoint, log_L_constraint: FloatArray,
                             sampler_state: SamplerState) -> Tuple[Sample, Sample]:
        _state = sampler_state[0]
        U_samples = _state.U_samples
        # jax.debug.print(f"{seed_point.U0.shape}, {U_samples.shape}")

        class XType(NamedTuple):
            key: jax.Array
            point1: UType
            point2: UType

        def propose_op(sample: Sample, x: XType) -> Sample:
            U_sample, log_L, num_likelihood_evaluations = _new_proposal(
                key=x.key,
                seed_point=SeedPoint(
                    # U0=seed_point.U0,
                    U0=sample.U_sample,
                    log_L0=sample.log_L,
                ),
                log_L_constraint=log_L_constraint,
                model=self.model,
                point1=x.point1,
                point2=x.point2,
            )
            return Sample(
                U_sample=U_sample,
                log_L_constraint=log_L_constraint,
                log_L=log_L,
                num_likelihood_evaluations=num_likelihood_evaluations + sample.num_likelihood_evaluations
            )

        init_sample = Sample(
            U_sample=seed_point.U0,
            log_L_constraint=log_L_constraint,
            log_L=seed_point.log_L0,
            num_likelihood_evaluations=jnp.asarray(0, int_type)
        )
        key, subkey = random.split(key)
        idxs = jax.vmap(random.choice, in_axes=(0, None, None, None))(
            random.split(subkey, self.num_steps),
            len(U_samples),
            (2,),
            False
        ).T
        xs = XType(
            key=random.split(key, self.num_steps),
            point1=U_samples[idxs[0]],
            point2=U_samples[idxs[1]],
        )
        final_sample, cumulative_samples = cumulative_op_static(
            op=propose_op,
            init=init_sample,
            xs=xs
        )

        # Last sample is the final sample, the rest are potential phantom samples
        # Take only the last num_phantom_save phantom samples
        phantom_samples: Sample = jax.tree.map(lambda x: x[-(self.num_phantom_save + 1):-1], cumulative_samples)

        # Due to the cumulative nature of the sampler, the final number of likelihood evaluations should be divided
        # equally among the accepted sample and retained phantom samples.
        num_likelihood_evaluations_per_phantom_sample = (
                final_sample.num_likelihood_evaluations / (self.num_phantom_save + 1)
        ).astype(int_type)
        num_likelihood_evaluations_per_accepted_sample = (
                final_sample.num_likelihood_evaluations - num_likelihood_evaluations_per_phantom_sample * self.num_phantom_save
        )
        final_sample = final_sample._replace(
            num_likelihood_evaluations=num_likelihood_evaluations_per_accepted_sample
        )
        phantom_samples = phantom_samples._replace(
            num_likelihood_evaluations=jnp.full(
                phantom_samples.num_likelihood_evaluations.shape,
                num_likelihood_evaluations_per_phantom_sample,
                phantom_samples.num_likelihood_evaluations.dtype
            )
        )
        return final_sample, phantom_samples
