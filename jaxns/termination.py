from typing import Optional, Union

from jax import numpy as jnp

from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.stats import linear_to_log_stats, effective_sample_size
from jaxns.types import TerminationCondition, SampleCollection, EvidenceCalculation, LivePoints, \
    int_type, TerminationConditionConjunction, TerminationConditionDisjunction

__all__ = ['determine_termination']


def determine_termination(
        term_cond: Union[TerminationConditionDisjunction, TerminationConditionConjunction, TerminationCondition],
        sample_collection: Optional[SampleCollection] = None,
        evidence_calculation: Optional[EvidenceCalculation] = None,
        live_points: Optional[LivePoints] = None
):
    """
    Determine if termination should happen. Termination Flags are bits:
        0-bit -> 1: used maximum allowed number of samples
        1-bit -> 2: evidence uncert below threshold
        2-bit -> 4: live points evidence below threshold
        3-bit -> 8: effective sample size big enough
        4-bit -> 16: used maxmimum allowed number of likelihood evaluations
        5-bit -> 32: maximum log-likelihood contour reached
        6-bit -> 64: sampler efficiency too low
        7-bit -> 128: entire live-points set is a single plateau

    Multiple flags are summed together

    Args:
        term_cond: termination condition

    Returns:
        boolean done signal, and termination reason
    """

    termination_reason = jnp.asarray(0, int_type)
    done = jnp.asarray(False, jnp.bool_)

    def _set_done_bit(bit_done, bit_reason, done, termination_reason):
        done = jnp.bitwise_or(bit_done, done)
        termination_reason += jnp.where(bit_done,
                                        jnp.asarray(2 ** bit_reason, int_type),
                                        jnp.asarray(0, int_type))
        return done, termination_reason

    if isinstance(term_cond, TerminationConditionConjunction):
        for c in term_cond.conds:
            _done, _reason = determine_termination(term_cond=c, sample_collection=sample_collection,
                                                   evidence_calculation=evidence_calculation, live_points=live_points)
            done = jnp.bitwise_and(_done, done)
            termination_reason = jnp.bitwise_and(_reason, termination_reason)
        return done, termination_reason

    if isinstance(term_cond, TerminationConditionDisjunction):
        for c in term_cond.conds:
            _done, _reason = determine_termination(term_cond=c, sample_collection=sample_collection,
                                                   evidence_calculation=evidence_calculation, live_points=live_points)
            done = jnp.bitwise_or(_done, done)
            termination_reason = jnp.bitwise_or(_reason, termination_reason)
        return done, termination_reason

    if term_cond.max_samples is not None:
        if sample_collection is None:
            raise ValueError("sample_collections must not be None.")
        # used all points
        num_samples = sample_collection.sample_idx
        reached_max_samples = num_samples >= term_cond.max_samples
        done, termination_reason = _set_done_bit(reached_max_samples, 0,
                                                 done=done, termination_reason=termination_reason)
    if term_cond.evidence_uncert is not None:
        if evidence_calculation is None:
            raise ValueError("evidence_calculation must not be None.")
        _, log_Z_var = linear_to_log_stats(
            log_f_mean=evidence_calculation.log_Z_mean,
            log_f2_mean=evidence_calculation.log_Z2_mean)
        evidence_uncert_low_enough = log_Z_var <= jnp.square(term_cond.evidence_uncert)
        done, termination_reason = _set_done_bit(evidence_uncert_low_enough, 1,
                                                 done=done, termination_reason=termination_reason)
    if term_cond.live_evidence_frac is not None:
        if evidence_calculation is None:
            raise ValueError("evidence_calculation must not be None.")
        if live_points is None:
            raise ValueError("live_points must not be None.")
        # Z_remaining/(Z_remaining + Z_current) < delta => 1 + Z_current/Z_remaining > 1/delta
        remaining_evidence_upper_bound = LogSpace(jnp.max(live_points.reservoir.log_L)) * LogSpace(
            evidence_calculation.log_X_mean)
        log_Z_remaining_upper = remaining_evidence_upper_bound.log_abs_val

        evidence_upper_bound = remaining_evidence_upper_bound + LogSpace(evidence_calculation.log_Z_mean)
        log_Z_upper = evidence_upper_bound.log_abs_val

        small_remaining_evidence = log_Z_remaining_upper - log_Z_upper < jnp.log(term_cond.live_evidence_frac)
        done, termination_reason = _set_done_bit(small_remaining_evidence, 2,
                                                 done=done, termination_reason=termination_reason)
    if term_cond.ess is not None:
        if evidence_calculation is None:
            raise ValueError("evidence_calculation must not be None.")
        # Kish's ESS = [sum weights]^2 / [sum weights^2]
        ess = effective_sample_size(evidence_calculation.log_Z_mean,
                                    evidence_calculation.log_dZ2_mean)
        ess_reached = ess >= term_cond.ess
        done, termination_reason = _set_done_bit(ess_reached, 3,
                                                 done=done, termination_reason=termination_reason)
    if term_cond.max_num_likelihood_evaluations is not None:
        if sample_collection is None:
            raise ValueError("sample_collections must not be None.")
        num_likelihood_evaluations = jnp.sum(sample_collection.reservoir.num_likelihood_evaluations)
        too_max_likelihood_evaluations = num_likelihood_evaluations >= term_cond.max_num_likelihood_evaluations
        done, termination_reason = _set_done_bit(too_max_likelihood_evaluations, 4,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.log_L_contour is not None:
        if sample_collection is None:
            raise ValueError("sample_collections must not be None.")
        log_L_max = jnp.max(
            jnp.where(jnp.arange(sample_collection.reservoir.log_L.size) < sample_collection.sample_idx,
                      sample_collection.reservoir.log_L,
                      -jnp.inf
                      )
        )
        likeihood_contour_reached = log_L_max >= term_cond.log_L_contour
        done, termination_reason = _set_done_bit(likeihood_contour_reached, 5,
                                                 done=done, termination_reason=termination_reason)

    if term_cond.efficiency_threshold is not None:
        if live_points is None:
            raise ValueError("live_points must be provided.")
        efficiency = jnp.reciprocal(jnp.mean(live_points.reservoir.num_likelihood_evaluations))
        efficiency_too_low = efficiency <= term_cond.efficiency_threshold
        done, termination_reason = _set_done_bit(efficiency_too_low, 6,
                                                 done=done, termination_reason=termination_reason)
    if (live_points is not None) and (live_points.reservoir.log_L.size > 1):
        all_plateau = jnp.min(live_points.reservoir.log_L) == jnp.max(live_points.reservoir.log_L)
        done, termination_reason = _set_done_bit(all_plateau, 7,
                                                 done=done, termination_reason=termination_reason)

    return done, termination_reason
