from abc import ABC, abstractmethod
from functools import cached_property
from typing import TypeVar, NamedTuple, Optional, Union, Tuple

from jax import numpy as jnp

from jaxns.types import FloatArray, float_type, PRNGKey, IntArray, NestedSamplerState, LivePoints, Sample, \
    TerminationCondition, UType, XType

__all__ = [
    'AbstractModel',
    'AbstractSampler',
    'AbstractRejectionSampler',
    'AbstractMarkovSampler',
    'AbstractNestedSampler',
    'PreProcessType',
    'SeedPoint'
]

PreProcessType = TypeVar('PreProcessType')


class SeedPoint(NamedTuple):
    U0: FloatArray
    log_L0: FloatArray


class AbstractModel(ABC):
    """
        Represents a Bayesian model in terms of a generative prior, and likelihood function.
        """

    @property
    def U_placeholder(self) -> UType:
        """
        A placeholder for U-space sample.
        """
        return self.parsed_prior[0]

    @property
    def X_placeholder(self) -> XType:
        """
        A placeholder for X-space sample.
        """
        return self.parsed_prior[1]

    @property
    def U_ndims(self) -> int:
        """
        The prior dimensionality.
        """
        return self.U_placeholder.size

    @abstractmethod
    def _parsed_prior(self) -> Tuple[UType, XType]:
        """
        The parsed prior.

        Returns:
            U-space sample, X-space sample
        """
        ...

    @cached_property
    def parsed_prior(self) -> Tuple[UType, XType]:
        """
        The parsed prior.

        Returns:
            U-space sample, X-space sample
        """
        return self._parsed_prior()

    @abstractmethod
    def __hash__(self):
        """
        Hash of the model.
        """
        ...

    @abstractmethod
    def sample_U(self, key: PRNGKey) -> FloatArray:
        """
        Sample uniformly from the prior in U-space.

        Args:
            key: PRNGKey

        Returns:
            U-space sample
        """
        ...

    @abstractmethod
    def transform(self, U: UType) -> XType:
        """
        Compute the prior sample.

        Args:
            U: U-space sample

        Returns:
            prior sample
        """
        ...

    @abstractmethod
    def forward(self, U: UType, allow_nan: bool = False) -> FloatArray:
        """
        Compute the log-likelihood.

        Args:
            U: U-space sample
            allow_nan: whether to allow nans in likelihood

        Returns:
            log likelihood at the sample
        """
        ...

    @abstractmethod
    def log_prob_prior(self, U: UType) -> FloatArray:
        """
        Computes the log-probability of the prior.

        Args:
            U: The U-space sample

        Returns:
            the log probability of prior
        """
        ...

    @abstractmethod
    def sanity_check(self, key: PRNGKey, S: int):
        """
        Performs a sanity check on the model.

        Args:
            key: PRNGKey
            S: number of samples to check

        Raises:
            AssertionError: if any of the samples are nan.
        """
        ...


class AbstractSampler(ABC):
    def __init__(self, model: AbstractModel, efficiency_threshold: Optional[FloatArray] = None):
        self.model = model
        if efficiency_threshold is None:
            efficiency_threshold = 0.
        if efficiency_threshold < 0. or efficiency_threshold >= 1.:
            raise ValueError(f"{efficiency_threshold} must be in [0., 1.), got {efficiency_threshold}.")
        efficiency_threshold = jnp.asarray(efficiency_threshold, float_type)
        self.efficiency_threshold = efficiency_threshold

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def preprocess(self, state: NestedSamplerState, live_points: Union[LivePoints, None] = None) -> PreProcessType:
        """
        Produces a data structure that is necessary for sampling to run.
        Typically this is where clustering happens.

        Args:
            state: nested sampler state

        Returns:
            any valid pytree
        """
        ...

    @abstractmethod
    def get_sample(self, key: PRNGKey, log_L_constraint: FloatArray, live_points: LivePoints,
                   preprocess_data: PreProcessType) -> Sample:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            log_L_constraint: the constraint to sample within
            live_points: the current live points reservoir
            preprocess_data: the data pytree needed and produced by the sampler

        Returns:
            an i.i.d. sample
        """
        ...


class AbstractRejectionSampler(AbstractSampler):
    """
    Samplers that are based on rejection sampling. They usually first-lines of attack, and are stopped once efficiency
    gets too low.
    """
    pass


class AbstractMarkovSampler(AbstractSampler):
    """
    A sampler that conditions off a known satisfying point, e.g. a seed point.
    """

    @abstractmethod
    def get_sample_from_seed(self, key: PRNGKey, seed_point: SeedPoint, log_L_constraint: FloatArray,
                             preprocess_data: PreProcessType) -> Sample:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            seed_point: function that gets the next sample from a seed point
            log_L_constraint: the constraint to sample within
            preprocess_data: the data pytree needed and produced by the sampler

        Returns:
            an i.i.d. sample
        """
        ...


class AbstractNestedSampler(ABC):
    @abstractmethod
    def __call__(self, key: PRNGKey, term_cond: TerminationCondition, *,
                 init_state: Optional[NestedSamplerState] = None) -> Tuple[IntArray, NestedSamplerState]:
        """
        Performs approximate nested sampling followed by adaptive refinement.

        Args:
            key: PRNGKey
            term_cond: termination condition
            init_state: optional initial state

        Returns:
            termination reason, and exact state
        """
        ...
