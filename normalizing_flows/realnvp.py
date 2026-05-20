import torch
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.nn.nets import ResidualNet
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.permutations import ReversePermutation
from sbi.neural_nets.estimators import ConditionalDensityEstimator


class RealNVP_SBI(ConditionalDensityEstimator):
    """
    Wraps an nflows Flow to satisfy the SBI ConditionalDensityEstimator interface.

    SBI shape conventions
    ---------------------
    log_prob(input, condition):
        input     : (batch, theta_dim)  — during training
                    (samples, batch, theta_dim)  — during evaluation with multiple samples
        condition : (batch, x_dim)
        returns   : (batch,) or (samples * batch,)   [per-sample log-probs]

    sample(sample_shape, condition):
        condition : (batch, x_dim)
        returns   : (num_samples, batch, theta_dim)

    nflows shape conventions
    ------------------------
    flow.log_prob(inputs, context):
        inputs  : (N, theta_dim)
        context : (N, x_dim)          — must match inputs batch size exactly
        returns : (N,)

    flow.sample(num_samples, context):
        context : (batch, x_dim)
        returns : (batch, num_samples, theta_dim)   ← note axis order!
    """

    def __init__(self, flow: Flow, theta_dim: int, x_dim: int):
        super().__init__(
            net=flow,
            input_shape=torch.Size([theta_dim]),
            condition_shape=torch.Size([x_dim]),
        )
        self.flow = flow

    def log_prob(
        self, input: torch.Tensor, condition: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Compute per-sample log probabilities.

        Handles both the training shape (batch, dim) and the evaluation shape
        (samples, batch, dim) by flattening the leading two dims and tiling the
        condition to match, then returning the flat log-prob vector.  SBI
        internally reshapes the result as needed.
        """
        if input.ndim == 2:
            # Standard training call: (batch, theta_dim)
            return self.flow.log_prob(inputs=input, context=condition)

        if input.ndim == 3:
            # Evaluation call: (num_samples, batch, theta_dim)
            num_samples, batch, theta_dim = input.shape
            # Flatten to (num_samples * batch, theta_dim)
            flat_input = input.reshape(num_samples * batch, theta_dim)
            # Tile condition so each sample in a batch shares the same observation
            # (batch, x_dim) -> (num_samples * batch, x_dim)
            tiled_condition = condition.repeat_interleave(num_samples, dim=0)
            # Note: repeat_interleave(n) tiles each row n times consecutively,
            # which correctly pairs every sample with its corresponding condition.
            return self.flow.log_prob(
                inputs=flat_input, context=tiled_condition
            )

        raise ValueError(
            f"Unexpected input shape: {input.shape}. Expected 2D or 3D tensor."
        )

    def sample(
        self, sample_shape: torch.Size, condition: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Draw samples from the posterior.

        nflows returns (batch, num_samples, theta_dim); SBI expects
        (num_samples, batch, theta_dim), so we transpose the first two axes.
        """
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)  # (x_dim,) -> (1, x_dim)

        num_samples = sample_shape[0]

        # nflows: (batch, num_samples, theta_dim)
        samples = self.flow.sample(num_samples=num_samples, context=condition)

        # SBI: (num_samples, batch, theta_dim)
        return samples.permute(1, 0, 2)

    def loss(
        self, input: torch.Tensor, condition: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Return per-sample negative log-likelihood (not reduced).

        SBI's trainer calls .mean() on whatever loss() returns, so we must
        hand back a (batch,) vector, not a scalar.
        """
        return -self.log_prob(input, condition)


def build_flow(
    hidden_features,
    num_transforms,
    theta_dim: int,
    x_dim: int,
) -> Flow:
    """
    Build a RealNVP normalizing flow using affine coupling layers.

    nflows AffineCouplingTransform uses a boolean mask where True = "pass-through"
    (identity) and False = "transformed". Alternating the mask each layer ensures
    every dimension gets transformed.
    """
    transforms = []
    for i in range(num_transforms):
        # Alternate which half is transformed each layer
        mask = torch.tensor(
            [j % 2 == i % 2 for j in range(theta_dim)],
            dtype=torch.bool,
        )
        transforms.append(
            AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=lambda in_f, out_f: ResidualNet(
                    in_features=in_f,
                    out_features=out_f,
                    hidden_features=hidden_features,
                    context_features=x_dim,
                    num_blocks=2,
                ),
            )
        )
        transforms.append(ReversePermutation(features=theta_dim))

    return Flow(
        transform=CompositeTransform(transforms),
        distribution=StandardNormal([theta_dim]),
    )


def build_realnvp(
    batch_theta: torch.Tensor,
    batch_x: torch.Tensor,
    hidden_features: int,
    num_transforms: int,
) -> RealNVP_SBI:
    """
    Factory function matching the signature SBI expects for a custom
    density_estimator builder: f(batch_theta, batch_x) -> estimator.
    """
    theta_dim = batch_theta.shape[-1]
    x_dim = batch_x.shape[-1]
    flow = build_flow(
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        theta_dim=theta_dim,
        x_dim=x_dim,
    )
    return RealNVP_SBI(flow=flow, theta_dim=theta_dim, x_dim=x_dim)
