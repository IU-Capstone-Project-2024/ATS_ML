import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from typing import Callable, Tuple

class CustomTradingNetwork(nn.Module):
    def __init__(self, feature_dim_ts: int, feature_dim_nf: int, combined_dim: int = 64):
        super(CustomTradingNetwork, self).__init__()

        # Sub-network for time series data
        self.time_series_net = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim_ts, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.time_series_length, 64),
            nn.ReLU()
        )

        # Sub-network for normal features
        self.normal_features_net = nn.Sequential(
            nn.Linear(feature_dim_nf, 64),
            nn.ReLU()
        )

        # Combined processing network
        self.combined_net = nn.Sequential(
            nn.Linear(64 + 64, combined_dim),
            nn.ReLU()
        )

        # Separate heads for policy and value functions
        self.policy_net = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU()
        )

        self.value_net = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU()
        )

        # Save output dimensions for creating distributions
        self.latent_dim_pi = 64
        self.latent_dim_vf = 64

    def forward(self, features: Tuple[th.Tensor, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        time_series_data, normal_features = features

        # Process time series data
        ts_features = self.time_series_net(time_series_data)

        # Process normal features
        nf_features = self.normal_features_net(normal_features)

        # Combine the processed features
        combined_features = th.cat((ts_features, nf_features), dim=1)
        combined_features = self.combined_net(combined_features)

        # Compute policy and value features
        policy_features = self.policy_net(combined_features)
        value_features = self.value_net(combined_features)

        return policy_features, value_features

class CustomTradingPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        # Create the custom network with appropriate input dimensions
        feature_dim_ts = self.observation_space['env_states'].shape[1]
        feature_dim_nf = self.observation_space['agent_data'].shape[0]
        self.mlp_extractor = CustomTradingNetwork(feature_dim_ts, feature_dim_nf)
