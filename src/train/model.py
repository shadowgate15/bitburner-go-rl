"""Neural network models for the Go-playing PPO agent.

Architecture overview
---------------------
The shared CNN backbone processes the 4-channel board observation and
produces a flat feature vector.  Two separate heads are attached to this
backbone:

* **Policy head** (GoActorNet) - outputs action logits, masking illegal
  moves so the agent never samples an invalid placement.
* **Value head** (GoValueNet) - outputs a single scalar state-value
  estimate used by the GAE advantage estimator.
"""

import torch
import torch.nn as nn


class GoCNN(nn.Module):
    """Convolutional backbone for the Go board.

    Processes a 4-channel spatial observation of shape
    ``(batch, 4, board_size, board_size)`` through a stack of
    Conv2d + BatchNorm + ReLU layers followed by a fully-connected
    projection, producing a flat feature vector.

    Args:
        board_size: Side length of the square board.
        in_channels: Number of input channels (default 4).
        n_filters: Number of convolutional filters per layer.
        n_cnn_layers: Number of convolutional layers.
        n_fc: Size of the fully-connected output feature vector.
    """

    def __init__(
        self,
        board_size: int,
        in_channels: int = 4,
        n_filters: int = 64,
        n_cnn_layers: int = 3,
        n_fc: int = 256,
    ) -> None:
        """Initialise the CNN backbone.

        Args:
            board_size: Side length of the board.
            in_channels: Number of input observation channels.
            n_filters: Convolutional filter count per layer.
            n_cnn_layers: Number of residual-style conv blocks.
            n_fc: Output feature-vector size.
        """
        super().__init__()

        self.board_size = board_size
        self.n_filters = n_filters
        self.n_fc = n_fc

        # ---- Convolutional feature extractor ----
        conv_layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(n_cnn_layers):
            conv_layers += [
                nn.Conv2d(
                    current_channels,
                    n_filters,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=True),
            ]
            current_channels = n_filters
        self.conv = nn.Sequential(*conv_layers)

        # ---- Fully-connected projection ----
        flat_dim = n_filters * board_size * board_size
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, n_fc),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract features from a board observation.

        Args:
            obs: Float32 tensor of shape
                ``(batch, in_channels, board_size, board_size)``.

        Returns:
            Feature tensor of shape ``(batch, n_fc)``.
        """
        # obs: (batch, C, B, B)
        x = self.conv(obs)  # (batch, n_filters, B, B)
        x = x.flatten(start_dim=1)  # (batch, n_filters * B * B)
        return self.fc(x)  # (batch, n_fc)


class GoActorNet(nn.Module):
    """Actor (policy) network for the Go agent.

    Produces *masked* action logits: illegal board positions are set to a
    large negative value so that they receive negligible probability under
    a softmax / Categorical distribution.  The PASS action (last index) is
    always considered legal because Go rules allow a pass at any time.

    The legal-move mask is derived directly from channel 3 of the
    observation tensor (see :func:`~src.env.go_env.encode_board`), so no
    additional input is required.

    Args:
        board_size: Side length of the board.
        n_filters: Convolutional filters per layer (passed to GoCNN).
        n_cnn_layers: Number of convolutional layers.
        n_fc: Fully-connected feature size.
    """

    def __init__(
        self,
        board_size: int,
        n_filters: int = 64,
        n_cnn_layers: int = 3,
        n_fc: int = 256,
    ) -> None:
        """Initialise the actor network.

        Args:
            board_size: Side length of the board.
            n_filters: Convolutional filter count.
            n_cnn_layers: Number of convolutional layers.
            n_fc: Fully-connected feature size.
        """
        super().__init__()

        self.board_size = board_size
        self.n_actions = board_size * board_size + 1

        # Shared CNN backbone
        self.cnn = GoCNN(
            board_size=board_size,
            in_channels=4,
            n_filters=n_filters,
            n_cnn_layers=n_cnn_layers,
            n_fc=n_fc,
        )

        # Policy head: features → logits
        self.policy_head = nn.Linear(n_fc, self.n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute masked logits from a board observation.

        Args:
            obs: Float32 tensor of shape
                ``(batch, 4, board_size, board_size)``.

        Returns:
            Logits tensor of shape ``(batch, n_actions)`` with illegal
            moves set to ``-1e9``.
        """
        features = self.cnn(obs)  # (batch, n_fc)
        logits = self.policy_head(features)  # (batch, n_actions)

        # --- Build legal-move mask from channel 3 ---
        # Channel 3: spatial legal-move map (board_size x board_size).
        # Flatten to (batch, board_size^2) and append PASS=True.
        legal_spatial = obs[:, 3, :, :]  # (batch, B, B)
        legal_flat = legal_spatial.reshape(obs.shape[0], -1)  # (batch, B*B)
        # PASS is always legal in Go.
        pass_mask = torch.ones(
            obs.shape[0], 1, dtype=torch.float32, device=obs.device
        )
        # legal_mask: (batch, n_actions)
        legal_mask = torch.cat([legal_flat, pass_mask], dim=-1)

        # Mask illegal positions with a large negative value.
        # Using -1e9 rather than -inf avoids NaN in softmax edge cases.
        logits = logits.masked_fill(legal_mask < 0.5, -1e9)
        return logits


class GoValueNet(nn.Module):
    """Value (critic) network for the Go agent.

    Produces a scalar state-value estimate from the board observation,
    used by the GAE advantage estimator and PPO critic loss.

    Args:
        board_size: Side length of the board.
        n_filters: Convolutional filters per layer.
        n_cnn_layers: Number of convolutional layers.
        n_fc: Fully-connected feature size.
    """

    def __init__(
        self,
        board_size: int,
        n_filters: int = 64,
        n_cnn_layers: int = 3,
        n_fc: int = 256,
    ) -> None:
        """Initialise the value network.

        Args:
            board_size: Side length of the board.
            n_filters: Convolutional filter count.
            n_cnn_layers: Number of convolutional layers.
            n_fc: Fully-connected feature size.
        """
        super().__init__()

        # Shared CNN backbone (separate weights from actor)
        self.cnn = GoCNN(
            board_size=board_size,
            in_channels=4,
            n_filters=n_filters,
            n_cnn_layers=n_cnn_layers,
            n_fc=n_fc,
        )

        # Value head: features → scalar
        self.value_head = nn.Sequential(
            nn.Linear(n_fc, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute a state-value estimate.

        Args:
            obs: Float32 tensor of shape
                ``(batch, 4, board_size, board_size)``.

        Returns:
            Value tensor of shape ``(batch, 1)``.
        """
        features = self.cnn(obs)  # (batch, n_fc)
        return self.value_head(features)  # (batch, 1)


__all__ = ["GoActorNet", "GoCNN", "GoValueNet"]
