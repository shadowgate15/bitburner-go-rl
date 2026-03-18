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
                nn.BatchNorm2d(n_filters, track_running_stats=False),
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
                ``(batch, in_channels, board_size, board_size)`` or
                ``(in_channels, board_size, board_size)`` for a single
                unbatched observation (e.g. during collector
                initialisation).

        Returns:
            Feature tensor of shape ``(batch, n_fc)``, or ``(n_fc,)``
            when the input was unbatched.
        """
        unbatched = obs.dim() == 3
        if unbatched:
            obs = obs.unsqueeze(0)
        # obs: (batch, C, B, B)
        x = self.conv(obs)  # (batch, n_filters, B, B)
        x = x.flatten(start_dim=1)  # (batch, n_filters * B * B)
        features = self.fc(x)  # (batch, n_fc)
        if unbatched:
            features = features.squeeze(0)
        return features


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
                ``(batch, 4, board_size, board_size)`` or
                ``(4, board_size, board_size)`` for a single unbatched
                observation (e.g. during collector initialisation).

        Returns:
            Logits tensor of shape ``(batch, n_actions)`` with illegal
            moves set to ``-1e9``, or ``(n_actions,)`` when the input
            was unbatched.
        """
        unbatched = obs.dim() == 3
        if unbatched:
            obs = obs.unsqueeze(0)

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

        if unbatched:
            logits = logits.squeeze(0)
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
                ``(batch, 4, board_size, board_size)`` or
                ``(4, board_size, board_size)`` for a single unbatched
                observation (e.g. during collector initialisation).

        Returns:
            Value tensor of shape ``(batch, 1)``, or ``(1,)`` when the
            input was unbatched.
        """
        unbatched = obs.dim() == 3
        if unbatched:
            obs = obs.unsqueeze(0)
        features = self.cnn(obs)  # (batch, n_fc)
        value = self.value_head(features)  # (batch, 1)
        if unbatched:
            value = value.squeeze(0)
        return value


def transfer_conv_weights(
    old_actor: GoActorNet,
    new_actor: GoActorNet,
    old_critic: GoValueNet,
    new_critic: GoValueNet,
) -> None:
    """Copy board-size-independent weights from old networks to new ones.

    When the curriculum moves to a different board size the CNN
    backbone and value head weights learned on the previous board size
    can be re-used.  The convolutional layers are purely spatial filters
    whose weight shapes depend only on *in_channels* and *n_filters*
    (not on *board_size*), and the value head only depends on *n_fc*.

    **Components transferred:**

    * ``GoActorNet.cnn.conv`` - conv + BN stack (actor and critic)
    * ``GoValueNet.cnn.conv`` - conv + BN stack
    * ``GoValueNet.value_head`` - value MLP (input = *n_fc*, constant)

    **Components intentionally left fresh** (board-size-dependent):

    * ``GoCNN.fc`` - ``Linear(n_filters * board_size^2, n_fc)``
    * ``GoActorNet.policy_head`` - ``Linear(n_fc, board_size^2 + 1)``

    Args:
        old_actor: Actor network trained on the previous board size.
        new_actor: Freshly initialised actor for the new board size.
        old_critic: Critic network trained on the previous board size.
        new_critic: Freshly initialised critic for the new board size.
    """
    new_actor.cnn.conv.load_state_dict(old_actor.cnn.conv.state_dict())
    new_critic.cnn.conv.load_state_dict(old_critic.cnn.conv.state_dict())
    new_critic.value_head.load_state_dict(
        old_critic.value_head.state_dict()
    )


__all__ = ["GoActorNet", "GoCNN", "GoValueNet", "transfer_conv_weights"]
