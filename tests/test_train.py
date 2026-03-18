"""Tests for the PPO training pipeline (model and build helpers).

These tests do *not* require a live Bitburner WebSocket server.  They
exercise:

* Neural-network output shapes and dtype.
* Illegal-move masking in the actor network.
* ``build_network`` returns correctly-typed TorchRL modules.
* ``TrainConfig`` default values and field types.
* Checkpoint loading (``load_checkpoint`` field and ``--load-checkpoint`` CLI).
"""

import sys
from pathlib import Path

import pytest
import torch

from src.train.model import GoActorNet, GoCNN, GoValueNet
from src.train.train import TrainConfig, _parse_args, build_network

BOARD_SIZE = 5  # Use a small board to keep tests fast
BATCH = 4


def _make_obs(
    batch: int = BATCH,
    board_size: int = BOARD_SIZE,
    all_legal: bool = True,
) -> torch.Tensor:
    """Return a random 4-channel observation tensor.

    Args:
        batch: Batch size.
        board_size: Side length of the board.
        all_legal: If True, channel 3 (legal mask) is all ones.

    Returns:
        Float32 tensor of shape ``(batch, 4, board_size, board_size)``.
    """
    obs = torch.zeros(batch, 4, board_size, board_size)
    obs[:, 2, :, :] = 1.0  # current player = black
    if all_legal:
        obs[:, 3, :, :] = 1.0  # all moves legal
    return obs


# ---------------------------------------------------------------------------
# GoCNN
# ---------------------------------------------------------------------------


class TestGoCNN:
    """Tests for the convolutional backbone."""

    def test_output_shape(self) -> None:
        """GoCNN must return features of shape (batch, n_fc)."""
        model = GoCNN(board_size=BOARD_SIZE, n_fc=128)
        obs = _make_obs()
        out = model(obs)
        assert out.shape == (BATCH, 128)

    def test_output_dtype(self) -> None:
        """GoCNN output must be float32."""
        model = GoCNN(board_size=BOARD_SIZE)
        assert model(_make_obs()).dtype == torch.float32

    def test_custom_filters_and_layers(self) -> None:
        """GoCNN must accept custom filter counts and layer depth."""
        model = GoCNN(
            board_size=BOARD_SIZE,
            n_filters=32,
            n_cnn_layers=2,
            n_fc=64,
        )
        out = model(_make_obs())
        assert out.shape == (BATCH, 64)

    def test_batch_size_one(self) -> None:
        """GoCNN must handle batch size of 1."""
        model = GoCNN(board_size=BOARD_SIZE)
        out = model(_make_obs(batch=1))
        assert out.shape[0] == 1

    def test_unbatched_input(self) -> None:
        """GoCNN must accept a single unbatched (3-D) observation."""
        model = GoCNN(board_size=BOARD_SIZE, n_fc=128)
        obs = _make_obs(batch=1).squeeze(0)  # (4, B, B) — no batch dim
        assert obs.dim() == 3
        out = model(obs)
        assert out.shape == (128,)


# ---------------------------------------------------------------------------
# GoActorNet
# ---------------------------------------------------------------------------


class TestGoActorNet:
    """Tests for the actor (policy) network."""

    def test_output_shape(self) -> None:
        """GoActorNet must return logits of shape (batch, n_actions)."""
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        model = GoActorNet(board_size=BOARD_SIZE)
        logits = model(_make_obs())
        assert logits.shape == (BATCH, n_actions)

    def test_output_dtype(self) -> None:
        """GoActorNet output must be float32."""
        model = GoActorNet(board_size=BOARD_SIZE)
        assert model(_make_obs()).dtype == torch.float32

    def test_illegal_moves_masked(self) -> None:
        """Logits for illegal positions must equal -1e9."""
        model = GoActorNet(board_size=BOARD_SIZE)
        # Build an obs where only position (0,0) is legal on the board.
        obs = _make_obs(all_legal=False)  # channel 3 all zeros
        obs[:, 3, 0, 0] = 1.0  # only (row=0, col=0) is legal
        logits = model(obs)
        # Action 0 corresponds to (row=0, col=0) and should NOT be masked.
        assert (logits[:, 0] != -1e9).all()
        # Action 1 corresponds to (row=0, col=1) and should be masked.
        assert (logits[:, 1] == -1e9).all()
        # PASS (last action) is always legal.
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        assert (logits[:, n_actions - 1] != -1e9).all()

    def test_pass_always_legal(self) -> None:
        """PASS action must never be masked regardless of channel 3."""
        model = GoActorNet(board_size=BOARD_SIZE)
        obs = _make_obs(all_legal=False)  # all board moves illegal
        logits = model(obs)
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        assert (logits[:, n_actions - 1] != -1e9).all()

    def test_all_legal_no_mask(self) -> None:
        """When all moves are legal, no logit should be masked."""
        model = GoActorNet(board_size=BOARD_SIZE)
        logits = model(_make_obs(all_legal=True))
        assert (logits != -1e9).all()

    def test_gradient_flows(self) -> None:
        """Gradients must flow through the actor for backpropagation."""
        model = GoActorNet(board_size=BOARD_SIZE)
        obs = _make_obs(all_legal=True)
        logits = model(obs)
        logits.sum().backward()
        # At least one parameter should have a gradient.
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )

    def test_unbatched_input(self) -> None:
        """GoActorNet must accept a single unbatched (3-D) observation.

        TorchRL's SyncDataCollector calls the policy with an unbatched
        observation during initialisation.  The network must not raise
        a ValueError from BatchNorm2d.
        """
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        model = GoActorNet(board_size=BOARD_SIZE)
        obs = _make_obs(batch=1, all_legal=True).squeeze(0)  # (4, B, B)
        assert obs.dim() == 3
        logits = model(obs)
        assert logits.shape == (n_actions,)


# ---------------------------------------------------------------------------
# GoValueNet
# ---------------------------------------------------------------------------


class TestGoValueNet:
    """Tests for the value (critic) network."""

    def test_output_shape(self) -> None:
        """GoValueNet must return values of shape (batch, 1)."""
        model = GoValueNet(board_size=BOARD_SIZE)
        val = model(_make_obs())
        assert val.shape == (BATCH, 1)

    def test_output_dtype(self) -> None:
        """GoValueNet output must be float32."""
        model = GoValueNet(board_size=BOARD_SIZE)
        assert model(_make_obs()).dtype == torch.float32

    def test_batch_size_one(self) -> None:
        """GoValueNet must handle batch size of 1."""
        model = GoValueNet(board_size=BOARD_SIZE)
        assert model(_make_obs(batch=1)).shape == (1, 1)

    def test_gradient_flows(self) -> None:
        """Gradients must flow through the critic for backpropagation."""
        model = GoValueNet(board_size=BOARD_SIZE)
        obs = _make_obs()
        val = model(obs)
        val.sum().backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )

    def test_unbatched_input(self) -> None:
        """GoValueNet must accept a single unbatched (3-D) observation.

        TorchRL's SyncDataCollector calls the critic with an unbatched
        observation during initialisation.  The network must not raise
        a ValueError from BatchNorm2d.
        """
        model = GoValueNet(board_size=BOARD_SIZE)
        obs = _make_obs(batch=1).squeeze(0)  # (4, B, B)
        assert obs.dim() == 3
        val = model(obs)
        assert val.shape == (1,)


# ---------------------------------------------------------------------------
# build_network
# ---------------------------------------------------------------------------


class TestBuildNetwork:
    """Tests for the build_network factory."""

    def _make_cfg(self) -> TrainConfig:
        """Return a TrainConfig suitable for fast testing."""
        return TrainConfig(
            board_size=BOARD_SIZE,
            n_filters=16,
            n_cnn_layers=1,
            n_fc=32,
        )

    def test_returns_two_modules(self) -> None:
        """build_network must return (actor, critic) tuple."""
        cfg = self._make_cfg()
        actor, critic = build_network(cfg, torch.device("cpu"))
        assert actor is not None
        assert critic is not None

    def test_actor_is_probabilistic_actor(self) -> None:
        """build_network must return a ProbabilisticActor."""
        from torchrl.modules import ProbabilisticActor

        cfg = self._make_cfg()
        actor, _ = build_network(cfg, torch.device("cpu"))
        assert isinstance(actor, ProbabilisticActor)

    def test_critic_is_value_operator(self) -> None:
        """build_network must return a ValueOperator."""
        from torchrl.modules import ValueOperator

        cfg = self._make_cfg()
        _, critic = build_network(cfg, torch.device("cpu"))
        assert isinstance(critic, ValueOperator)

    def test_actor_forward(self) -> None:
        """Actor must produce action and sample_log_prob from observation."""
        from tensordict import TensorDict

        cfg = self._make_cfg()
        actor, _ = build_network(cfg, torch.device("cpu"))
        obs = _make_obs(board_size=BOARD_SIZE)
        td = TensorDict({"observation": obs}, batch_size=[BATCH])
        out = actor(td)
        assert "action" in out
        # ProbabilisticActor stores log_prob as "action_log_prob".
        assert "action_log_prob" in out
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        assert out["action"].shape == (BATCH,)
        assert out["action"].max().item() < n_actions

    def test_critic_forward(self) -> None:
        """Critic must produce state_value from observation."""
        from tensordict import TensorDict

        cfg = self._make_cfg()
        _, critic = build_network(cfg, torch.device("cpu"))
        obs = _make_obs(board_size=BOARD_SIZE)
        td = TensorDict({"observation": obs}, batch_size=[BATCH])
        out = critic(td)
        assert "state_value" in out
        assert out["state_value"].shape == (BATCH, 1)


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------


class TestTrainConfig:
    """Tests for TrainConfig defaults and types."""

    def test_default_board_size(self) -> None:
        """Default board size must be 9."""
        assert TrainConfig().board_size == 9

    def test_default_websocket_uri(self) -> None:
        """Default WebSocket URI must point to localhost:8765."""
        assert "8765" in TrainConfig().websocket_uri

    def test_custom_fields(self) -> None:
        """TrainConfig must accept and store custom values."""
        cfg = TrainConfig(board_size=13, n_filters=128, lr=1e-3)
        assert cfg.board_size == 13
        assert cfg.n_filters == 128
        assert cfg.lr == 1e-3

    def test_frames_per_batch_positive(self) -> None:
        """frames_per_batch must be positive in default config."""
        assert TrainConfig().frames_per_batch > 0

    def test_total_frames_positive(self) -> None:
        """total_frames must be positive in default config."""
        assert TrainConfig().total_frames > 0

    def test_default_load_checkpoint_is_none(self) -> None:
        """load_checkpoint must default to None."""
        assert TrainConfig().load_checkpoint is None

    def test_custom_load_checkpoint(self) -> None:
        """TrainConfig must accept a custom load_checkpoint path."""
        cfg = TrainConfig(load_checkpoint="checkpoints/checkpoint_00010.pt")
        assert cfg.load_checkpoint == "checkpoints/checkpoint_00010.pt"


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


class TestCheckpointLoading:
    """Tests for the checkpoint save/load round-trip."""

    def _make_cfg(self) -> TrainConfig:
        """Return a TrainConfig suitable for fast testing."""
        return TrainConfig(
            board_size=BOARD_SIZE,
            n_filters=16,
            n_cnn_layers=1,
            n_fc=32,
        )

    def test_load_checkpoint_restores_actor_weights(self, tmp_path: Path) -> None:
        """Actor weights loaded from a checkpoint must match the saved ones."""
        cfg = self._make_cfg()
        device = torch.device("cpu")
        actor, critic = build_network(cfg, device)

        # Modify actor weights so they differ from freshly-initialised ones.
        with torch.no_grad():
            for p in actor.parameters():
                p.fill_(0.42)

        # Save checkpoint.
        ckpt_path = tmp_path / "checkpoint_test.pt"
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()))
        torch.save(
            {
                "iter": 5,
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg,
            },
            ckpt_path,
        )

        # Build fresh networks and load the checkpoint.
        actor2, critic2 = build_network(cfg, device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        actor2.load_state_dict(ckpt["actor_state_dict"])
        critic2.load_state_dict(ckpt["critic_state_dict"])

        # All actor parameters must match.
        for p_orig, p_loaded in zip(actor.parameters(), actor2.parameters()):
            assert torch.allclose(p_orig, p_loaded)

    def test_load_checkpoint_restores_critic_weights(self, tmp_path: Path) -> None:
        """Critic weights loaded from a checkpoint must match the saved ones."""
        cfg = self._make_cfg()
        device = torch.device("cpu")
        actor, critic = build_network(cfg, device)

        with torch.no_grad():
            for p in critic.parameters():
                p.fill_(1.23)

        ckpt_path = tmp_path / "checkpoint_critic.pt"
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()))
        torch.save(
            {
                "iter": 10,
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg,
            },
            ckpt_path,
        )

        actor2, critic2 = build_network(cfg, device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        actor2.load_state_dict(ckpt["actor_state_dict"])
        critic2.load_state_dict(ckpt["critic_state_dict"])

        for p_orig, p_loaded in zip(critic.parameters(), critic2.parameters()):
            assert torch.allclose(p_orig, p_loaded)

    def test_load_checkpoint_iter_is_stored(self, tmp_path: Path) -> None:
        """The checkpoint must store and restore the iteration counter."""
        cfg = self._make_cfg()
        device = torch.device("cpu")
        actor, critic = build_network(cfg, device)
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()))

        ckpt_path = tmp_path / "checkpoint_iter.pt"
        torch.save(
            {
                "iter": 42,
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg,
            },
            ckpt_path,
        )

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        assert ckpt["iter"] == 42


# ---------------------------------------------------------------------------
# CLI --load-checkpoint argument
# ---------------------------------------------------------------------------


class TestParseArgsLoadCheckpoint:
    """Tests for the ``--load-checkpoint`` CLI argument."""

    def test_default_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--load-checkpoint`` must default to None when not provided."""
        monkeypatch.setattr(sys, "argv", ["train"])
        cfg = _parse_args()
        assert cfg.load_checkpoint is None

    def test_load_checkpoint_arg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``--load-checkpoint`` must set load_checkpoint on the config."""
        ckpt = str(tmp_path / "ckpt.pt")
        monkeypatch.setattr(sys, "argv", ["train", "--load-checkpoint", ckpt])
        cfg = _parse_args()
        assert cfg.load_checkpoint == ckpt
