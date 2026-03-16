"""PPO training script for the Bitburner IPvGO agent.

Pipeline
--------
1. Build CNN-based actor and critic networks.
2. Wrap networks in TorchRL's ``ProbabilisticActor`` and
   ``ValueOperator`` so they speak the TensorDict protocol.
3. Collect experience with ``SyncDataCollector`` (on-policy rollouts).
4. Compute GAE advantages and value targets.
5. Store the annotated rollouts in a ``ReplayBuffer``.
6. Run *K* epochs of mini-batch PPO updates via ``ClipPPOLoss``.
7. Log statistics and periodically save checkpoints.

Usage (CLI)::

    python -m src.train.train --board-size 9 --total-frames 200000

All hyper-parameters are collected in :class:`TrainConfig` and can be
overridden from the command line (see ``--help``).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import (
    SamplerWithoutReplacement,
)
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from src.env.go_env import TorchRLGoEnv
from src.train.model import GoActorNet, GoValueNet

try:
    from torch.distributions import Categorical
except ImportError:  # pragma: no cover
    from torch.distributions.categorical import Categorical  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """All hyper-parameters for a PPO training run.

    Attributes:
        board_size: Side length of the Go board.
        websocket_uri: URI of the Bitburner IPvGO WebSocket server.
        n_filters: Number of Conv2d filters per layer in the CNN.
        n_cnn_layers: Number of convolutional layers.
        n_fc: Size of the fully-connected feature vector.
        clip_epsilon: PPO clip ratio ``ε``.
        entropy_coeff: Weight for the entropy bonus in the PPO loss.
        critic_coeff: Weight for the critic loss in the PPO loss.
        gamma: Discount factor for GAE.
        lmbda: GAE λ smoothing parameter.
        frames_per_batch: Rollout length per iteration.
        total_frames: Total environment frames to collect.
        n_epochs: PPO update epochs per iteration.
        minibatch_size: Mini-batch size for each PPO gradient step.
        lr: Adam learning rate.
        max_grad_norm: Gradient clipping norm.
        log_interval: Print statistics every N iterations.
        save_interval: Save a checkpoint every N iterations.
        checkpoint_dir: Directory for model checkpoints.
    """

    # ---- Environment ----
    board_size: int = 9
    websocket_uri: str = "ws://localhost:8765"

    # ---- Network ----
    n_filters: int = 64
    n_cnn_layers: int = 3
    n_fc: int = 256

    # ---- PPO loss ----
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    critic_coeff: float = 0.5

    # ---- GAE ----
    gamma: float = 0.99
    lmbda: float = 0.95

    # ---- Data collection ----
    frames_per_batch: int = 256
    total_frames: int = 100_000

    # ---- PPO updates ----
    n_epochs: int = 4
    minibatch_size: int = 64
    lr: float = 3e-4
    max_grad_norm: float = 0.5

    # ---- Logging / checkpointing ----
    log_interval: int = 1
    save_interval: int = 10
    checkpoint_dir: str = "checkpoints"

    # Internal: populated by build_network() / train()
    _extra: dict = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_env(
    board_size: int = 9,
    websocket_uri: str = "ws://localhost:8765",
) -> TorchRLGoEnv:
    """Create a :class:`~src.env.go_env.TorchRLGoEnv` instance.

    This thin factory exists so that ``SyncDataCollector`` can receive a
    *callable* and construct the environment in worker processes.

    Args:
        board_size: Side length of the board.
        websocket_uri: WebSocket URI of the Bitburner server.

    Returns:
        A freshly-constructed :class:`~src.env.go_env.TorchRLGoEnv`.
    """
    return TorchRLGoEnv(
        board_size=board_size,
        websocket_uri=websocket_uri,
    )


def build_network(
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[ProbabilisticActor, ValueOperator]:
    """Build the actor and critic TorchRL modules from *cfg*.

    The actor is a ``ProbabilisticActor`` that:

    1. Runs ``GoActorNet`` on ``"observation"`` → ``"logits"`` (with
       illegal-move masking baked in).
    2. Samples an action from a ``Categorical`` distribution and stores
       ``"action"`` and ``"sample_log_prob"`` in the output TensorDict.

    The critic is a ``ValueOperator`` that:

    1. Runs ``GoValueNet`` on ``"observation"`` → ``"state_value"``.

    Args:
        cfg: Training configuration.
        device: Torch device to place the networks on.

    Returns:
        Tuple of ``(actor, critic)`` TorchRL modules.
    """
    n_actions = cfg.board_size * cfg.board_size + 1

    # --- Actor ---
    # Step 1: TensorDictModule wrapping GoActorNet.
    #   observation → logits
    actor_net = TensorDictModule(
        GoActorNet(
            board_size=cfg.board_size,
            n_filters=cfg.n_filters,
            n_cnn_layers=cfg.n_cnn_layers,
            n_fc=cfg.n_fc,
        ).to(device),
        in_keys=["observation"],
        out_keys=["logits"],
    )

    # Step 2: ProbabilisticActor that samples from the Categorical
    #   distribution defined by the logits.
    #   logits → action, sample_log_prob
    env_for_spec = TorchRLGoEnv(board_size=cfg.board_size)
    actor = ProbabilisticActor(
        module=actor_net,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        distribution_kwargs={},
        return_log_prob=True,
        spec=env_for_spec.action_spec,
    )

    # --- Critic ---
    # ValueOperator wrapping GoValueNet.
    #   observation → state_value
    critic = ValueOperator(
        module=GoValueNet(
            board_size=cfg.board_size,
            n_filters=cfg.n_filters,
            n_cnn_layers=cfg.n_cnn_layers,
            n_fc=cfg.n_fc,
        ).to(device),
        in_keys=["observation"],
        out_keys=["state_value"],
    )

    # Infer action count from the actor (convenience check)
    assert n_actions == cfg.board_size ** 2 + 1

    return actor, critic


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig | None = None) -> None:
    """Run a PPO training loop for the Go agent.

    The function follows these steps each iteration:

    1. A ``SyncDataCollector`` rolls out the current policy for
       ``cfg.frames_per_batch`` environment steps.
    2. GAE computes per-step advantages and value targets.
    3. Advantage normalisation stabilises learning.
    4. A ``ReplayBuffer`` stores the annotated rollout.
    5. ``cfg.n_epochs`` passes of mini-batch PPO gradient updates are
       applied (clip + entropy + critic losses).
    6. Statistics are logged and checkpoints saved according to
       ``cfg.log_interval`` / ``cfg.save_interval``.

    Args:
        cfg: Training configuration.  Defaults to ``TrainConfig()``
            (all default hyper-parameters).
    """
    if cfg is None:
        cfg = TrainConfig()

    # ------------------------------------------------------------------
    # 1. Device selection
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # ------------------------------------------------------------------
    # 2. Build actor and critic networks
    # ------------------------------------------------------------------
    actor, critic = build_network(cfg, device)
    print("[train] Networks built.")
    print(f"  Actor:  {sum(p.numel() for p in actor.parameters()):,} params")
    print(f"  Critic: {sum(p.numel() for p in critic.parameters()):,} params")

    # ------------------------------------------------------------------
    # 3. Combined policy for data collection
    #    The TensorDictSequential runs actor first (adds action &
    #    sample_log_prob), then critic (adds state_value).
    # ------------------------------------------------------------------
    collection_policy = TensorDictSequential(actor, critic)

    # ------------------------------------------------------------------
    # 4. Advantage estimator (GAE)
    # ------------------------------------------------------------------
    # GAE reads "state_value" and produces "advantage" + "value_target".
    advantage_module = GAE(
        gamma=cfg.gamma,
        lmbda=cfg.lmbda,
        value_network=critic,
        average_gae=False,
    )

    # ------------------------------------------------------------------
    # 5. PPO loss module
    # ------------------------------------------------------------------
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.clip_epsilon,
        entropy_bonus=True,
        entropy_coeff=cfg.entropy_coeff,
        critic_coeff=cfg.critic_coeff,
        normalize_advantage=False,  # we normalise manually below
    )

    # ------------------------------------------------------------------
    # 6. Optimizer (all actor + critic parameters)
    # ------------------------------------------------------------------
    all_params = list(loss_module.parameters())
    optimizer = torch.optim.Adam(all_params, lr=cfg.lr)

    # ------------------------------------------------------------------
    # 7. Data collector
    # ------------------------------------------------------------------
    # create_env_fn must be a callable so that SyncDataCollector can
    # instantiate (and optionally recreate) the environment internally.
    board_size = cfg.board_size
    uri = cfg.websocket_uri

    def _make_env() -> TorchRLGoEnv:
        return make_env(board_size=board_size, websocket_uri=uri)

    collector = SyncDataCollector(
        create_env_fn=_make_env,
        policy=collection_policy,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=device,
    )

    # ------------------------------------------------------------------
    # 8. Replay buffer (on-policy: cleared each iteration)
    # ------------------------------------------------------------------
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=cfg.frames_per_batch,
            device=device,
        ),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.minibatch_size,
    )

    # ------------------------------------------------------------------
    # 9. Checkpointing setup
    # ------------------------------------------------------------------
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 10. Training loop
    # ------------------------------------------------------------------
    total_iters = cfg.total_frames // cfg.frames_per_batch
    iter_idx = 0
    t0 = time.time()

    print(f"[train] Starting training: {total_iters} iterations, "
          f"{cfg.total_frames:,} total frames")

    for data in collector:
        # data shape: (frames_per_batch,) TensorDict
        # Contains: observation, action, action_log_prob, state_value,
        #           next.{observation, reward, done, terminated}

        iter_idx += 1

        # ----------------------------------------------------------
        # Compute GAE advantages and value targets (no gradients)
        # ----------------------------------------------------------
        with torch.no_grad():
            advantage_module(data)

        # Normalise advantages for training stability.
        adv = data["advantage"]
        data["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ----------------------------------------------------------
        # Populate the replay buffer with this rollout
        # ----------------------------------------------------------
        replay_buffer.empty()
        replay_buffer.extend(data)

        # ----------------------------------------------------------
        # PPO update: K epochs of mini-batch gradient steps
        # ----------------------------------------------------------
        epoch_losses: list[float] = []

        for _epoch in range(cfg.n_epochs):
            for batch in replay_buffer:
                batch = batch.to(device)

                # Forward pass through loss module.
                # ClipPPOLoss internally re-evaluates the actor to get
                # the *new* log-probs and computes the clip ratio.
                loss_dict = loss_module(batch)
                total_loss = (
                    loss_dict["loss_objective"]
                    + loss_dict["loss_critic"]
                    + loss_dict["loss_entropy"]
                )

                optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping for stable training
                nn.utils.clip_grad_norm_(
                    all_params, cfg.max_grad_norm
                )
                optimizer.step()

                epoch_losses.append(total_loss.item())

        # ----------------------------------------------------------
        # Logging
        # ----------------------------------------------------------
        if iter_idx % cfg.log_interval == 0:
            elapsed = time.time() - t0
            mean_reward = (
                data["next", "reward"].mean().item()
            )
            mean_loss = (
                sum(epoch_losses) / len(epoch_losses)
                if epoch_losses
                else float("nan")
            )
            frames_collected = iter_idx * cfg.frames_per_batch
            fps = frames_collected / elapsed
            print(
                f"[iter {iter_idx:4d}/{total_iters}] "
                f"frames={frames_collected:7,} "
                f"fps={fps:6.0f} "
                f"reward={mean_reward:+.4f} "
                f"loss={mean_loss:.4f}"
            )

        # ----------------------------------------------------------
        # Checkpointing
        # ----------------------------------------------------------
        if iter_idx % cfg.save_interval == 0:
            ckpt_path = ckpt_dir / f"checkpoint_{iter_idx:05d}.pt"
            torch.save(
                {
                    "iter": iter_idx,
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg,
                },
                ckpt_path,
            )
            print(f"[train] Checkpoint saved → {ckpt_path}")

    # Final checkpoint
    final_path = ckpt_dir / "checkpoint_final.pt"
    torch.save(
        {
            "iter": iter_idx,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": cfg,
        },
        final_path,
    )
    print(f"[train] Training complete. Final checkpoint → {final_path}")

    collector.shutdown()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> TrainConfig:
    """Parse command-line arguments into a :class:`TrainConfig`.

    Returns:
        Populated :class:`TrainConfig` with any CLI overrides applied.
    """
    parser = argparse.ArgumentParser(
        description="Train a PPO Go agent against the Bitburner IPvGO engine."
    )

    # Environment
    parser.add_argument(
        "--board-size", type=int, default=9,
        help="Side length of the Go board (default: 9).",
    )
    parser.add_argument(
        "--websocket-uri", type=str, default="ws://localhost:8765",
        help="WebSocket URI of the Bitburner IPvGO server.",
    )

    # Network
    parser.add_argument(
        "--n-filters", type=int, default=64,
        help="CNN filter count per layer.",
    )
    parser.add_argument(
        "--n-cnn-layers", type=int, default=3,
        help="Number of convolutional layers.",
    )
    parser.add_argument(
        "--n-fc", type=int, default=256,
        help="Fully-connected feature size.",
    )

    # PPO
    parser.add_argument(
        "--clip-epsilon", type=float, default=0.2,
        help="PPO clipping parameter ε.",
    )
    parser.add_argument(
        "--entropy-coeff", type=float, default=0.01,
        help="Entropy bonus coefficient.",
    )
    parser.add_argument(
        "--critic-coeff", type=float, default=0.5,
        help="Critic loss coefficient.",
    )

    # GAE
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor gamma.",
    )
    parser.add_argument(
        "--lmbda", type=float, default=0.95,
        help="GAE λ parameter.",
    )

    # Data collection
    parser.add_argument(
        "--frames-per-batch", type=int, default=256,
        help="Rollout steps per PPO iteration.",
    )
    parser.add_argument(
        "--total-frames", type=int, default=100_000,
        help="Total environment steps to train for.",
    )

    # PPO updates
    parser.add_argument(
        "--n-epochs", type=int, default=4,
        help="PPO update epochs per iteration.",
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=64,
        help="Mini-batch size for PPO updates.",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5,
        help="Gradient clipping norm.",
    )

    # Logging / checkpointing
    parser.add_argument(
        "--log-interval", type=int, default=1,
        help="Log every N iterations.",
    )
    parser.add_argument(
        "--save-interval", type=int, default=10,
        help="Save checkpoint every N iterations.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory for model checkpoints.",
    )

    args = parser.parse_args()

    return TrainConfig(
        board_size=args.board_size,
        websocket_uri=args.websocket_uri,
        n_filters=args.n_filters,
        n_cnn_layers=args.n_cnn_layers,
        n_fc=args.n_fc,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        critic_coeff=args.critic_coeff,
        gamma=args.gamma,
        lmbda=args.lmbda,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        n_epochs=args.n_epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    train(_parse_args())
