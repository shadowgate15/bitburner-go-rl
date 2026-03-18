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

Curriculum training adds periodic evaluation phases that feed metrics
into :class:`~src.curriculum.curriculum.GoCurriculumManager`, which
dynamically adjusts opponent difficulty and board size (see
:func:`train_with_curriculum`).

Usage (CLI)::

    python -m src.train.train --board-size 9 --total-frames 200000

All hyper-parameters are collected in :class:`TrainConfig` and can be
overridden from the command line (see ``--help``).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import (
    SamplerWithoutReplacement,
)
from torchrl.envs.utils import step_mdp
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from src.curriculum.curriculum import BOARD_SIZES, GoCurriculumManager
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
        load_checkpoint: Path to a checkpoint file to resume training from.
            When set, actor, critic, and optimizer states are restored and
            the iteration counter resumes from the checkpoint's saved value.
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
    load_checkpoint: str | None = None

    # Internal: populated by build_network() / train()
    _extra: dict = field(default_factory=dict, repr=False)


@dataclass
class CurriculumTrainConfig(TrainConfig):
    """Training configuration with curriculum-learning settings.

    Extends :class:`TrainConfig` with parameters that control how
    often evaluation phases are run and how many episodes are used for
    each evaluation.

    Attributes:
        eval_interval: Run an evaluation phase every N training
            iterations (PPO updates).  After evaluation,
            :class:`~src.curriculum.curriculum.GoCurriculumManager`
            decides whether to advance or retreat the difficulty.
        eval_episodes: Number of deterministic episodes to run during
            each evaluation phase.  More episodes give a more reliable
            win-rate estimate at the cost of more environment
            interactions.
        curriculum: Pre-constructed
            :class:`~src.curriculum.curriculum.GoCurriculumManager`
            instance.  When ``None``, a manager with default settings
            is created automatically at the start of training.
    """

    # ---- Curriculum ----
    eval_interval: int = 100
    eval_episodes: int = 20
    curriculum: GoCurriculumManager | None = field(default=None, repr=False)


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
    assert n_actions == cfg.board_size**2 + 1

    return actor, critic


# ---------------------------------------------------------------------------
# Curriculum evaluation helper
# ---------------------------------------------------------------------------


def run_evaluation_episodes(
    actor: ProbabilisticActor,
    eval_env: TorchRLGoEnv,
    n_episodes: int,
) -> dict[str, float]:
    """Run evaluation episodes with a deterministic (greedy) policy.

    The actor is switched to ``eval()`` mode so that BatchNorm uses its
    running statistics and no stochastic noise is applied.  Greedy
    action selection is approximated by argmax over the masked logits
    produced by the actor's underlying
    :class:`~src.train.model.GoActorNet`.

    The logit module is accessed via ``actor.module`` (the
    ``TensorDictModule`` wrapping ``GoActorNet``), which is the
    documented public attribute of ``ProbabilisticActor``.

    Win detection: a positive cumulative episode reward is counted as a
    win.  The Bitburner IPvGO server returns a positive terminal reward
    when the agent (black) wins and a negative reward when it loses.

    Args:
        actor: The :class:`~torchrl.modules.ProbabilisticActor` that
            wraps the policy network.
        eval_env: A fully constructed
            :class:`~src.env.go_env.TorchRLGoEnv` instance with the
            desired opponent and board size already set via
            ``eval_env.opponent`` / ``eval_env.board_size``.
        n_episodes: Number of complete episodes to play.  Must be > 0.

    Returns:
        Dictionary with keys:

        * ``"win_rate"`` - fraction of episodes won (float in [0, 1]).
        * ``"avg_reward"`` - mean cumulative reward across episodes.
        * ``"game_length"`` - mean number of steps per episode.

    Raises:
        ValueError: If *n_episodes* is not positive.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")

    # actor.module is the TensorDictModule(GoActorNet) that was passed
    # to ProbabilisticActor.  This is a documented attribute of
    # TensorDictModuleBase (the parent class of ProbabilisticActor).
    # Calling it with a TensorDict containing "observation" populates
    # "logits" - the masked log-unnormalized policy scores.
    logit_module = actor.module  # TensorDictModule

    actor.eval()
    wins = 0
    total_rewards: list[float] = []
    game_lengths: list[int] = []

    with torch.no_grad():
        for _ in range(n_episodes):
            # Reset with current curriculum settings already on env.
            td = eval_env.reset()
            episode_reward = 0.0
            steps = 0
            done = False

            while not done:
                # Greedy action: forward through logit module, argmax.
                logit_td = TensorDict(
                    {"observation": td["observation"].unsqueeze(0)},
                    batch_size=[1],
                )
                logit_td = logit_module(logit_td)
                action = int(logit_td["logits"].argmax(dim=-1).item())
                td["action"] = torch.tensor(action, dtype=torch.int64)

                # Step the environment.
                td = eval_env.step(td)
                done = bool(td["next", "done"].item())
                episode_reward += float(td["next", "reward"].item())
                steps += 1

                # Prepare TensorDict for the next step.
                td = step_mdp(td)

            # Positive terminal reward means the agent (black) won.
            if episode_reward > 0:
                wins += 1
            total_rewards.append(episode_reward)
            game_lengths.append(steps)

    actor.train()

    return {
        "win_rate": wins / n_episodes,
        "avg_reward": sum(total_rewards) / n_episodes,
        "game_length": sum(game_lengths) / n_episodes,
    }


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
    # 3. Policy for data collection (actor only).
    #    Only the actor runs during rollouts to produce action &
    #    sample_log_prob.  GAE calls the critic internally on both the
    #    current and next observations, so pre-computing state_value
    #    here would cause a TensorDict key mismatch when stacking the
    #    two steps (current has state_value; next does not).
    # ------------------------------------------------------------------
    collection_policy = actor

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
    # 7. Load checkpoint (optional)
    # ------------------------------------------------------------------
    start_iter = 0
    if cfg.load_checkpoint is not None:
        ckpt_path = Path(cfg.load_checkpoint)
        print(f"[train] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt["actor_state_dict"])
        critic.load_state_dict(ckpt["critic_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iter = ckpt["iter"]
        print(f"[train] Resumed from iteration {start_iter}.")

    # ------------------------------------------------------------------
    # 8. Data collector
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
    # 9. Replay buffer (on-policy: cleared each iteration)
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
    # 10. Checkpointing setup
    # ------------------------------------------------------------------
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 11. Training loop
    # ------------------------------------------------------------------
    total_iters = cfg.total_frames // cfg.frames_per_batch
    iter_idx = start_iter
    t0 = time.time()

    print(
        f"[train] Starting training: {total_iters} iterations, "
        f"{cfg.total_frames:,} total frames"
    )

    for data in collector:
        # data shape: (frames_per_batch,) TensorDict
        # Contains: observation, action, action_log_prob,
        #           next.{observation, reward, done, terminated}
        # Note: state_value is NOT pre-computed here; GAE adds it.

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
                nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm)
                optimizer.step()

                epoch_losses.append(total_loss.item())

        # ----------------------------------------------------------
        # Logging
        # ----------------------------------------------------------
        if iter_idx % cfg.log_interval == 0:
            elapsed = time.time() - t0
            mean_reward = data["next", "reward"].mean().item()
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
# Curriculum training loop
# ---------------------------------------------------------------------------


def train_with_curriculum(
    cfg: CurriculumTrainConfig | None = None,
) -> None:
    """Run a PPO training loop with integrated curriculum scheduling.

    This function extends :func:`train` with periodic evaluation phases.
    Every ``cfg.eval_interval`` iterations the training collector is
    paused, ``cfg.eval_episodes`` episodes are played with a greedy
    (deterministic) policy, and the resulting metrics are fed to the
    :class:`~src.curriculum.curriculum.GoCurriculumManager`.

    Curriculum progression:

    * **Advance**: when the smoothed win rate exceeds the upper
      threshold the opponent difficulty increases; if already at the
      hardest opponent, the board size increases.
    * **Retreat**: when the smoothed win rate falls below the lower
      threshold the opponent difficulty decreases; if already at the
      easiest opponent, the board size decreases.
    * **No change**: when the win rate is between the two thresholds.

    Board-size changes require the actor, critic, loss module,
    optimizer, advantage module, replay buffer, and
    ``SyncDataCollector`` to be **fully rebuilt** because the CNN's
    FC layer is fixed to ``n_filters * board_size**2`` inputs at
    construction time and cannot handle a different board size.
    Opponent-only changes just update ``train_env.opponent`` so the
    next automatic episode reset picks up the new setting.

    When no explicit ``cfg.curriculum`` is provided, the curriculum
    manager is automatically initialised with a starting board-size
    index that matches ``cfg.board_size``.  This ensures the initial
    training environment is compatible with the network.

    Args:
        cfg: Curriculum training configuration.  Defaults to
            ``CurriculumTrainConfig()`` (all default hyper-parameters).
    """
    if cfg is None:
        cfg = CurriculumTrainConfig()

    # Initialise curriculum manager.
    # When no explicit curriculum is supplied, auto-select the starting
    # board-size index so it matches cfg.board_size.  This is critical:
    # the network is built once for cfg.board_size (the FC layer is
    # fixed to n_filters * board_size**2 inputs), so the initial
    # training environment MUST use the same board size, otherwise the
    # SyncDataCollector immediately crashes with a shape mismatch.
    if cfg.curriculum is not None:
        curriculum = cfg.curriculum
    else:
        try:
            initial_board_size_idx = BOARD_SIZES.index(cfg.board_size)
        except ValueError:
            # cfg.board_size is not in BOARD_SIZES; pick the nearest.
            initial_board_size_idx = min(
                range(len(BOARD_SIZES)),
                key=lambda i: abs(BOARD_SIZES[i] - cfg.board_size),
            )
        curriculum = GoCurriculumManager(
            initial_board_size_idx=initial_board_size_idx
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_curriculum] Using device: {device}")

    # ------------------------------------------------------------------
    # Build networks
    # ------------------------------------------------------------------
    actor, critic = build_network(cfg, device)
    print("[train_curriculum] Networks built.")

    # ------------------------------------------------------------------
    # Load checkpoint (optional)
    # ------------------------------------------------------------------
    start_iter = 0
    if cfg.load_checkpoint is not None:
        ckpt_path = Path(cfg.load_checkpoint)
        print(f"[train_curriculum] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt["actor_state_dict"])
        critic.load_state_dict(ckpt["critic_state_dict"])
        start_iter = ckpt["iter"]
        print(f"[train_curriculum] Resumed from iteration {start_iter}.")

    # ------------------------------------------------------------------
    # Advantage estimator and PPO loss
    # ------------------------------------------------------------------
    advantage_module = GAE(
        gamma=cfg.gamma,
        lmbda=cfg.lmbda,
        value_network=critic,
        average_gae=False,
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.clip_epsilon,
        entropy_bonus=True,
        entropy_coeff=cfg.entropy_coeff,
        critic_coeff=cfg.critic_coeff,
        normalize_advantage=False,
    )

    all_params = list(loss_module.parameters())
    optimizer = torch.optim.Adam(all_params, lr=cfg.lr)

    if cfg.load_checkpoint is not None:
        ckpt = torch.load(
            Path(cfg.load_checkpoint),
            map_location=device,
            weights_only=False,
        )
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # ------------------------------------------------------------------
    # Initialise training environment and collector
    # ------------------------------------------------------------------
    # Keep a direct reference to the training env so we can update
    # env.opponent between iterations without rebuilding the collector.
    def _build_collector(
        train_env: TorchRLGoEnv,
    ) -> SyncDataCollector:
        """Create a SyncDataCollector bound to *train_env*."""
        return SyncDataCollector(
            create_env_fn=train_env,
            policy=actor,
            frames_per_batch=cfg.frames_per_batch,
            total_frames=cfg.total_frames,
            device=device,
        )

    init_cfg = curriculum.get_current_config()
    train_env = TorchRLGoEnv(
        board_size=int(init_cfg["board_size"]),
        websocket_uri=cfg.websocket_uri,
        opponent=str(init_cfg["opponent"]),
    )
    collector = _build_collector(train_env)

    # Separate environment used exclusively for evaluation episodes.
    # It is created fresh so it doesn't interfere with the collector.
    eval_env = TorchRLGoEnv(
        board_size=int(init_cfg["board_size"]),
        websocket_uri=cfg.websocket_uri,
        opponent=str(init_cfg["opponent"]),
    )

    # ------------------------------------------------------------------
    # Replay buffer (on-policy: cleared each iteration)
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
    # Checkpointing setup
    # ------------------------------------------------------------------
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    total_iters = cfg.total_frames // cfg.frames_per_batch
    iter_idx = start_iter
    t0 = time.time()

    print(
        f"[train_curriculum] Starting: {total_iters} iterations, "
        f"{cfg.total_frames:,} total frames\n"
        f"  initial opponent  = {curriculum.current_opponent}\n"
        f"  initial board_size = {curriculum.current_board_size}"
    )

    for data in collector:
        iter_idx += 1

        # --------------------------------------------------------------
        # GAE + advantage normalisation
        # --------------------------------------------------------------
        with torch.no_grad():
            advantage_module(data)
        adv = data["advantage"]
        data["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        # --------------------------------------------------------------
        # Replay buffer + PPO update
        # --------------------------------------------------------------
        replay_buffer.empty()
        replay_buffer.extend(data)
        epoch_losses: list[float] = []

        for _epoch in range(cfg.n_epochs):
            for batch in replay_buffer:
                batch = batch.to(device)
                loss_dict = loss_module(batch)
                total_loss = (
                    loss_dict["loss_objective"]
                    + loss_dict["loss_critic"]
                    + loss_dict["loss_entropy"]
                )
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm)
                optimizer.step()
                epoch_losses.append(total_loss.item())

        # --------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------
        if iter_idx % cfg.log_interval == 0:
            elapsed = time.time() - t0
            mean_reward = data["next", "reward"].mean().item()
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
                f"loss={mean_loss:.4f} "
                f"opponent={curriculum.current_opponent!r} "
                f"board={curriculum.current_board_size}"
            )

        # --------------------------------------------------------------
        # Curriculum evaluation phase (every eval_interval iterations)
        # --------------------------------------------------------------
        if iter_idx % cfg.eval_interval == 0:
            print(
                f"[train_curriculum] Evaluation at iter {iter_idx} "
                f"({cfg.eval_episodes} episodes) ..."
            )
            # Sync eval env to current curriculum settings.
            eval_env.opponent = curriculum.current_opponent
            if eval_env.board_size != curriculum.current_board_size:
                eval_env.board_size = curriculum.current_board_size
                eval_env.rebuild_specs()

            metrics = run_evaluation_episodes(
                actor=actor,
                eval_env=eval_env,
                n_episodes=cfg.eval_episodes,
            )
            print(
                f"[train_curriculum]   win_rate={metrics['win_rate']:.3f}"
                f"  avg_reward={metrics['avg_reward']:.4f}"
                f"  game_length={metrics['game_length']:.1f}"
            )

            # Let curriculum decide the next difficulty level.
            prev_board_size = curriculum.current_board_size
            curriculum.update(metrics)
            new_cfg = curriculum.get_current_config()

            # Apply curriculum changes to the training environment.
            train_env.opponent = str(new_cfg["opponent"])
            new_board_size = int(new_cfg["board_size"])

            if new_board_size != prev_board_size:
                # Board-size change: the CNN's FC layer is fixed to
                # n_filters * board_size**2 inputs, so the actor and
                # critic cannot process observations from a different
                # board size.  Rebuild every size-dependent component:
                # network, loss, optimizer, advantage module, replay
                # buffer, environment, and collector.
                print(
                    f"[train_curriculum] Board size changed "
                    f"{prev_board_size} → {new_board_size}. "
                    "Rebuilding networks and collector ..."
                )
                collector.shutdown()

                # New networks for the new board size.
                size_cfg = replace(cfg, board_size=new_board_size)
                actor, critic = build_network(size_cfg, device)
                advantage_module = GAE(
                    gamma=cfg.gamma,
                    lmbda=cfg.lmbda,
                    value_network=critic,
                    average_gae=False,
                )
                loss_module = ClipPPOLoss(
                    actor_network=actor,
                    critic_network=critic,
                    clip_epsilon=cfg.clip_epsilon,
                    entropy_bonus=True,
                    entropy_coeff=cfg.entropy_coeff,
                    critic_coeff=cfg.critic_coeff,
                    normalize_advantage=False,
                )
                all_params = list(loss_module.parameters())
                optimizer = torch.optim.Adam(all_params, lr=cfg.lr)

                # Fresh replay buffer (old one has wrong obs shape).
                replay_buffer = ReplayBuffer(
                    storage=LazyTensorStorage(
                        max_size=cfg.frames_per_batch,
                        device=device,
                    ),
                    sampler=SamplerWithoutReplacement(),
                    batch_size=cfg.minibatch_size,
                )

                # New env + collector.
                # _build_collector() reads `actor` from the closure,
                # so it will use the freshly built network above.
                train_env = TorchRLGoEnv(
                    board_size=new_board_size,
                    websocket_uri=cfg.websocket_uri,
                    opponent=str(new_cfg["opponent"]),
                )
                collector = _build_collector(train_env)

        # --------------------------------------------------------------
        # Checkpointing
        # --------------------------------------------------------------
        if iter_idx % cfg.save_interval == 0:
            ckpt_path = ckpt_dir / f"checkpoint_{iter_idx:05d}.pt"
            torch.save(
                {
                    "iter": iter_idx,
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg,
                    "curriculum_opponent_idx": (curriculum.opponent_idx),
                    "curriculum_board_size_idx": (curriculum.board_size_idx),
                },
                ckpt_path,
            )
            print(f"[train_curriculum] Checkpoint saved → {ckpt_path}")

    # Final checkpoint
    final_path = ckpt_dir / "checkpoint_final.pt"
    torch.save(
        {
            "iter": iter_idx,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": cfg,
            "curriculum_opponent_idx": curriculum.opponent_idx,
            "curriculum_board_size_idx": curriculum.board_size_idx,
        },
        final_path,
    )
    print(
        f"[train_curriculum] Training complete. "
        f"Final checkpoint → {final_path}"
    )

    collector.shutdown()


def _train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a PPO Go agent against the Bitburner IPvGO engine."
    )

    # Environment
    parser.add_argument(
        "--board-size",
        type=int,
        default=9,
        help="Side length of the Go board (default: 9).",
    )
    parser.add_argument(
        "--websocket-uri",
        type=str,
        default="ws://localhost:8765",
        help="WebSocket URI of the Bitburner IPvGO server.",
    )

    # Network
    parser.add_argument(
        "--n-filters",
        type=int,
        default=64,
        help="CNN filter count per layer.",
    )
    parser.add_argument(
        "--n-cnn-layers",
        type=int,
        default=3,
        help="Number of convolutional layers.",
    )
    parser.add_argument(
        "--n-fc",
        type=int,
        default=256,
        help="Fully-connected feature size.",
    )

    # PPO
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
        help="PPO clipping parameter ε.",
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient.",
    )
    parser.add_argument(
        "--critic-coeff",
        type=float,
        default=0.5,
        help="Critic loss coefficient.",
    )

    # GAE
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor gamma.",
    )
    parser.add_argument(
        "--lmbda",
        type=float,
        default=0.95,
        help="GAE λ parameter.",
    )

    # Data collection
    parser.add_argument(
        "--frames-per-batch",
        type=int,
        default=256,
        help="Rollout steps per PPO iteration.",
    )
    parser.add_argument(
        "--total-frames",
        type=int,
        default=100_000,
        help="Total environment steps to train for.",
    )

    # PPO updates
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=4,
        help="PPO update epochs per iteration.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=64,
        help="Mini-batch size for PPO updates.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Gradient clipping norm.",
    )

    # Logging / checkpointing
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Log every N iterations.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save checkpoint every N iterations.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for model checkpoints.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint file (.pt) to resume training from. "
            "Restores actor, critic, and optimizer state."
        ),
    )

    return parser


def _parse_args() -> TrainConfig:
    """Parse command-line arguments into a :class:`TrainConfig`.

    Returns:
        Populated :class:`TrainConfig` with any CLI overrides applied.
    """
    args = _train_parser().parse_args()

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
        load_checkpoint=args.load_checkpoint,
    )


def _parse_args_curriculum() -> CurriculumTrainConfig:
    """Parse command-line arguments into a :class:`CurriculumTrainConfig`.

    Returns:
        Populated :class:`CurriculumTrainConfig` with any CLI overrides applied.
    """
    parser = _train_parser()

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Run an evaluation phase every N training",
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Number of deterministic episodes to run during",
    )

    args = parser.parse_args()

    return CurriculumTrainConfig(
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
        load_checkpoint=args.load_checkpoint,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
    )


if __name__ == "__main__":  # pragma: no cover
    train(_parse_args())
