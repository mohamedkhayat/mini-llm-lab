"""The log backend: one logger interface, two adapters.

``resolve_log_backend`` picks the backend for a run (``training.log_backend``
with alias normalization, rejection of unknown backends, and the
``WANDB_MODE=disabled`` escape hatch). ``create_logger`` maps the resolved
backend to an adapter:

- ``WandbLogger`` — the W&B adapter. It owns the wandb import (isolated from
  the rest of the codebase), the "a resumed run keeps the original run's
  config and name" behavior, the computed-value config update, HTML sample
  logging, and the optional never-fail artifact upload.
- ``TerminalLogger`` — prints only. It holds no reference to W&B, so a
  terminal-mode run performs zero W&B interactions.

The interface covers exactly five responsibilities: run initialization,
metrics logging at a step, sample-text logging, checkpoint/artifact logging
by kind, and finish.
"""

import os
from abc import ABC, abstractmethod

# Aliases accepted for ``training.log_backend``.
_BACKEND_ALIASES = {"term": "terminal", "console": "terminal"}


def resolve_log_backend(training_cfg, wandb_mode=None):
    """Resolve whether metrics should be sent to W&B or the terminal."""
    configured_backend = getattr(training_cfg, "log_backend", "wandb")
    if configured_backend is None:
        configured_backend = "wandb"

    normalized_backend = str(configured_backend).strip().lower()
    log_backend = _BACKEND_ALIASES.get(normalized_backend, normalized_backend)
    if log_backend not in {"wandb", "terminal"}:
        raise ValueError(
            "training.log_backend must be either 'wandb' or 'terminal'; "
            f"got {configured_backend!r}"
        )

    # Keep the existing WANDB_MODE=disabled escape hatch useful even when the
    # config still has its default backend.
    selected_wandb_mode = (
        os.environ.get("WANDB_MODE", "") if wandb_mode is None else wandb_mode
    )
    if str(selected_wandb_mode).strip().lower() == "disabled":
        return "terminal"
    return log_backend


class RunLogger(ABC):
    """One interface for run logging, one adapter per backend.

    A run holds exactly one adapter and tears it down with a single
    ``finish()`` call, so trainer code contains no backend branches.
    """

    @abstractmethod
    def init(
        self,
        *,
        name,
        config,
        computed,
        stage_marker,
        parameter_metrics,
        step,
        saved_config=None,
        resumed_run_id=None,
    ) -> None:
        """Start the run: name, resolved config, computed budget values,
        the stage marker, parameter metrics, and an optional resumed W&B run
        id. A resumed run keeps the original run's config and name."""

    @abstractmethod
    def log_metrics(self, metrics, step) -> None:
        """Log a metrics dict at ``step``."""

    @abstractmethod
    def log_sample(self, text, step, loss=None) -> None:
        """Log a generated sample: HTML in W&B, one print line in terminal."""

    @abstractmethod
    def log_checkpoint(self, kind, name, path, metadata) -> None:
        """Record a checkpoint save of ``kind`` (``latest`` / ``best`` /
        ``final`` / ``periodic``); the W&B adapter uploads it as an artifact."""

    @property
    def run_url(self):
        """The run's public URL, when the backend has one (W&B)."""
        return None

    @abstractmethod
    def finish(self) -> None:
        """Flush and close the run."""


class TerminalLogger(RunLogger):
    """Prints only; holds no reference to W&B at all."""

    def init(
        self,
        *,
        name,
        config,
        computed,
        stage_marker,
        parameter_metrics,
        step,
        saved_config=None,
        resumed_run_id=None,
    ) -> None:
        # Terminal runs carry no run-level log output at init; the console
        # lines (budget, model summary, per-step progress) are printed by the
        # trainer itself.
        return None

    def log_metrics(self, metrics, step) -> None:
        return None

    def log_sample(self, text, step, loss=None) -> None:
        loss_text = "loss n/a" if loss is None else f"val loss {loss:.4f}"
        print(f"sample | step {step} | {loss_text} | text: {text}")

    def log_checkpoint(self, kind, name, path, metadata) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbLogger(RunLogger):
    """The W&B adapter: owns the wandb import and every W&B interaction."""

    def __init__(self, wandb_module=None, upload_artifacts=False):
        if wandb_module is None:
            import wandb  # noqa: F401 — the import stays inside this adapter

            wandb_module = wandb
        self._wandb = wandb_module
        self.upload_artifacts = bool(upload_artifacts)
        self.run_id = None
        self._run = None

    def init(
        self,
        *,
        name,
        config,
        computed,
        stage_marker,
        parameter_metrics,
        step,
        saved_config=None,
        resumed_run_id=None,
    ) -> None:
        if saved_config is not None:
            # Keep the resumed run's original configuration in W&B; the
            # resume path itself is an invocation detail, not a new
            # experiment configuration.
            config = saved_config
            name = saved_config.get("training", {}).get("exp_name", name)

        wandb_kwargs = {
            "project": "mini-llm-lab",
            "name": name,
            "config": config,
        }
        if resumed_run_id is not None:
            wandb_kwargs.update(id=resumed_run_id, resume="must")
            print(f"Resuming W&B run: {resumed_run_id}")
        self._wandb.init(**wandb_kwargs)
        self._run = self._wandb.run
        if self._run is not None:
            self.run_id = self._run.id

        self._wandb.config.update(
            {
                **computed,
                **stage_marker,
                "model_parameters_total": parameter_metrics["model/parameters_total"],
                "model_parameters_trainable": parameter_metrics[
                    "model/parameters_trainable"
                ],
                "model_parameters_non_trainable": parameter_metrics[
                    "model/parameters_non_trainable"
                ],
                "model_parameter_memory_mb": parameter_metrics[
                    "model/parameter_memory_mb"
                ],
            }
        )
        self._wandb.log({**parameter_metrics, "step": step}, step=step)

    def log_metrics(self, metrics, step) -> None:
        self._wandb.log(metrics, step=step)

    def log_sample(self, text, step, loss=None) -> None:
        self._wandb.log(
            {"sample_text": self._wandb.Html(f"<pre>{text}</pre>"), "step": step},
            step=step,
        )

    def log_checkpoint(self, kind, name, path, metadata) -> None:
        if not self.upload_artifacts:
            return
        # ``log_artifact`` never raises: an artifact failure only warns, so
        # a W&B outage cannot abort a long training run.
        log_artifact(self._wandb, path, name, metadata)

    @property
    def run_url(self):
        if self._run is None:
            return None
        return getattr(self._run, "url", None)

    def finish(self) -> None:
        if self._run is not None:
            self._wandb.finish()


def create_logger(log_backend: str, upload_artifacts=False) -> RunLogger:
    """Map the resolved backend to its adapter."""
    if log_backend == "wandb":
        return WandbLogger(upload_artifacts=upload_artifacts)
    return TerminalLogger()


def should_log_artifacts(log_backend, upload_artifacts=False) -> bool:
    """Return whether this run should upload checkpoint artifacts to W&B."""
    return log_backend == "wandb" and bool(upload_artifacts)


def build_artifact_metadata(
    kind, stage, step, tokens_seen, best_val_loss, git_commit, local_path, wandb_run_id
):
    """Identifying metadata attached to a checkpoint W&B artifact."""
    return {
        "kind": kind,
        "stage": stage,
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "git_commit": git_commit,
        "local_path": local_path,
        "wandb_run_id": wandb_run_id,
    }


def log_artifact(wandb, path, name, metadata) -> bool:
    """Upload ``path`` to W&B as artifact ``name`` with ``metadata``.

    Never raises: an artifact failure (connection drop, server error) only
    prints a warning so it cannot abort a long training run. Returns True
    when the artifact was logged.
    """
    try:
        artifact = wandb.Artifact(name, type="models")
        artifact.add_file(str(path))
        artifact.metadata.update(metadata)
        wandb.log_artifact(artifact)
        return True
    except Exception as error:
        print(f"Warning: failed to log W&B artifact {name!r}: {error}")
        return False
