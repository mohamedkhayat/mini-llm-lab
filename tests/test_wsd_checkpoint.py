from training.checkpointing import (
    atomic_torch_save,
    build_checkpoint_payload,
    capture_data_state,
    load_checkpoint,
    restore_data_state,
)


def _payload(**overrides):
    kwargs = dict(
        model_state={"w": 1},
        model_cfg={"emb_dim": 16},
        optimizer_state={"state": 1},
        scheduler_state={"base_lrs": [5e-4]},
        step=30,
        best_val_loss=1.2,
        tokens_seen=777,
        run_elapsed_seconds=1.5,
        steps_per_pass=40,
        tokens_per_step=256,
        total_steps=38,
        wandb_run_id="abc123",
        rng_state={"python": None},
        cfg_container={"training": {"lr": 5e-4}},
        wsd_state={"triggered": True, "step": 30},
    )
    kwargs.update(overrides)
    return kwargs


def test_checkpoint_payload_carries_the_triggered_wsd_state():
    payload = build_checkpoint_payload(**_payload())

    assert payload["wsd"] == {"triggered": True, "step": 30}
    assert payload["step"] == 30
    assert payload["total_steps"] == 38
    assert payload["wandb_run_id"] == "abc123"


def test_checkpoint_payload_carries_the_untriggered_wsd_state():
    payload = build_checkpoint_payload(
        **_payload(wsd_state={"triggered": False, "step": None})
    )

    assert payload["wsd"] == {"triggered": False, "step": None}


def test_data_state_capture_and_restore_use_the_loader_seam():
    class CursorLoader:
        """The memmap loader's state seam: state_dict / load_state_dict."""

        def __init__(self):
            self.current_step = 0

        def __len__(self):
            return 5

        def state_dict(self):
            return {"current_step": self.current_step}

        def load_state_dict(self, state):
            self.current_step = state["current_step"]

    loader = CursorLoader()
    loader.current_step = 3
    saved = capture_data_state(loader)
    loader.current_step = 0
    restore_data_state(loader, saved)

    assert saved == {"kind": "memmap", "state": {"current_step": 3}}
    assert loader.current_step == 3


def test_saved_checkpoint_roundtrips_the_wsd_state(tmp_path):
    path = tmp_path / "latest.pt"
    atomic_torch_save(build_checkpoint_payload(**_payload()), path)

    checkpoint = load_checkpoint(path)
    assert checkpoint["wsd"] == {"triggered": True, "step": 30}


def test_memmap_resume_is_silent(capsys):
    class CursorLoader:
        current_step = 0

        def __len__(self):
            return 5

        def load_state_dict(self, state):
            self.current_step = state["current_step"]

    loader = CursorLoader()
    restore_data_state(loader, {"kind": "memmap", "state": {"current_step": 4}})

    assert capsys.readouterr().out == ""
    assert loader.current_step == 4


def test_legacy_checkpoint_without_wsd_key_is_tolerated_on_resume():
    # Loaders tolerate the absence of the wsd state: a checkpoint saved
    # before this feature simply has no "wsd" key.
    from training.schedule import resolve_wsd_state

    assert resolve_wsd_state(None, start_decay=False, resume_step=30) == {
        "triggered": False,
        "step": None,
    }
