"""T1.3: streamed atomic writes and strict model restore."""

import io

import pytest
import torch
import torch.nn as nn

from training.checkpointing import atomic_torch_save, restore_model
from training.run_manifest import atomic_write


def test_atomic_write_streams_chunks_to_the_destination(tmp_path):
    """The writer receives an open binary file and may stream it in
    chunks — the destination ends up with exactly the concatenated
    chunks and no temp file is left behind."""
    destination = tmp_path / "out.bin"
    seen = {}

    def writer(f):
        seen["is_file"] = hasattr(f, "fileno")
        seen["is_bytesio"] = isinstance(f, io.BytesIO)
        for chunk in (b"first-", b"second-", b"third"):
            f.write(chunk)

    atomic_write(destination, writer)

    assert destination.read_bytes() == b"first-second-third"
    assert seen == {"is_file": True, "is_bytesio": False}
    assert not list(tmp_path.glob(".out.bin.tmp.*"))


def test_atomic_write_failure_removes_the_temp_and_keeps_the_destination(
    tmp_path,
):
    """A writer exception removes the temp file; an existing destination
    is left untouched (a failed write never truncates the good file)."""
    destination = tmp_path / "out.bin"
    destination.write_bytes(b"good")

    def boom(f):
        f.write(b"partial")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        atomic_write(destination, boom)

    assert destination.read_bytes() == b"good"
    assert not list(tmp_path.glob(".out.bin.tmp.*"))


def test_atomic_write_failure_leaves_no_destination(tmp_path):
    destination = tmp_path / "missing.bin"

    def boom(f):
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError):
        atomic_write(destination, boom)

    assert not destination.exists()
    assert not list(tmp_path.glob(".missing.bin.tmp.*"))


def test_atomic_torch_save_streams_directly_to_a_file(tmp_path, monkeypatch):
    """torch.save receives the temp file itself, not an in-memory BytesIO
    copy of the full payload (the P10 peak-RAM fix)."""
    captured = {}
    real_save = torch.save

    def spy(payload, f):
        captured["stream"] = f
        real_save(payload, f)

    import training.checkpointing as checkpointing

    monkeypatch.setattr(checkpointing.torch, "save", spy)
    path = tmp_path / "latest.pt"
    atomic_torch_save({"step": 12, "tensor": torch.tensor([1, 2, 3])}, path)

    stream = captured["stream"]
    assert not isinstance(stream, io.BytesIO)
    assert hasattr(stream, "fileno")
    # The saved file is readable back (the stream carried the real bytes).
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["step"] == 12


def test_restore_model_is_strict():
    """A state dict with a missing key surfaces torch's native error
    (strict=True), not the old hand-rolled mismatch message."""
    model = nn.Linear(4, 4, bias=False)

    with pytest.raises(RuntimeError, match="Missing key"):
        restore_model(model, {"model_state_dict": {}})


def test_restore_model_still_strips_the_orig_mod_prefix():
    """Checkpoints written by the current code may carry the torch.compile
    ``_orig_mod.`` prefix; it is stripped before the strict load."""
    model = nn.Linear(4, 4, bias=False)
    prefixed = {
        f"_orig_mod.{k}": v for k, v in model.state_dict().items()
    }

    restore_model(model, {"model_state_dict": prefixed})

    torch.testing.assert_close(
        model.state_dict()["weight"], prefixed["_orig_mod.weight"]
    )
