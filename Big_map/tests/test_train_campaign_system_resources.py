import json
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from train_campaign import _collect_lscpu_fields
from train_campaign import _format_binary_size
from train_campaign import _read_linux_meminfo


def test_format_binary_size_uses_gibibytes():
    assert _format_binary_size(16 * 1024**3) == "16.00 GiB"


def test_read_linux_meminfo_converts_kibibytes_to_bytes(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       16384 kB\nMemAvailable:    8192 kB\n",
        encoding="utf-8",
    )

    assert _read_linux_meminfo(meminfo) == {
        "MemTotal": 16384 * 1024,
        "MemAvailable": 8192 * 1024,
    }


def test_collect_lscpu_fields_parses_json(monkeypatch):
    payload = {
        "lscpu": [
            {"field": "CPU(s):", "data": "4"},
            {"field": "Model name:", "data": "AMD EPYC Test CPU"},
        ]
    }

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _collect_lscpu_fields() == {
        "CPU(s)": "4",
        "Model name": "AMD EPYC Test CPU",
    }
