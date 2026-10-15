import numpy as np
import torch

from decide.calibrate import Calibration, ece, fit_temperature, fit_temperatures
from decide.model import Logits


def test_fit_temperature_recovers_a_known_temperature():
    g = torch.Generator().manual_seed(0)
    true = torch.randn(4000, 5, generator=g) * 3
    y = torch.multinomial(torch.softmax(true, -1), 1, generator=g).squeeze(1)
    logits = true * 2.5  # overconfident by a factor 2.5
    loss_at = lambda t: torch.nn.functional.cross_entropy(logits / t, y).item()  # noqa: E731
    assert abs(fit_temperature(loss_at) - 2.5) < 0.15


def test_fit_temperatures_by_kind():
    g = torch.Generator().manual_seed(1)
    z = torch.randn(3000, generator=g) * 2
    y = (torch.rand(3000, generator=g) < torch.sigmoid(z)).long()
    out = Logits(torch.zeros(0, 3), z * 3, torch.zeros(0), torch.zeros(0, 1))
    temps = fit_temperatures(out, {"choice": torch.zeros(0, dtype=torch.long), "yesno": y, "score": torch.zeros(0, dtype=torch.long)})
    assert set(temps) == {"yesno"} and abs(temps["yesno"] - 3) < 0.4


def test_ece():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1, 20000)
    calibrated = rng.uniform(size=conf.size) < conf
    assert ece(conf, calibrated) < 0.01
    assert abs(ece(np.full(100, 0.9), np.zeros(100)) - 0.9) < 1e-9


def test_calibration_round_trip(tmp_path):
    cal = Calibration({"choice": 1.2}, {"choice": 0.9}, alpha=0.05)
    cal.save(tmp_path / "c.json")
    assert Calibration.load(tmp_path / "c.json") == cal
