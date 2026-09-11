"""BSS 外部气候概率参考 Reader（ref/MMDDHH.000）。"""

from datetime import datetime

import numpy as np
import pytest

from xmetai_evaluation.core.errors import ConfigError
from xmetai_evaluation.io.ref_probability_reader import REF_THRESHOLDS, RefProbabilityReader


def _write_ref(root):
    (root / "010106.000").write_text(
        "1 0.5 0.5 0.5 0.5\n2 0.1 0.2 0.3 0.4\n", encoding="ascii"
    )
    (root / "010112.000").write_text(
        "1 0.6 0.6 0.6 0.6\n", encoding="ascii"
    )


def test_probabilities_lookup_and_missing_station(tmp_path):
    _write_ref(tmp_path)
    reader = RefProbabilityReader(root_dir=tmp_path)

    probs = reader.probabilities(datetime(2025, 1, 1, 6), [1, 2, 999])
    assert probs.shape == (3, 4)
    np.testing.assert_allclose(probs[0], [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(probs[1], [0.1, 0.2, 0.3, 0.4])
    assert np.isnan(probs[2]).all()  # 缺 ref 站 → 整行 NaN

    assert reader.thresholds == REF_THRESHOLDS


def test_missing_time_raises(tmp_path):
    _write_ref(tmp_path)
    reader = RefProbabilityReader(root_dir=tmp_path)
    with pytest.raises(ConfigError, match="缺少时次"):
        reader.probabilities(datetime(2025, 1, 1, 18), [1])


def test_empty_dir_raises(tmp_path):
    with pytest.raises(ConfigError, match=".000"):
        RefProbabilityReader(root_dir=tmp_path)
