import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
NN_DATA_DIR = ROOT_DIR / "NN Data"
sys.path.insert(0, str(NN_DATA_DIR))

from src import data_engine


def test_load_data_auto_merges_csvs(tmp_path, monkeypatch):
    monkeypatch.setattr(data_engine, "DATA_RAW_DIR", str(tmp_path))

    df_first = pd.DataFrame(
        {
            "Frequency_GHz": [1.0, 2.0],
            "via_pitch": [10.0, 20.0],
            "via_height": [30.0, 40.0],
            "r_cu": [1.0, 1.1],
            "t_sio2": [0.1, 0.2],
            "r_D": [2.0, 2.1],
            "r_cu_1": [1.5, 1.6],
            "array_size": [4, 4],
        }
    )
    df_second = pd.DataFrame(
        {
            "Frequency_GHz": [3.0],
            "via_pitch": [30.0],
            "via_height": [50.0],
            "r_cu": [1.2],
            "t_sio2": [0.3],
            "r_D": [2.2],
            "r_cu_1": [1.7],
            "array_size": [4],
        }
    )

    df_first.to_csv(tmp_path / "data_a.csv", index=False)
    df_second.to_csv(tmp_path / "data_b.csv", index=False)

    df_master, msg = data_engine.DataEngine.load_data_auto()

    assert df_master is not None
    assert len(df_master) == 3
    assert "成功加载" in msg
