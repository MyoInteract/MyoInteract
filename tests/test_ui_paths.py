from pathlib import Path

from myosuite.envs.myo.myouser import paths


def test_bundled_model_and_presets_are_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = paths.bundled_path("myosuite/envs/myo/assets/arm/mobl_arms_index_universal_myouser.xml")
    assert model.is_file()
    assert paths.get_bundled_config_dir().is_dir()
    assert len(list(paths.get_bundled_config_dir().glob("*.json"))) == 3


def test_explicit_custom_model_paths_keep_their_cwd_semantics(tmp_path, monkeypatch):
    custom = tmp_path / "custom.xml"
    custom.write_text("<mujoco/>")
    monkeypatch.chdir(tmp_path)
    assert paths.bundled_path("custom.xml") == custom
    assert paths.bundled_path(custom) == custom


def test_output_paths_share_explicit_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MYOINTERACT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("MYOINTERACT_LOG_DIR", raising=False)
    monkeypatch.delenv("MYOINTERACT_CONFIG_DIR", raising=False)
    assert paths.get_logs_dir() == tmp_path / "data" / "logs"
    assert paths.get_user_config_dir() == tmp_path / "data" / "gradio_configs"
    monkeypatch.setenv("MYOINTERACT_LOG_DIR", str(tmp_path / "runs"))
    assert paths.get_logs_dir() == tmp_path / "runs"
