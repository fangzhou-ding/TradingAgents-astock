"""Web 侧栏 API Key 输入（v0.5.22）的回归测试。

要锁住三件事：
1. provider → 环境变量映射与各 LLM 客户端实际读取的变量名一致；
2. 保存的 Key 在启动时注入环境变量（apply_stored_secrets）；
3. 没配 Key 时在起线程之前就被 validate_api_key 拦住。
"""

from __future__ import annotations

import json
import os
import types

import pytest

from web.components import sidebar


@pytest.fixture()
def fake_secrets(monkeypatch, tmp_path):
    """把凭据文件指到临时目录，并在用例结束后还原被改动的环境变量。"""
    monkeypatch.setattr(sidebar, "_SECRETS_PATH", tmp_path / "home" / ".tradingagents" / "secrets.json")
    touched = [env for env in sidebar._PROVIDER_KEY_ENV.values() if env]
    saved = {env: os.environ.get(env) for env in touched}
    yield
    for env, value in saved.items():
        if value is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = value


@pytest.mark.unit
def test_provider_env_map_matches_clients():
    # 与 tradingagents/llm_clients/openai_client.py 的 _PROVIDER_CONFIG 对齐；
    # 映射错了的表现是「页面填了 key，客户端却仍报缺失」，极难排查。
    assert sidebar._PROVIDER_KEY_ENV["deepseek"] == "DEEPSEEK_API_KEY"
    assert sidebar._PROVIDER_KEY_ENV["qwen"] == "DASHSCOPE_API_KEY"
    assert sidebar._PROVIDER_KEY_ENV["glm"] == "ZHIPU_API_KEY"
    assert sidebar._PROVIDER_KEY_ENV["minimax"] == "MINIMAX_API_KEY"
    assert sidebar._PROVIDER_KEY_ENV["openai_compatible"] == "OPENAI_COMPATIBLE_API_KEY"
    assert sidebar._PROVIDER_KEY_ENV["anthropic"] == "ANTHROPIC_API_KEY"
    assert sidebar._PROVIDER_KEY_ENV["ollama"] is None
    # 每个可选供应商都得有映射，否则界面上根本渲染不出输入框
    assert set(sidebar._PROVIDER_KEYS) <= set(sidebar._PROVIDER_KEY_ENV)


@pytest.mark.unit
def test_secrets_roundtrip(fake_secrets):
    sidebar._write_secrets({"deepseek": "sk-test"})
    assert sidebar._read_secrets() == {"deepseek": "sk-test"}


@pytest.mark.unit
def test_corrupt_secrets_file_is_treated_as_empty(fake_secrets):
    sidebar._SECRETS_PATH.parent.mkdir(parents=True)
    sidebar._SECRETS_PATH.write_text("{not json", encoding="utf-8")
    assert sidebar._read_secrets() == {}


@pytest.mark.unit
def test_write_failure_only_warns(fake_secrets, caplog):
    blocker = sidebar._SECRETS_PATH.parent.parent
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("not a dir")
    sidebar._write_secrets({"deepseek": "sk-test"})  # must not raise
    assert any("持久化失败" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
def test_apply_stored_secrets_sets_env(fake_secrets, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    sidebar._SECRETS_PATH.parent.mkdir(parents=True)
    sidebar._SECRETS_PATH.write_text(json.dumps({"deepseek": "sk-from-ui"}), encoding="utf-8")

    sidebar.apply_stored_secrets()

    assert os.environ["DEEPSEEK_API_KEY"] == "sk-from-ui"


@pytest.mark.unit
def test_apply_stored_secrets_skips_empty_values(fake_secrets, monkeypatch):
    # 空值不能覆盖 .env 里正常配置的 key
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    sidebar._SECRETS_PATH.parent.mkdir(parents=True)
    sidebar._SECRETS_PATH.write_text(json.dumps({"deepseek": "   "}), encoding="utf-8")

    sidebar.apply_stored_secrets()

    assert os.environ["DEEPSEEK_API_KEY"] == "sk-from-env"


@pytest.mark.unit
def test_validate_api_key_blocks_missing_key(fake_secrets, monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    err = sidebar.validate_api_key({"llm_provider": "minimax"})
    assert err is not None
    assert "MINIMAX_API_KEY" in err


@pytest.mark.unit
def test_validate_api_key_passes_when_env_present(fake_secrets, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-x")
    assert sidebar.validate_api_key({"llm_provider": "minimax"}) is None


@pytest.mark.unit
def test_validate_api_key_skips_ollama(fake_secrets):
    assert sidebar.validate_api_key({"llm_provider": "ollama"}) is None


@pytest.mark.unit
def test_validate_api_key_reads_session_state_when_config_missing(fake_secrets, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        sidebar, "st", types.SimpleNamespace(session_state={"llm_provider": "deepseek"})
    )
    assert sidebar.validate_api_key() is not None


@pytest.mark.unit
def test_app_validates_key_at_the_consumption_point():
    """结构性回归：Key 校验必须挂在 `start_analysis` 的唯一消费点。"""
    from pathlib import Path

    app_src = (Path(sidebar.__file__).resolve().parent.parent / "app.py").read_text(
        encoding="utf-8"
    )

    pop_at = app_src.index('st.session_state.pop("start_analysis"')
    guard_at = app_src.index("validate_api_key(config)")
    run_at = app_src.index("run_analysis_in_thread(")

    assert pop_at < guard_at < run_at
