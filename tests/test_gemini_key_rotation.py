from types import SimpleNamespace

from src.config import Settings
from src.gemini_client import GeminiClient, GeminiError
from src.models import ImageCheckSchema


class RateLimitError(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("rate limit exceeded")


class FakeModels:
    def __init__(self, key, calls):
        self.key = key
        self.calls = calls

    def generate_content(self, *, model, contents, config):
        self.calls.append((self.key, model))
        if self.key in {"key-one", "key-two"}:
            raise RateLimitError()
        return SimpleNamespace(
            parsed=ImageCheckSchema(
                relevant_to_story=True,
                contains_text_or_watermark=False,
                obvious_defects=False,
                reason="ok",
            )
        )


class FakeClient:
    def __init__(self, key, calls):
        self.models = FakeModels(key, calls)


def test_gemini_rotates_to_next_key_after_rate_limit(monkeypatch):
    calls = []

    def make_client(key, timeout_seconds):
        return FakeClient(key, calls)

    monkeypatch.setattr(GeminiClient, "_make_client", staticmethod(make_client))
    cfg = Settings(
        gemini_api_key="key-one",
        gemini_api_key_2="key-two",
        gemini_api_key_3="key-three",
        gemini_model="gemini-test",
        gemini_fallback_model="gemini-test-fallback",
        max_retries=0,
    )
    client = GeminiClient(cfg)

    result = client.generate_json("check this", ImageCheckSchema, tag="IMGCHECK")

    assert result.relevant_to_story is True
    assert [key for key, _ in calls] == ["key-one", "key-two", "key-three"]
    assert client._exhausted_keys == {0, 1}


def test_gemini_falls_back_after_model_error_without_name_error(monkeypatch):
    monkeypatch.setattr(
        GeminiClient,
        "_make_client",
        staticmethod(lambda _key, _timeout: FakeClient("key-one", [])),
    )
    cfg = Settings(
        gemini_api_key="key-one",
        gemini_model="gemini-primary",
        gemini_fallback_model="gemini-fallback",
        max_retries=0,
    )
    client = GeminiClient(cfg)
    models = []

    def fake_generate(model, _prompt, schema, **_kwargs):
        models.append(model)
        if model == "gemini-primary":
            raise GeminiError("temporary 503")
        return schema(
            relevant_to_story=True,
            contains_text_or_watermark=False,
            obvious_defects=False,
            reason="fallback succeeded",
        )

    monkeypatch.setattr(client, "_generate_json_model", fake_generate)
    result = client.generate_json("check this image", ImageCheckSchema, tag="IMGCHECK")

    assert result.relevant_to_story is True
    assert models == ["gemini-primary", "gemini-fallback"]


def test_gemini_falls_back_after_unexpected_name_error(monkeypatch):
    monkeypatch.setattr(
        GeminiClient,
        "_make_client",
        staticmethod(lambda _key, _timeout: FakeClient("key-one", [])),
    )
    cfg = Settings(
        gemini_api_key="key-one",
        gemini_model="gemini-primary",
        gemini_fallback_model="gemini-fallback",
        max_retries=0,
    )
    client = GeminiClient(cfg)
    models = []

    def fake_generate(model, _prompt, schema, **_kwargs):
        models.append(model)
        if model == "gemini-primary":
            raise NameError("name 'is_last' is not defined")
        return schema(
            relevant_to_story=True,
            contains_text_or_watermark=False,
            obvious_defects=False,
            reason="fallback succeeded",
        )

    monkeypatch.setattr(client, "_generate_json_model", fake_generate)
    result = client.generate_json("check this image", ImageCheckSchema, tag="IMGCHECK")

    assert result.relevant_to_story is True
    assert models == ["gemini-primary", "gemini-fallback"]
