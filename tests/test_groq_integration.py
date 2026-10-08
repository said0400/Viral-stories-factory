import json
import re

from pydantic import BaseModel

from src.config import Settings
from src.groq_client import GroqClient
from src.llm_router import LLMRouter
from src.models import SourceArticle
from src.photo_composer import (
    PhotoAssessmentBatch,
    PhotoAssessmentSchema,
    analyze_source_photos,
)


class ResultSchema(BaseModel):
    value: str


class StubProvider:
    def __init__(self, value, *, fail=False, configured=True):
        self.value = value
        self.fail = fail
        self.configured = configured
        self.calls = []

    def is_configured(self):
        return self.configured

    def generate_json(self, prompt, schema, **kwargs):
        self.calls.append(kwargs.get("tag"))
        if self.fail:
            raise RuntimeError("simulated provider failure")
        return schema(value=self.value)


def test_router_assigns_writing_to_gemini_and_analysis_to_groq():
    gemini = StubProvider("gemini")
    groq = StubProvider("groq")
    router = LLMRouter(gemini, groq)

    assert router.generate_json("p", ResultSchema, tag="ARTICLE").value == "gemini"
    assert router.generate_json("p", ResultSchema, tag="METADATA").value == "gemini"
    assert router.generate_json("p", ResultSchema, tag="FACTCHECK").value == "groq"
    assert gemini.calls == ["ARTICLE", "METADATA"]
    assert groq.calls == ["FACTCHECK"]


def test_router_falls_back_in_both_directions():
    gemini = StubProvider("gemini-fallback")
    groq = StubProvider("groq-fallback", fail=True)
    router = LLMRouter(gemini, groq)
    assert router.generate_json("p", ResultSchema, tag="FACTCHECK").value == "gemini-fallback"

    gemini_article = StubProvider("gemini-primary", fail=True)
    groq_article = StubProvider("groq-article")
    router = LLMRouter(gemini_article, groq_article)
    assert router.generate_json("p", ResultSchema, tag="ARTICLE").value == "groq-article"


def test_groq_client_uses_chat_completions_json_and_separate_vision_model(monkeypatch):
    requests_seen = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"value":"ok"}'}}]}

    def fake_post(url, **kwargs):
        requests_seen.append((url, kwargs))
        return Response()

    monkeypatch.setattr("src.groq_client.requests.post", fake_post)
    client = GroqClient(Settings(
        groq_api_key="test-groq-secret",
        groq_model="text-test-model",
        groq_vision_model="vision-test-model",
    ))

    assert client.generate_json("return JSON", ResultSchema, tag="METADATA").value == "ok"
    assert client.generate_json(
        "describe this image", ResultSchema,
        images=[(b"image-bytes", "image/jpeg")], tag="VISUAL",
    ).value == "ok"

    text_url, text_request = requests_seen[0]
    image_url, image_request = requests_seen[1]
    assert text_url == image_url == "https://api.groq.com/openai/v1/chat/completions"
    assert text_request["json"]["model"] == "text-test-model"
    assert image_request["json"]["model"] == "vision-test-model"
    assert text_request["json"]["response_format"] == {"type": "json_object"}
    image_content = image_request["json"]["messages"][1]["content"]
    assert image_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert text_request["headers"]["Authorization"] == "Bearer test-groq-secret"


def test_groq_only_settings_are_valid_and_secret_is_redacted():
    settings = Settings.from_env({"GROQ_API_KEY": "groq-secret-value"})
    assert settings.groq_api_key == "groq-secret-value"
    assert not settings.validate()
    assert "groq-secret-value" in settings.secret_values()


def test_photo_analysis_splits_six_candidates_into_vision_batches():
    class BatchAnalyzer:
        def __init__(self):
            self.batch_sizes = []
            self.indices = []

        def generate_json(self, prompt, schema, **kwargs):
            images = kwargs["images"]
            self.batch_sizes.append(len(images))
            match = re.search(r"global indices (\[[^\]]+\])", prompt)
            indices = json.loads(match.group(1))
            self.indices.append(indices)
            return PhotoAssessmentBatch(items=[
                PhotoAssessmentSchema(
                    index=i,
                    relevance_score=90 - i,
                    visual_impact_score=80 - i,
                    focus_center_x=500,
                    focus_center_y=500,
                    focus_width=350,
                    focus_height=350,
                    focal_description=f"subject {i}",
                )
                for i in indices
            ])

    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    article = SourceArticle(
        source_name="Example",
        original_title="A verified story",
        original_url="https://example.com/story",
        normalized_url="https://example.com/story",
        discovered_at=now,
        article_text="A short verified context for selecting relevant images.",
    )
    photos = []
    from PIL import Image
    for i in range(6):
        photos.append(Image.new("RGB", (400, 400), (i * 30, 40, 80)))

    analyzer = BatchAnalyzer()
    selected = analyze_source_photos(analyzer, article, photos, limit=4)
    assert analyzer.batch_sizes == [3, 3]
    assert analyzer.indices == [[0, 1, 2], [3, 4, 5]]
    assert [photo.reason for photo in selected] == ["subject 0", "subject 1", "subject 2", "subject 3"]


def test_gpt_oss_uses_low_hidden_reasoning_for_json(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"value":"ok"}'}}]}

    def fake_post(url, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr("src.groq_client.requests.post", fake_post)
    client = GroqClient(Settings(groq_api_key="secret", groq_model="openai/gpt-oss-120b"))
    assert client.generate_json("return JSON", ResultSchema).value == "ok"
    assert captured["reasoning_effort"] == "low"
    assert captured["reasoning_format"] == "hidden"
    assert len(captured["messages"]) == 1
    assert captured["messages"][0]["role"] == "user"
    assert "JSON Schema" in captured["messages"][0]["content"]


def test_groq_retries_once_after_schema_validation_error(monkeypatch):
    outputs = iter(['{"value": 123}', '{"value": "fixed"}'])
    requests_seen = []

    class Response:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": next(outputs)}}]}

    def fake_post(url, **kwargs):
        requests_seen.append(kwargs["json"])
        return Response()

    monkeypatch.setattr("src.groq_client.requests.post", fake_post)
    monkeypatch.setattr("src.groq_client.time.sleep", lambda _seconds: None)
    client = GroqClient(Settings(groq_api_key="secret", groq_model="test-model"))
    assert client.generate_json("return JSON", ResultSchema).value == "fixed"
    assert len(requests_seen) == 2
    assert "Try again from scratch" in requests_seen[1]["messages"][-1]["content"]


def test_groq_http_error_includes_safe_status_and_provider_reason():
    import pytest
    from src.groq_client import GroqError

    class Response:
        status_code = 401
        text = ""

        @staticmethod
        def json():
            return {"error": {"code": "invalid_api_key", "message": "API key rejected"}}

    client = GroqClient(Settings(groq_api_key="secret-value"))
    error = client._http_error(Response(), "METADATA")
    assert isinstance(error, GroqError)
    assert error.status_code == 401
    assert not error.retryable
    assert "invalid_api_key" in str(error)
    assert "secret-value" not in str(error)


def test_qwen_vision_keeps_analysis_prompt_when_merging_json_instructions(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"value":"ok"}'}}]}

    def fake_post(url, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr("src.groq_client.requests.post", fake_post)
    client = GroqClient(Settings(groq_api_key="secret", groq_vision_model="qwen/qwen3.8-27b"))
    assert client.generate_json(
        "Describe the image's focal subject", ResultSchema,
        images=[(b"image-bytes", "image/jpeg")],
    ).value == "ok"
    content = captured["messages"][0]["content"]
    assert "Describe the image's focal subject" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert captured["reasoning_effort"] == "low"



def test_groq_retries_provider_json_validation_failure(monkeypatch):
    requests_seen = []

    class BadResponse:
        status_code = 400

        @staticmethod
        def json():
            return {"error": {"code": "json_validate_failed", "message": "Failed to generate JSON"}}

    class GoodResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"value":"ok"}'}}]}

    def fake_post(_url, **kwargs):
        requests_seen.append(json.loads(json.dumps(kwargs["json"])))
        return BadResponse() if len(requests_seen) == 1 else GoodResponse()

    monkeypatch.setattr("src.groq_client.requests.post", fake_post)
    monkeypatch.setattr("src.groq_client.time.sleep", lambda _seconds: None)
    client = GroqClient(Settings(groq_api_key="secret"))

    assert client.generate_json("return JSON", ResultSchema, tag="TRIAGE").value == "ok"
    assert len(requests_seen) == 2
    assert requests_seen[1]["temperature"] == 0.1
    assert any(
        "Try again from scratch" in message.get("content", "")
        for message in requests_seen[1]["messages"]
    )


def test_groq_limits_completion_budget_and_reduces_it_after_tpm_413(monkeypatch):
    requests_seen = []

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            if self.status_code == 413:
                return {
                    "error": {
                        "code": "rate_limit_exceeded",
                        "message": "Request too large on tokens per minute (TPM): Limit 8000, Requested 10621",
                    }
                }
            return {"choices": [{"message": {"content": '{"value":"ok"}'}}]}

    def fake_post(_url, **kwargs):
        requests_seen.append(json.loads(json.dumps(kwargs["json"])))
        return Response(413 if len(requests_seen) == 1 else 200)

    monkeypatch.setattr("src.groq_client.requests.post", fake_post)
    monkeypatch.setattr("src.groq_client.time.sleep", lambda _seconds: None)
    client = GroqClient(Settings(groq_api_key="secret"))

    assert client.generate_json("check claims", ResultSchema, tag="FACTCHECK").value == "ok"
    assert len(requests_seen) == 2
    assert requests_seen[0]["max_completion_tokens"] == 1024
    assert requests_seen[1]["max_completion_tokens"] == 512


def test_groq_uses_task_specific_completion_budgets(monkeypatch):
    budgets = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"value":"ok"}'}}]}

    def fake_post(_url, **kwargs):
        budgets.append((kwargs["json"]["messages"][-1]["content"], kwargs["json"]["max_completion_tokens"]))
        return Response()

    monkeypatch.setattr("src.groq_client.requests.post", fake_post)
    client = GroqClient(Settings(groq_api_key="secret", groq_model="test-model"))
    for tag in ("ARTICLE", "TRIAGE", "METADATA", "FACTCHECK"):
        client.generate_json("short request", ResultSchema, tag=tag)

    assert [budget for _, budget in budgets] == [4096, 4096, 2048, 1024]
