"""
Unit tests for Plain Text Mode with paragraph fallback disabled
(DISABLE_PARAGRAPH_FALLBACK=true).

Validates that:
- When the LLM merges or changes paragraph counts, the translated segment is accepted
  without triggering the expensive per-paragraph repair loop.
- Only the original translation call is made (zero fallback calls).
- The reassembly emits all translated paragraphs without padding missing slots with
  empty strings (which would otherwise fall back to English source text).
- Empty source slots (image-only anchors) are preserved in their relative positions.
"""
import re
import pytest

import src.core.common.plain_text_pipeline as plain_pipeline


MERGED = ["Alpha paragraph.", "Beta paragraph."]
THREE = ["First paragraph.", "Second paragraph.", "Third paragraph."]


def _recording_merging_llm(prefix="T::"):
    """LLM that folds multi-paragraph inputs into a single merged paragraph."""
    calls = []

    async def fake_request(*, main_content, **kwargs):
        calls.append(main_content)
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", main_content) if p.strip()]
        return prefix + " ".join(paragraphs)

    fake_request.calls = calls
    return fake_request


async def _run(paragraphs, max_tokens=1000, workers=1, prompt_options=None, **overrides):
    """Translate `paragraphs` and return (output, stats, logs)."""
    logs = []
    kwargs = dict(
        paragraphs=paragraphs,
        source_language="English",
        target_language="Russian",
        model_name="test-model",
        llm_client=object(),
        max_tokens_per_chunk=max_tokens,
        parallel_workers=workers,
        prompt_options=prompt_options,
        log_callback=lambda key, msg: logs.append((key, msg)),
    )
    kwargs.update(overrides)
    out, stats, interrupted = await plain_pipeline.translate_paragraphs_plain(**kwargs)
    assert not interrupted
    return out, stats, logs


@pytest.mark.asyncio
async def test_disable_fallback_via_prompt_options(monkeypatch):
    """When disable_paragraph_fallback=True, a merged segment is accepted in 1 call."""
    fake = _recording_merging_llm()
    monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake)
    monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s: s)

    out, stats, logs = await _run(
        MERGED,
        prompt_options={"disable_paragraph_fallback": True},
    )

    # Exactly 1 LLM call made: no retry and no per-paragraph repair calls
    assert len(fake.calls) == 1
    assert stats.paragraph_count_mismatches == 1
    assert stats.paragraph_repair_failed == 0

    # Log recorded the accepted mismatch
    accepted_logs = [msg for key, msg in logs if key == "plain_text_paragraph_mismatch_accepted"]
    assert len(accepted_logs) == 1

    # Output contains the translated text; NO English source leftovers
    assert out == ["T::Alpha paragraph. Beta paragraph."]


@pytest.mark.asyncio
async def test_disable_fallback_via_env_var(monkeypatch):
    """When DISABLE_PARAGRAPH_FALLBACK=true in env, fallback is bypassed."""
    monkeypatch.setenv("DISABLE_PARAGRAPH_FALLBACK", "true")
    fake = _recording_merging_llm()
    monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake)
    monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s: s)

    out, stats, logs = await _run(THREE)

    # Exactly 1 LLM call made
    assert len(fake.calls) == 1
    assert stats.paragraph_count_mismatches == 1
    assert stats.paragraph_repair_failed == 0

    # Output is the single merged translation; no English leftovers
    assert out == ["T::First paragraph. Second paragraph. Third paragraph."]


@pytest.mark.asyncio
async def test_disable_fallback_preserves_empty_image_anchors(monkeypatch):
    """Empty slots (image anchors) are preserved in relative position."""
    monkeypatch.setenv("DISABLE_PARAGRAPH_FALLBACK", "true")
    fake = _recording_merging_llm()
    monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake)
    monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s: s)

    source = ["", "Chapter Title", "", "Body paragraph 1.", "Body paragraph 2."]
    out, stats, logs = await _run(source)

    # Exactly 1 call (empty paragraphs were excluded from chunk input)
    assert len(fake.calls) == 1

    # Empty slots stay empty; translated paragraphs land in order
    assert out[0] == ""
    assert out[2] == ""
    assert "T::Chapter Title" in out[1]
    assert "Body paragraph 1. Body paragraph 2." in out[1]
    assert len(out) == 3
    # No untranslated English text in any slot
    for slot in out:
        if slot:
            assert slot.startswith("T::")
