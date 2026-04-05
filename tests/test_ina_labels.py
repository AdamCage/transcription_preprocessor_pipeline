"""inaSpeechSegmenter label normalization (female/male -> speech for ASR)."""

from audio_asr_pipeline.segmenters import _map_ina_label, _parse_ina_result


def test_map_ina_gender_labels_are_speech() -> None:
    assert _map_ina_label("female") == "speech"
    assert _map_ina_label("male") == "speech"
    assert _map_ina_label("speech") == "speech"


def test_parse_ina_female_male_produces_speech_segments() -> None:
    raw = [("female", 0.0, 1.5), ("male", 1.5, 3.0), ("noise", 4.0, 5.0)]
    segs = _parse_ina_result(raw, duration_sec=10.0)
    by_label = {s.label for s in segs}
    assert "speech" in by_label
    assert segs[0].label == "speech" and segs[1].label == "speech"
    assert segs[2].label == "noise"
