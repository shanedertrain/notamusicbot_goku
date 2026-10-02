"""Unit tests for the RVC worker IPC protocol framing helpers.

The framing helpers are pure functions: request encoding and reply parsing
(including malformed replies and non-object JSON) are pinned here without any
process involvement.
"""

from __future__ import annotations

import json

import pytest

from notamusicbot.speech.voice_conversion.rvc import (
    WorkerError,
    encode_request,
    parse_reply,
)


def test_encode_request_is_one_json_line_with_id_and_action() -> None:
    line = encode_request(7, "convert", input_wav_path="/a.wav", pitch_shift=2)
    assert "\n" not in line
    payload = json.loads(line)
    assert payload == {
        "id": 7,
        "action": "convert",
        "input_wav_path": "/a.wav",
        "pitch_shift": 2,
    }


def test_encode_request_json_roundtrip_keeps_unicode() -> None:
    payload = json.loads(encode_request(1, "load-model", name="Goku"))
    assert payload["name"] == "Goku"


def test_parse_reply_accepts_object_reply() -> None:
    reply = parse_reply('{"id": 3, "status": "ok", "model": "Goku"}')
    assert reply["id"] == 3
    assert reply["status"] == "ok"


def test_parse_reply_rejects_malformed_json() -> None:
    with pytest.raises(WorkerError, match="malformed"):
        parse_reply("this is not json")


def test_parse_reply_rejects_non_object_reply() -> None:
    with pytest.raises(WorkerError, match="not a JSON object"):
        parse_reply("[1, 2, 3]")
