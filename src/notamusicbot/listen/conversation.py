"""Listen-mode conversation state and text guards (plan section 5).

Pure string work: the rolling in-memory history, the chat content request,
reply trimming, the echo-text guard and the ASR noise filter. Transcripts are
never written to disk and nothing here imports discord.py.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import NamedTuple

from notamusicbot.content.base import ContentKind, ContentRequest
from notamusicbot.content.trim import split_sentences

BOT_SPEAKER = "Goku"

# Echo guard: ratio above this against the bot's last sentence means the bot
# heard itself.
ECHO_RATIO_THRESHOLD = 0.6

# Whisper-style hallucinations (compared lowercased after strip).
_NOISE_TRANSCRIPTS = frozenset({"thank you.", "thanks for watching.", "."})

_NON_ALNUM = re.compile(r"[^\w\s]")


@dataclass(slots=True)
class ConversationTurn:
    """One history entry; ``interrupted_after`` is set when barge-in cut a bot reply."""

    speaker: str
    text: str
    is_bot: bool = False
    interrupted_after: int | None = None

    def render(self) -> str:
        line = f"{self.speaker}: {self.text}"
        if self.interrupted_after is not None:
            line += f" [interrupted after sentence {self.interrupted_after}]"
        return line


class Conversation:
    """Rolling in-memory history of the last ``history_turns`` turns."""

    def __init__(self, history_turns: int = 8) -> None:
        self._max = max(0, history_turns)
        self._turns: list[ConversationTurn] = []

    def _append(self, turn: ConversationTurn) -> None:
        self._turns.append(turn)
        # Keep only the newest N (N == 0 keeps nothing).
        del self._turns[: max(0, len(self._turns) - self._max)]

    def add_user(self, speaker_name: str, text: str) -> None:
        self._append(ConversationTurn(speaker=speaker_name, text=text.strip()))

    def add_bot(self, text: str) -> None:
        self._append(ConversationTurn(speaker=BOT_SPEAKER, text=text.strip(), is_bot=True))

    def extend_bot(self, text: str) -> None:
        """Append spoken text to the newest turn when that turn is the bot's."""
        if self._turns and self._turns[-1].is_bot and text.strip():
            last = self._turns[-1]
            last.text = f"{last.text} {text.strip()}".strip()

    def mark_interrupted(self, sentence_index: int) -> None:
        """Flag the newest bot turn as cut off after ``sentence_index`` sentences."""
        for turn in reversed(self._turns):
            if turn.is_bot:
                turn.interrupted_after = sentence_index
                return

    def history(self) -> list[ConversationTurn]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()


def build_chat_request(
    persona_description: str,
    speaker_name: str,
    transcript_text: str,
    history: Sequence[ConversationTurn],
) -> ContentRequest:
    """Content request for one spoken reply; ``history`` is rendered oldest first."""
    return ContentRequest(
        kind=ContentKind.CHAT,
        persona_description=persona_description,
        requester_name=speaker_name,
        transcript_text=transcript_text,
        chat_history=tuple(t.render() for t in history),
    )


# --- spoken-text cleanup and chunking -----------------------------------------

_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\u2600-\u27bf\u2b00-\u2bff\u2300-\u23ff"
    "\ufe00-\ufe0f\u200d\u20e3\U000e0000-\U000e007f]"
)
_ASTERISK_SPAN = re.compile(r"\*+([^*\n]+?)\*+")
_PAREN_SPAN = re.compile(r"\([^()]*\)")
_LEADING_MARKUP = re.compile(r"^[ \t]*(?:#+[ \t]*|[-*+\u2022][ \t]+)", re.MULTILINE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")
_STAGE_WORDS = frozenset(
    {
        "laughs", "laughing", "sighs", "chuckles", "grins", "smiles", "giggles",
        "gasps", "cheers", "yells", "shouts", "whispers", "nods", "winks", "claps",
        "pauses", "groans", "screams", "cries", "sniffs", "coughs", "snickers",
        "smirks", "grunts", "hums", "sings", "flexes", "bows", "waves", "shrugs",
    }
)  # fmt: skip

_STOP_WORDS = frozenset(
    {
        "the", "a", "an", "of", "in", "on", "to", "and", "or", "but",
        "that", "is", "with", "for", "at", "by",
    }
)  # fmt: skip
_DASHES = "\u2014\u2013"
_MIN_CHUNK_WORDS = 3


def _asterisk_replacement(match: re.Match[str]) -> str:
    inner = match.group(1).strip()
    words = inner.split()
    if len(words) >= 2 or inner.lower().rstrip(".,!?") in _STAGE_WORDS:
        return " "  # stage direction like "*laughs*" or "*clears throat*"
    return inner  # plain emphasis


def clean_for_speech(text: str) -> str:
    """Strip emoji, markdown and stage directions so the TTS reads only words."""
    text = _EMOJI.sub(" ", text)
    text = _ASTERISK_SPAN.sub(_asterisk_replacement, text)
    text = _PAREN_SPAN.sub(" ", text)
    text = _LEADING_MARKUP.sub("", text)
    text = text.replace("`", "").replace("*", "").replace("__", "")
    text = " ".join(text.split())
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)


def _is_interjection(sentence: str) -> bool:
    return len(sentence.split()) <= 2 and sentence.rstrip()[-1:] in ("!", "?")


def split_reply_sentences(text: str) -> list[str]:
    """Sentence units; a short "Hey!" merges into the next sentence.

    Each returned unit counts as ONE real sentence toward a sentence cap. A
    trailing interjection with nothing after it stays a unit of its own.
    """
    units: list[str] = []
    pending: list[str] = []
    for sentence in split_sentences(text):
        if _is_interjection(sentence):
            pending.append(sentence)
            continue
        units.append(" ".join([*pending, sentence]))
        pending = []
    if pending:
        units.append(" ".join(pending))
    return units


def trim_reply(text: str, max_sentences: int) -> str:
    """Keep at most ``max_sentences`` sentences (``< 1`` returns the text stripped)."""
    text = text.strip()
    if max_sentences < 1:
        return text
    return " ".join(split_reply_sentences(text)[:max_sentences])


def _is_dash(token: str) -> bool:
    return token in ("\u2014", "\u2013")


def _count_words(tokens: list[str]) -> int:
    return sum(1 for t in tokens if not _is_dash(t))


def _last_word(token: str) -> str:
    return token.strip(".,!?;:\"'()" + _DASHES).lower()


def _boundary_strength(tokens: list[str], b: int) -> int:
    """Lower is a better place to cut after ``tokens[b - 1]``."""
    last = tokens[b - 1]
    if _is_dash(last) or last[-1] in _DASHES:
        return 1
    return {";": 2, ":": 3, ",": 4}.get(last[-1], 5)


_PLAIN_CUT = 5  # _boundary_strength of a cut that is not at a clause mark


def _chunk_end(tokens: list[str], start: int, cap: int) -> tuple[int, int]:
    """End index and boundary strength of the chunk starting at ``start``.

    Only valid when more than ``cap`` words remain. The decision reads at most
    ``cap + 1`` words past ``start`` (plus the tail check of a plain word cut,
    which needs ``cap + 3``), which is what lets StreamChunker decide early.
    """
    min_words = min(_MIN_CHUNK_WORDS, cap)
    cands: list[int] = []  # absolute boundaries b: chunk is tokens[start:b]
    for b in range(start + 1, len(tokens)):
        if _is_dash(tokens[b]):
            continue  # a lone dash stays glued to the word before it
        n = _count_words(tokens[start:b])
        if n > cap:
            break
        if n >= min_words:
            cands.append(b)
    if not cands:
        cands = [start + cap]
    non_stop = [b for b in cands if _last_word(tokens[b - 1]) not in _STOP_WORDS]
    pool = non_stop or cands
    best = min(_boundary_strength(tokens, b) for b in pool)
    end = max(b for b in pool if _boundary_strength(tokens, b) == best)
    if best == _PLAIN_CUT:
        # Plain word cut: avoid stranding a 1-2 word tail.
        tail = _count_words(tokens[end:])
        if 0 < tail < _MIN_CHUNK_WORDS:
            shifted = [b for b in pool if _count_words(tokens[b:]) >= _MIN_CHUNK_WORDS]
            if shifted:
                end = max(shifted)
    return end, best


def _split_long(sentence: str, first_cap: int, later_cap: int, is_first: bool) -> list[str]:
    """Split one sentence into chunks; the first chunk of the reply uses ``first_cap``."""
    tokens = sentence.split()
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        cap = first_cap if (is_first and not chunks) else later_cap
        rest = tokens[start:]
        if _count_words(rest) <= cap:
            chunks.append(" ".join(rest))
            break
        end, _ = _chunk_end(tokens, start, cap)
        chunks.append(" ".join(tokens[start:end]))
        start = end
    return chunks


def split_spoken_chunks(
    text: str,
    max_sentences: int,
    first_chunk_max_words: int = 10,
    later_chunk_max_words: int = 18,
) -> list[str]:
    """Clean ``text`` and cut it into short chunks for streamed speech.

    Interjections ("Hey!") merge into the next sentence and do not count toward
    ``max_sentences`` (``< 1`` keeps every sentence). Long sentences split at a
    dash, semicolon, colon, comma, then a plain word boundary; the first chunk
    holds at most ``first_chunk_max_words`` words, later ones
    ``later_chunk_max_words``.
    """
    units = split_reply_sentences(clean_for_speech(text))
    if max_sentences >= 1:
        units = units[:max_sentences]
    first_cap = max(1, first_chunk_max_words)
    later_cap = max(1, later_chunk_max_words)
    chunks: list[str] = []
    for unit in units:
        chunks.extend(_split_long(unit, first_cap, later_cap, is_first=not chunks))
    return chunks


class StreamChunk(NamedTuple):
    """One finished chunk and the 1-based index of the sentence it belongs to."""

    text: str
    sentence_index: int


# Terminator (+ optional closing quote) + whitespace, then the first character
# of the NEXT sentence. Requiring that character is what makes the boundary
# final: a later "...", "!!" or " ," could otherwise still rewrite it.
_STABLE_BOUNDARY = re.compile(r"[.!?][\"']?\s+(?=[^\W_])")


def _balanced(text: str) -> bool:
    """No open ``*`` span or ``(`` that clean_for_speech could still rewrite."""
    return text.count("*") % 2 == 0 and text.count("(") <= text.count(")")


class StreamChunker:
    """Incremental :func:`split_spoken_chunks` for text that arrives in deltas.

    ``feed`` returns the chunks that became FINAL with this delta; ``finish``
    flushes the rest. For any split of a text into deltas the joined output
    equals ``split_spoken_chunks(text, ...)`` joined. A chunk is final when

    * its sentence is complete: a terminator, whitespace and the first character
      of the next sentence have arrived (one delta of lookahead), or
    * it is cut from a still-open sentence that already holds enough words for
      the cut not to depend on what follows (``cap + 1`` words for a clause
      cut, ``cap + 3`` for a plain word cut).

    A short "Hey!" is held and merged into the next sentence. After
    ``max_sentences`` real sentences everything else is ignored. An unclosed
    ``(`` or ``*`` holds output until it closes or ``finish`` is called.
    """

    def __init__(
        self,
        max_sentences: int,
        first_chunk_max_words: int = 10,
        later_chunk_max_words: int = 18,
    ) -> None:
        self._max = max_sentences
        self._first_cap = max(1, first_chunk_max_words)
        self._later_cap = max(1, later_chunk_max_words)
        self._buf = ""
        self._pending: list[str] = []  # held interjections
        self._units = 0  # real sentence units closed
        self._chunks = 0  # chunks emitted so far
        self._consumed = 0  # tokens of the open unit already emitted
        self._done = False
        self._unit_text: dict[int, str] = {}
        self._partial: dict[int, list[str]] = {}

    @property
    def done(self) -> bool:
        """True once ``max_sentences`` sentences were taken (or after finish)."""
        return self._done

    def sentence_text(self, index: int) -> str:
        """The sentence ``index`` (1-based) as far as it is known."""
        if index in self._unit_text:
            return self._unit_text[index]
        return " ".join(self._partial.get(index, []))

    def feed(self, delta: str) -> list[str]:
        return [c.text for c in self.feed_chunks(delta)]

    def finish(self) -> list[str]:
        return [c.text for c in self.finish_chunks()]

    def feed_chunks(self, delta: str) -> list[StreamChunk]:
        if self._done or not delta:
            return []
        self._buf += delta
        out: list[StreamChunk] = []
        cut = self._last_boundary()
        if cut:
            prefix, self._buf = self._buf[:cut], self._buf[cut:]
            out.extend(self._take_sentences(prefix))
        if not self._done:
            out.extend(self._cut_open_sentence())
        return out

    def finish_chunks(self) -> list[StreamChunk]:
        if self._done:
            return []
        rest, self._buf = self._buf, ""
        out = self._take_sentences(rest)
        if not self._done and self._pending:
            out.extend(self._close_unit(" ".join(self._pending)))
            self._pending = []
        self._done = True
        return out

    def _last_boundary(self) -> int:
        for match in reversed(list(_STABLE_BOUNDARY.finditer(self._buf))):
            if _balanced(self._buf[: match.end()]):
                return match.end()
        return 0

    def _cap(self) -> int:
        return self._first_cap if self._chunks == 0 else self._later_cap

    def _emit(self, tokens: list[str], start: int, end: int) -> StreamChunk:
        text = " ".join(tokens[start:end])
        index = self._units + 1
        self._partial.setdefault(index, []).append(text)
        self._chunks += 1
        self._consumed = end
        return StreamChunk(text, index)

    def _take_sentences(self, text: str) -> list[StreamChunk]:
        out: list[StreamChunk] = []
        for sentence in split_sentences(clean_for_speech(text)):
            if self._done:
                break
            if _is_interjection(sentence):
                self._pending.append(sentence)
                continue
            unit = " ".join([*self._pending, sentence])
            self._pending = []
            out.extend(self._close_unit(unit))
        return out

    def _close_unit(self, unit: str) -> list[StreamChunk]:
        """Emit what is left of a complete sentence unit and count it."""
        tokens = unit.split()
        out: list[StreamChunk] = []
        start = self._consumed
        while start < len(tokens):
            cap = self._cap()
            if _count_words(tokens[start:]) <= cap:
                out.append(self._emit(tokens, start, len(tokens)))
                break
            end, _ = _chunk_end(tokens, start, cap)
            out.append(self._emit(tokens, start, end))
            start = end
        self._units += 1
        self._unit_text[self._units] = unit
        self._partial.pop(self._units, None)
        self._consumed = 0
        if self._max >= 1 and self._units >= self._max:
            self._done = True
        return out

    def _cut_open_sentence(self) -> list[StreamChunk]:
        """Emit leading chunks of the still-open sentence once they cannot change."""
        if not _balanced(self._buf):
            return []
        cleaned = clean_for_speech(self._buf)
        if len(split_sentences(cleaned)) > 1:
            return []  # an unresolved boundary inside; wait for it to settle
        # The last token may still grow or pick up punctuation: never cut on it.
        tokens = [*" ".join(self._pending).split(), *cleaned.split()[:-1]]
        out: list[StreamChunk] = []
        start = self._consumed
        while True:
            cap = self._cap()
            words = _count_words(tokens[start:])
            if words < cap + 1:
                break
            end, best = _chunk_end(tokens, start, cap)
            if best == _PLAIN_CUT and words < cap + 3:
                break  # the tail check of a plain cut still needs more words
            out.append(self._emit(tokens, start, end))
            start = end
        return out


def _normalise(text: str) -> str:
    return " ".join(_NON_ALNUM.sub("", text.lower()).split())


def echo_guard(transcript: str, last_bot_sentence: str) -> bool:
    """True if ``transcript`` looks like the bot hearing its own last sentence."""
    heard = _normalise(transcript)
    spoken = _normalise(last_bot_sentence)
    if not heard or not spoken:
        return False
    if heard in spoken:
        return True
    return SequenceMatcher(None, heard, spoken).ratio() > ECHO_RATIO_THRESHOLD


def is_noise_transcript(text: str) -> bool:
    """True for empty, one-word or known-hallucination transcripts."""
    stripped = text.strip()
    if len(stripped.split()) < 2:
        return True
    return stripped.lower() in _NOISE_TRANSCRIPTS
