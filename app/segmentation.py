import json
from dataclasses import dataclass, asdict
from typing import List, Optional
import logging

HNS_PER_SECOND = 10_000_000  # Hundreds of nanoseconds units used by SDK


@dataclass
class Segment:
    id: int
    language: str
    start_hns: int
    end_hns: int

    @property
    def start_sec(self) -> float:
        return self.start_hns / HNS_PER_SECOND

    @property
    def end_sec(self) -> float:
        return self.end_hns / HNS_PER_SECOND

    def to_public_dict(self):
        d = asdict(self)
        d.update({"start_sec": self.start_sec, "end_sec": self.end_sec})
        return d


class SegmentBuilder:
    """Build segments based on language detection events.

    Starts a segment on first detection, closes a segment when language changes or at finalize.
    Optional minimum segment duration filter (segments shorter than threshold are dropped).
    """

    def __init__(
        self, min_duration_hns: int = 0, logger: Optional[logging.Logger] = None
    ):
        self._segments: List[Segment] = []
        self._current_lang: Optional[str] = None
        self._current_start: Optional[int] = None
        self._min_duration_hns = min_duration_hns
        self._log = logger or logging.getLogger(__name__)

    def on_detection(self, language: str, start_hns: int, end_hns: int):
        # First detection
        if self._current_lang is None:
            self._current_lang = language
            self._current_start = start_hns
            self._log.debug(f"Start first segment lang={language} start={start_hns}")
            return
        # Language switch
        if language != self._current_lang:
            self._log.debug(
                f"Language change {self._current_lang}->{language} at {start_hns}"
            )
            self._close(prev_end=start_hns)
            self._current_lang = language
            self._current_start = start_hns

    def finalize(self, final_end_hns: int):
        if self._current_lang is not None and self._current_start is not None:
            self._close(prev_end=final_end_hns)

    def _close(self, prev_end: int):
        if self._current_start is None:
            return
        duration = prev_end - self._current_start
        if duration < 0:
            self._log.warning("Negative duration encountered; skipping segment")
            return
        if duration < self._min_duration_hns:
            self._log.info(
                f"Dropping short segment lang={self._current_lang} dur_hns={duration} < {self._min_duration_hns}"
            )
            return
        seg_id = len(self._segments)
        self._segments.append(
            Segment(
                id=seg_id,
                language=self._current_lang or "unknown",
                start_hns=self._current_start or 0,
                end_hns=prev_end,
            )
        )
        self._log.debug(
            f"Closed segment id={seg_id} lang={self._current_lang} start={self._current_start} end={prev_end}"
        )

    def segments(self) -> List[Segment]:
        return self._segments

    def to_json(self, path: str, audio_file: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "audio": audio_file,
                    "segments": [s.to_public_dict() for s in self._segments],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
