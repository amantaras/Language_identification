import json
from dataclasses import dataclass, asdict
from typing import List, Optional

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
    """Builds segments based on language detection events.

    Adds a new segment when language changes. Keeps previous segment end = current start.
    """
    def __init__(self):
        self._segments: List[Segment] = []
        self._current_lang: Optional[str] = None
        self._current_start: Optional[int] = None

    def on_detection(self, language: str, start_hns: int, end_hns: int):
        # Start of first segment
        if self._current_lang is None:
            self._current_lang = language
            self._current_start = start_hns
            return
        # If language changed, close previous and start new
        if language != self._current_lang:
            self._close(prev_end=start_hns)
            self._current_lang = language
            self._current_start = start_hns
        # Extend current segment end dynamically with end_hns (optional)

    def finalize(self, final_end_hns: int):
        if self._current_lang is not None and self._current_start is not None:
            self._close(prev_end=final_end_hns)

    def _close(self, prev_end: int):
        seg_id = len(self._segments)
        self._segments.append(Segment(
            id=seg_id,
            language=self._current_lang or "unknown",
            start_hns=self._current_start or 0,
            end_hns=prev_end
        ))

    def segments(self) -> List[Segment]:
        return self._segments

    def to_json(self, path: str, audio_file: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "audio": audio_file,
                "segments": [s.to_public_dict() for s in self._segments]
            }, f, ensure_ascii=False, indent=2)
