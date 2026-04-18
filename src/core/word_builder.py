"""
word_builder.py
Accumulates per-frame predictions into stable characters and words.

Decoupled from webcam / display code so it can be tested independently.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WordBuilderState:
    current_word: str = ""
    last_committed_char: Optional[str] = None
    new_char_committed: bool = False
    completed_word: str = ""           # The word just finished when space is detected; else ""
    current_segment: str = ""          # Letters accumulated since the last space


class WordBuilder:
    """
    Converts a stream of (possibly noisy) per-frame label predictions
    into a growing word string.

    A character is committed only when the same label appears in
    *buffer_size* consecutive frames with confidence ≥ *min_confidence*.

    Parameters
    ----------
    buffer_size : int
        How many consecutive frames must agree before a character is accepted.
    min_confidence : float
        Predictions below this threshold are ignored (buffer is cleared).
    """

    def __init__(self, buffer_size: int = 6, min_confidence: float = 0.8):
        self._buffer_size = buffer_size
        self._min_confidence = min_confidence
        self._buffer: deque = deque(maxlen=buffer_size)
        self.state = WordBuilderState()

    # ------------------------------------------------------------------
    def update(self, label: str, confidence: float) -> WordBuilderState:
        """
        Feed one prediction.  Returns the updated WordBuilderState.

        Call this once per frame; read `state.new_char_committed` to know
        whether a character was just added this frame.
        """
        self.state.new_char_committed = False
        self.state.completed_word = ""

        if confidence >= self._min_confidence:
            self._buffer.append(label)
        else:
            self._buffer.clear()
            return self.state

        if len(self._buffer) < self._buffer_size:
            return self.state

        # Buffer is full — pick the majority label
        stable_label = max(set(self._buffer), key=self._buffer.count)

        if stable_label == self.state.last_committed_char:
            return self.state   # same char as before, don't repeat

        # Commit the new character and clear the buffer so the next
        # letter starts accumulating from scratch rather than being
        # blocked by the just-committed label still sitting in the deque.
        self.state.last_committed_char = stable_label
        self._buffer.clear()

        if stable_label.lower() == "space":
            self.state.completed_word = self.state.current_segment
            self.state.current_segment = ""
            self.state.current_word += " "
        elif stable_label.lower() != "nothing":
            self.state.current_word += stable_label
            self.state.current_segment += stable_label

        self.state.new_char_committed = True
        return self.state

    def reset(self):
        """Clear the current word and buffer (e.g. when user presses 'r')."""
        self._buffer.clear()
        self.state = WordBuilderState()