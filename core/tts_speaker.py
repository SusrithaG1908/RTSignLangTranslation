"""
tts_speaker.py
Thread-safe, non-blocking text-to-speech.

Uses Windows SAPI via a PowerShell subprocess — one fresh process per
utterance.  This completely avoids the pyttsx3 singleton/runAndWait bug
where only the first letter is spoken.  Every committed character gets
its own clean speech process so nothing is ever silently dropped.
"""

import queue
import subprocess
import threading


_STOP_SENTINEL = object()


class TTSSpeaker:
    """
    Speaks text asynchronously.  Every speak() call is queued and spoken
    in order on a dedicated worker thread.

    Parameters
    ----------
    rate : int
        Speech rate (0–200, Windows SAPI scale; default 0 = normal).
    volume : float
        Volume in [0.0, 1.0].
    """

    def __init__(self, rate: int = 0, volume: float = 1.0):
        self._rate   = rate
        self._volume = int(volume * 100)
        self._queue: queue.Queue = queue.Queue()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def speak(self, text: str):
        """Enqueue *text* for speech.  Returns immediately."""
        if not text:
            return
        self._queue.put(text)

    def stop(self):
        """Shut down the worker thread gracefully."""
        self._queue.put(_STOP_SENTINEL)
        self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    def _worker(self):
        while True:
            item = self._queue.get()
            if item is _STOP_SENTINEL:
                break
            try:
                # Fresh PowerShell process per utterance — no engine state to corrupt
                script = (
                    f"Add-Type -AssemblyName System.Speech; "
                    f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Rate = {self._rate}; "
                    f"$s.Volume = {self._volume}; "
                    f"$s.Speak('{item}');"
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as exc:
                print(f"[TTSSpeaker] error: {exc}")