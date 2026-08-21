"""Shared canary protocol constants/helpers (r0b0bench protocol, rc2).

JSON canary probe question — FIXED 2026-08-21:
  Previously: "Return one compact JSON object with keys alpha and beta,
  set to integers 2 and 3. No prose." — thinking-mode models spent the
  budget on reasoning before emitting JSON (finish=length, empty content),
  requiring a known-artifact exclusion in canary.py. The question now
  explicitly forbids thinking AND tells the model to output the JSON
  immediately, and the scoring accepts the JSON anywhere in the visible
  content (some models still inline short reasoning before the object).
  The artifact exclusion is removed once the probe is validated to finish
  with content on the model classes in the qualification matrix.
"""
JSON_CANARY_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Do not think or reason at all. Do not write any prose. "
            "Immediately output ONLY this exact JSON object, nothing else:\n"
            '{"alpha": 2, "beta": 3}'
        ),
    }
]
JSON_CANARY_EXPECT = {"alpha": 2, "beta": 3}
