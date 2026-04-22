"""Fast sanity tests for extract_pairs() — no GPU / no model load.

Run BEFORE the expensive generation step to verify the parser handles realistic
teacher outputs (markdown fences, preamble prose, nested strings, truncation).

    python synth/test_extract.py

Exits non-zero on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_instructions import extract_pairs  # noqa: E402


# Sample 1: clean JSON (happy path)
SAMPLE_CLEAN = """\
{"pairs": [
  {"instruction": "What is Vx?", "response": "Vx is the best angle of climb speed, giving the greatest altitude gain per horizontal distance. It is used when obstacle clearance after takeoff is needed. Refer to the POH for the specific Vx of your aircraft."},
  {"instruction": "Explain the impossible turn.", "response": "The impossible turn refers to attempting a 180-degree return to the runway following an engine failure shortly after takeoff. It is widely discouraged below 400-800 feet AGL due to altitude loss, stall risk, and misjudgment of wind. FAA guidance typically recommends landing straight ahead or slightly off-heading, accepting a forced landing rather than attempting a turn that often ends in stall/spin."}
]}
"""

# Sample 2: wrapped in markdown fence (very common with chat models)
SAMPLE_FENCED = """\
Here are the Q&A pairs:

```json
{"pairs": [
  {"instruction": "What does MAYDAY indicate?", "response": "MAYDAY is the international distress signal indicating a life-threatening emergency. It is transmitted three times (\\"MAYDAY MAYDAY MAYDAY\\") followed by aircraft identification, position, nature of emergency, and intentions. Use it when immediate assistance is required."}
]}
```

Let me know if you need more!
"""

# Sample 3: prose preamble (model forgot strict JSON)
SAMPLE_WITH_PROSE = """\
Sure! Below are 2 pairs:
{"pairs": [{"instruction": "What is PAVE?", "response": "PAVE is a risk assessment mnemonic covering Pilot, Aircraft, enVironment, and External pressures. It is used during preflight planning to evaluate risks systematically. It is part of the broader ADM (aeronautical decision-making) framework taught per FAA standards."}, {"instruction": "What is IMSAFE?", "response": "IMSAFE is a pilot self-assessment checklist: Illness, Medication, Stress, Alcohol, Fatigue, Emotion/Eating. It is intended to be run before every flight to catch factors that could impair performance. See FAA AC 60-22."}]}

Hope this helps!
"""

# Sample 4: nested braces inside a string value (previously broken regex case)
SAMPLE_NESTED_BRACES = """\
{"pairs": [
  {"instruction": "Describe the emergency JSON schema.", "response": "Some systems log emergencies as JSON objects with fields like {severity, timestamp, location}. However for FAA reporting you should follow the NASA ASRS form rather than any informal JSON structure. The key point is factual reporting, not schema."}
]}
"""

# Sample 5: completely broken output (should return [])
SAMPLE_BROKEN = "I don't have enough information to generate training pairs for this topic."

# Sample 6: empty
SAMPLE_EMPTY = ""

# Sample 7: truncated (model hit max_new_tokens) — should return 0 because JSON not closed
SAMPLE_TRUNCATED = """\
{"pairs": [
  {"instruction": "What is the best glide speed?", "response": "Best glide speed is the airspeed that gives the maximum lift-to-drag ratio, providing the greatest distance per altitude lost in a no-thrust glide. It is listed in the POH as Vbg and typically occurs at a specific gross weight. For example, for a typical C172, it is around 68 KIAS at max gross weight, but actual values vary by aircraft and loading."},
  {"instruction": "Explain the four forces", "response": "The four forces acting on an airplane in flight are lift, weight, thrust, and drag. In steady level flight, lift"""


def check(name: str, pairs: list[dict], expected_min: int, expected_max: int | None = None) -> bool:
    lo, hi = expected_min, (expected_max if expected_max is not None else expected_min)
    n = len(pairs)
    ok = lo <= n <= hi
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got {n} pairs (expected {lo}..{hi})")
    if not ok:
        print(f"         first pair: {pairs[0] if pairs else None}")
    return ok


def main() -> int:
    print("=== extract_pairs() smoke tests ===")
    results: list[bool] = []

    pairs = extract_pairs(SAMPLE_CLEAN)
    results.append(check("clean JSON", pairs, 2))

    pairs = extract_pairs(SAMPLE_FENCED)
    results.append(check("markdown-fenced", pairs, 1))

    pairs = extract_pairs(SAMPLE_WITH_PROSE)
    results.append(check("prose preamble", pairs, 2))

    pairs = extract_pairs(SAMPLE_NESTED_BRACES)
    results.append(check("nested braces in string", pairs, 1))

    pairs = extract_pairs(SAMPLE_BROKEN)
    results.append(check("broken output → empty", pairs, 0, 0))

    pairs = extract_pairs(SAMPLE_EMPTY)
    results.append(check("empty string → empty", pairs, 0, 0))

    pairs = extract_pairs(SAMPLE_TRUNCATED)
    # Truncated may yield 1 complete pair if parser recovers partial; accept 0 or 1.
    results.append(check("truncated → degrade gracefully", pairs, 0, 1))

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n=== {n_pass}/{n_total} passed ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
