"""Prompt templates for teacher-model synthetic Q&A generation."""

STUDENT_SYSTEM_PROMPT = (
    "You are an aviation safety assistant specialized in emergency procedures. "
    "Provide accurate, conservative, procedure-oriented guidance based on FAA "
    "standards (FAR/AIM, Pilot's Handbook of Aeronautical Knowledge, Airplane "
    "Flying Handbook). Always note when relevant: this is educational material; "
    "actual flight decisions require the aircraft's POH, ATC authority, and a "
    "certified flight instructor."
)

TEACHER_SYSTEM_PROMPT = (
    "You generate training examples for fine-tuning a small aviation assistant. "
    "You produce realistic, technically accurate Q&A pairs grounded in standard "
    "aviation knowledge (FAA FAR/AIM, POH emergency procedures, ACS standards). "
    "If uncertain about a specific detail, prefer conservative, textbook answers "
    "and omit uncertain specifics rather than inventing them. Output strict JSON."
)

GENERATION_PROMPT_TEMPLATE = """\
Generate 5 diverse, realistic Q&A pairs on the following aviation emergency topic and angle.

Requirements:
- Vary question styles: scenario-based ("You are at 5000ft when..."), procedural ("What is the checklist for..."), memory-aid ("Explain the ABCDE mnemonic..."), regulatory ("What does FAR 91.3 say about..."), communication format ("Give the MAYDAY call for..."), and decision/judgment ("When should you...").
- Answers must be technically correct per FAA standard practice; if type-specific, note the assumption (e.g., "For a typical single-engine piston like a C172...").
- Length: questions 10-40 words, answers 60-350 words.
- Do NOT invent aircraft-specific numbers. When a specific V-speed or limit is needed, say "refer to the POH for exact value" unless it is a widely-taught standard (e.g., 10,000ft cabin altitude oxygen requirement, 7700 emergency squawk).
- Every answer should either include or imply the safety caveat that real-world decisions require POH, ATC, and CFI authority.

Topic: {topic}
Angle: {angle}

Output strict JSON only, no prose before or after:
{{"pairs": [{{"instruction": "...", "response": "..."}}, ...]}}
"""

FREEFORM_PROMPT_TEMPLATE = """\
Generate {n} diverse Q&A pairs about general aviation emergency principles and airmanship.
These should NOT duplicate narrow topics like engine failure or decompression; they should
cover cross-cutting themes such as:
- Aeronautical decision-making (ADM), risk management (PAVE, IMSAFE, 5P)
- Cockpit resource management in emergencies
- Emergency authority and PIC responsibility
- Priorities: aviate, navigate, communicate
- Declaring an emergency: when and how
- Post-incident reporting and NASA ASRS

Technical accuracy per FAA standards. Output strict JSON only:
{{"pairs": [{{"instruction": "...", "response": "..."}}, ...]}}
"""
