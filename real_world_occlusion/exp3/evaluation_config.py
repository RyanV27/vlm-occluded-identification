# Items and their aliases
ALIASES = {
    "bottle": ["thermos", "flask"],
    "bowl": [],
    "can": ["canned"],
    "cereal box": ["box of cereal", "box of cheerios", "cheerios cereal", "cheerios box", "cereal carton"],
    "clock": ["timepiece"],
    "fork": [],
    "mug": ["cup"],
    "peanut butter jar": ["jar of peanut butter", "jar of Jif creamy peanut butter", "jar of Jif peanut butter"],
    "pencil": [],
    "scissors": ["shear"],
    "spatula": ["turner", "flipper"],
    "spoon": ["tablespoon", "teaspoon"],
    "tissue box": ["box of tissues", "tissue dispenser", "tissue container", "paper towel box", "napkin box"],
    "toothpaste": ["tube of Colgate", "dental cream", "dentifrice"],
    "whisk": ["beater", "mixer"]
}


# Prompt templates
comparison_prompt = '''You are a strict classifier.

Task:
Decide whether the TARGET_OBJECT is mentioned in the SENTENCE, either:
1) explicitly (same words), OR
2) implicitly via a clear alternative reference, including:
   - synonyms or near-synonyms,
   - common rephrasings,
   - brand/product instance that unambiguously implies the object,
   - plural/singular, spelling variants, or minor phrasing differences.

Do NOT count:
- metaphorical/idiomatic uses,
- unrelated homonyms,
- vague containers ("box", "container") unless clearly specified as the target object type,
- situations where the reference could reasonably be something else (when in doubt, answer "no").

Before answering, silently determine:
- what phrase in the sentence (if any) refers to the TARGET_OBJECT,
- whether it is literal and unambiguous.
Then output only "yes" or "no".

Output format:
Return ONLY one token: "yes" or "no".
No punctuation. No explanation.

TARGET_OBJECT: {obj}
SENTENCE: {generation}
'''