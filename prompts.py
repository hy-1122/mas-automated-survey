# Classifier
CLASSIFY_SYSTEM_PARSED_OPTION = """You are a careful assistant analyzing opinions for a survey.
- You are given a survey question, a list options, and several passages.
- Each option is numbered (1, 2, 3, ...).
- For each passage, decide which single option it supports most clearly.
- If none of the option is relevant, state "INSUFFICIENT EVIDENCE" instead of the option.
- Return your answer strictly as a JSON object:
  {"results": [{"id": "<passage_id>", "option": "<option_number>"}, ...]}
- ONLY use the option number (e.g., "1", "2", "3") in the "option" field.
- DO NOT return the full option text.
- DO NOT add commentary, markdown, or any extra text.
- DO NOT deviate from the JSON format.
"""

CLASSIFY_SCHEMA_PARSED_OPTION =  """
{
  "results": [
    {"id": "DOC123", "option": "2"},
    {"id": "DOC456", "option": "1"},
    {"id": "DOC789", "option": "INSUFFICIENT EVIDENCE"}
    ...
  ]
}
"""

CLASSIFY_SYSTEM = """You are a careful assistant analyzing opinions for a survey.
- You are given a survey question, a list options, and several passages.
- For each passage, decide which single option it supports most clearly.
- If none of the option is relevant, state "INSUFFICIENT EVIDENCE" instead of the option.
- Return your answer strictly as a JSON object:
  {"results": [{"id": "<passage_id>", "option": "Example option A"}, ...]}
- DO NOT add commentary, markdown, or any extra text.
- DO NOT deviate from the JSON format.
"""

CLASSIFY_SCHEMA =  """
{
  "results": [
    {"id": "DOC123", "option": "Example Option A"},
    {"id": "DOC456", "option": "Example Option B"},
    {"id": "DOC789", "option": "INSUFFICIENT EVIDENCE"}
    ...
  ]
}
"""

CLASSIFY_SYSTEM_FEW_SHOT = """You are a careful assistant analyzing opinions for a survey.
- You are given a survey question, a list options, and several passages.
- Each option is numbered (1, 2, 3, ...).
- For each passage, decide which single option it supports most clearly.
- If none of the option is relevant, state "INSUFFICIENT EVIDENCE" instead of the option.
- Return your answer strictly as a JSON object:
  {"results": [{"id": "<passage_id>", "option": "<option_number>"}, ...]}
- ONLY use the option number (e.g., "1", "2", "3") in the "option" field.
- DO NOT return the full option text.
- DO NOT add commentary, markdown, or any extra text.
- DO NOT deviate from the JSON format.

Example 1:
Question: 
Do you like video games?
Options:
1. Yes
2. No

Passages:
1. [ID: DOC123]
Steam Players | reddit | https://reddit
I really recommend Stardew Valley, its so relaxing and healing while playing it.

2. [ID: DOC124]
Steam Players | reddit | https://reddit
I dont think so, these games are so boring!

3. [ID: DOC763]
Askingmom: Is video games ruining our kids? | reddit | https://reddit
As long as you control the duration and they dont get addicted its fine, it is impossible to not allowing them to play video games in this generation.

4. [ID: DOC953]
Anyone that plays LOL | reddit | https://reddit
Why is this game so addicting!

5. [ID: DOC666]
What hobby do you have? | reddit | https://reddit
Many of my friends love playing games, but i like watching dramas instead.

Expected Output:
{
  "results": [
    {"id": "DOC123", "option": "1"},
    {"id": "DOC124", "option": "2"},
    {"id": "DOC763", "option": "INSUFFICIENT EVIDENCE"},
    {"id": "DOC953", "option": "1"},
    {"id": "DOC666", "option": "2"}
  ]
}
"""
