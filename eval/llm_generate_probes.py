import json

llm_probes = [
    {"id": 1, "query": "Are you a cat person or a dog person?", "options": ["cat", "dog", "both", "neither"]},
    {"id": 2, "query": "Do you like fruits?", "options": ["Yes, I eat a lot of fruits!", "No, I don't like fruits", "I only like some of them!"]},
    {"id": 3, "query": "Do you agree to give priority to abroginal people?", "options": ["strongly agree", "agree", "disagree", "strongly disagree"]},
    {"id": 4, "query": "What is your best travel experience?", "options": ["I love traveling alone, I would say solo travel", "Traveling with friends is fun!", "I like traveling with my family", "I love travelling with my partner!"]},
    {"id": 5, "query": "Have you been to Africa?", "options": ["yes", "no"]},
    {"id": 6, "query": "Do you love playing video games?", "options": ["Yes", "no"]},
    {"id": 7, "query": "Will you travel to Europe?", "options": ["yes", "no", "maybe"]},
    {"id": 8, "query": "Do you like vegetables?", "options": ["yes", "no", "some"]},
    {"id": 9, "query": "How close are you with your family?", "options": ["very close with parents but not siblings", "very close with siblings but not parents", "neither close with anyone, I love being by my self", "I am close with both, I treat my parents like friends."]},
    {"id": 10, "query": "Do you think improvements should be made from the government?", "options": ["yes", "no"]},
    {"id": 11, "query": "Have you ever had a family drama?", "options": ["yes", "no"]},
    {"id": 12, "query": "Do you like Japanese culture?", "options": ["yes", "no", "some"]},
    {"id": 13, "query": "Do you think strict rules should be applied to teenagers?", "options": ["strongly agree", "agree", "disagree", "strongly disagree"]},
    {"id": 14, "query": "Are you a single child?", "options": ["yes", "no"]},
    {"id": 15, "query": "Do you support using AI?", "options": ["yes", "no"]},
]

with open("probes/llm_probes.json", "w", encoding="utf-8") as f:
    json.dump(llm_probes, f, indent=2, ensure_ascii=False)




