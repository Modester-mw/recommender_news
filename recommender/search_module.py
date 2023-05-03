
sample_data = [
    {"title": "Apple announces new iPhone", "content": "Apple has announced the latest iPhone..."},
    {"title": "Google releases new version of Android", "content": "Google has released a new version of Android..."},
    {"title": "Microsoft unveils new Surface devices", "content": "Microsoft has unveiled new Surface devices..."},
    {"title": "Tesla launches new electric vehicle", "content": "Tesla has launched a new electric vehicle..."},
    {"title": "Amazon reports record-breaking profits", "content": "Amazon has reported record-breaking profits..."},
]
def search(query):
    query = query.lower()
    results = [item for item in sample_data if query in item["title"].lower() or query in item["content"].lower()]
    return results
