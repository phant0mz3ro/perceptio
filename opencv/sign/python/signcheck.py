import json

with open("wlasl_raw/WLASL_v0.3.json", "r") as f:
    data = json.load(f)

all_glosses = [entry["gloss"].lower() for entry in data]

MY_SIGNS = [
    "hello", "my", "name", "yes", "no",
    "please", "thank you", "help", "finish", "want"
]

print("=== Sign Check ===")
for sign in MY_SIGNS:
    found = sign in all_glosses
    count = len(data[all_glosses.index(sign)]["instances"]) if found else 0
    print(f"  {sign:<15} {'✓  ' + str(count) + ' videos' if found else '✗  NOT FOUND'}")

print(f"\nTotal glosses in dataset: {len(all_glosses)}")