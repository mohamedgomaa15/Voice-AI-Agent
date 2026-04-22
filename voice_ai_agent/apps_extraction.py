import json

with open("data/apps.json", encoding="utf-8") as f:
    data = json.load(f)

apps = []

for path, app in data.items():
    name = app.get("Name")
    exec_cmd = app.get("Exec")

    if not name or not exec_cmd:
        continue

    aliases = []

    # add name
    aliases.append(name.lower())

    # add keywords if exist
    if "Keywords" in app:
        aliases.extend([k.lower() for k in app["Keywords"]])

    # remove duplicates
    aliases = list(set(aliases))

    apps.append({
        "name": name.lower(),
        "exec": exec_cmd,
        "aliases": aliases
    })

# save cleaned version
with open("data/clean_apps.json", "w") as f:
    json.dump(apps, f, indent=2)