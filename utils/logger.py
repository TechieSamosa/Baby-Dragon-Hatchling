import json

def load_logs(log_file):
    """
    Loads JSONL log file into a dictionary grouped by model.
    """
    data = {}
    try:
        with open(log_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    model = entry["model"]
                    if model not in data:
                        data[model] = {"steps": [], "losses": []}
                    data[model]["steps"].append(entry["step"])
                    data[model]["losses"].append(entry["loss"])
    except FileNotFoundError:
        pass
    return data
