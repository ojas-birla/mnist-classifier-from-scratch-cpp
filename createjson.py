import json
import numpy as np

weights = {
    "W1": np.loadtxt("./data/W1.txt").tolist(),
    "b1": np.loadtxt("./data/b1.txt").tolist(),
    "W2": np.loadtxt("./data/W2.txt").tolist(),
    "b2": np.loadtxt("./data/b2.txt").tolist(),
}

with open("./web-page/weights.json", "w") as f: json.dump(weights, f)