import json
import os
from datetime import datetime
from typing import List, Dict, Any


def parse_time(ts: str) -> datetime:
    # ISO 8601 with timezone is supported by fromisoformat
    return datetime.fromisoformat(ts)


class UGCDataset:
    """
    CoX-3D Dataset: {text, image(optional), timestamp}
    This is the *constructed* dataset artifact (Algorithm 1 output).
    """

    def __init__(self, path_jsonl: str, image_root: str):
        self.path_jsonl = path_jsonl
        self.image_root = image_root
        self.items: List[Dict[str, Any]] = []

        with open(path_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                obj["timestamp"] = parse_time(obj["timestamp"])
                # normalize image path
                imgp = obj.get("image_path", "")
                if imgp:
                    obj["image_path"] = os.path.join(image_root, imgp)
                self.items.append(obj)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]
