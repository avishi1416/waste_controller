from fnmatch import fnmatch
from pathlib import Path

import yaml

from wastectl.config import ROOT

CLASS_MAP = ROOT / "configs" / "class_map.yaml"
DROP = "drop"


class ClassMap:
    def __init__(self, classes, path=CLASS_MAP):
        self.classes = list(classes)
        self.index = {c: i for i, c in enumerate(self.classes)}
        self.rules = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        for source, rules in self.rules.items():
            targets = [t for _, t in rules] if isinstance(rules, list) else rules.values()
            bad = set(targets) - set(self.classes) - {DROP}
            if bad:
                raise ValueError(f"class_map[{source}] maps to unknown classes {sorted(bad)}")

    def name(self, source, label):
        rules = self.rules[source]
        if isinstance(rules, list):
            for pattern, target in rules:
                if fnmatch(label, pattern):
                    return target
        elif label in rules:
            return rules[label]
        raise KeyError(f"unmapped {source} label {label!r}: add it to configs/class_map.yaml")

    def id(self, source, label):
        """Class index, or None when the label is mapped to `drop`."""
        name = self.name(source, label)
        return None if name == DROP else self.index[name]
