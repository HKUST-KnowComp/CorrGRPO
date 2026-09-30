from dataclasses import dataclass

@dataclass(frozen=True)
class WorkItem:
    item_id: str
    prompt: str
    source_prompt: str = ""
    entry_point: str = ""
