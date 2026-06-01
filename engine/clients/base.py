from dataclasses import dataclass, field

@dataclass
class GenerationResult:
    content:       dict
    model:         str
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    raw_text:      str = ""
    error:         str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.content)