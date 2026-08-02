"""Abstract base classes for loss models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LossResult:
    """Output of any loss model.

    Attributes:
        component: name of the component (e.g. "MOSFET", "Inductor")
        power_loss_W: total power loss in Watts
        sub_losses: breakdown dict (e.g. {"conduction": 21.4, "switching": 1.7})
        metadata: optional debug/design info
    """
    component: str
    power_loss_W: float
    sub_losses: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __str__(self):
        parts = [f"{self.component}: {self.power_loss_W:.3f} W"]
        for name, val in self.sub_losses.items():
            parts.append(f"  {name}: {val:.3f} W")
        return "\n".join(parts)


class LossModel(ABC):
    """Abstract base class for all loss models."""

    @abstractmethod
    def compute(self, *args, **kwargs) -> LossResult:
        """Compute loss and return a LossResult."""
        ...
