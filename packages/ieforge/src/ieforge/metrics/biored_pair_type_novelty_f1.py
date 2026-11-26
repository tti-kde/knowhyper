from ieforge.metrics.relation_f1 import RelationF1
from ieforge.registry import register


@register("metric", name="biored_pair_type_novelty_f1")
class BioREDPairTypeNoveltyF1(RelationF1):
    """Pair+type+novelty F1 using exact (head, tail, label) match."""

    def __init__(
        self,
        negative_label: str = "None",
        exclude_negative: bool = True,
        average: str = "micro",
        name: str = "biored_pair_type_novelty_f1",
    ) -> None:
        """Initialise the relation-style metric with BioRED defaults."""
        super().__init__(
            negative_label=negative_label,
            exclude_negative=exclude_negative,
            average=average,
            name=name,
        )
