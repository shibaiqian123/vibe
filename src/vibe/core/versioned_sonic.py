"""Vision actor retaining the ORCS base-weight resume contract."""
from orcs.core.sonic.model import SonicContractMixin
from rsl_rl.models import ExtractorSonicAdapterModel


class VersionedExtractorSonicAdapterModel(SonicContractMixin, ExtractorSonicAdapterModel):
    pass
