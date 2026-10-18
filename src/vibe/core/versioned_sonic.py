"""Visual actor with base-weight and experiment resume contracts."""
import torch
from orcs.core.sonic.model import SonicContractMixin
from rsl_rl.models import ExtractorSonicAdapterModel


class VersionedExtractorSonicAdapterModel(SonicContractMixin, ExtractorSonicAdapterModel):
    def __init__(self, *args, visual_contract_id, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('_vibe_visual_contract', torch.tensor(
            list(bytes.fromhex(visual_contract_id)), dtype=torch.uint8))

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if hasattr(self, '_vibe_visual_contract'):
            incoming = state_dict.get('_vibe_visual_contract')
            if incoming is None or not torch.equal(incoming.cpu(), self._vibe_visual_contract.cpu()):
                raise ValueError('refusing resume: visual dataset/camera/asset contract differs')
        return super().load_state_dict(state_dict, strict=strict, **kwargs)
