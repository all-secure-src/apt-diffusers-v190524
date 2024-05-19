from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["pipeline_omegapicasso"] = ["OmegaPicassoPipeline"]
    _import_structure["pipeline_omegapicasso_combined"] = [
        "OmegaPicassoCombinedPipeline",
        "OmegaPicassoImg2ImgCombinedPipeline",
        "OmegaPicassoInpaintCombinedPipeline",
    ]
    _import_structure["pipeline_omegapicasso_controlnet"] = ["OmegaPicassoControlnetPipeline"]
    _import_structure["pipeline_omegapicasso_controlnet_img2img"] = ["OmegaPicassoControlnetImg2ImgPipeline"]
    _import_structure["pipeline_omegapicasso_img2img"] = ["OmegaPicassoImg2ImgPipeline"]
    _import_structure["pipeline_omegapicasso_inpainting"] = ["OmegaPicassoInpaintPipeline"]
    _import_structure["pipeline_omegapicasso_prior"] = ["OmegaPicassoBaselinePipeline"]
    _import_structure["pipeline_omegapicasso_prior_emb2emb"] = ["OmegaPicassoBaselineEmb2EmbPipeline"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        from .pipeline_omegapicasso import OmegaPicassoPipeline
        from .pipeline_omegapicasso_combined import (
            OmegaPicassoCombinedPipeline,
            OmegaPicassoImg2ImgCombinedPipeline,
            OmegaPicassoInpaintCombinedPipeline,
        )
        from .pipeline_omegapicasso_controlnet import OmegaPicassoControlnetPipeline
        from .pipeline_omegapicasso_controlnet_img2img import OmegaPicassoControlnetImg2ImgPipeline
        from .pipeline_omegapicasso_img2img import OmegaPicassoImg2ImgPipeline
        from .pipeline_omegapicasso_inpainting import OmegaPicassoInpaintPipeline
        from .pipeline_omegapicasso_prior import OmegaPicassoBaselinePipeline
        from .pipeline_omegapicasso_prior_emb2emb import OmegaPicassoBaselineEmb2EmbPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
