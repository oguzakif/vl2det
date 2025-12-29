import functools
import pathlib
import warnings
from collections import defaultdict
from typing import Any
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def allowlist_checkpoint_globals(ckpt_path: Optional[str] = None) -> None:
    try:
        import torch.serialization
    except Exception:
        return

    allowlist = [functools.partial, pathlib.Path, pathlib.PosixPath, pathlib.WindowsPath, dict, defaultdict]
    allowlist.append(Any)

    try:
        import torch.optim
        import torch.optim.lr_scheduler
        import torch.optim.adamw as adamw
        from torch.nn.modules.container import Sequential, ModuleList
        from torch.nn.modules.conv import Conv2d
        from torch.nn.modules.batchnorm import BatchNorm2d
        from torch.nn.modules.activation import Hardswish, ReLU, Hardsigmoid, ReLU6
        from torch.nn.modules.pooling import AdaptiveAvgPool2d
        from torch.nn.modules.linear import Identity, Linear
        from torch.nn.modules.dropout import Dropout
        from torch.nn.modules.flatten import Flatten


        allowlist.extend(
            [
                torch.optim.AdamW,
                adamw.AdamW,
                torch.optim.lr_scheduler.StepLR,
                Sequential,
                Conv2d,
                BatchNorm2d,
                Hardswish,
                ReLU,
                AdaptiveAvgPool2d,
                Hardsigmoid,
                ModuleList,
                ReLU6,
                Identity,
                Linear,
                Dropout,
                Flatten,
            ]
        )
    except Exception:
        pass

    try:
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor
        from torchvision.transforms.functional import InterpolationMode

        allowlist.extend([Compose, Normalize, Resize, ToTensor, InterpolationMode])
    except Exception:
        pass

    try:
        from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation

        allowlist.extend([Conv2dNormActivation, SqueezeExcitation])
    except Exception:
        pass


    try:
        from torchvision.models.detection.ssd import SSD
        from torchvision.models.detection.ssdlite import SSDLiteHead, SSDLiteFeatureExtractorMobileNet, SSDLiteClassificationHead, SSDLiteRegressionHead
        from torchvision.models.mobilenetv3 import InvertedResidual
        from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
        from torchvision.models.detection.transform import GeneralizedRCNNTransform
        from torchvision.models.detection._utils import BoxCoder, SSDMatcher

        allowlist.extend([SSD, SSDLiteHead, SSDLiteFeatureExtractorMobileNet, InvertedResidual, DefaultBoxGenerator, SSDLiteClassificationHead, SSDLiteRegressionHead, GeneralizedRCNNTransform, BoxCoder, SSDMatcher])
    except Exception:
        pass

    try:
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode

        allowlist.extend([DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode])
    except Exception:
        pass

    try:
        from src.models.components.detection import (
            DetectionStudent,
            DetectionTeacher,
            DetectionTeacherStudent,
        )

        allowlist.extend([DetectionStudent, DetectionTeacherStudent, DetectionTeacher])
    except Exception:
        pass

    try:
        from src.models.components.campus import TeacherStudent, TeacherNet

        allowlist.extend([TeacherStudent, TeacherNet])
    except Exception:
        pass

    try:
        from open_clip.model import CLIP
        from open_clip.timm_model import TimmModel
        from open_clip.transformer import Transformer

        allowlist.extend([CLIP, TimmModel, Transformer])
    except Exception:
        pass

    try:
        from timm.models.convnext import ConvNeXt, ConvNeXtStage, ConvNeXtBlock
        from timm.layers.norm import LayerNorm2d, LayerNorm
        from timm.layers.mlp import Mlp
        from timm.layers.activations import GELU
        from timm.layers.drop import DropPath
        from timm.layers.classifier import NormMlpClassifierHead
        from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d

        allowlist.extend([ConvNeXt, ConvNeXtStage, ConvNeXtBlock, LayerNorm2d, LayerNorm, Mlp, GELU, DropPath, NormMlpClassifierHead, SelectAdaptivePool2d])
    except Exception:
        pass

    try:
        from src.data.yolo_detection_dataset import YoloDetectionDataset

        allowlist.append(YoloDetectionDataset)
    except Exception:
        pass

    if ckpt_path:
        try:
            import importlib

            get_unsafe_globals = getattr(
                torch.serialization, "get_unsafe_globals_in_checkpoint", None
            )
            if get_unsafe_globals is not None:
                unsafe_globals = get_unsafe_globals(ckpt_path)
                for name in unsafe_globals:
                    if not isinstance(name, str):
                        continue
                    module_name, _, attr = name.rpartition(".")
                    if not module_name:
                        continue
                    try:
                        module = importlib.import_module(module_name)
                    except Exception:
                        continue
                    obj = getattr(module, attr, None)
                    if obj is not None:
                        allowlist.append(obj)
        except Exception:
            pass

    torch.serialization.add_safe_globals(allowlist)


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
