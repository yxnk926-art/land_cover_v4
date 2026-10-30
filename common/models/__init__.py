# -*- coding: utf-8 -*-
"""九种模型的统一工厂。

``unet_resnet``、``unetpp_resnet``、``deeplabv3plus``+``DEEPLAB_BACKBONE``、
``segformer``、``swin_upernet``、``segmenter``、``mobilenetv3_lraspp``、``pidnet`` 和 ``ddrnet``。
SegFormer 用 ``SEGFORMER_VARIANT`` 选 B0 或 B2。
"""


def build_model(cfg):
    if not hasattr(cfg, "MODEL_NAME"):
        raise ValueError("配置必须显式声明 MODEL_NAME")
    name = cfg.MODEL_NAME

    if name == "deeplabv3plus":
        from .deeplabv3plus import DeepLabV3Plus
        return DeepLabV3Plus(
            num_classes=cfg.NUM_CLASSES,
            in_channels=getattr(cfg, "IN_CHANNELS", 3),
            width_mult=getattr(cfg, "MOBILENET_WIDTH_MULT", 1.0),
            output_stride=getattr(cfg, "OUTPUT_STRIDE", 16),
            aspp_channels=getattr(cfg, "ASPP_CHANNELS", 256),
            decoder_channels=getattr(cfg, "DECODER_CHANNELS", 256),
            low_channels=getattr(cfg, "LOW_LEVEL_CHANNELS", 48),
            aspp_rates=getattr(cfg, "ASPP_RATES", None),
            dropout=getattr(cfg, "MODEL_DROPOUT", 0.1),
            backbone=getattr(cfg, "DEEPLAB_BACKBONE", "mobilenetv3"),
            mobilenet_bn_eps=getattr(cfg, "MOBILENET_BN_EPS", 1e-3),
        )

    if name in ("unet_resnet", "unetpp_resnet"):
        from .resnet_unet import ResNetUNet, ResNetUNetPlusPlus
        model_class = ResNetUNet if name == "unet_resnet" else ResNetUNetPlusPlus
        return model_class(
            num_classes=cfg.NUM_CLASSES,
            in_channels=getattr(cfg, "IN_CHANNELS", 3),
            encoder=getattr(cfg, "ENCODER_NAME", "resnet50"),
            decoder_channels=getattr(cfg, "UNET_DECODER_CHANNELS",
                                     (256, 128, 64, 32, 16)),
        )

    if name == "swin_upernet":
        from .upernet import SwinUperNet
        return SwinUperNet(
            num_classes=cfg.NUM_CLASSES,
            in_channels=getattr(cfg, "IN_CHANNELS", 3),
            variant=getattr(cfg, "SWIN_VARIANT", "tiny"),
            window_size=getattr(cfg, "SWIN_WINDOW_SIZE", 7),
            drop_path_rate=getattr(cfg, "DROP_PATH_RATE", 0.3),
            channels=getattr(cfg, "UPERNET_CHANNELS", 512),
            pool_scales=getattr(cfg, "UPERNET_POOL_SCALES", (1, 2, 3, 6)),
            dropout=getattr(cfg, "MODEL_DROPOUT", 0.1),
            auxiliary=getattr(cfg, "UPERNET_AUXILIARY", True),
            aux_channels=getattr(cfg, "AUX_CHANNELS", 256),
        )

    if name == "segmenter":
        from .segmenter import Segmenter
        return Segmenter(
            num_classes=cfg.NUM_CLASSES,
            in_channels=getattr(cfg, "IN_CHANNELS", 3),
            variant=getattr(cfg, "SEGMENTER_VARIANT", "s16"),
            decoder_layers=getattr(cfg, "SEGMENTER_DECODER_LAYERS", 2),
            dropout=getattr(cfg, "MODEL_DROPOUT", 0.1),
            drop_path_rate=getattr(cfg, "DROP_PATH_RATE", 0.1),
        )

    if name == "ddrnet":
        from .ddrnet import DDRNet
        return DDRNet(
            num_classes=cfg.NUM_CLASSES,
            in_channels=getattr(cfg, "IN_CHANNELS", 3),
            variant=getattr(cfg, "DDRNET_VARIANT", "23_slim"),
            augment=getattr(cfg, "DDRNET_AUXILIARY", True),
        )

    if name == "mobilenetv3_lraspp":
        from .mobilenetv3_lraspp import MobileNetV3LRASPP
        return MobileNetV3LRASPP(
            num_classes=cfg.NUM_CLASSES,
            in_channels=getattr(cfg, "IN_CHANNELS", 3),
            width_mult=getattr(cfg, "MOBILENET_WIDTH_MULT", 1.0),
            output_stride=getattr(cfg, "OUTPUT_STRIDE", 16),
            head_channels=getattr(cfg, "LRASPP_CHANNELS", 128),
            dropout=getattr(cfg, "MODEL_DROPOUT", 0.0),
            context_pool=getattr(cfg, "LRASPP_CONTEXT_POOL", "adaptive"),
            reduced_tail=getattr(cfg, "MOBILENET_REDUCED_TAIL", True),
            norm_eps=getattr(cfg, "MOBILENET_BN_EPS", 1e-3),
        )

    if name == "segformer":
        from .segformer import SegFormer
        return SegFormer(
            num_classes=cfg.NUM_CLASSES,
            in_channels=getattr(cfg, "IN_CHANNELS", 3),
            variant=getattr(cfg, "SEGFORMER_VARIANT", "b0"),
            decoder_dim=getattr(cfg, "SEGFORMER_DECODER_DIM", None),
            dropout=getattr(cfg, "MODEL_DROPOUT", 0.1),
            drop_path_rate=getattr(cfg, "DROP_PATH_RATE", 0.1),
        )

    if name == "pidnet":
        from .pidnet import PIDNet
        return PIDNet(
            num_classes=cfg.NUM_CLASSES,
            in_channels=getattr(cfg, "IN_CHANNELS", 3),
            variant=getattr(cfg, "PIDNET_VARIANT", "s"),
            augment=getattr(cfg, "PIDNET_AUXILIARY", True),
        )

    choices = (
        "unet_resnet, unetpp_resnet, deeplabv3plus, segformer, swin_upernet, segmenter, "
        "mobilenetv3_lraspp, pidnet, ddrnet"
    )
    raise ValueError(f"未知 MODEL_NAME: {name!r}（可选: {choices}）")
