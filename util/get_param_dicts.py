import json
import torch
import torch.nn as nn


def match_name_keywords(n: str, name_keywords: list):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def _get_name_keywords(args, attr_name, default_value=None):
    value = getattr(args, attr_name, default_value)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def get_param_dict(args, model_without_ddp: nn.Module):
    try:
        param_dict_type = args.param_dict_type
    except:
        param_dict_type = 'default'
    assert param_dict_type in ['default', 'ddetr_in_mmdet', 'large_wd']

    # by default
    # import pdb;pdb.set_trace()
    if param_dict_type == 'default':
        lr_special = getattr(args, "lr_special", None)
        lr_special_names = _get_name_keywords(args, "lr_special_names", [])
        if lr_special is not None and lr_special_names:
            param_dicts = [
                {
                    "params": [
                        p for n, p in model_without_ddp.named_parameters()
                        if "backbone" not in n
                        and not match_name_keywords(n, lr_special_names)
                        and p.requires_grad
                    ]
                },
                {
                    "params": [
                        p for n, p in model_without_ddp.named_parameters()
                        if "backbone" in n
                        and not match_name_keywords(n, lr_special_names)
                        and p.requires_grad
                    ],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [
                        p for n, p in model_without_ddp.named_parameters()
                        if match_name_keywords(n, lr_special_names)
                        and p.requires_grad
                    ],
                    "lr": lr_special,
                },
            ]
            return param_dicts
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            }
        ]
        return param_dicts

    if param_dict_type == 'ddetr_in_mmdet':
        lr_backbone_names = _get_name_keywords(args, "lr_backbone_names", [])
        lr_linear_proj_names = _get_name_keywords(args, "lr_linear_proj_names", [])
        lr_text_head_names = _get_name_keywords(args, "lr_text_head_names", ["feat_map", "text_adapter_ffn"])
        lr_text_head = getattr(args, "lr_text_head", None)
        lr_special = getattr(args, "lr_special", None)
        lr_special_names = _get_name_keywords(args, "lr_special_names", [])

        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                        if not match_name_keywords(n, lr_backbone_names)
                        and not match_name_keywords(n, lr_linear_proj_names)
                        and not (lr_text_head is not None and match_name_keywords(n, lr_text_head_names))
                        and not (lr_special is not None and match_name_keywords(n, lr_special_names))
                        and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() 
                        if match_name_keywords(n, lr_backbone_names) and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() 
                        if match_name_keywords(n, lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr_linear_proj_mult,
            }
        ]
        if lr_text_head is not None:
            param_dicts.append(
                {
                    "params": [
                        p for n, p in model_without_ddp.named_parameters()
                        if match_name_keywords(n, lr_text_head_names) and p.requires_grad
                    ],
                    "lr": lr_text_head,
                }
            )
        if lr_special is not None and lr_special_names:
            param_dicts.append(
                {
                    "params": [
                        p for n, p in model_without_ddp.named_parameters()
                        if match_name_keywords(n, lr_special_names) and p.requires_grad
                    ],
                    "lr": lr_special,
                }
            )
        return param_dicts

    if param_dict_type == 'large_wd':
        param_dicts = [
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, ['backbone']) and not match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, ['backbone']) and match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr_backbone,
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, ['backbone']) and not match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr_backbone,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, ['backbone']) and match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr,
                    "weight_decay": 0.0,
                }
            ]

        # print("param_dicts: {}".format(param_dicts))

    return param_dicts
