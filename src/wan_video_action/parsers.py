import argparse
import json
import os
import yaml

def merge_yaml_and_args(yaml_path, parser, args):
    # priority: CLI args > YAML config > parser defaults
    if not yaml_path or not os.path.exists(yaml_path):
        return args

    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_dict = yaml.safe_load(f) or {}

    cli_overrides = {}
    for key, value in vars(args).items():
        if value != parser.get_default(key):
            cli_overrides[key] = value

    for key, value in yaml_dict.items():
        if key not in cli_overrides:
            setattr(args, key, value)

    return args


def prepare_runtime_config(args):
    cfg = {}
    model_config_path = getattr(args, "model_config_path", "")
    if model_config_path and os.path.exists(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    enabled_mods = [m for m in ["dit", "vae", "image"] if getattr(args, f"enable_{m}", True)]
    
    text_mode = getattr(args, "text_mode", "emb")
    enabled_mods.append("text" if text_mode == "t5" else f"text:{text_mode}")
    
    action_mode = getattr(args, "action_mode", "none")
    if action_mode != "none":
        enabled_mods.append(f"action:{action_mode}")

    model_paths = getattr(args, "model_paths", "")
    yaml_modules_map = cfg.get("modules", {})
    
    paths_list = [
        os.path.join(model_paths, yaml_modules_map[m.split(":")[0]])
        for m in enabled_mods if m.split(":")[0] in yaml_modules_map
    ]

    tokenizer_path = None
    if text_mode == "t5" and model_paths:
        subdir = getattr(args, "tokenizer_subdir", cfg.get("tokenizer_subdir", "tokenizer"))
        tokenizer_path = os.path.join(model_paths, subdir)

    raw_keys = getattr(args, "data_file_keys", "image,video")
    data_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    return {
        "modules": enabled_mods,
        "model_paths_list": paths_list,
        "tokenizer_path": tokenizer_path,
        "data_file_keys": data_keys,
        "action_enabled": action_mode != "none"
    }


def add_dataset_base_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("dataset")
    group.add_argument("--dataset_base_path", type=str, default="", help="[REQUIRED] Base path of the dataset.")
    group.add_argument("--dataset_metadata_path", type=str, default=None, help="[OPTIONAL] Path to the metadata file of the dataset.")
    group.add_argument("--dataset_repeat", type=int, default=1, help="[TUNABLE] Number of times to repeat the dataset per epoch.")
    group.add_argument("--dataset_num_workers", type=int, default=0, help="[OPTIONAL] Number of workers for data loading.")
    group.add_argument("--data_file_keys", type=str, default="image,video", help="[OPTIONAL] Data file keys in the metadata. Comma-separated.")
    return parser


def add_video_size_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("video")
    group.add_argument("--height", type=int, default=None, help="[KEY] Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    group.add_argument("--width", type=int, default=None, help="[KEY] Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    group.add_argument("--max_pixels", type=int, default=1048576, help="[OPTIONAL] Maximum number of pixels per frame, used for dynamic resolution.")
    group.add_argument("--num_frames", type=int, default=81, help="[KEY] Number of frames per video. Frames are sampled from the video prefix.")
    group.add_argument("--resize_mode", type=str, default="fit", choices=["crop", "fit"], help="[OPTIONAL] Resize behavior: crop (center crop), fit (no crop).")
    group.add_argument("--num_history_frames", type=int, default=1, help="[KEY] Number of conditioning history frames. Must satisfy 1 <= num_history_frames < num_frames.")
    return parser


def add_model_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("model")
    group.add_argument("--model_paths", type=str, default=None, help="[REQUIRED] Paths to load models. In JSON format.")
    group.add_argument("--model_id_with_origin_paths", type=str, default=None, help="[OPTIONAL] Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    group.add_argument("--extra_inputs", type=str, default=None, help="[OPTIONAL] Additional model inputs, comma-separated.")
    group.add_argument("--fp8_models", type=str, default=None, help="[OPTIONAL] Models with FP8 precision, comma-separated.")
    group.add_argument("--offload_models", type=str, default=None, help="[OPTIONAL] Models with offload, comma-separated. Only used in splited training.")
    group.add_argument("--model_config_path", type=str, default="configs/model/wan2_1_fun_1_3b_inp.yaml", help="[KEY] Path to model config YAML.")
    group.add_argument("--initialize_model_on_cpu", action="store_true", default=False, help="[OPTIONAL] Whether to initialize models on CPU.")
    group.add_argument("--text_mode", type=str, default="emb", choices=["t5", "emb", "none"], help="[KEY] Execution mode for the text module: 't5' (full inference), 'emb' (use pre-extracted embeddings), 'none' (disabled).")
    group.add_argument("--enable_dit", action="store_true", default=True, help="[OPTIONAL] Enable DiT diffusion model.")
    group.add_argument("--enable_vae", action="store_true", default=True, help="[OPTIONAL] Enable VAE model.")
    group.add_argument("--enable_image", action="store_true", default=True, help="[OPTIONAL] Enable image encoder (CLIP).")
    group.add_argument("--action_mode", type=str, default="noise", choices=["noise", "adaln", "none"], help="[KEY] Action injection mode: noise (per-frame), adaln (global), none (disabled).")
    return parser


def add_action_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("action")
    group.add_argument("--action_type", type=str, choices=["joint_abs", "eef_abs", "joint_delta", "eef_delta"], default="eef_delta", help='[KEY] Action/state representation: joint/eef × abs/delta. (choices: "joint_abs", "eef_abs", "joint_delta", "eef_delta")')
    group.add_argument("--action_stat_path", type=str, default=None, help="[OPTIONAL] Path to robot normalization stats (stat.json).")
    group.add_argument("--action_dim", type=int, default=14, help="[OPTIONAL] Action dimension.")
    return parser


def add_training_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("training")
    group.add_argument("--learning_rate", type=float, default=1e-4, help="[TUNABLE] Learning rate.")
    group.add_argument("--num_epochs", type=int, default=1, help="[TUNABLE] Number of epochs.")
    group.add_argument("--trainable_models", type=str, default=None, help="[KEY] Models to train, e.g., dit, vae, text_encoder.")
    group.add_argument("--find_unused_parameters", action="store_true", default=False, help="[OPTIONAL] Whether to find unused parameters in DDP.")
    group.add_argument("--weight_decay", type=float, default=0.01, help="[TUNABLE] Weight decay.")
    group.add_argument("--task", type=str, default="sft", help="[OPTIONAL] Task type.")
    group.add_argument("--seed", type=int, default=42, help="[OPTIONAL] Random seed for python/numpy/torch.")
    group.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="[OPTIONAL] Mixed precision mode.")
    group.add_argument("--max_timestep_boundary", type=float, default=1.0, help="[OPTIONAL] Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    group.add_argument("--min_timestep_boundary", type=float, default=0.0, help="[OPTIONAL] Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    group.add_argument("--batch_size", type=int, default=1, help="[TUNABLE] Batch size per GPU.")
    return parser


def add_output_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("output")
    group.add_argument("--output_path", type=str, default="./models", help="[KEY] Output save path.")
    group.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help='[OPTIONAL] Remove prefix in ckpt. (default: "pipe.dit.")')
    group.add_argument("--save_steps", type=int, default=None, help="[OPTIONAL] Number of checkpoint saving intervals. If None, checkpoints will be saved every epoch.")
    group.add_argument("--ckpt_path", type=str, default=None, help="[OPTIONAL] Path to model checkpoint (.safetensors) used to initialize training weights (model-only resume).")
    group.add_argument("--resume_from", type=str, default=None, help="[OPTIONAL] Path to a checkpoint directory saved by accelerator (e.g., output_path/epoch-0).")
    return parser


def add_lora_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("lora")
    group.add_argument("--lora_base_model", type=str, default=None, help="[OPTIONAL] Which model LoRA is added to.")
    group.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="[OPTIONAL] Which layers LoRA is added to.")
    group.add_argument("--lora_rank", type=int, default=32, help="[TUNABLE] Rank of LoRA.")
    group.add_argument("--lora_checkpoint", type=str, default=None, help="[OPTIONAL] Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    group.add_argument("--preset_lora_path", type=str, default=None, help="[OPTIONAL] Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model.")
    group.add_argument("--preset_lora_model", type=str, default=None, help="[OPTIONAL] Which model the preset LoRA is fused to.")
    return parser


def add_gradient_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("gradient")
    group.add_argument("--use_gradient_checkpointing", action="store_true", default=False, help="[KEY] Whether to use gradient checkpointing.")
    group.add_argument("--use_gradient_checkpointing_offload", action="store_true", default=False, help="[KEY] Whether to offload gradient checkpointing to CPU memory.")
    group.add_argument("--gradient_accumulation_steps", type=int, default=1, help="[TUNABLE] Gradient accumulation steps.")
    group.add_argument("--max_grad_norm", type=float, default=0.5, help="[OPTIONAL] Maximum gradient norm for clipping. (default: 0.5)")
    return parser


def add_tracking_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("tracking")
    group.add_argument("--use_wandb", action="store_true", default=False, help="[OPTIONAL] Enable Weights & Biases tracking.")
    group.add_argument("--use_swanlab", action="store_true", default=False, help="[OPTIONAL] Enable SwanLab tracking.")
    group.add_argument("--swanlab_experiment_name", type=str, default=None, help="[OPTIONAL] SwanLab experiment name. Defaults to output_path.")
    return parser


def add_infer_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("infer")
    group.add_argument("--cfg_scale", type=float, default=5.0, help="[OPTIONAL] CFG scale for generation.")
    group.add_argument("--num_inference_steps", type=int, default=50, help="[OPTIONAL] Number of inference steps.")
    group.add_argument("--negative_prompt", type=str, default="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion", help="[OPTIONAL] Negative prompt for generation.")
    group.add_argument("--negative_prompt_emb", type=str, default=None, help="[OPTIONAL] Path to the pre-extracted negative prompt embedding.")
    group.add_argument("--quality", type=int, default=5, help="[OPTIONAL] Output video quality.")
    group.add_argument("--disable_chunk_infer", dest="chunk_infer", action="store_false", default=True, help="[OPTIONAL] Disable chunked inference with 81-frame segments.")
    group.add_argument("--fps", type=int, default=24, help="[OPTIONAL] Output video FPS.")
    group.add_argument("--disable_metrics", dest="enable_metrics", action="store_false", default=True, help="[OPTIONAL] Disable evaluation metrics.")
    return parser


def add_config_support(parser: argparse.ArgumentParser):
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file. CLI args override YAML config.")
    return parser


def add_general_config(parser: argparse.ArgumentParser):
    parser = add_config_support(parser)
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_tracking_config(parser)
    parser = add_action_config(parser)
    return parser
