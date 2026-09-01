import torch
from tqdm import tqdm
from typing import Optional, Union
from einops import rearrange

from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.diffusion.base_pipeline import PipelineUnit
from diffsynth.core.device.npu_compatible_device import get_device_type
from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

from ..models.wan_video_action_encoder import WanVideoActionEncoder
from ..models.wan_video_vae import apply_wan_vae_compat


ACTION_INJECTION_MODES = {"adaln", "noise", "none"}


def _validate_action_injection_mode(action_injection_mode: str) -> str:
    if action_injection_mode not in ACTION_INJECTION_MODES:
        expected = ", ".join(sorted(ACTION_INJECTION_MODES))
        raise ValueError(
            f"Unsupported action injection mode '{action_injection_mode}'. "
            f"Expected one of: {expected}."
        )
    return action_injection_mode


def _prepare_history_condition_latents(
    self: WanVideoPipeline,
    inputs_shared: dict,
    *,
    use_history_condition_noise_in_inference: bool,
):
    first_frame_latents = inputs_shared.get("first_frame_latents")
    if first_frame_latents is None:
        return None, 0
    latents = inputs_shared.get("latents")
    if latents is None:
        return None, 0

    if first_frame_latents.ndim == 4:
        first_frame_latents = first_frame_latents.unsqueeze(0)

    history_t = min(int(first_frame_latents.shape[2]), int(latents.shape[2]))
    if history_t <= 0:
        return None, 0

    conditioning_latents = first_frame_latents[:, :, :history_t].clone()
    inputs_shared["latents"][:, :, :history_t] = conditioning_latents

    if (
        use_history_condition_noise_in_inference
        and getattr(self, "action_injection_mode", "none") == "adaln"
        and history_t > 1
    ):
        noise = inputs_shared.get("noise")
        if not isinstance(noise, torch.Tensor):
            raise RuntimeError("Expected `noise` tensor for history-conditioned inference, but it was missing.")
        small_timestep_idx = max(0, len(self.scheduler.timesteps) - 50)
        small_timestep = self.scheduler.timesteps[small_timestep_idx].unsqueeze(0).to(
            dtype=self.torch_dtype,
            device=self.device,
        )
        conditioning_latents[:, :, 1:history_t] = self.scheduler.add_noise(
            conditioning_latents[:, :, 1:history_t],
            noise[:, :, 1:history_t],
            small_timestep,
        )
        inputs_shared["latents"][:, :, 1:history_t] = conditioning_latents[:, :, 1:history_t]
    return conditioning_latents, history_t


def _restore_history_condition_latents(
    inputs_shared: dict,
    *,
    conditioning_latents: Optional[torch.Tensor],
    history_t: int,
) -> None:
    if conditioning_latents is None or history_t <= 0:
        return
    inputs_shared["latents"][:, :, :history_t] = conditioning_latents[:, :, :history_t]


def _build_wan2_action_units(pipe: WanVideoPipeline):
    selected = [
        unit for unit in pipe.units
        if unit.__class__.__name__ in {
            "WanVideoUnit_ShapeChecker",
            "WanVideoUnit_NoiseInitializer",
        }
    ]
    selected.append(WanVideoUnit_InputVideoEmbedder())
    selected.append(WanVideoUnit_ImageEmbedderFused())
    selected.append(WanVideoUnit_ActionEmbedder())
    return selected


def _install_wan_video_action_call(pipeline: WanVideoPipeline) -> None:
    @torch.no_grad()
    def __call__(
        self: WanVideoPipeline,
        input_video: Optional[torch.Tensor] = None,
        denoising_strength: Optional[float] = 1.0,
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: int = 81,
        num_history_frames: int = 1,
        action: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        use_history_condition_noise_in_inference: bool = False,
        progress_bar_cmd=tqdm,
        output_type: str = "quantized",
        **_: dict,
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        input_video = input_video.to(dtype=self.torch_dtype, device=self.device)

        inputs_posi = {}
        inputs_nega = {}
        inputs_shared = {
            "vace_reference_image": None,
            "input_video": input_video,
            "num_views": int(input_video.shape[0]),
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_history_frames": num_history_frames,
            "action": action,
            "cfg_scale": cfg_scale,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        conditioning_latents, history_t = _prepare_history_condition_latents(
            self,
            inputs_shared,
            use_history_condition_noise_in_inference=use_history_condition_noise_in_inference,
        )
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        use_gradient_checkpointing = self.use_gradient_checkpointing
        use_gradient_checkpointing_offload = self.use_gradient_checkpointing_offload

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred_posi = self.model_fn(
                **models,
                **inputs_shared,
                timestep=timestep,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
            noise_pred = noise_pred_posi
            inputs_shared["latents"] = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                inputs_shared["latents"],
            )
            _restore_history_condition_latents(
                inputs_shared,
                conditioning_latents=conditioning_latents,
                history_t=history_t,
            )

        self.load_models_to_device(["vae"])
        latents = inputs_shared["latents"]
        num_views = int(inputs_shared.get("num_views", 1))
        if latents.shape[-2] % num_views != 0:
            raise ValueError(
                f"Latent height {latents.shape[-2]} is not divisible by num_views={num_views}."
            )
        latents_by_view = rearrange(latents, "b c t (v h) w -> (b v) c t h w", v=num_views, h=latents.shape[-2] // num_views)
        video = self.vae.decode(latents_by_view, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        else:
            raise ValueError(f"Unsupported output_type='{output_type}', expected 'quantized' or 'floatpoint'.")
        if use_history_condition_noise_in_inference and history_t > 0:
            history_to_copy = min(
                int(num_history_frames),
                int(video.shape[2]),
                int(input_video.shape[2]),
            )
            if history_to_copy > 0:
                video[:, :, :history_to_copy] = input_video[:, :, :history_to_copy].to(
                    dtype=video.dtype,
                    device=video.device,
                )

        return video

    if getattr(pipeline, "_wrapped_call_class", None) is not None:
        WrappedPipeline = pipeline._wrapped_call_class
    else:
        WrappedPipeline = type(
            f"{pipeline.__class__.__name__}ActionPatched",
            (pipeline.__class__,),
            {},
        )
        WrappedPipeline.__call__ = __call__
        pipeline._wrapped_call_class = WrappedPipeline
    pipeline.__class__ = WrappedPipeline


def configure_ti2v_text_off_dit(dit):
    dit.use_text_embedding = False
    dit.has_text_input = True
    dit.has_image_input = False
    dit.fuse_vae_embedding_in_latents = True


def load_checkpoint_weights(pipe, ckpt_path: str):
    print(f"Loading training weights from checkpoint: {ckpt_path}")
    state_dict = load_state_dict(ckpt_path, torch_dtype=pipe.torch_dtype, device="cpu")

    dit = pipe.dit
    action_encoder = pipe.action_encoder

    action_state = {}
    dit_state = {}
    for key, value in state_dict.items():
        if key.startswith("pipe.action_encoder."):
            action_state[key.removeprefix("pipe.action_encoder.")] = value
        elif key.startswith("action_encoder."):
            action_state[key.removeprefix("action_encoder.")] = value
        elif key.startswith("pipe.dit."):
            dit_state[key.removeprefix("pipe.dit.")] = value
        elif key.startswith("dit."):
            dit_state[key.removeprefix("dit.")] = value
        else:
            dit_state[key] = value

    dit_result = dit.load_state_dict(dit_state, strict=False)
    print(
        f"  - Loaded dit keys: {len(dit_state)} "
        f"(missing={len(dit_result.missing_keys)}, unexpected={len(dit_result.unexpected_keys)})"
    )

    if action_encoder is not None:
        # A DiT-only checkpoint is a valid partial override, but once action
        # weights are present they must completely match the active encoder
        # architecture. Otherwise a legacy checkpoint can silently leave a
        # TI2V encoder (or vice versa) randomly initialized.
        action_result = action_encoder.load_state_dict(
            action_state,
            strict=bool(action_state),
        )
        print(
            f"  - Loaded action_encoder keys: {len(action_state)} "
            f"(missing={len(action_result.missing_keys)}, unexpected={len(action_result.unexpected_keys)})"
        )


def build_wan_video_action_pipeline(
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = get_device_type(),
    model_configs: list[ModelConfig] = None,
    tokenizer_config: ModelConfig = None,
    redirect_common_files: bool = True,
    vram_limit: float = None,
    ckpt_path: Optional[str] = None,
    action_dim: int = 14,
    action_mode: str = "adaln",
    text_enabled: bool = True,
):
    action_mode = _validate_action_injection_mode(action_mode)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=redirect_common_files,
        vram_limit=vram_limit,
    )
    apply_wan_vae_compat(pipe.vae)
    ti2v2_action_conditioning = bool(
        getattr(pipe.dit, "seperated_timestep", False)
        and getattr(pipe.dit, "fuse_vae_embedding_in_latents", False)
    )
    if ti2v2_action_conditioning and action_mode != "adaln":
        raise ValueError(
            "Wan2.2 TI2V action conditioning currently supports only "
            "action_mode='adaln'. Noise and action-free modes require a "
            "text or image cross-attention context."
        )

    if ti2v2_action_conditioning:
        configure_ti2v_text_off_dit(pipe.dit)

    pipe.action_encoder = None
    if action_mode != "none":
        pipe.action_encoder = WanVideoActionEncoder(
            action_dim=int(action_dim),
            dim=pipe.dit.dim,
            num_action_per_chunk=81 if action_mode == "adaln" else None,
            ti2v2=ti2v2_action_conditioning,
        )
        pipe.action_encoder = pipe.action_encoder.to(
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )
        pipe.action_encoder.eval()

    if ckpt_path is not None:
        load_checkpoint_weights(pipe, ckpt_path)

    if ti2v2_action_conditioning:
        pipe.units = _build_wan2_action_units(pipe)
        _install_wan_video_action_call(pipe)
    else:
        if not text_enabled:
            pipe.units = [
                unit
                for unit in pipe.units
                if unit.__class__.__name__ != "WanVideoUnit_PromptEmbedder"
            ]
        if action_mode != "none":
            pipe.units.append(WanVideoUnit_ActionEmbedder())

    pipe.action_injection_mode = action_mode
    pipe.ti2v2_action_conditioning = ti2v2_action_conditioning

    pipe.model_fn = model_fn_wan_video_action
    return pipe


class WanVideoUnit_ActionEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("action", "num_frames"),
            output_params=(
                "action_emb",
                "action_mod_emb",
                "action_injection_mode",
                "ti2v2_action_conditioning",
            ),
            onload_model_names=("action_encoder",)
        )

    def process(self, pipe, action=None, num_frames=None):
        action_injection_mode = _validate_action_injection_mode(
            getattr(pipe, "action_injection_mode", "none")
        )
        ti2v2_action_conditioning = bool(
            getattr(pipe, "ti2v2_action_conditioning", False)
        )
        if action_injection_mode == "none":
            return {
                "action_injection_mode": action_injection_mode,
                "ti2v2_action_conditioning": ti2v2_action_conditioning,
            }
        if action is None:
            raise ValueError(
                "`action` is required when action injection mode is "
                f"'{action_injection_mode}'."
            )
        if pipe.action_encoder is None:
            raise ValueError("Action encoder is not available in the pipeline.")
        if any(
            param.device != pipe.device for param in pipe.action_encoder.parameters()
        ):
            pipe.action_encoder = pipe.action_encoder.to(
                device=pipe.device,
                dtype=pipe.torch_dtype,
            )

        pipe.load_models_to_device(self.onload_model_names)
        action = torch.as_tensor(action, device=pipe.device, dtype=pipe.torch_dtype)

        target_groups = (int(num_frames) - 1) // 4 + 1
        target_action_frames = 1 + 4 * (target_groups - 1)
        current_action_frames = int(action.shape[1])
        if current_action_frames > target_action_frames:
            action = action[:, :target_action_frames]
        elif current_action_frames < target_action_frames:
            raise ValueError(
                f"Action sequence too short for latent groups: action_frames={current_action_frames}, "
                f"required={target_action_frames}, target_groups={target_groups}"
            )

        if action_injection_mode == "noise":
            grouped_action = torch.cat(
                [action[:, 0:1].repeat(1, 4, 1), action[:, 1:]],
                dim=1,
            )
            grouped_action = grouped_action.reshape(
                action.shape[0], target_groups, 4, action.shape[-1]
            ).mean(dim=2)
            action_emb = pipe.action_encoder(grouped_action)
            return {
                "action_emb": action_emb,
                "action_injection_mode": action_injection_mode,
                "ti2v2_action_conditioning": ti2v2_action_conditioning,
            }

        if not ti2v2_action_conditioning:
            expected_frames = pipe.action_encoder.num_action_per_chunk
            if target_action_frames != expected_frames:
                raise ValueError(
                    "Legacy adaln action length mismatch: "
                    f"got {target_action_frames}, expected {expected_frames}."
                )
            action_emb = pipe.action_encoder(
                rearrange(action, "b f d -> b (f d)").contiguous()
            )
            return {
                "action_emb": action_emb,
                "action_injection_mode": action_injection_mode,
                "ti2v2_action_conditioning": ti2v2_action_conditioning,
            }

        action_emb, action_mod_emb = pipe.action_encoder.encode_ti2v2(action)
        return {
            "action_emb": action_emb,
            "action_mod_emb": action_mod_emb,
            "action_injection_mode": action_injection_mode,
            "ti2v2_action_conditioning": ti2v2_action_conditioning,
        }


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    """
    Input frame embedder aligned to target history-conditioning behavior:
    - For short input (<=1 or < num_frames), skip VAE-conditioned noise injection
      and use pure noise as latents.
    - Otherwise encode and add scheduler initial noise as usual.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "precomputed_latents", "noise", "num_frames", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_video,
        precomputed_latents,
        noise,
        num_frames,
        tiled,
        tile_size,
        tile_stride,
    ):
        if precomputed_latents is not None:
            input_latents_views = precomputed_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = rearrange(
                input_latents_views,
                "v c t h w -> 1 c t (v h) w",
            )
            if pipe.scheduler.training:
                return {"latents": noise, "input_latents": input_latents}
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}

        if input_video is None:
            return {"latents": noise}

        if int(input_video.shape[2]) <= 1 or (not pipe.scheduler.training and int(input_video.shape[2]) < int(num_frames)):
            return {"latents": noise}

        pipe.load_models_to_device(self.onload_model_names)
        input_video = input_video.to(dtype=pipe.torch_dtype, device=pipe.device)
        input_latents_views = pipe.vae.encode(
            input_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        input_latents = rearrange(input_latents_views, "v c t h w -> 1 c t (v h) w")

        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        return {"latents": latents, "input_latents": input_latents}


def model_fn_wan_video_action(
    dit,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    action_emb: Optional[torch.Tensor] = None,
    action_mod_emb: Optional[torch.Tensor] = None,
    action_injection_mode: str = "none",
    ti2v2_action_conditioning: bool = False,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    fuse_vae_embedding_in_latents: bool = False,
    fused_condition_latent_frames: Optional[int] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    action_injection_mode = _validate_action_injection_mode(action_injection_mode)
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        condition_t = 1 if fused_condition_latent_frames is None else int(fused_condition_latent_frames)
        condition_t = max(0, min(condition_t, latents.shape[2]))
        spatial_token_count = latents.shape[3] * latents.shape[4] // 4
        t = torch.concat(
            [
                torch.zeros((condition_t, spatial_token_count), dtype=latents.dtype, device=latents.device),
                torch.ones((latents.shape[2] - condition_t, spatial_token_count), dtype=latents.dtype, device=latents.device) * timestep,
            ]
        ).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, t).unsqueeze(0))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))

    text_token_count = 0
    use_text_embedding = getattr(dit, "use_text_embedding", getattr(dit, "has_text_input", True))
    has_text_input = getattr(dit, "has_text_input", True)

    if use_text_embedding and context is not None:
        context = dit.text_embedding(context)
        text_token_count = context.shape[1]
    elif not has_text_input:
        context = None
    elif not use_text_embedding:
        context = None

    if action_injection_mode == "adaln":
        if action_emb is None:
            raise ValueError("`action:adaln` requires `action_emb`.")
        if ti2v2_action_conditioning:
            if action_mod_emb is None:
                raise ValueError("TI2V `action:adaln` requires `action_mod_emb`.")
            if context is None:
                context = action_emb
            else:
                context = torch.cat([context, action_emb], dim=1)
            text_token_count = context.shape[1]
            if t.shape[1] % action_mod_emb.shape[1] != 0:
                raise RuntimeError(
                    f"Temporal group mismatch: t.shape={tuple(t.shape)}, "
                    f"action_mod_emb.shape={tuple(action_mod_emb.shape)}. "
                    "Expected t.shape[1] to be divisible by "
                    "action_mod_emb.shape[1]."
                )
            num_spatial_tokens = t.shape[1] // action_mod_emb.shape[1]
            action_mod_emb = action_mod_emb.unsqueeze(2).repeat(
                1, 1, num_spatial_tokens, 1
            ).flatten(1, 2)
            t = t + action_mod_emb
        else:
            if action_emb.ndim != 2 or tuple(action_emb.shape) != tuple(t.shape):
                raise ValueError(
                    "Legacy `action:adaln` expects action_emb to match the "
                    f"timestep embedding, got {tuple(action_emb.shape)} and "
                    f"{tuple(t.shape)}."
                )
            t = t + action_emb
    elif action_injection_mode == "noise" and action_emb is None:
        raise ValueError("`action:noise` requires `action_emb`.")

    if t.ndim == 3:
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    x = latents

    if y is not None and dit.has_image_input and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)

    if clip_feature is not None and dit.has_image_input and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        if context is None:
            context = clip_embdding
        else:
            context = torch.cat([clip_embdding, context], dim=1)

    if context is None and len(dit.blocks) > 0:
        raise ValueError(
            "Wan DiT cross-attention requires a text, image, or TI2V action context."
        )

    x = dit.patchify(x)
    f, h, w = x.shape[2:]

    if action_injection_mode == "noise":
        if action_emb.ndim != 3:
            raise ValueError(
                "`action:noise` expects action_emb with shape (B, F, D), "
                f"got {tuple(action_emb.shape)}."
            )
        expected_shape = (x.shape[0], f, x.shape[1])
        if tuple(action_emb.shape) != expected_shape:
            raise ValueError(
                "`action:noise` temporal embedding mismatch: "
                f"got {tuple(action_emb.shape)}, "
                f"expected {expected_shape} to match patchified video tokens."
            )
        x = x + rearrange(action_emb, "b f d -> b d f 1 1")

    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block in dit.blocks:
        if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "text_token_count"):
            block.cross_attn.text_token_count = text_token_count
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))

    return x


class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode the conditioning frame directly into latents for Wan2.2 TI2V.
    """
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "precomputed_latents",
                "input_latents",
                "latents",
                "num_history_frames",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=(
                "latents",
                "fuse_vae_embedding_in_latents",
                "first_frame_latents",
                "fused_condition_latent_frames",
            ),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_video,
        precomputed_latents,
        input_latents,
        latents,
        num_history_frames,
        tiled,
        tile_size,
        tile_stride,
    ):
        if not getattr(pipe.dit, "fuse_vae_embedding_in_latents", False):
            return {}

        if precomputed_latents is not None:
            target_history = max(1, (int(num_history_frames) - 1) // 4 + 1)
            z = input_latents[:, :, :target_history].clone()
            history_t = z.shape[2]
            latents[:, :, :history_t] = z
            return {
                "latents": latents,
                "fuse_vae_embedding_in_latents": True,
                "first_frame_latents": z,
                "fused_condition_latent_frames": int(history_t),
            }

        if input_video is None:
            return {}

        num_history_frames = int(num_history_frames)
        if num_history_frames <= 0:
            raise ValueError("`input_video` must include at least one history frame.")
        if input_video.shape[2] < num_history_frames:
            raise ValueError(
                f"`num_history_frames` ({num_history_frames}) exceeds input video frames ({input_video.shape[2]})."
            )

        pipe.load_models_to_device(self.onload_model_names)
        history_frames = input_video[:, :, :num_history_frames]
        first_frame_latents = history_frames.to(dtype=pipe.torch_dtype, device=pipe.device)
        z_views = pipe.vae.encode(
            first_frame_latents,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        z_views = z_views.to(dtype=pipe.torch_dtype, device=pipe.device)
        z = rearrange(z_views, "v c t h w -> 1 c t (v h) w")

        history_t = z.shape[2]
        latents[:, :, :history_t] = z

        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "first_frame_latents": z,
            "fused_condition_latent_frames": int(history_t),
        }
