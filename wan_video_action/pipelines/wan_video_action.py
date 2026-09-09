import torch
from tqdm import tqdm
from typing import Optional, Union
from einops import rearrange

from diffsynth.pipelines.wan_video import WanVideoPipeline, WanVideoUnit_ShapeChecker
from diffsynth.diffusion.base_pipeline import PipelineUnit
from diffsynth.core.device.npu_compatible_device import get_device_type
from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

from ..models.wan_video_action_encoder import WanVideoActionEncoder


class WanVideoActionPipeline(WanVideoPipeline):
    @classmethod
    def from_pretrained(
        cls,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = None,
        tokenizer_config: ModelConfig = None,
        redirect_common_files: bool = True,
        vram_limit: float = None,
        ckpt_path: Optional[str] = None,
        action_dim: int = 14,
        action_mode: str = "adaln",
    ):
        pipe = super().from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            redirect_common_files=redirect_common_files,
            vram_limit=vram_limit,
        )
        pipe.__class__ = cls
        pipe.dit.use_text_embedding = False
        pipe.dit.has_text_input = True
        pipe.dit.has_image_input = False
        pipe.dit.fuse_vae_embedding_in_latents = True

        pipe.action_encoder = WanVideoActionEncoder(
            action_dim=int(action_dim),
            dim=pipe.dit.dim,
        )
        pipe.action_encoder = pipe.action_encoder.to(dtype=pipe.torch_dtype, device=pipe.device)
        pipe.action_encoder.eval()
        pipe.action_injection_mode = action_mode

        if ckpt_path is not None:
            load_checkpoint_weights(pipe, ckpt_path)

        pipe.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_ActionEmbedder(),
        ]

        pipe.model_fn = model_fn_wan_video_action

        return pipe

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

        history_condition_latents = inputs_shared.get("history_condition_latents")
        history_t = int(inputs_shared.get("fused_condition_latent_frames") or 0)
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        use_gradient_checkpointing = self.use_gradient_checkpointing
        use_gradient_checkpointing_offload = self.use_gradient_checkpointing_offload

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            if history_t > 0:
                inputs_shared["latents"][:, :, :history_t] = history_condition_latents[:, :, :history_t]
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

        if history_t > 0:
            inputs_shared["latents"][:, :, :history_t] = history_condition_latents[:, :, :history_t]

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
        if history_t > 0:
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


def model_fn_wan_video_action(
    dit,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    action_emb: Optional[torch.Tensor] = None,
    action_mod_emb: Optional[torch.Tensor] = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    fuse_vae_embedding_in_latents: bool = False,
    fused_condition_latent_frames: Optional[int] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
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

    if context is None:
        context = action_emb
    else:
        context = torch.cat([context, action_emb], dim=1)
    text_token_count = context.shape[1]
    num_spatial_tokens = t.shape[1] // action_mod_emb.shape[1]
    action_mod_emb = action_mod_emb.unsqueeze(2).repeat(1, 1, num_spatial_tokens, 1).flatten(1, 2)
    t = t + action_mod_emb

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

    x = dit.patchify(x)
    f, h, w = x.shape[2:]

    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    for block in dit.blocks:
        if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "text_token_count"):
            block.cross_attn.text_token_count = text_token_count
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                block,
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))

    return x


def load_checkpoint_weights(pipe, ckpt_path: str):
    print(f"Loading training weights from checkpoint: {ckpt_path}")
    state_dict = load_state_dict(ckpt_path, torch_dtype=pipe.torch_dtype, device="cpu")

    dit = pipe.dit
    action_encoder = pipe.action_encoder

    action_prefixes = ("pipe.action_encoder.", "action_encoder.")
    dit_prefix = "pipe.dit."
    action_state = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        for prefix in action_prefixes
        if key.startswith(prefix)
    }
    dit_state = {}
    for key, value in state_dict.items():
        if any(key.startswith(prefix) for prefix in action_prefixes):
            continue
        if key.startswith(dit_prefix):
            key = key[len(dit_prefix):]
        dit_state[key] = value

    dit_result = dit.load_state_dict(dit_state, strict=False)
    print(
        f"  - Loaded dit keys: {len(dit_state)} "
        f"(missing={len(dit_result.missing_keys)}, unexpected={len(dit_result.unexpected_keys)})"
    )

    action_result = action_encoder.load_state_dict(action_state, strict=False)
    print(
        f"  - Loaded action_encoder keys: {len(action_state)} "
        f"(missing={len(action_result.missing_keys)}, unexpected={len(action_result.unexpected_keys)})"
    )


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "precomputed_latents", "height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: WanVideoPipeline, input_video, precomputed_latents, height, width, num_frames, seed, rand_device):
        if precomputed_latents is not None:
            shape = (
                1,
                int(precomputed_latents.shape[1]),
                int(precomputed_latents.shape[2]),
                int(precomputed_latents.shape[0]) * int(precomputed_latents.shape[3]),
                int(precomputed_latents.shape[4]),
            )
        else:
            num_views = int(input_video.shape[0]) if input_video is not None else 1
            length = (int(num_frames) - 1) // 4 + 1
            latent_height = (int(height) * num_views) // pipe.vae.upsampling_factor
            latent_width = int(width) // pipe.vae.upsampling_factor
            shape = (1, pipe.vae.model.z_dim, length, latent_height, latent_width)

        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class WanVideoUnit_ActionEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("action", "num_frames"),
            output_params=("action_emb", "action_mod_emb"),
            onload_model_names=("action_encoder",)
        )

    def process(self, pipe, action=None, num_frames=None):
        if action is None:
            return {}

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
        action_emb, action_mod_emb = pipe.action_encoder(action)
        return {"action_emb": action_emb, "action_mod_emb": action_mod_emb}


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
                "noise",
                "num_history_frames",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=(
                "latents",
                "fuse_vae_embedding_in_latents",
                "history_condition_latents",
                "fused_condition_latent_frames",
            ),
            onload_model_names=("vae",)
        )

    def _add_history_condition_noise(
        self,
        pipe: WanVideoPipeline,
        history_condition_latents: torch.Tensor,
        noise: torch.Tensor,
        history_t: int,
    ):
        if not (
            pipe.scheduler.training
            and pipe.action_injection_mode == "adaln"
            and history_t > 1
        ):
            return history_condition_latents

        training_sigmas = pipe.scheduler.sigmas
        small_sigma_idx = max(0, len(training_sigmas) - 50)
        small_sigma = training_sigmas[small_sigma_idx]
        history_condition_latents[:, :, 1:history_t] = (
            (1 - small_sigma) * history_condition_latents[:, :, 1:history_t].float()
            + small_sigma * noise[:, :, 1:history_t].float()
        ).to(dtype=history_condition_latents.dtype)
        return history_condition_latents

    def process(
        self,
        pipe: WanVideoPipeline,
        input_video,
        precomputed_latents,
        input_latents,
        latents,
        noise,
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

            z = self._add_history_condition_noise(pipe, z, noise, history_t)
            return {
                "latents": latents,
                "fuse_vae_embedding_in_latents": True,
                "history_condition_latents": z,
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
        history_frames = input_video[:, :, :num_history_frames].to(dtype=pipe.torch_dtype, device=pipe.device)
        z_views = pipe.vae.encode(
            history_frames,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        z_views = z_views.to(dtype=pipe.torch_dtype, device=pipe.device)
        z = rearrange(z_views, "v c t h w -> 1 c t (v h) w")

        history_t = z.shape[2]
        latents[:, :, :history_t] = z

        z = self._add_history_condition_noise(pipe, z, noise, history_t)

        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "history_condition_latents": z,
            "fused_condition_latent_frames": int(history_t),
        }
