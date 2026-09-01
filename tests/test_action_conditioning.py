import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def _load_action_pipeline_module():
    package = "_boundless_test_package"
    root_package = _install_module(package)
    root_package.__path__ = [str(REPO_ROOT / "wan_video_action")]
    models_package = _install_module(f"{package}.models")
    models_package.__path__ = [str(REPO_ROOT / "wan_video_action" / "models")]
    pipelines_package = _install_module(f"{package}.pipelines")
    pipelines_package.__path__ = [str(REPO_ROOT / "wan_video_action" / "pipelines")]

    diffsynth = _install_module("diffsynth")
    diffsynth.__path__ = []
    diffsynth_pipelines = _install_module("diffsynth.pipelines")
    diffsynth_pipelines.__path__ = []
    _install_module(
        "diffsynth.pipelines.wan_video",
        WanVideoPipeline=type("WanVideoPipeline", (), {}),
    )

    class PipelineUnit:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    diffsynth_diffusion = _install_module("diffsynth.diffusion")
    diffsynth_diffusion.__path__ = []
    _install_module("diffsynth.diffusion.base_pipeline", PipelineUnit=PipelineUnit)

    diffsynth_core = _install_module(
        "diffsynth.core",
        ModelConfig=type("ModelConfig", (), {}),
        load_state_dict=lambda *args, **kwargs: {},
    )
    diffsynth_core.__path__ = []
    diffsynth_core_device = _install_module("diffsynth.core.device")
    diffsynth_core_device.__path__ = []
    _install_module(
        "diffsynth.core.device.npu_compatible_device",
        get_device_type=lambda: "cpu",
    )

    diffsynth_models = _install_module("diffsynth.models")
    diffsynth_models.__path__ = []

    def sinusoidal_embedding_1d(dim, timestep):
        return timestep.reshape(-1, 1).expand(-1, dim).to(dtype=torch.float32)

    _install_module(
        "diffsynth.models.wan_video_dit",
        sinusoidal_embedding_1d=sinusoidal_embedding_1d,
    )

    encoder_name = f"{package}.models.wan_video_action_encoder"
    encoder_spec = importlib.util.spec_from_file_location(
        encoder_name,
        REPO_ROOT / "wan_video_action" / "models" / "wan_video_action_encoder.py",
    )
    encoder_module = importlib.util.module_from_spec(encoder_spec)
    sys.modules[encoder_name] = encoder_module
    encoder_spec.loader.exec_module(encoder_module)

    _install_module(
        f"{package}.models.wan_video_vae",
        apply_wan_vae_compat=lambda vae: vae,
    )

    pipeline_name = f"{package}.pipelines.wan_video_action"
    pipeline_spec = importlib.util.spec_from_file_location(
        pipeline_name,
        REPO_ROOT / "wan_video_action" / "pipelines" / "wan_video_action.py",
    )
    pipeline_module = importlib.util.module_from_spec(pipeline_spec)
    sys.modules[pipeline_name] = pipeline_module
    pipeline_spec.loader.exec_module(pipeline_module)
    return pipeline_module


@pytest.fixture(scope="module")
def action_pipeline():
    prefixes = ("diffsynth", "_boundless_test_package")
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith(prefixes)
    }
    try:
        yield _load_action_pipeline_module()
    finally:
        for name in list(sys.modules):
            if name.startswith(prefixes):
                sys.modules.pop(name)
        sys.modules.update(previous_modules)


class _RecordingActionEncoder(torch.nn.Module):
    def __init__(self, num_action_per_chunk=9):
        super().__init__()
        self.num_action_per_chunk = num_action_per_chunk
        self.noise_input = None
        self.adaln_input = None

    def forward(self, action):
        self.noise_input = action.clone()
        return action

    def encode_ti2v2(self, action):
        self.adaln_input = action.clone()
        return action + 10, action + 20


class _ActionPipe:
    def __init__(self, mode, ti2v2=False):
        self.action_injection_mode = mode
        self.ti2v2_action_conditioning = ti2v2
        self.action_encoder = _RecordingActionEncoder()
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.loaded_models = []

    def load_models_to_device(self, names):
        self.loaded_models.append(tuple(names))


def test_noise_actions_are_aligned_to_vae_temporal_groups(action_pipeline):
    pipe = _ActionPipe("noise")
    action = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1)

    result = action_pipeline.WanVideoUnit_ActionEmbedder().process(
        pipe,
        action=action,
        num_frames=9,
    )

    expected = torch.tensor([[[0.0], [2.5], [6.5]]])
    torch.testing.assert_close(pipe.action_encoder.noise_input, expected)
    torch.testing.assert_close(result["action_emb"], expected)
    assert result["action_injection_mode"] == "noise"
    assert pipe.action_encoder.adaln_input is None


def test_adaln_actions_keep_framewise_context_and_report_mode(action_pipeline):
    pipe = _ActionPipe("adaln", ti2v2=True)
    action = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1)

    result = action_pipeline.WanVideoUnit_ActionEmbedder().process(
        pipe,
        action=action,
        num_frames=9,
    )

    torch.testing.assert_close(pipe.action_encoder.adaln_input, action)
    torch.testing.assert_close(result["action_emb"], action + 10)
    torch.testing.assert_close(result["action_mod_emb"], action + 20)
    assert result["action_injection_mode"] == "adaln"
    assert result["ti2v2_action_conditioning"] is True
    assert pipe.action_encoder.noise_input is None


def test_legacy_adaln_flattens_the_complete_action_chunk(action_pipeline):
    pipe = _ActionPipe("adaln")
    action = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1)

    result = action_pipeline.WanVideoUnit_ActionEmbedder().process(
        pipe,
        action=action,
        num_frames=9,
    )

    expected = action.flatten(1)
    torch.testing.assert_close(pipe.action_encoder.noise_input, expected)
    torch.testing.assert_close(result["action_emb"], expected)
    assert result["ti2v2_action_conditioning"] is False


def test_action_encoder_only_registers_the_selected_injection_branch(action_pipeline):
    encoder_class = action_pipeline.WanVideoActionEncoder
    noise_encoder = encoder_class(action_dim=2, dim=4, num_action_per_chunk=None)
    legacy_adaln_encoder = encoder_class(
        action_dim=2,
        dim=4,
        num_action_per_chunk=9,
    )
    ti2v2_encoder = encoder_class(
        action_dim=2,
        dim=4,
        num_action_per_chunk=9,
        ti2v2=True,
    )

    noise_parameter_names = {name for name, _ in noise_encoder.named_parameters()}
    legacy_adaln_parameter_names = {
        name for name, _ in legacy_adaln_encoder.named_parameters()
    }
    ti2v2_parameter_names = {name for name, _ in ti2v2_encoder.named_parameters()}

    assert noise_parameter_names
    assert all(name.startswith("action_embedding.") for name in noise_parameter_names)
    assert legacy_adaln_parameter_names
    assert all(
        name.startswith("action_embedding.") for name in legacy_adaln_parameter_names
    )
    assert ti2v2_parameter_names
    assert all(
        name.startswith(("action_mlp1.", "action_mlp2."))
        for name in ti2v2_parameter_names
    )

    noise_checkpoint = {
        **noise_encoder.state_dict(),
        **ti2v2_encoder.state_dict(),
    }
    ti2v2_checkpoint = {
        **legacy_adaln_encoder.state_dict(),
        **ti2v2_encoder.state_dict(),
    }
    noise_result = noise_encoder.load_state_dict(noise_checkpoint)
    ti2v2_result = ti2v2_encoder.load_state_dict(ti2v2_checkpoint)
    assert not noise_result.missing_keys and not noise_result.unexpected_keys
    assert not ti2v2_result.missing_keys and not ti2v2_result.unexpected_keys

    with pytest.raises(RuntimeError, match="missing_keys"):
        ti2v2_encoder.load_state_dict(legacy_adaln_encoder.state_dict())

    noise_encoder(torch.randn(1, 3, 2)).sum().backward()
    legacy_adaln_encoder(torch.randn(1, 18)).sum().backward()
    action_context, action_mod = ti2v2_encoder.encode_ti2v2(torch.randn(1, 9, 2))
    (action_context.sum() + action_mod.sum()).backward()
    assert all(parameter.grad is not None for parameter in noise_encoder.parameters())
    assert all(
        parameter.grad is not None for parameter in legacy_adaln_encoder.parameters()
    )
    assert all(parameter.grad is not None for parameter in ti2v2_encoder.parameters())


@pytest.mark.parametrize(
    ("mode", "expected_chunk_size"),
    (("noise", None), ("adaln", 81), ("none", None)),
)
def test_pipeline_builds_only_the_encoder_needed_by_each_mode(
    action_pipeline,
    monkeypatch,
    mode,
    expected_chunk_size,
):
    class WanVideoUnit_ShapeChecker:
        pass

    class WanVideoUnit_NoiseInitializer:
        pass

    class WanVideoUnit_PromptEmbedder:
        pass

    class LoadedPipeline:
        def __init__(self):
            self.torch_dtype = torch.float32
            self.device = torch.device("cpu")
            self.dit = types.SimpleNamespace(dim=4)
            self.vae = object()
            self.units = [
                WanVideoUnit_ShapeChecker(),
                WanVideoUnit_NoiseInitializer(),
                WanVideoUnit_PromptEmbedder(),
            ]

    monkeypatch.setattr(
        action_pipeline.WanVideoPipeline,
        "from_pretrained",
        lambda **kwargs: LoadedPipeline(),
        raising=False,
    )

    pipe = action_pipeline.build_wan_video_action_pipeline(
        action_mode=mode, action_dim=2
    )

    assert pipe.action_injection_mode == mode
    unit_names = [unit.__class__.__name__ for unit in pipe.units]
    assert "WanVideoUnit_PromptEmbedder" in unit_names
    if mode == "none":
        assert pipe.action_encoder is None
        assert "WanVideoUnit_ActionEmbedder" not in unit_names
    else:
        assert pipe.action_encoder.num_action_per_chunk == expected_chunk_size
        assert pipe.action_encoder.ti2v2 is False
        assert unit_names[-1] == "WanVideoUnit_ActionEmbedder"


def test_legacy_builder_removes_prompt_unit_only_when_text_is_disabled(
    action_pipeline,
    monkeypatch,
):
    class WanVideoUnit_PromptEmbedder:
        pass

    loaded_pipeline = types.SimpleNamespace(
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
        dit=types.SimpleNamespace(dim=4),
        vae=object(),
        units=[WanVideoUnit_PromptEmbedder()],
    )
    monkeypatch.setattr(
        action_pipeline.WanVideoPipeline,
        "from_pretrained",
        lambda **kwargs: loaded_pipeline,
        raising=False,
    )

    pipe = action_pipeline.build_wan_video_action_pipeline(
        action_mode="noise",
        action_dim=2,
        text_enabled=False,
    )

    unit_names = [unit.__class__.__name__ for unit in pipe.units]
    assert "WanVideoUnit_PromptEmbedder" not in unit_names
    assert unit_names[-1] == "WanVideoUnit_ActionEmbedder"


def test_ti2v2_builder_accepts_only_the_trained_adaln_contract(
    action_pipeline,
    monkeypatch,
):
    class WanVideoUnit_ShapeChecker:
        pass

    class WanVideoUnit_NoiseInitializer:
        pass

    class WanVideoUnit_PromptEmbedder:
        pass

    class LoadedPipeline:
        def __init__(self):
            self.torch_dtype = torch.float32
            self.device = torch.device("cpu")
            self.dit = types.SimpleNamespace(
                dim=4,
                seperated_timestep=True,
                fuse_vae_embedding_in_latents=True,
            )
            self.vae = object()
            self.units = [
                WanVideoUnit_ShapeChecker(),
                WanVideoUnit_NoiseInitializer(),
                WanVideoUnit_PromptEmbedder(),
            ]

    monkeypatch.setattr(
        action_pipeline.WanVideoPipeline,
        "from_pretrained",
        lambda **kwargs: LoadedPipeline(),
        raising=False,
    )

    pipe = action_pipeline.build_wan_video_action_pipeline(
        action_mode="adaln",
        action_dim=2,
    )
    assert pipe.ti2v2_action_conditioning is True
    assert pipe.action_encoder.ti2v2 is True
    assert "WanVideoUnit_PromptEmbedder" not in {
        unit.__class__.__name__ for unit in pipe.units
    }

    for unsupported_mode in ("noise", "none"):
        with pytest.raises(ValueError, match="supports only"):
            action_pipeline.build_wan_video_action_pipeline(
                action_mode=unsupported_mode,
                action_dim=2,
            )


def test_none_mode_checkpoint_loads_dit_and_ignores_action_weights(
    action_pipeline,
    monkeypatch,
):
    checkpoint = {
        "dit.weight": torch.tensor([1.0]),
        "pipe.action_encoder.action_embedding.0.weight": torch.tensor([2.0]),
    }
    monkeypatch.setattr(
        action_pipeline,
        "load_state_dict",
        lambda *args, **kwargs: checkpoint,
    )

    class RecordingDiT:
        dim = 4

        def __init__(self):
            self.loaded_state = None

        def load_state_dict(self, state_dict, strict):
            self.loaded_state = state_dict
            return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

    loaded_pipeline = types.SimpleNamespace(
        dit=RecordingDiT(),
        vae=object(),
        units=[],
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        action_pipeline.WanVideoPipeline,
        "from_pretrained",
        lambda **kwargs: loaded_pipeline,
        raising=False,
    )

    pipe = action_pipeline.build_wan_video_action_pipeline(
        action_mode="none",
        ckpt_path="checkpoint.safetensors",
    )

    assert pipe.dit.loaded_state == {"weight": checkpoint["dit.weight"]}
    assert pipe.action_encoder is None


def test_checkpoint_load_rejects_action_weights_for_the_wrong_architecture(
    action_pipeline,
    monkeypatch,
):
    encoder_class = action_pipeline.WanVideoActionEncoder
    legacy_encoder = encoder_class(
        action_dim=2,
        dim=4,
        num_action_per_chunk=9,
    )
    ti2v2_encoder = encoder_class(
        action_dim=2,
        dim=4,
        num_action_per_chunk=9,
        ti2v2=True,
    )
    checkpoint = {
        f"action_encoder.{key}": value
        for key, value in legacy_encoder.state_dict().items()
    }
    monkeypatch.setattr(
        action_pipeline,
        "load_state_dict",
        lambda *args, **kwargs: checkpoint,
    )

    class EmptyDiT:
        def load_state_dict(self, state_dict, strict):
            assert state_dict == {}
            return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

    pipe = types.SimpleNamespace(
        dit=EmptyDiT(),
        action_encoder=ti2v2_encoder,
        torch_dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="missing_keys"):
        action_pipeline.load_checkpoint_weights(pipe, "legacy.safetensors")


class _IdentityPatchDiT:
    dim = 1
    freq_dim = 1
    seperated_timestep = False
    has_text_input = True
    use_text_embedding = True
    has_image_input = False
    require_vae_embedding = False
    require_clip_embedding = False

    def __init__(self):
        self.blocks = []
        self.freqs = tuple(torch.zeros(8, 1) for _ in range(3))

    def time_embedding(self, timestep):
        return torch.zeros_like(timestep)

    def time_projection(self, timestep):
        return torch.zeros(*timestep.shape[:-1], 6, dtype=timestep.dtype)

    def text_embedding(self, context):
        return context

    def patchify(self, latents):
        return latents

    def head(self, tokens, timestep):
        return tokens

    def unpatchify(self, tokens, shape):
        frames, height, width = shape
        return rearrange(
            tokens,
            "b (f h w) c -> b c f h w",
            f=frames,
            h=height,
            w=width,
        )


class _CaptureBlock:
    def __init__(self):
        self.context = None

    def __call__(self, tokens, context, timestep_mod, freqs):
        if context is None:
            raise TypeError("cross-attention context cannot be None")
        self.context = context
        return tokens


class _CaptureAdalnDiT(_IdentityPatchDiT):
    seperated_timestep = True
    has_text_input = False
    use_text_embedding = False

    def __init__(self):
        super().__init__()
        self.capture_block = _CaptureBlock()
        self.blocks = [self.capture_block]
        self.head_timestep = None

    def head(self, tokens, timestep):
        self.head_timestep = timestep
        return tokens


class _CaptureLegacyAdalnDiT(_IdentityPatchDiT):
    def __init__(self):
        super().__init__()
        self.capture_block = _CaptureBlock()
        self.blocks = [self.capture_block]
        self.head_timestep = None

    def head(self, tokens, timestep):
        self.head_timestep = timestep
        return tokens


def test_noise_mode_injects_each_action_group_into_matching_video_tokens(
    action_pipeline,
):
    dit = _IdentityPatchDiT()
    dit.capture_block = _CaptureBlock()
    dit.blocks = [dit.capture_block]
    latents = torch.zeros(1, 1, 3, 2, 2)
    action_emb = torch.tensor([[[1.0], [2.0], [3.0]]])
    text_context = torch.tensor([[[7.0]]])

    output = action_pipeline.model_fn_wan_video_action(
        dit=dit,
        latents=latents,
        timestep=torch.tensor([1.0]),
        context=text_context,
        action_emb=action_emb,
        action_injection_mode="noise",
    )

    expected = action_emb.transpose(1, 2).unsqueeze(-1).unsqueeze(-1).expand_as(latents)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(dit.capture_block.context, text_context)


def test_adaln_mode_keeps_context_and_timestep_conditioning(action_pipeline):
    dit = _CaptureAdalnDiT()
    latents = torch.zeros(1, 1, 3, 2, 2)
    action_context = torch.tensor([[[1.0], [2.0]]])
    action_mod = torch.tensor([[[3.0], [4.0], [5.0]]])

    action_pipeline.model_fn_wan_video_action(
        dit=dit,
        latents=latents,
        timestep=torch.tensor([1.0]),
        action_emb=action_context,
        action_mod_emb=action_mod,
        action_injection_mode="adaln",
        ti2v2_action_conditioning=True,
        fuse_vae_embedding_in_latents=True,
        fused_condition_latent_frames=1,
    )

    torch.testing.assert_close(dit.capture_block.context, action_context)
    torch.testing.assert_close(dit.head_timestep, action_mod)


def test_legacy_adaln_adds_global_action_and_preserves_text_context(action_pipeline):
    dit = _CaptureLegacyAdalnDiT()
    text_context = torch.tensor([[[7.0]]])
    action_emb = torch.tensor([[3.0]])

    action_pipeline.model_fn_wan_video_action(
        dit=dit,
        latents=torch.zeros(1, 1, 3, 2, 2),
        timestep=torch.tensor([1.0]),
        context=text_context,
        action_emb=action_emb,
        action_injection_mode="adaln",
    )

    torch.testing.assert_close(dit.capture_block.context, text_context)
    torch.testing.assert_close(dit.head_timestep, action_emb)


def test_none_mode_keeps_text_context_without_action_embeddings(action_pipeline):
    dit = _IdentityPatchDiT()
    dit.capture_block = _CaptureBlock()
    dit.blocks = [dit.capture_block]
    latents = torch.zeros(1, 1, 3, 2, 2)
    text_context = torch.tensor([[[7.0]]])

    output = action_pipeline.model_fn_wan_video_action(
        dit=dit,
        latents=latents,
        timestep=torch.tensor([1.0]),
        context=text_context,
        action_injection_mode="none",
    )

    torch.testing.assert_close(output, latents)
    torch.testing.assert_close(dit.capture_block.context, text_context)


def test_action_free_model_rejects_missing_cross_attention_context(action_pipeline):
    dit = _IdentityPatchDiT()
    dit.blocks = [_CaptureBlock()]

    with pytest.raises(ValueError, match="cross-attention requires"):
        action_pipeline.model_fn_wan_video_action(
            dit=dit,
            latents=torch.zeros(1, 1, 3, 2, 2),
            timestep=torch.tensor([1.0]),
            action_injection_mode="none",
        )


@pytest.mark.parametrize("mode", ("noise", "none"))
def test_pipeline_unit_outputs_can_feed_model_without_optional_keys(
    action_pipeline,
    mode,
):
    pipe = _ActionPipe(mode)
    inputs_shared = {
        "action": torch.arange(9, dtype=torch.float32).reshape(1, 9, 1),
        "num_frames": 9,
        "context": torch.tensor([[[7.0]]]),
    }
    unit = action_pipeline.WanVideoUnit_ActionEmbedder()
    unit_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
    inputs_shared.update(unit.process(pipe, **unit_inputs))

    latents = torch.zeros(1, 1, 3, 2, 2)
    dit = _IdentityPatchDiT()
    dit.blocks = [_CaptureBlock()]
    output = action_pipeline.model_fn_wan_video_action(
        dit=dit,
        latents=latents,
        timestep=torch.tensor([1.0]),
        **inputs_shared,
    )

    if mode == "noise":
        expected = torch.tensor([[[0.0], [2.5], [6.5]]])
        expected = expected.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        torch.testing.assert_close(output, expected.expand_as(latents))
    else:
        torch.testing.assert_close(output, latents)


def test_training_pipeline_uses_explicit_action_configuration():
    tree = ast.parse((REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_wan_video_action_pipeline"
    ]
    assert len(calls) == 1
    keyword_names = {keyword.arg for keyword in calls[0].keywords}
    assert "args" not in keyword_names
    assert {"action_mode", "action_dim", "ckpt_path", "text_enabled"} <= keyword_names


def test_action_free_training_skips_action_statistics_and_paths():
    tree = ast.parse((REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8"))

    def is_action_enabled_guard(node):
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "runtime_config"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "action_enabled"
        )

    guarded_blocks = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and is_action_enabled_guard(node.test)
    ]
    guarded_source = "\n".join(guarded_blocks)

    assert "open(args.action_stat_path" in guarded_source
    assert "path_groups.append" in guarded_source
    assert "special_operator_map" in guarded_source
    assert "LoadCobotAction" in guarded_source


def test_runtime_config_always_loads_actions_for_action_training():
    class FakeOmegaConf:
        @staticmethod
        def load(path):
            return {"modules": {}}

        @staticmethod
        def to_container(value, resolve=True):
            return value

    previous_omegaconf = sys.modules.get("omegaconf")
    _install_module("omegaconf", OmegaConf=FakeOmegaConf)
    try:
        parser_spec = importlib.util.spec_from_file_location(
            "_boundless_test_parsers",
            REPO_ROOT / "wan_video_action" / "parsers.py",
        )
        parser_module = importlib.util.module_from_spec(parser_spec)
        parser_spec.loader.exec_module(parser_module)

        args = types.SimpleNamespace(
            model_config_path="unused.yaml",
            enable_dit=True,
            enable_vae=True,
            enable_image=True,
            enable_text=True,
            text_mode="emb",
            action_mode="noise",
            model_paths="",
            data_file_keys="video",
        )
        runtime_config = parser_module.prepare_runtime_config(args)
        assert runtime_config["data_file_keys"] == ["video", "action"]

        args.action_mode = "none"
        runtime_config = parser_module.prepare_runtime_config(args)
        assert runtime_config["data_file_keys"] == ["video"]
    finally:
        if previous_omegaconf is None:
            sys.modules.pop("omegaconf", None)
        else:
            sys.modules["omegaconf"] = previous_omegaconf
