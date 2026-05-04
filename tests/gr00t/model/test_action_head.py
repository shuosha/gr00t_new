"""
Test Gr00tN1d6ActionHead: flow matching forward, get_action, feature encoding,
training-time RTC.

These tests instantiate the action head directly (no backbone required)
and feed it synthetic backbone output tensors.
"""

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _small_config(**overrides) -> Gr00tN1d6Config:
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        state_additive_noise_scale=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d6Config(**defaults)


@pytest.fixture
def action_head():
    config = _small_config()
    head = Gr00tN1d6ActionHead(config)
    head.eval()
    return head, config


def _make_backbone_output(config, batch_size=2, seq_len=8):
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
    )


def _make_action_input(config, batch_size=2):
    return BatchFeature(
        data={
            "state": torch.randn(batch_size, 1, config.max_state_dim),
            "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
            "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
        }
    )


class TestActionHeadForward:
    """Test training forward pass."""

    def test_forward_returns_loss(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "loss" in out
        assert out["loss"].dim() == 0
        assert torch.isfinite(out["loss"])

    def test_forward_loss_shape(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert out["action_loss"].shape == (2, config.action_horizon, config.max_action_dim)


class TestActionHeadGetAction:
    """Test inference (denoising loop)."""

    def test_get_action_output_shape(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]  # get_action doesn't need ground-truth action
        out = head.get_action(_make_backbone_output(config), action_input)
        assert "action_pred" in out
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_get_action_no_grad(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert not out["action_pred"].requires_grad


class TestActionHeadRTC:
    """Training-time Real-Time Chunking (mirrors openpi model_test.py RTC tests)."""

    _RTC_METRIC_KEYS = {
        "rtc_mean_delay",
        "rtc_postfix_mse",
        "rtc_boundary_mse",
        "rtc_pred_boundary_jump",
        "rtc_true_boundary_jump",
    }

    def test_forward_shape_and_metrics(self):
        config = _small_config(rtc_max_delay=2)
        head = Gr00tN1d6ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert out["loss"].dim() == 0
        assert torch.isfinite(out["loss"])
        assert out["action_loss"].shape == (2, config.action_horizon, config.max_action_dim)
        assert set(out["rtc_metrics"].keys()) == self._RTC_METRIC_KEYS
        for k, v in out["rtc_metrics"].items():
            assert v.dim() == 0, f"{k} should be scalar, got {v.shape}"
            assert torch.isfinite(v), f"{k} is not finite: {v}"

    def test_rtc_metrics_empty_without_rtc(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert out["rtc_metrics"] == {}

    def test_full_prefix_zero_loss(self):
        # Force d == action_horizon for every example (entire chunk is prefix).
        # Loss should be exactly 0 because the per-example mask is all zeros and the
        # numerator is 0 (denominator gets +1e-6).
        H = 4
        config = _small_config(rtc_max_delay=H, action_horizon=H)
        head = Gr00tN1d6ActionHead(config)
        head.train()

        original_randint = torch.randint

        def all_h_randint(low, high, size, **kwargs):  # noqa: ARG001
            return torch.full(size, H, dtype=torch.long, **kwargs)

        torch.randint = all_h_randint
        try:
            out = head.forward(_make_backbone_output(config), _make_action_input(config))
        finally:
            torch.randint = original_randint
        assert torch.isfinite(out["loss"])
        torch.testing.assert_close(out["loss"], torch.zeros_like(out["loss"]))

    def test_sample_actions_clamps_prefix(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]
        delay = 2
        prev = torch.full(
            (2, config.action_horizon, config.max_action_dim), 7.0, dtype=torch.float32
        )
        out = head.get_action(
            _make_backbone_output(config),
            action_input,
            options={"prev_action_chunk": prev, "inference_delay": delay},
        )
        actions = out["action_pred"]
        assert actions.shape == (2, config.action_horizon, config.max_action_dim)
        torch.testing.assert_close(actions[:, :delay], prev[:, :delay])
        assert not torch.all(actions[:, delay:] == 7.0)

    def test_legacy_path_unchanged_without_rtc(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "loss" in out and "action_loss" in out and "action_mask" in out
        assert out["rtc_metrics"] == {}
        assert torch.isfinite(out["loss"])

    def test_config_rejects_invalid_rtc_max_delay(self):
        with pytest.raises(ValueError, match="non-negative"):
            _small_config(rtc_max_delay=-1)
        with pytest.raises(ValueError, match="action_horizon"):
            _small_config(rtc_max_delay=999)
