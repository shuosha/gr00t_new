# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test Gr00tN1d7ActionHead: flow matching forward, get_action, feature encoding.

These tests instantiate the action head directly (no backbone required)
and feed it synthetic backbone output tensors.
"""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _small_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
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
    return Gr00tN1d7Config(**defaults)


@pytest.fixture
def action_head():
    config = _small_config()
    head = Gr00tN1d7ActionHead(config)
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
            "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
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

    def test_forward_with_state_dropout(self):
        config = _small_config(state_dropout_prob=0.5)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert torch.isfinite(out["loss"])


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

    def test_get_action_single_sample(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config, batch_size=1)
        del action_input["action"]
        out = head.get_action(
            _make_backbone_output(config, batch_size=1),
            action_input,
        )
        assert out["action_pred"].shape[0] == 1


class TestActionHeadEncodeFeatures:
    """Test feature encoding helper."""

    def test_encode_features_shapes(self, action_head):
        head, config = action_head
        result = head._encode_features(
            _make_backbone_output(config),
            _make_action_input(config),
        )
        assert result["backbone_features"].shape == (2, 8, config.backbone_embedding_dim)
        assert result["state_features"].shape == (2, 1, config.input_embedding_dim)


class TestActionHeadTrainableParams:
    """Test parameter freezing."""

    def test_all_trainable_by_default(self, action_head):
        head, _ = action_head
        head.set_trainable_parameters(True, True, True)
        assert all(p.requires_grad for p in head.parameters())

    def test_freeze_projector(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(False, True, True)
        for p in head.state_encoder.parameters():
            assert not p.requires_grad
        for p in head.action_encoder.parameters():
            assert not p.requires_grad

    def test_freeze_diffusion(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(True, False, True)
        for p in head.model.parameters():
            assert not p.requires_grad


class TestActionHeadRTC:
    """Test RTC (real-time chunking) inpainting in the denoising loop.

    RTC activates when "action" is present in action_input at get_action time:
    the first rtc_overlap_steps rows are seeded from the previous chunk, rows
    [0, rtc_frozen_steps) get vel_strength=0 (hard-frozen), and rows
    [rtc_frozen_steps, rtc_overlap_steps) get a ramped partial denoise.
    """

    @staticmethod
    def _rtc_options(config, frozen, overlap=None, ramp_rate=5.0):
        return {
            "action_horizon": config.action_horizon,
            "rtc_overlap_steps": overlap if overlap is not None else frozen,
            "rtc_frozen_steps": frozen,
            "rtc_ramp_rate": ramp_rate,
        }

    def test_frozen_rows_equal_seed(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        prev_chunk = action_input["action"].clone()
        frozen = 2
        out = head.get_action(
            _make_backbone_output(config),
            action_input,
            options=self._rtc_options(config, frozen=frozen),
        )
        # Seed for new rows [0, overlap) is the LAST `overlap` rows of the valid
        # horizon of the input chunk; frozen rows never receive velocity updates.
        expected = prev_chunk[:, config.action_horizon - frozen : config.action_horizon]
        torch.testing.assert_close(out["action_pred"][:, :frozen], expected)

    def test_overlap_equals_frozen_empty_ramp(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        out = head.get_action(
            _make_backbone_output(config),
            action_input,
            options=self._rtc_options(config, frozen=2, overlap=2),
        )
        assert torch.isfinite(out["action_pred"]).all()

    def test_ramp_region_partially_denoised(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        prev_chunk = action_input["action"].clone()
        frozen, overlap = 1, 3
        out = head.get_action(
            _make_backbone_output(config),
            action_input,
            options=self._rtc_options(config, frozen=frozen, overlap=overlap),
        )
        pred = out["action_pred"]
        seed = prev_chunk[:, config.action_horizon - overlap : config.action_horizon]
        torch.testing.assert_close(pred[:, :frozen], seed[:, :frozen])
        # Ramp rows start from the seed but receive nonzero velocity updates.
        assert not torch.allclose(pred[:, frozen:overlap], seed[:, frozen:overlap])
        assert torch.isfinite(pred).all()

    def test_missing_options_raises(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)  # contains "action" -> RTC path
        with pytest.raises(AssertionError):
            head.get_action(_make_backbone_output(config), action_input, options=None)
        incomplete = self._rtc_options(config, frozen=2)
        del incomplete["rtc_ramp_rate"]
        with pytest.raises(AssertionError):
            head.get_action(_make_backbone_output(config), action_input, options=incomplete)

    def test_no_rtc_without_action_key(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input, options=None)
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)
