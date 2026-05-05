from typing import Any, Tuple

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree


class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Initialize components directly from config
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            print("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg, cross_attention_dim=config.backbone_embedding_dim
            )
            print("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.rtc_max_delay = config.rtc_max_delay

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features.
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features.
        if self.training and self.state_additive_noise_scale > 0:
            print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory.
        actions = action_input.action
        B, H, _ = actions.shape
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t_scalar = self.sample_time(B, device=actions.device, dtype=actions.dtype)  # [B]

        if self.rtc_max_delay is None:
            # Legacy path: per-batch scalar tau, broadcast across all action tokens.
            t = t_scalar[:, None, None]  # (B, 1, 1)
            noisy_trajectory = (1 - t) * noise + t * actions
            t_discretized_action = (t_scalar * self.num_timestep_buckets).long()  # [B]
            t_discretized_full = t_discretized_action  # DiT sees per-batch scalar
            prefix_action_mask = None
            delay = None
            t_per_tok = None
        else:
            # Training-time RTC: per-example delay d ~ Unif{0, ..., rtc_max_delay}, prefix
            # slots [0, d) are clamped to clean GT (per-token tau=1.0), postfix slots
            # share the sampled scalar tau. Loss is masked to the postfix.
            delay = torch.randint(
                0, self.rtc_max_delay + 1, (B,), device=actions.device
            )  # [B]
            arange_h = torch.arange(H, device=actions.device)
            prefix_action_mask = arange_h[None, :] < delay[:, None]  # [B, H] bool
            ones_per_tok = torch.ones_like(t_scalar)[:, None].expand(-1, H)
            t_postfix_per_tok = t_scalar[:, None].expand(-1, H)
            t_per_tok = torch.where(
                prefix_action_mask, ones_per_tok, t_postfix_per_tok
            )  # [B, H]
            t_b = t_per_tok[..., None]  # [B, H, 1]
            noisy_trajectory = (1 - t_b) * noise + t_b * actions
            t_discretized_action = (t_per_tok * self.num_timestep_buckets).long()  # [B, H]
            # State token (slot 0 of sa_embs) gets the postfix scalar — same tau as the
            # tokens generated alongside it.
            t_discretized_state = (t_scalar * self.num_timestep_buckets).long()[:, None]  # [B, 1]
            t_discretized_full = torch.cat(
                [t_discretized_state, t_discretized_action], dim=1
            )  # [B, 1+H]

        velocity = actions - noise

        action_features = self.action_encoder(
            noisy_trajectory, t_discretized_action, embodiment_id
        )

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized_full,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized_full,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        if prefix_action_mask is not None:
            # Mask out the prefix slots and rescale per example so the per-example mean
            # over [B, H] stays a clean per-postfix-token mean. Mirrors openpi pi0.py:
            #   chunked_loss = per_tok_se * loss_mask * (action_horizon / postfix_count)
            keep = (~prefix_action_mask)[..., None].to(action_mask.dtype)  # [B, H, 1]
            postfix_count = keep.sum(dim=1).clamp_min(1.0)  # [B, 1]
            scale = (float(H) / postfix_count)[..., None]  # [B, 1, 1]
            eff_mask = action_mask * keep * scale
        else:
            eff_mask = action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * eff_mask
        loss = action_loss.sum() / (eff_mask.sum() + 1e-6)

        if prefix_action_mask is not None:
            rtc_metrics = self._compute_rtc_continuity_metrics(
                v_pred=pred_actions,
                actions=actions,
                x_t=noisy_trajectory,
                t_per_tok=t_per_tok,
                prefix_mask=prefix_action_mask,
                delay=delay,
                action_mask=action_mask,
            )
        else:
            rtc_metrics = {}

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "rtc_metrics": rtc_metrics,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    @torch.no_grad()
    def _compute_rtc_continuity_metrics(
        self,
        *,
        v_pred: torch.Tensor,
        actions: torch.Tensor,
        x_t: torch.Tensor,
        t_per_tok: torch.Tensor,
        prefix_mask: torch.Tensor,
        delay: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Continuity metrics measured at the prefix-postfix boundary (log-only).

        These mirror openpi `pi0.py:_compute_rtc_continuity_metrics`. The key concern in
        RTC is that the model's first postfix prediction (slot d) joins smoothly onto
        the clamped prefix tail (slot d-1).

        Predicted clean action under gr00t's convention `x_t = (1 - t) * noise + t * a`,
        velocity target `v = a - noise`:
            x_hat = x_t + (1 - t) * v_pred
        (Differs from openpi only in sign because openpi has t=0 clean, t=1 noise.)

        Returned scalars (mean over the batch unless noted; gradient stopped):
            rtc_mean_delay:        average sampled delay (sanity check on the distribution).
            rtc_postfix_mse:       MSE of x_hat vs ground truth, averaged over postfix
                                   tokens — overall postfix accuracy.
            rtc_boundary_mse:      MSE of x_hat vs ground truth at slot d only, averaged
                                   over examples where d < H. The discontinuity at the
                                   join the robot would feel.
            rtc_pred_boundary_jump: ||x_hat[:, d, :] - actions[:, d-1, :]||_2 averaged
                                   over examples where 1 <= d < H. The clamped prefix
                                   ensures actions[:, d-1, :] equals what the prefix tail
                                   contains, so this is the predicted "step" at the join.
            rtc_true_boundary_jump: ||actions[:, d, :] - actions[:, d-1, :]||_2 over the
                                   same examples — the ground-truth step magnitude.
                                   Compare with rtc_pred_boundary_jump: ratio near 1
                                   means the model preserves natural trajectory smoothness.
        """
        B, H, _ = actions.shape
        # Predicted clean action at every slot. At prefix slots (t_per_tok=1) this
        # collapses to x_t = actions exactly (clamped GT), so x_hat there is trivial.
        x_hat = x_t + (1.0 - t_per_tok)[..., None] * v_pred  # [B, H, A]
        # Mask out padded action dims (e.g. ALOHA uses 14 of max_action_dim=29) so that
        # unconstrained model outputs on padded dims don't pollute MSE / L2 jump.
        amask = action_mask.to(x_hat.dtype)  # [B, H, A] in {0, 1}
        per_tok_dim_count = amask.sum(dim=-1).clamp_min(1.0)  # [B, H]
        clean_se = (((x_hat - actions) * amask) ** 2).sum(dim=-1) / per_tok_dim_count  # [B, H]

        postfix = (~prefix_mask).to(clean_se.dtype)  # [B, H]
        postfix_count_total = postfix.sum().clamp_min(1.0)
        rtc_postfix_mse = (clean_se * postfix).sum() / postfix_count_total

        boundary_valid = (delay < H).to(clean_se.dtype)  # [B]
        d_clamped = delay.clamp_max(H - 1)
        bidx = torch.arange(B, device=delay.device)
        rtc_boundary_mse = (clean_se[bidx, d_clamped] * boundary_valid).sum() / boundary_valid.sum().clamp_min(1.0)

        # Boundary jump magnitude: only meaningful when 1 <= d < H.
        jump_valid = ((delay >= 1) & (delay < H)).to(clean_se.dtype)  # [B]
        prev = (d_clamped - 1).clamp_min(0)
        amask_at_d = amask[bidx, d_clamped]  # [B, A] — assume same active dims at d-1
        x_hat_at_d = x_hat[bidx, d_clamped] * amask_at_d
        a_at_prev = actions[bidx, prev] * amask_at_d
        a_at_d = actions[bidx, d_clamped] * amask_at_d
        pred_jump = (x_hat_at_d - a_at_prev).norm(dim=-1)
        true_jump = (a_at_d - a_at_prev).norm(dim=-1)
        jw = jump_valid.sum().clamp_min(1.0)
        rtc_pred_boundary_jump = (pred_jump * jump_valid).sum() / jw
        rtc_true_boundary_jump = (true_jump * jump_valid).sum() / jw

        return {
            "rtc_mean_delay": delay.float().mean().detach(),
            "rtc_postfix_mse": rtc_postfix_mse.detach(),
            "rtc_boundary_mse": rtc_boundary_mse.detach(),
            "rtc_pred_boundary_jump": rtc_pred_boundary_jump.detach(),
            "rtc_true_boundary_jump": rtc_true_boundary_jump.detach(),
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, state_horizon, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
            options: Optional kwargs for training-time RTC inference. Recognized keys
                (mirrors openpi `Policy.infer(prev_action_chunk=..., inference_delay=...)`):
                  - "prev_action_chunk": tensor (B, action_horizon, action_dim) — first
                    `inference_delay` slots are clamped to this chunk during sampling;
                    the model only generates the postfix.
                  - "inference_delay": int. Defaults to 0 (plain sampling).
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        # Training-time RTC inference path (mirrors openpi pi0.py:sample_actions).
        # Activated when the caller passes `prev_action_chunk` in options. Clamps the
        # first `inference_delay` action slots to the previous chunk every Euler step
        # and uses per-token tau=1 on those slots; the model only generates the postfix.
        prev_action_chunk = (options or {}).get("prev_action_chunk")
        if prev_action_chunk is not None:
            inference_delay = int((options or {}).get("inference_delay", 0))
            return self._sample_actions_training_time_rtc(
                vl_embeds=vl_embeds,
                state_features=state_features,
                embodiment_id=embodiment_id,
                backbone_output=backbone_output,
                prev_action_chunk=prev_action_chunk.to(device=device, dtype=actions.dtype),
                inference_delay=inference_delay,
                batch_size=batch_size,
                device=device,
                dt=dt,
                actions=actions,
            )

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def _sample_actions_training_time_rtc(
        self,
        *,
        vl_embeds: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        prev_action_chunk: torch.Tensor,
        inference_delay: int,
        batch_size: int,
        device: torch.device,
        dt: float,
        actions: torch.Tensor,
    ) -> BatchFeature:
        """Training-time RTC sampling.

        Mirrors openpi `pi0.py:sample_actions` when `prev_action_chunk` is provided:
        clamps the first `inference_delay` slots of the chunk to the previous chunk's
        predictions at every Euler step and sets per-token tau=1 there. The model only
        generates the postfix. Matches the runtime contract `(action_prefix, d) ->
        action_postfix` from the RTC paper.
        """
        H = self.action_horizon
        ah = torch.arange(H, device=device)
        action_prefix_mask = ah < inference_delay  # [H] bool
        action_prefix_mask_b = action_prefix_mask[None, :].expand(batch_size, -1)  # [B, H]

        # Make sure prev chunk has a batch dim and matches dtype.
        if prev_action_chunk.dim() == 2:
            prev_action_chunk = prev_action_chunk[None, ...]
        prev_action_chunk = prev_action_chunk.to(dtype=actions.dtype)

        # Clamp initial state so the first denoising step sees clean prefix slots.
        actions = torch.where(
            action_prefix_mask[None, :, None], prev_action_chunk, actions
        )

        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)  # 0 -> 1 (noise -> clean)
            t_disc_postfix = int(t_cont * self.num_timestep_buckets)
            t_disc_prefix = int(self.num_timestep_buckets)  # tau=1 (clean) on prefix slots

            # Per-token discretized timestep for the action slots.
            t_disc_action = torch.where(
                action_prefix_mask_b,
                torch.full_like(action_prefix_mask_b, t_disc_prefix, dtype=torch.long),
                torch.full_like(action_prefix_mask_b, t_disc_postfix, dtype=torch.long),
            )  # [B, H]
            # State token (slot 0 of sa_embs) carries the postfix scalar.
            t_disc_state = torch.full(
                (batch_size, 1), t_disc_postfix, device=device, dtype=torch.long
            )
            t_disc_full = torch.cat([t_disc_state, t_disc_action], dim=1)  # [B, 1+H]

            action_features = self.action_encoder(actions, t_disc_action, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1)

            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=t_disc_full,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=t_disc_full,
                )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]

            # Re-clamp prefix BEFORE the Euler update (defensive — already clamped at
            # init / previous iter), then take the step, then re-clamp AFTER to keep the
            # prefix exact across the update.
            actions = torch.where(
                action_prefix_mask[None, :, None], prev_action_chunk, actions
            )
            actions = actions + dt * pred_velocity
            actions = torch.where(
                action_prefix_mask[None, :, None], prev_action_chunk, actions
            )

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)
            options: Optional kwargs forwarded to get_action_with_features. Use to pass
                training-time RTC kwargs (prev_action_chunk, inference_delay).

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d6Config):
    if "NVEagle" in config.model_name or "nvidia/Eagle" in config.model_name:
        return EagleBackbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d6(PreTrainedModel):
    """Gr00tN1d6: Vision-Language-Action model with backbone."""

    config_class = Gr00tN1d6Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d6Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d6 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d6ActionHead(config)
        from .processing_gr00t_n1d6 import Gr00tN1d6DataCollator

        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Eagle inputs (prefixed with 'eagle_')
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
