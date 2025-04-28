# src/finetune/model.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import torch
import torch.nn as nn
from transformers import ViTModel
from config import Config

class ViTBasePatch16_224(nn.Module):
    def __init__(self, num_classes=37, hidden_dim=768, depth=12, num_heads=12, mlp_dim=3072,
                 dropout=0.2, attention_dropout=0.2):
        super(ViTBasePatch16_224, self).__init__()

        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (224 // 16) * (224 // 16) + 1, hidden_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=attention_dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        if hasattr(nn.init, "trunc_normal_"):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.head.weight, std=0.02)
        else:
            nn.init.normal_(self.pos_embed, std=0.02)
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.head.weight, std=0.02)
        
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x)
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)

        return logits

def build_vit_model(pretrained=True):
    """Builds the ViT model. If pretrained=True, loads Huggingface weights fully including head resizing."""
    model = ViTBasePatch16_224(dropout=0.2, attention_dropout=0.2).to(Config.DEVICE)

    if pretrained:
        print("🔍 Loading Huggingface pre-trained weights directly into custom model...")

        hf_model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2
        )

        state_dict = model.state_dict()

        # Embeddings
        state_dict['patch_embed.weight'] = hf_model.embeddings.patch_embeddings.projection.weight
        state_dict['patch_embed.bias'] = hf_model.embeddings.patch_embeddings.projection.bias
        state_dict['cls_token'] = hf_model.embeddings.cls_token
        state_dict['pos_embed'] = hf_model.embeddings.position_embeddings

        # Transformer Blocks
        for i in range(12):
            prefix_custom = f"transformer.layers.{i}"
            prefix_hf = f"encoder.layer.{i}"

            state_dict[f'{prefix_custom}.self_attn.in_proj_weight'] = torch.cat([
                hf_model.encoder.layer[i].attention.attention.query.weight,
                hf_model.encoder.layer[i].attention.attention.key.weight,
                hf_model.encoder.layer[i].attention.attention.value.weight
            ], dim=0)
            state_dict[f'{prefix_custom}.self_attn.in_proj_bias'] = torch.cat([
                hf_model.encoder.layer[i].attention.attention.query.bias,
                hf_model.encoder.layer[i].attention.attention.key.bias,
                hf_model.encoder.layer[i].attention.attention.value.bias
            ], dim=0)

            state_dict[f'{prefix_custom}.self_attn.out_proj.weight'] = hf_model.encoder.layer[i].attention.output.dense.weight
            state_dict[f'{prefix_custom}.self_attn.out_proj.bias'] = hf_model.encoder.layer[i].attention.output.dense.bias

            state_dict[f'{prefix_custom}.linear1.weight'] = hf_model.encoder.layer[i].intermediate.dense.weight
            state_dict[f'{prefix_custom}.linear1.bias'] = hf_model.encoder.layer[i].intermediate.dense.bias
            state_dict[f'{prefix_custom}.linear2.weight'] = hf_model.encoder.layer[i].output.dense.weight
            state_dict[f'{prefix_custom}.linear2.bias'] = hf_model.encoder.layer[i].output.dense.bias

            state_dict[f'{prefix_custom}.norm1.weight'] = hf_model.encoder.layer[i].layernorm_before.weight
            state_dict[f'{prefix_custom}.norm1.bias'] = hf_model.encoder.layer[i].layernorm_before.bias
            state_dict[f'{prefix_custom}.norm2.weight'] = hf_model.encoder.layer[i].layernorm_after.weight
            state_dict[f'{prefix_custom}.norm2.bias'] = hf_model.encoder.layer[i].layernorm_after.bias

        # Final LayerNorm
        state_dict['norm.weight'] = hf_model.layernorm.weight
        state_dict['norm.bias'] = hf_model.layernorm.bias

        # Load the mapped weights
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")

        print("✅ Pre-trained backbone loaded into custom ViT model!")

    return model

if __name__ == "__main__":
    model = build_vit_model(pretrained=True).to(Config.DEVICE)
    print("Dummy test run...")
    dummy_input = torch.randn(1, 3, 224, 224).to(Config.DEVICE)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f"Dummy output shape: {dummy_output.shape}")
