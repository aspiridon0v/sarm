# convert_weights.py
import os

import numpy as np
import open_clip


def get_vitb32_visual(model):
    # Works for both open_clip and OpenAI CLIP variants that open_clip exposes.
    return model.visual


def get_language_transformer(model):
    return model.text


def extract_qkv_and_out_from_mha(attn):
    # attn is torch.nn.MultiheadAttention
    # QKV are packed along dim 0: [3*d, d]
    W = attn.in_proj_weight.detach().cpu().numpy()  # (3d, d)
    b = attn.in_proj_bias.detach().cpu().numpy()  # (3d,)
    Wq, Wk, Wv = np.split(W, 3, axis=0)
    bq, bk, bv = np.split(b, 3, axis=0)

    Wout = attn.out_proj.weight.detach().cpu().numpy()  # (d, d)
    bout = attn.out_proj.bias.detach().cpu().numpy()  # (d,)
    return (Wq, bq), (Wk, bk), (Wv, bv), (Wout, bout)


def to_numpy(t):
    return t.detach().cpu().numpy()


def main():
    # 1) Load ViT-B/32 CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True)
    model.eval()

    params = {}

    # ===== VISION ENCODER =====
    visual = get_vitb32_visual(model)

    # Sanity: basic shapes
    embed_dim = visual.output_dim  # 512
    vision_width = visual.conv1.out_channels  # 768
    patch_size = visual.conv1.kernel_size[0]  # 32
    grid = visual.grid_size  # 7 (for 224x224)
    vision_layers = len(visual.transformer.resblocks)  # 12
    vision_heads = visual.transformer.width // 64  # 12

    # Patchify conv -> Equinox Conv2d expects (out_c, in_c, kh, kw)
    conv = visual.conv1
    params["visual.patch.weight"] = to_numpy(conv.weight)  # [width, 3, 32, 32]

    # Class token + positional embedding
    params["visual.cls"] = to_numpy(visual.class_embedding)[None, :]  # (1,768)
    params["visual.pos"] = to_numpy(visual.positional_embedding)  # (50, 768)

    # Pre-transformer LayerNorm
    params["visual.ln_pre.weight"] = to_numpy(visual.ln_pre.weight)
    params["visual.ln_pre.bias"] = to_numpy(visual.ln_pre.bias)

    # Post-transformer LayerNorm
    params["visual.ln_post.weight"] = to_numpy(visual.ln_post.weight)
    params["visual.ln_post.bias"] = to_numpy(visual.ln_post.bias)

    # Projection to embed_dim (768 -> 512)
    params["visual.proj.weight"] = to_numpy(visual.proj)  # [768, 512]

    # Vision Transformer blocks
    for i, blk in enumerate(visual.transformer.resblocks):
        (Wq, bq), (Wk, bk), (Wv, bv), (Wout, bout) = extract_qkv_and_out_from_mha(blk.attn)
        base = f"visual.blocks.{i}"
        params[f"{base}.attn.q.weight"] = Wq
        params[f"{base}.attn.q.bias"] = bq
        params[f"{base}.attn.k.weight"] = Wk
        params[f"{base}.attn.k.bias"] = bk
        params[f"{base}.attn.v.weight"] = Wv
        params[f"{base}.attn.v.bias"] = bv
        params[f"{base}.attn.out.weight"] = Wout
        params[f"{base}.attn.out.bias"] = bout
        # LayerNorms
        params[f"{base}.ln1.weight"] = to_numpy(blk.ln_1.weight)
        params[f"{base}.ln1.bias"] = to_numpy(blk.ln_1.bias)
        params[f"{base}.ln2.weight"] = to_numpy(blk.ln_2.weight)
        params[f"{base}.ln2.bias"] = to_numpy(blk.ln_2.bias)
        # MLP: fc1 (Linear), act=QuickGELU, fc2 (Linear)
        params[f"{base}.mlp.fc1.weight"] = to_numpy(blk.mlp.c_fc.weight)
        params[f"{base}.mlp.fc1.bias"] = to_numpy(blk.mlp.c_fc.bias)
        params[f"{base}.mlp.fc2.weight"] = to_numpy(blk.mlp.c_proj.weight)
        params[f"{base}.mlp.fc2.bias"] = to_numpy(blk.mlp.c_proj.bias)

    # ===== TEXT ENCODER =====
    # For standard CLIP, text components are directly on model
    # token_embedding, positional_embedding, transformer, ln_final, text_projection

    text_width = model.transformer.width  # 512
    text_layers = len(model.transformer.resblocks)  # 12
    text_heads = text_width // 64  # 8
    context_length = model.context_length  # 77
    vocab_size = model.vocab_size  # 49408

    # Token embedding
    params["text.token_embedding.weight"] = to_numpy(model.token_embedding.weight)  # (49408, 512)

    # Positional embedding
    params["text.positional_embedding"] = to_numpy(model.positional_embedding)  # (77, 512)

    # Text transformer blocks
    for i, blk in enumerate(model.transformer.resblocks):
        (Wq, bq), (Wk, bk), (Wv, bv), (Wout, bout) = extract_qkv_and_out_from_mha(blk.attn)
        base = f"text.blocks.{i}"
        params[f"{base}.attn.q.weight"] = Wq
        params[f"{base}.attn.q.bias"] = bq
        params[f"{base}.attn.k.weight"] = Wk
        params[f"{base}.attn.k.bias"] = bk
        params[f"{base}.attn.v.weight"] = Wv
        params[f"{base}.attn.v.bias"] = bv
        params[f"{base}.attn.out.weight"] = Wout
        params[f"{base}.attn.out.bias"] = bout
        # LayerNorms
        params[f"{base}.ln1.weight"] = to_numpy(blk.ln_1.weight)
        params[f"{base}.ln1.bias"] = to_numpy(blk.ln_1.bias)
        params[f"{base}.ln2.weight"] = to_numpy(blk.ln_2.weight)
        params[f"{base}.ln2.bias"] = to_numpy(blk.ln_2.bias)
        # MLP
        params[f"{base}.mlp.fc1.weight"] = to_numpy(blk.mlp.c_fc.weight)
        params[f"{base}.mlp.fc1.bias"] = to_numpy(blk.mlp.c_fc.bias)
        params[f"{base}.mlp.fc2.weight"] = to_numpy(blk.mlp.c_proj.weight)
        params[f"{base}.mlp.fc2.bias"] = to_numpy(blk.mlp.c_proj.bias)

    # Final layer norm
    params["text.ln_final.weight"] = to_numpy(model.ln_final.weight)
    params["text.ln_final.bias"] = to_numpy(model.ln_final.bias)

    # Text projection
    # In standard CLIP, text_projection is a Parameter (not nn.Linear)
    if model.text_projection is not None:
        params["text.text_projection"] = to_numpy(model.text_projection)  # (512, 512)

    # 3) Save model meta + weights
    meta = dict(
        embed_dim=embed_dim,
        vision_width=vision_width,
        vision_layers=vision_layers,
        vision_heads=vision_heads,
        patch_size=patch_size,
        grid=grid,
        image_size=224,
        text_width=text_width,
        text_layers=text_layers,
        text_heads=text_heads,
        context_length=context_length,
        vocab_size=vocab_size,
    )

    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    np.savez(
        os.path.join(checkpoints_dir, "clip_vit_b32_openai.npz"),
        **params,
        **{f"meta.{k}": v for k, v in meta.items()},
    )
    print("Saved clip_vit_b32_openai.npz")


if __name__ == "__main__":
    main()
