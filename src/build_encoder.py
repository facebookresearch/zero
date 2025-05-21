from transformers.models.clip.modeling_clip import CLIPVisionModel
from transformers.models.clip.image_processing_clip import CLIPImageProcessor


def load_clip(cfg: object, model_name: str = "openai/clip-vit-large-patch14-336"):
    print(f"loading clip from hf: {model_name}")

    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
    model = CLIPVisionModel.from_pretrained(
        model_name,
        attn_implementation="sdpa" if cfg.enable_sdpa else None,
        device_map=None if cfg.fsdp_mode else cfg.device,
        torch_dtype=cfg.ptdtype,
        cache_dir=cfg.hf_cache_dir,
    )
    model.to(cfg.ptdtype)

    processor = CLIPImageProcessor.from_pretrained(
        model_name, cache_dir=cfg.hf_cache_dir
    )
    image_input_size = processor.size["shortest_edge"]

    return model, processor, image_input_size
