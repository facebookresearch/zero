"""
HF repo: https://huggingface.co/kaiyuyue/zero-checkpoints/tree/main
"""

BASE_REPO_URL = "https://huggingface.co/kaiyuyue/zero-checkpoints/resolve/main/"

HF_HUB_LINKS = {
    "3": {
        "encoder": "llama3.2-3b_surrogate-trained-encoder/ckpt_01_0004880.pth",
        "translator": "llama3.2-3b_adapter_translator/ckpt_01_0003568.pth",
    },
    "8": {
        "encoder": "llama3.1-8b_surrogate-trained-encoder/ckpt_01_0004880.pth",
        "translator": "llama3.1-8b_adapter_translator/ckpt_01_0004393.pth",
    },
    "70": {
        "encoder": "llama3.1-70b_surrogate-trained-encoder/ckpt_01_0004880.pth",
        "translator": "llama3.1-70b_adapter_translator/ckpt_01_0004393.pth",
    },
    "tokenizer": {
        "model": "llama3.x-tokenizer/tokenizer.model",
    },
}
