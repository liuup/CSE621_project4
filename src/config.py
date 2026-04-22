from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeConfig:
    encoder_model_name: str = "distilbert-base-uncased"
    decoder_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    encoder_batch_size: int = 16
    decoder_batch_size: int = 1
    encoder_max_length: int = 256
    decoder_max_length: int = 256
    decoder_max_new_tokens: int = 8
    train_epochs: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    validation_size: float = 0.1
    random_seed: int = 42
    shots_per_class: int = 1
    tfidf_max_features: int = 20000
    output_dir: str = "results"
    require_cuda: bool = True
    prefer_bf16: bool = True
