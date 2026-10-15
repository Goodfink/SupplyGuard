import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from supplyguard.config import DEFAULT_CONFIG, ModelConfig
from supplyguard.errors import ConfigurationError, ModelGenerationError, ModelLoadError


class HuggingFaceLLM:
    def __init__(self, config: ModelConfig = DEFAULT_CONFIG.models):
        """Load the configured local language model."""
        self.config = config

        print(f"Loading model: {config.language_model_name}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.language_model_name,
                revision=config.language_model_revision,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                config.language_model_name,
                revision=config.language_model_revision,
                dtype=torch.float32,
            ).to(config.device)
            self.model.eval()
            print("✓ Model loaded successfully")
        except Exception as error:
            raise ModelLoadError(
                f"Could not load language model '{config.language_model_name}' at revision "
                f"'{config.language_model_revision}'. Check the model cache and network access. "
                f"Underlying error: {error}"
            ) from error

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Generate a deterministic response or raise an actionable model error."""
        if max_tokens is None:
            max_tokens = self.config.answer_max_tokens
        if max_tokens <= 0:
            raise ConfigurationError("max_tokens must be greater than 0")

        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.prompt_max_length,
            )
            inputs = {key: value.to(self.config.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            response = response.strip()
            if not response:
                raise ModelGenerationError("The language model returned an empty response")
            return response

        except ModelGenerationError:
            raise
        except Exception as error:
            raise ModelGenerationError(f"Language model generation failed: {error}") from error
