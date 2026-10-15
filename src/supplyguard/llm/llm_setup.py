from supplyguard.reproducibility import DEVICE
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

class HuggingFaceLLM:
    def __init__(self, model_name=MODEL_NAME, model_revision=MODEL_REVISION):
        """Initialize with a very small, fast model"""
        self.model_name = model_name
        self.model_revision = model_revision
        
        print(f"Loading model: {model_name}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=model_revision,
                dtype=torch.float32
            ).to(DEVICE)
            self.model.eval()
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def generate(self, prompt, temperature=0.3, max_tokens=150):
        """Generate response - fast and simple"""
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=16384
            )
            inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return response.strip()
        
        except Exception as e:
            print(f"Error: {e}")
            return None
