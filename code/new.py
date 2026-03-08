# new.py
import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_aime24_dataset():
    dataset_path = "datasets/aime24.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Please place AIME24 dataset at {dataset_path}")
    with open(dataset_path, "r") as f:
        return json.load(f)

def build_qwen_prompt(problem: str) -> str:
    return f"<|im_start|>system\nYou are a helpful math assistant.<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"

def apply_fast_thinking_intervention(model, interventor, coeff, start_layer, end_layer):
    hidden_size = interventor.shape[0]
    print(f">>> Applying intervention (shape={hidden_size}, coeff={coeff}) to layers [{start_layer}, {end_layer})")

    def make_hook(ivt_vector, c):
        def hook_fn(module, input, output):
            # Handle both tuple and tensor outputs
            if isinstance(output, tuple):
                hidden_states = output[0]
                is_tuple = True
            else:
                hidden_states = output
                is_tuple = False

            # Apply intervention based on dimension
            if hidden_states.dim() == 3:
                # Prefill phase: [batch, seq_len, hidden]
                hidden_states[:, -1, :] += c * ivt_vector.to(
                    hidden_states.device, dtype=hidden_states.dtype
                )
            elif hidden_states.dim() == 2:
                # Autoregressive decoding: [batch, hidden] → current token
                hidden_states += c * ivt_vector.to(
                    hidden_states.device, dtype=hidden_states.dtype
                )
            else:
                raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")

            # Reconstruct output
            if is_tuple:
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        return hook_fn

    for idx in range(start_layer, min(end_layer, len(model.model.layers))):
        layer = model.model.layers[idx]
        hook = make_hook(interventor, coeff)
        layer.register_forward_hook(hook)
        print(f"  → Hook on layer {idx}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="aime24")
    parser.add_argument("--result_save_path", type=str, required=True)
    parser.add_argument("--max_output_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--enable_fast_thinking", action="store_true")
    parser.add_argument("--interventor_ckpt", type=str, default="")
    parser.add_argument("--intervention_coeff", type=float, default=0.0)
    parser.add_argument("--intervention_start_layer", type=int, default=18)
    parser.add_argument("--intervention_end_layer", type=int, default=26)
    args = parser.parse_args()

    print("#" * 100)
    print(args)
    print("#" * 100)

    # Load data
    if args.dataset == "aime24":
        data = load_aime24_dataset()
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported.")
    prompts = [build_qwen_prompt(item["problem"]) for item in data]
    #prompts = [build_qwen_prompt("What is 2+2?")]
    #data = [{"problem": "What is 2+2?", "answer": "4"}]

    if args.enable_fast_thinking:
        print(">>> Loading HuggingFace model for intervention...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to("cuda").eval()

        # ========== 干预加载逻辑（优先用纯张量，回退到兼容模式）==========
        interventor = None
        if args.intervention_coeff != 0.0 and args.interventor_ckpt:
            ckpt_path = args.interventor_ckpt

            # 尝试 1: 先看是否有 *_tensor.pt（推荐）
            if ckpt_path.endswith(".pkl"):
                tensor_path = ckpt_path.replace(".pkl", "_tensor.pt")
                if os.path.exists(tensor_path):
                    print(f">>> Found pure tensor file: {tensor_path}")
                    interventor = torch.load(tensor_path, map_location="cpu", weights_only=True)
                else:
                    print(f">>> Pure tensor file not found. Loading from .pkl with fallback...")

            # 尝试 2: 直接加载 .pkl（兼容模式）
            if interventor is None:
                if ckpt_path.endswith(".pkl"):
                    import sys
                    class PCARepReader:
                        def __init__(self, *args, **kwargs): pass
                    sys.modules['__main__'].PCARepReader = PCARepReader
                    raw_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    if hasattr(raw_obj, 'intervenor'):
                        interventor = raw_obj.intervenor
                    else:
                        raise ValueError("Loaded object has no 'intervenor' attribute.")
                else:
                    interventor = torch.load(ckpt_path, map_location="cpu", weights_only=True)

            # 验证
            if interventor.ndim != 1:
                raise ValueError(f"Expected 1D vector, got {interventor.shape}")
            print(f">>> Intervention vector loaded: {interventor.shape}")

            # 应用干预（注意：end_layer 是开区间，所以传入 args.intervention_end_layer + 1）
            apply_fast_thinking_intervention(
                model, interventor,
                coeff=args.intervention_coeff,
                start_layer=args.intervention_start_layer,
                end_layer=args.intervention_end_layer + 1
            )

        # Generate
        results = []
        for idx, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_output_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append({
                "problem": data[idx]["problem"],
                "model_answer": response.strip(),
                "ground_truth": data[idx].get("answer", "")
            })
            ####
            print(f"Processed {idx + 1}/{len(prompts)}: {data[idx]['problem'][:50]}...")
            ###
            if (idx + 1) % 5 == 0:
                print(f"Processed {idx + 1}/{len(prompts)}")

    else:
        # vLLM path (unchanged)
        print(">>> Loading vLLM model (no intervention)...")
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=args.model_name_or_path,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.num_gpus,
            enforce_eager=True,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            max_tokens=args.max_output_tokens,
            stop=["<|im_end|>", "\n\n"]
        )
        outputs = llm.generate(prompts, sampling_params)
        results = [{
            "problem": data[i]["problem"],
            "model_answer": out.outputs[0].text.strip(),
            "ground_truth": data[i].get("answer", "")
        } for i, out in enumerate(outputs)]

    # Save
    os.makedirs(os.path.dirname(args.result_save_path), exist_ok=True)
    with open(args.result_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Inference completed. Results saved to {args.result_save_path}")

if __name__ == "__main__":
    main()