def debug_generation(model, tokenizer, prompt, max_new_tokens=50):
    """Debug version to see what's happening with generation"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Input tokens: {inputs['input_ids'][0].tolist()}")
    print(f"Decoded input: {tokenizer.decode(inputs['input_ids'][0])}")
    
    with torch.no_grad():
        # Get raw outputs first
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs.sequences[0]
    input_length = inputs['input_ids'].shape[1]
    new_tokens = generated_ids[input_length:]
    
    print(f"Generated sequence length: {len(generated_ids)}")
    print(f"Input length: {input_length}")
    print(f"New tokens: {new_tokens.tolist()}")
    print(f"New tokens decoded: '{tokenizer.decode(new_tokens, skip_special_tokens=True)}'")
    
    # Check if first new token is EOS
    if len(new_tokens) > 0:
        first_token = new_tokens[0].item()
        print(f"First generated token: {first_token}")
        print(f"Is EOS? {first_token == tokenizer.eos_token_id}")
        print(f"Token decoded: '{tokenizer.decode([first_token])}'")
    
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# Test this on a simple prompt first:
def test_debug():
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Test with simple prompt first
    simple_prompt = "Hello, my name is"
    print("=== Testing simple prompt ===")
    result1 = debug_generation(model, tokenizer, simple_prompt)
    print(f"Result: '{result1}'\n")
    
    # Test with chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("=== Testing chat template ===")
    result2 = debug_generation(model, tokenizer, chat_prompt)
    print(f"Result: '{result2}'\n")

if __name__ == "__main__":
    test_debug()