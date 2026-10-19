"""Day 1: simulate the life of an LLM inference request.

There's no real model here. Each stage sleeps for a realistic-ish amount of time so you
can see where latency comes from: queueing, tokenizing, prefill (scales with prompt
length) and decode (one token at a time).

Usage:
    python3 simulate_request.py "your prompt here"
    python3 simulate_request.py --prompt-tokens 2000 --max-tokens 30
"""

import argparse
import random
import time

# Toy timing assumptions (in seconds), roughly the shape of a small model on one GPU.
NETWORK = 0.020          # client -> server round trip
QUEUE = 0.010            # waiting for the scheduler
TOKENIZE_PER_TOKEN = 0.00001
PREFILL_PER_TOKEN = 0.0002   # prompt processed in parallel -> cheap per token
DECODE_PER_TOKEN = 0.025     # one forward pass per generated token -> expensive

VOCAB = ["Inference", " is", " running", " a", " trained", " model", " on", " new",
         " input", " to", " produce", " predictions", ",", " fast", " and", " cheaply", "."]
EOS = "<eos>"


def tokenize(text):
    """Very rough tokenizer: about 1 token per 4 characters (good enough for a demo)."""
    return [text[i:i + 4] for i in range(0, len(text), 4)]


def log(t0, stage, detail=""):
    print(f"[{(time.perf_counter() - t0) * 1000:8.1f} ms] {stage:<12} {detail}")


def run(prompt, prompt_tokens, max_tokens, seed):
    random.seed(seed)
    t0 = time.perf_counter()

    log(t0, "CLIENT", "POST /v1/chat/completions")
    time.sleep(NETWORK / 2)
    log(t0, "GATEWAY", "API key ok, rate limit ok")

    time.sleep(QUEUE)
    log(t0, "SCHEDULER", "request admitted to batch")

    tokens = tokenize(prompt) if prompt_tokens is None else ["tok"] * prompt_tokens
    time.sleep(TOKENIZE_PER_TOKEN * len(tokens))
    log(t0, "TOKENIZER", f"{len(tokens)} prompt tokens")

    time.sleep(PREFILL_PER_TOKEN * len(tokens))
    log(t0, "PREFILL", f"KV cache filled for {len(tokens)} tokens")

    output, ttft = [], None
    for step in range(max_tokens):
        time.sleep(DECODE_PER_TOKEN)
        # "Sampling": walk the toy vocab in order, ending early sometimes.
        token = VOCAB[step] if step < len(VOCAB) else EOS
        if token == EOS or (step > 8 and random.random() < 0.05):
            log(t0, "DECODE", "EOS -> stop, free KV cache")
            break
        if ttft is None:
            ttft = time.perf_counter() - t0 + NETWORK / 2
        output.append(token)
        log(t0, "DECODE", f"step {step + 1:>2}: {token!r}  (streamed to client)")

    total = time.perf_counter() - t0 + NETWORK / 2
    decode_time = total - ttft
    print("\n--- Response ---")
    print("".join(output))
    print("\n--- Metrics ---")
    print(f"Prompt tokens        : {len(tokens)}")
    print(f"Output tokens        : {len(output)}")
    print(f"TTFT                 : {ttft * 1000:.1f} ms")
    if len(output) > 1:
        print(f"Inter-token latency  : {decode_time / (len(output) - 1) * 1000:.1f} ms")
        print(f"Decode speed         : {(len(output) - 1) / decode_time:.1f} tokens/s")
    print(f"End-to-end latency   : {total * 1000:.1f} ms")


def main():
    parser = argparse.ArgumentParser(description="Simulate an LLM inference request.")
    parser.add_argument("prompt", nargs="?", default="What is AI inference?")
    parser.add_argument("--prompt-tokens", type=int, help="override prompt length in tokens")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.prompt, args.prompt_tokens, args.max_tokens, args.seed)


if __name__ == "__main__":
    main()
