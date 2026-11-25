# MemScope

**Does an AI agent's memory actually help, or does it only look like it helps?**

MemScope is a small Python library for testing memory systems in LLM agents. It checks whether a reported improvement comes from *what the agent remembered* or simply from having an extra block of text in the prompt.

## The idea

Adding memory to an agent changes two things at once. It adds **content** (the past experience that was retrieved) and a **slot** (an extra block in the prompt). MemScope works like a placebo in a drug trial: it keeps the slot exactly the same and swaps only the content. If the agent does just as well with meaningless filler as with its real memories, the memory was not the reason for the gain.

| Condition | What goes in the memory slot |
|---|---|
| `none` | Nothing (baseline) |
| `full` | The system's real memories |
| `random` | The same number of memories, drawn at random from the same bank |
| `neutral` | Filler text of the same length with no useful information |
| `oracle` | The correct answer (checks that the setup can detect useful content) |

If `full` beats `neutral`, the memory content helps. If they score the same, the gain came from the slot.

## Install

```bash
pip install -e .
```

## Usage

MemScope sits between a system's retrieval step and its prompt, so the rest of the system stays untouched:

```python
from memscope import MemScope

ms = MemScope("neutral", seed=0, bank=frozen_bank,
              telemetry_path="runs/telemetry.jsonl")

real  = host.retrieve(query)
texts = ms.substitute(key=query, real_texts=real, instance_id=qid)
prompt += host.render(texts)
```

For systems that insert a single memory block, use `ms.substitute_block(...)`.

## Command line

```bash
python -m memscope census --bank memory.jsonl --system reasoningbank   # inspect a memory bank
python -m memscope verify --telemetry runs/telemetry.jsonl             # check a run
```

## Supported systems

- ReasoningBank (`memscope.adapters.reasoningbank`)
- U-Mem and MemRL (`memscope.adapters.legacy`)

## License

Apache-2.0
