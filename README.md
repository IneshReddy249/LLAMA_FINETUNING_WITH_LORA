# LoRA Fine-Tuning Llama 3.2-3B for Cybersecurity Domain Adaptation

Fine-tuning Meta's Llama 3.2-3B base model using LoRA (Low-Rank Adaptation) on cybersecurity instruction data with **data scaling experiments** and **automated LLM-as-judge evaluation**.

A base text-completion model that produces gibberish → a domain-specific cybersecurity assistant scoring **4.45/5** on expert evaluation — with just **500 examples** and **5 minutes** of training.

---

## Results

| Model | Train Loss | Val Loss | GPT-4o-mini Judge Score | Training Time |
|:------|:----------:|:--------:|:-----------------------:|:-------------:|
| Baseline (no fine-tuning) | — | — | 2.30 / 5 | — |
| LoRA · 500 examples | 1.24 | 1.19 | **4.45 / 5** | 5 min |
| LoRA · 2,000 examples | 0.87 | 1.06 | **4.60 / 5** | 10 min |
| LoRA · 5,000 examples | 0.73 | 1.01 | **4.45 / 5** | 19 min |

![Results](results.png)

### Key Findings

**1. Massive quality jump with minimal data**  
500 examples were sufficient to transform a base text-completion model into a cybersecurity domain expert, improving automated judge scores from 2.30 to 4.45/5 — a **93% improvement**.

**2. Diminishing returns on data scaling**  
Going from 500 → 5,000 examples (10x data, 4x training time) yielded only +0.15 in judge score. The model learned the cybersecurity reasoning pattern quickly from the highly consistent dataset format.

**3. Loss vs quality divergence**  
Validation loss kept improving (1.19 → 1.01) across data sizes, but judge scores plateaued. This indicates the 3B model hits a quality ceiling — and that **loss alone is insufficient for evaluation**.

---

## Before vs After Fine-Tuning

**Question:** *"How would you detect lateral movement using Windows Event Logs and Sysmon telemetry?"*

**Before (base model):**
> Produces vague one-liners, repeats the question as a heading, generates blog-post intros, or outputs gibberish. No frameworks, no technical substance.

**After (LoRA fine-tuned, 500 examples):**
> Structured multi-paragraph response covering specific Event IDs (4624, 4688), Sysmon events (1, 9, 12, 13), behavioral analysis techniques, network connection monitoring, process creation tracking, and defensive recommendations with MITRE ATT&CK references.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Llama 3.2-3B Base                   │
│              3.2B parameters (frozen)                │
│                                                     │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐      │
│   │  Attn   │     │   MLP   │     │  Other  │      │
│   │ Layers  │     │ Layers  │     │ Linear  │      │
│   └────┬────┘     └────┬────┘     └────┬────┘      │
│        │               │               │            │
│   ┌────┴────┐     ┌────┴────┐     ┌────┴────┐      │
│   │  LoRA   │     │  LoRA   │     │  LoRA   │      │
│   │ r=16    │     │ r=16    │     │ r=16    │      │
│   │ A(16×d) │     │ A(16×d) │     │ A(16×d) │      │
│   │ B(d×16) │     │ B(d×16) │     │ B(d×16) │      │
│   └─────────┘     └─────────┘     └─────────┘      │
│                                                     │
│         24.3M trainable params (0.75%)              │
└─────────────────────────────────────────────────────┘
```

Each linear layer computes: `output = W·x + B·A·x` where W is frozen and only A, B are trained.

---

## Training Configuration

| Parameter | Value |
|:----------|:------|
| Base Model | `meta-llama/Llama-3.2-3B` |
| LoRA Rank (r) | 16 |
| LoRA Alpha (α) | 16 |
| Target Modules | all-linear |
| Trainable Parameters | 24.3M / 3.2B (0.75%) |
| Learning Rate | 2e-4 with cosine decay |
| Batch Size | 4 × 4 gradient accumulation = 16 effective |
| Epochs | 3 |
| Max Sequence Length | 1,024 tokens |
| Precision | FP16 |
| GPU | NVIDIA H100 80GB |

---

## Dataset

[Trendyol Cybersecurity Instruction-Tuning Dataset](https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset) — 53,200 instruction-response pairs covering:

- MITRE ATT&CK techniques and tactics
- NIST Cybersecurity Framework (CSF)
- Malware analysis and reverse engineering
- Incident response procedures
- Threat intelligence and hunting
- Zero Trust architecture
- API security and DLP evasion techniques

License: Apache 2.0

---

## Evaluation Method

### LLM-as-Judge (GPT-4o-mini)
20 held-out cybersecurity questions scored on a 1–5 rubric:

| Score | Criteria |
|:-----:|:---------|
| 1 | Gibberish, repeats question, completely irrelevant |
| 2 | Vaguely related, no frameworks, no actionable detail |
| 3 | Partially correct, some relevant concepts but shallow |
| 4 | Good technical depth, references frameworks/techniques |
| 5 | Expert-level, cites MITRE ATT&CK/NIST, comprehensive and actionable |

### Validation Perplexity
Cross-entropy loss on a fixed 50-example validation set (indices 10,000–10,050) — separate from all training splits to prevent leakage.

### Generalization Testing
Evaluation questions drawn from index 20,000+ in the dataset — never seen during training at any data size — to confirm the model learned reasoning patterns, not memorized answers.

---

## Experiment Design

```
Data Scaling Experiment:
─────────────────────────────────────────────────────────

  Base Llama 3.2-3B ──┬── LoRA + 500 examples  ──► Eval
                      ├── LoRA + 2000 examples ──► Eval
                      └── LoRA + 5000 examples ──► Eval

  Same LoRA config, same val set, same 20 eval questions
  Only variable: training data size
```

Each run reloads the base model from scratch to ensure independent experiments. All models evaluated on the same 20 unseen questions by the same GPT-4o-mini judge with temperature=0 for reproducibility.

---

## Quick Start

1. Open the notebook in Google Colab (A100 or H100 runtime)
2. Add your HuggingFace token (requires [Llama 3.2 gated access](https://huggingface.co/meta-llama/Llama-3.2-3B))
3. Add your OpenAI API key for LLM-as-judge evaluation
4. Run all cells — ~35 min total on H100, ~60 min on A100

---

## Repository Structure

```
├── LoRA_FineTuning_Llama3_2_CyberSecurity.ipynb   # Complete experiment notebook
├── results.png                                      # Evaluation charts
└── README.md
```

---

## Technologies

- **PEFT** — LoRA adapter injection and management
- **TRL** — SFTTrainer for supervised fine-tuning
- **HuggingFace Transformers** — Model loading, tokenization, inference
- **OpenAI API** — GPT-4o-mini as automated evaluator
- **Matplotlib** — Result visualization

---

## References

- Hu et al., 2021 — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Lee et al., 2026 — [Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning](https://arxiv.org/abs/2602.04998)

---

## Author

**Inesh Reddy Chappidi** — [GitHub](https://github.com/IneshReddy249) · [LinkedIn](https://linkedin.com/in/inesh-reddy)
