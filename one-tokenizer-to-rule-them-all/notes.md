## Paper

[Paper](./paper/paper.md)

## Abstract

Pretraining massively multilingual Large Language Models (LLMs) for many languages at once is challenging due to limited model capacity, scarce high-quality data, and compute constraints. Moreover, the lack of language coverage of the tokenizer makes it harder to address the gap for new languages purely at the post-training stage. In this work, we study what relatively cheap interventions early on in training improve "language plasticity", or adaptation capabilities of the model post-training to new languages. We focus on tokenizer design and propose using a universal tokenizer that is trained for more languages than the primary pretraining languages to enable efficient adaptation in expanding language coverage after pretraining. Our systematic experiments across diverse groups of languages and different training strategies show that a universal tokenizer enables significantly higher language adaptation, with up to 20.2% increase in win rates compared to tokenizers specific to pretraining languages. Furthermore, a universal tokenizer also leads to better plasticity towards languages that are completely unseen in the tokenizer and pretraining, by up to 5% win rate gain. We achieve this adaptation to an expanded set of languages with minimal compromise in performance on the majority of languages included in pre-training.

## Key Learning Notes (first read)

### The Core Problem

- **Post-training bottleneck**: Once a model is pretrained, expanding to new languages is expensive and difficult
- **Tokenizer limitation**: If the tokenizer has poor coverage for new languages, post-training adaptation becomes much harder because:
  - Text gets inefficiently tokenized (longer sequences, poor linguistic alignment)
  - Limited post-training budget gets wasted on tokenization artifacts instead of learning language patterns
  - We can't easily change the tokenizer after pretraining without expensive retraining

### The Solution: Strategic Early Investment

**Two-Phase Setup:**

1. **Pre-training Phase** (Before any model training):
   - Design **Universal tokenizer** trained on ALL 62 languages (not just primary training languages)
   - Use specialized weighting: balance natural data distribution with "language buckets" (languages sharing scripts/families)
   - Choose large vocabulary (250k tokens) for optimal Universal tokenizer performance

2. **Training Phase** (During actual model training):
   - Optional: Reallocate 5% of English data to expanded languages (but not necessary)
   - The Universal tokenizer works even with 0% expanded language data during pretraining

### Experimental Architecture

**Tokenizers:**

- **1 Universal tokenizer**: Trained on all 62 languages across all clusters
- **3 Cluster tokenizers**: One each for European, Asian, and Middle Eastern & Indic clusters (trained only on their primary languages)

**Language Organization:**

- **62 total languages** across 3 geographical clusters
- **Primary languages**: Languages used in initial pretraining for each cluster
- **Expanded languages**: Languages from other clusters used to test adaptation
- **Unseen languages**: 7 languages not in tokenizer or pretraining at all

### Results

**Adaptation Performance:**

- **8x faster adaptation** with Universal tokenizer
- **Up to 20.2% higher win rates** for expanded language adaptation
- **Up to 5% improvement** even for completely unseen languages
- **Minimal compromise** on primary languages (≤2% difference)

### Insight

Instead of optimizing tokenizer for current training needs, invest upfront in broader language coverage to enable much cheaper future scaling. The "relatively cheap early interventions" pay massive dividends in post-training language plasticity.
