## Abstract

Recent advancements in large language models (LLMs) have shifted focus toward scaling inference-time compute—improving performance without retraining the model. A common approach is to sample multiple outputs in parallel, and select one of these as the final output. However, work to date has focused on English and a handful of domains such as math and code. In contrast, we are most interested in techniques that generalize across open-ended tasks, formally verifiable tasks, and across languages. In this work, we study how to robustly scale inference-time compute for open-ended generative tasks in a multilingual, multi-task setting.

Our findings show that both sampling strategy—based on temperature variation—and selection strategy must be adapted to account for diverse domains and varied language settings. We evaluate existing selection methods, revealing that strategies effective in English often fail to generalize across languages. We propose novel sampling and selection strategies specifically adapted for multilingual and multi-task inference scenarios, and show they yield notable gains across languages and tasks. In particular, our combined sampling and selection methods lead to an average +6.8 jump in win-rates for our 8B models on m-ArenaHard-v2.0 prompts, against proprietary models such as Gemini. At larger scale, Command-A (111B model) equipped with our methods, shows +9.0 improvement in win-rates on the same benchmark with just five samples against single-sample decoding, a substantial increase at minimal cost. Our results underscore the need for language- and task-aware approaches to inference-time compute, aiming to democratize performance improvements in underrepresented languages.

---

## Key Learning Notes (from the abstract)

### **The Big Picture Shift**
- **Old approach**: Improve AI by building bigger/better models (expensive, months of training)
- **New approach**: Keep same model, use it more cleverly during inference (cheaper, more accessible)
- This **democratizes** AI improvement - more organizations can participate

### **Core Technique: Best-of-N Sampling**
- Generate multiple responses to same input parallelly, pick the best one
- **Technical terms**: "sampling and reranking" or "best-of-N sampling" 
- Uses more compute during inference but same underlying model

### **The Problem with Current Approaches**
- Most inference-time scaling work focused on **English + easily verifiable tasks** (math, code)
- **Real world needs**: Techniques that work across:
  - **Open-ended tasks**: Creative writing, conversation, subjective analysis
  - **Formally verifiable tasks**: Math, coding, factual questions
  - **All languages**: Not just English

### **Key Research Findings**
1. **Sampling strategy** (using different temperature settings) must be customized
   - **Temperature** = "creativity dial" (low = conservative, high = creative)
   - **Temperature variation** = use different temperatures for different samples
   
2. **Selection strategy** (how we pick the winner) must be language/task aware
   - What works for English math problems fails for Arabic creative writing
   - Cultural differences, language structures, and training biases all matter

3. **One-size-fits-all doesn't work** - need **language- and task-aware approaches**

### **Concrete Results**
- **+6.8 win-rate improvement** for 8B models on multilingual benchmark
- **+9.0 improvement** for larger Command-A 111B model with just 5 samples
- Tested on [**m-ArenaHard-v2.0**](https://huggingface.co/datasets/CohereLabs/m-ArenaHard-v2.0): 498 challenging prompts, each translated to 23 languages

### **Impact**
- **democratize performance improvements in underrepresented languages**
- Make inference-time scaling work beyond English-centric domains
- Enable smaller organizations to improve AI performance without massive retraining costs