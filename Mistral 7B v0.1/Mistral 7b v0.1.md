#mistral #llm #mistral-7b-v0_1 #gqa #swa 

![[mistral7b0-1.pdf]]
---

### Quotes from the paper

> Mistral 7B leverages grouped-query attention (GQA) [ 1 ], and sliding window attention (SWA) [6, 3]. GQA significantly accelerates the inference speed, and also reduces the memory requirement during decoding, allowing for higher batch sizes hence higher throughput, a crucial factor for real-time applications. In addition, SWA is designed to handle longer sequences more effectively at a reduced computational cost, thereby alleviating a common limitation in LLMs. These attention mechanisms collectively contribute to the enhanced performance and efficiency of Mistral 7B.

[[mistral7b0-1.pdf#page=1&selection=53,0,67,81|mistral7b0-1, page 1]]

> Figure 2: Rolling buffer cache. The cache has a fixed size of W = 4. Keys and values for position i are stored in position i mod W of the cache. When the position i is larger than W , past values in the cache are overwritten. The hidden state corresponding to the latest generated tokens are colored in orange.

[[mistral7b0-1.pdf#page=3&selection=0,31,29,84|mistral7b0-1, page 3]]


## Terms to go through
- Sliding Window Attention (SWA)
	- SWA and how it integrates with GQA to form the Mistral architecture
- Rolling Buffer Cache
- Pre-fill and Chunking
## Current Notes

### Sliding Window Attention
Mistral 7B uses SWA to tackle the cost problem with a normal Full Attention. Instead of allowing every token to look back at _all_ previous tokens, SWA restricts its view.

**The Core Idea:** Each token can only attend to a fixed number of immediately preceding tokens, plus itself. This fixed number is called the **Window Size (W)**.

For instance, let's say that I have to process the sentence: "**The quick brown fox jumps over the lazy dog**",

Let me assume a **Window Size (W) = 4**. This means each token can look back at itself and the previous 3 tokens (total = 4 tokens).

- **Token 1 ("The"):**
    - Can attend to: "The" (Token 1)
    - Window: [Token 1]
- **Token 2 ("quick"):**
    - Can attend to: "The" (Token 1), "quick" (Token 2)
    - Window: [Token 1, Token 2]
- **Token 3 ("brown"):**
    - Can attend to: "The" (Token 1), "quick" (Token 2), "brown" (Token 3)
    - Window: [Token 1, Token 2, Token 3]
- **Token 4 ("fox"):**
    - Can attend to: "The" (Token 1), "quick" (Token 2), "brown" (Token 3), "fox" (Token 4)
    - Window: [Token 1, Token 2, Token 3, Token 4]
- **Token 5 ("jumps"):**
    - Can attend to: "quick" (Token 2), "brown" (Token 3), "fox" (Token 4), "jumps" (Token 5)
    - Window: [Token 2, Token 3, Token 4, Token 5]
    - **Crucially, it _cannot_ directly attend to "The" (Token 1) because it's outside the window of size 4.**
- **Token 6 ("over"):**
    - Can attend to: "brown" (Token 3), "fox" (Token 4), "jumps" (Token 5), "over" (Token 6)
    - Window: [Token 3, Token 4, Token 5, Token 6]
- **Token 7 ("the"):**
    - Can attend to: "fox" (Token 4), "jumps" (Token 5), "over" (Token 6), "the" (Token 7)
    - Window: [Token 4, Token 5, Token 6, Token 7]

...and so on.

**Visualization:**

I can imagine an attention matrix where rows are the Query token and columns are the Key/Value tokens.

- **Full Attention:** For a sequence of length N, I would have a lower triangular matrix (including the diagonal) of size N x N filled with attention scores.
- **Sliding Window Attention (W=4):** I would have a _band_ along the diagonal. Each row `i` would only have non-zero scores in columns `max(0, i-W+1)` to `i`. Outside this band, the scores are effectively zero (they aren't calculated).

```bash
Full Attention (N=8):      Sliding Window (N=8, W=4):
  1 2 3 4 5 6 7 8            1 2 3 4 5 6 7 8   (Columns = Keys)
1 X                        1 X
2 X X                      2 X X
3 X X X                    3 X X X
4 X X X X                  4 X X X X
5 X X X X X                5   X X X X           <-- Token 5 attends to 2,3,4,5
6 X X X X X X              6     X X X X         <-- Token 6 attends to 3,4,5,6
7 X X X X X X X            7       X X X X       <-- Token 7 attends to 4,5,6,7
8 X X X X X X X X          8         X X X X     <-- Token 8 attends to 5,6,7,8
(Rows = Queries)           (X = Calculated Attention)
```

**Why would this be good for Mistral 7B?**

- **Efficiency:** The paper says that the computational cost drops significantly from O(N²) to roughly O(N * W). Since W (Mistral uses a large window, like 4096 or 8192) is fixed and often much smaller than the potential sequence length N, this is a huge saving in speed and memory.
- **Handling Long Sequences:** This efficiency allows Mistral 7B to process much longer sequences than models of similar size that use full attention over their entire context window.

>[!Important]
>But what about the performance? Does limiting a token's ability to view only till a certain distance of `W` not hamper it?

- While it seems like losing access to distant tokens might hurt, the Mistral authors seem to have found that this approach works pretty darn well. Information from tokens outside the window can still influence the current token _indirectly_. But how?
    - Token 5 is influenced by Tokens 2, 3, 4.
    - Token 9 (later in the sequence) will be influenced by Token 5 (if W=4).
    - So, information from Token 2 can indirectly reach Token 9 by passing through Token 5. This happens layer by layer, allowing information to propagate over longer distances, even if not via direct attention in a single layer.

So, basically the illustration below:

![Screenshot](./Screenshot%202025-05-12%20at%2021.28.12.png)

> At the last layer, using a window size of W = 4096, we have a theoretical attention span of approximately 131K tokens

[[mistral7b0-1.pdf#page=2&selection=157,36,168,6|mistral7b0-1, page 2]]

