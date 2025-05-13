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

### Rolling Buffer Cache

> A fixed attention span means that we can limit our cache size using a rolling buffer cache. The cache has a fixed size of W , and the keys and values for the timestep i are stored in position i mod W of the cache. As a result, when the position i is larger than W , past values in the cache are overwritten, and the size of the cache stops increasing

[[mistral7b0-1.pdf#page=2&selection=186,0,213,72|mistral7b0-1, page 2]]

With SWA, with a window size `W`, a token only needs to attend to the previous `W-1` tokens and itself. The Rolling Buffer Cache leverages this insight from SWA. Instead of storing K and V for _all_ preceding tokens, it only stores the K and V vectors for the most recent `W` tokens.

- **Fixed Size:** It uses a cache with a fixed size, determined by the window size `W`.
- **Mechanism:** I can think of it like a first-in, first-out (FIFO) queue, but implemented efficiently as a circular buffer.
    - The cache holds `W` slots for K/V pairs.
    - When a new token (say, token `t`) is processed, its K and V vectors (Kₜ, Vₜ) are calculated.
    - These new K/V vectors need to be added to the cache.
    - If the cache already holds `W` pairs, the _oldest_ pair (corresponding to token `t-W`) is removed (or overwritten) to make space.
    - The new pair (Kₜ, Vₜ) is added.
- **"Rolling" or "Circular":** This process happens continuously. As new tokens are generated, their K/V pairs are added, and the oldest ones (that fall outside the sliding window) are discarded. The buffer conceptually "rolls" forward. This is often implemented using modular arithmetic on the indices (`index % W`) so that I can keep writing into a fixed-size array, overwriting the oldest entries naturally.

**Visualization:**

Let me assume **Window Size (`W`) = 4** and try to visualize the rolling buffer cache logic I just discussed earlier by tracing the cache state during generation for the input prompt **"The quick brown fox"**:

- **Cache:** Pretty much a buffer designed to hold 4 K/V pairs. Let's represent slots as `[ Slot 0 | Slot 1 | Slot 2 | Slot 3 ]`. I'll use an index pointer that wraps around (modulo 4 since `W` = 4). Using `Pointer = 0` initially.
    
- **Process Token 1 ("The"):** Calculate K₁, V₁. Store at `Pointer=0`.
    
    - Cache: `[ (K₁,V₁) | - | - | - ]`
    - Pointer becomes `(0 + 1) % 4 = 1`.

- **Process Token 2 ("quick"):** Calculate K₂, V₂. Store at `Pointer=1`.
    
    - Cache: `[ (K₁,V₁) | (K₂,V₂) | - | - ]`
    - Pointer becomes `(1 + 1) % 4 = 2`.

- **Process Token 3 ("brown"):** Calculate K₃, V₃. Store at `Pointer=2`.
    
    - Cache: `[ (K₁,V₁) | (K₂,V₂) | (K₃,V₃) | - ]`
    - Pointer becomes `(2 + 1) % 4 = 3`.

- **Process Token 4 ("fox"):** Calculate K₄, V₄. Store at `Pointer=3`.
    
    - Cache: `[ (K₁,V₁) | (K₂,V₂) | (K₃,V₃) | (K₄,V₄) ]` (Cache is full)
    - Pointer becomes `(3 + 1) % 4 = 0`.

- **Generate Token 5 ("jumps"):**
    
    - _Attention Calculation:_ Needs K/V for tokens 1, 2, 3, 4 (relative position -3, -2, -1, 0). All are in the cache.
    - _Calculate K₅, V₅._
    - _Store:_ Store at `Pointer=0`. This **overwrites** (K₁, V₁), which is the oldest entry and no longer needed for the next step according to SWA.
    - Cache: `[ (K₅,V₅) | (K₂,V₂) | (K₃,V₃) | (K₄,V₄) ]`
    - Pointer becomes `(0 + 1) % 4 = 1`.

- **Generate Token 6 ("over"):**
    
    - _Attention Calculation:_ Needs K/V for tokens 2, 3, 4, 5 (relative position -3, -2, -1, 0). All are in the cache.
    - _Calculate K₆, V₆._
    - _Store:_ Store at `Pointer=1`. This **overwrites** (K₂, V₂).
    - Cache: `[ (K₅,V₅) | (K₆,V₆) | (K₃,V₃) | (K₄,V₄) ]`
    - Pointer becomes `(1 + 1) % 4 = 2`.

...and so on. The cache always contains the K/V pairs for the most recent `W=4` tokens, and its memory size never increases beyond `W`. In the paper, this is visualized like the following:

![Rolling Buffer Cache screenshot](./Pasted%20image%2020250513081338.png)
