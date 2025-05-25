# Bijective Modular Diffusion LLMs - Core Concepts

## Project Hypothesis

**Core Insight**: Bijective modular systems are better suited for non-autoregressive architectures (like diffusion LLMs) than autoregressive ones, providing theoretical and practical advantages over standard transformer approaches.

## Background

### Why Bijective Modules Failed in Autoregressive Models
- Autoregressive generation has inherent information asymmetry (past → future)
- Bijective constraints conflict with the causal masking required for sequential generation
- Empirical testing on causal problems showed poor performance

### Why They Should Work for Diffusion LLMs
- **Information preservation**: No data loss through denoising transformations
- **Exact likelihood computation**: Replace variational bounds with exact calculations
- **Parallel processing**: All tokens processed simultaneously, fitting bijective constraints
- **Theoretical reversibility**: Diffusion process is already conceptually invertible

## SEDD Foundation

### What is SEDD (Score Entropy Discrete Diffusion)
- **Input**: Corrupted/noisy text sequences
- **Process**: Bidirectional attention (no causal masking) 
- **Output**: Denoised text for ALL positions simultaneously
- **Training**: Gradually increase corruption, learn to reverse it
- **Advantage**: Parallel generation + better long-range coherence

### Key Architectural Differences from GPT-style Models
- Bidirectional transformers (BERT-like) instead of causal
- Score estimation rather than direct token prediction
- Entropy-based corruption strategies
- Multi-step denoising inference

## Bijective Module Integration

### Mathematical Foundation
- Each module f: X → Y has exact inverse f⁻¹: Y → X
- Composed transformation: F = f_n ∘ f_{n-1} ∘ ... ∘ f_1
- Jacobian determinant computable and non-zero everywhere
- Exact likelihood: log p(x) = log p(F(x)) + log|det(J_F(x))|

### Implementation Strategy
1. **Replace transformer blocks** with invertible coupling layers
2. **Maintain bidirectional attention** for context
3. **Preserve SEDD's denoising objective** while adding bijective constraints
4. **Enable exact likelihood computation** for improved training

### Expected Advantages
- **Improved training stability** through exact gradients
- **Better reconstruction quality** via information preservation
- **Reduced denoising steps** through more efficient reverse process
- **Controllable generation** with mathematical guarantees

## Technical Implementation Plan

### Phase 1: SEDD Baseline
- Get working SEDD implementation
- Understand discrete diffusion framework
- Establish performance benchmarks

### Phase 2: Bijective Layer Design
- Implement coupling layers for discrete tokens
- Design invertible transformations for text embeddings
- Test individual module reversibility

### Phase 3: Integration & Testing
- Replace SEDD transformer blocks with bijective modules
- Compare performance: standard vs bijective SEDD
- Measure improvements in likelihood computation and generation quality

## Key Research Questions

1. Can bijective constraints improve the denoising process quality?
2. Do exact likelihood computations provide better training signals?
3. What's the computational overhead vs accuracy tradeoff?
4. How do bijective modules handle discrete token spaces effectively?

## Success Metrics

- **Generation quality**: BLEU, perplexity compared to baseline SEDD
- **Training efficiency**: Convergence speed and stability
- **Likelihood accuracy**: Exact vs approximate computations
- **Computational efficiency**: Speed and memory usage

---

*This represents a novel approach to non-autoregressive text generation, combining the parallel processing advantages of diffusion models with the information-preserving properties of bijective transformations.*