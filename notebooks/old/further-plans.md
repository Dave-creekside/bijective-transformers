# Context Summary for Coding Assistant

## Background
User has built a **bijective discrete diffusion model** for text generation that:
- Uses invertible coupling layers in transformer blocks instead of standard feedforward layers
- Enables **exact likelihood computation** (no variational bounds)
- Shows unique **oscillatory training dynamics** - rapid loss fluctuations around exponential decay
- Converges in ~5 epochs vs 10-20 for standard transformers
- Maintains bidirectional attention for iterative denoising

## Key Architecture Details
- **Coupling layers**: Split hidden state h into h₁, h₂; transform as y₁=h₁, y₂=h₂+f(h₁;θ)
- **Unit Jacobian determinant**: log|det(J)|=0, so no computational overhead for exact likelihood
- **Training objective**: L_total = L_denoise + λ×L_likelihood + L_invert
- **Corruption process**: Masking, substitution, deletion with cosine schedules

## Metaparameters We Discussed

### Core Architecture
- `n_bijective_layers`: Number of bijective transformer blocks (6-8 recommended start)
- `bijective_ratio`: Fraction of layers that are bijective (1.0=fully bijective, 0.5=every other layer)
- `coupling_hidden_dim`: Hidden dimension in coupling function f (typically 2×embedding_dim)
- `coupling_depth`: Number of layers in coupling network (2-3 standard)

### Training Parameters
- `lambda_likelihood`: Weight for exact likelihood loss (0.1 standard)
- `lambda_invert`: Invertibility regularization weight (very small, ~0.01% of total loss)
- `T`: Number of diffusion timesteps (100-500)
- Corruption schedule coefficients for masking/substitution/deletion

### Multi-Scale Options
- `enable_multiscale`: Local/segment/global coupling at different resolutions
- `segment_length`: For segment-level operations

## Innovative Bijective MoE Idea

### Concept
Instead of traditional MoE with gating/routing, create **bijective MoE where experts are coupled through invertible transformations**:

```python
# Traditional MoE: output = Σ(gate_i * expert_i(input))
# Bijective MoE: All experts active, coupled through bijective layers

class BijectiveMoE(nn.Module):
    def __init__(self, n_experts, dim):
        self.experts = nn.ModuleList([TransformerBlock(dim) for _ in range(n_experts)])
        self.bijective_coupler = CouplingLayer(n_experts * dim)
    
    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]
        coupled = self.bijective_coupler(torch.cat(expert_outputs, dim=-1))
        return coupled
```

### Advantages Over Traditional MoE
- **No gating complexity**: All experts always active
- **No load balancing issues**: Bijective coupling prevents expert collapse
- **Information preservation**: Exact likelihood across entire MoE system
- **Coordinated specialization**: Experts specialize through coupling dynamics, not isolation
- **Stable training**: All experts get gradients every step

### Implementation Strategy
1. **Start simple**: 2-expert bijective MoE with single coupling layer
2. **Monitor**: Whether oscillatory training dynamics persist with multiple experts
3. **Scale up**: Gradually increase expert count
4. **Hybrid option**: Sparse selection of experts, but bijective coupling among active ones

## Next Steps
User wants to:
1. Experiment with changing number of bijective layers in existing model
2. Build a simple 2-expert bijective MoE as proof of concept
3. Compare training dynamics and convergence properties

## Key Implementation Note
User has existing bijective diffusion model code + notebook. The MoE extension would build on this foundation by duplicating transformer blocks and adding coupling between their outputs - much simpler than traditional MoE since no gating/routing needed.

The "final layers connect" concept could be the key innovation - getting MoE scaling benefits while maintaining bijective guarantees and the unique 5-epoch convergence property.