# SEDD Architecture Deep Dive

## Core Architecture Components

### 1. Bidirectional Transformer Backbone
**Unlike GPT (causal)**: Can see entire sequence context
- No causal masking in attention layers
- Each position attends to all other positions
- Similar to BERT but trained for generation, not understanding

### 2. Time Embedding
**Critical for diffusion process**:
- Encodes current "noise level" or corruption step
- Usually sinusoidal embeddings (like positional encoding)
- Added to token embeddings before processing
- Tells model "how noisy is this input right now?"

### 3. Discrete Noise Process
**Text corruption strategies**:
- **Masking**: Replace tokens with [MASK]
- **Substitution**: Replace with random tokens from vocabulary  
- **Deletion/Insertion**: Remove or add random tokens
- **Entropy-based**: Gradually increase randomness

### 4. Score-Based Prediction Head
**Instead of direct token prediction**:
- Outputs probability ratios between clean/noisy states
- More stable than raw probability estimation
- Enables better gradient flow during training

## Training Process

### Forward Process (Adding Noise)
```
Clean text: "The cat sat on the mat"
Step 1:     "The cat sat on the mat"  (t=0, no noise)
Step 2:     "The cat [MASK] on the mat"  (t=1, light noise)
Step 3:     "The [MASK] [MASK] [MASK] the mat"  (t=2, medium noise)
Step 4:     "[MASK] [MASK] [MASK] [MASK] [MASK] [MASK]"  (t=3, heavy noise)
```

### Reverse Process (Denoising)
Model learns to predict clean tokens given noisy input + time step:
- Input: noisy sequence + time embedding
- Output: denoised sequence
- Loss: Cross-entropy between predicted and actual clean tokens

## Key Differences from Standard Diffusion

### Discrete vs Continuous
**Image diffusion**: Gaussian noise on pixel values
**SEDD**: Discrete corruption on token sequences
- Can't use simple Gaussian transitions
- Requires categorical distributions
- More complex noise scheduling

### Vocabulary Constraints  
**Must respect language structure**:
- Can't interpolate between tokens like pixels
- Noise process must stay within vocabulary
- Denoising must produce valid tokens

## Inference Process

### Multi-Step Denoising
1. Start with heavily corrupted text (or pure noise)
2. Apply model to predict less noisy version
3. Repeat with decreasing noise levels
4. Final output: clean generated text

### Parallel Generation
**Key advantage over autoregressive**:
- All tokens generated simultaneously at each step
- No left-to-right dependency
- Faster inference for long sequences

## Connection Points for Bijective Modules

### Where Bijective Layers Could Replace Standard Transformers
1. **Between attention and feedforward**: Invertible mixing
2. **Residual connections**: Make them exactly invertible
3. **Embedding transformations**: Bijective token→vector mapping
4. **Cross-timestep transitions**: Exactly reversible noise/denoise steps

### Theoretical Advantages
- **Exact reverse process**: No approximation errors
- **Perfect information preservation**: Critical for high-quality denoising  
- **Stable training dynamics**: Exact gradients through bijective functions

---

# Implementation Notes

## Current State of SEDD Implementations

### Available Codebases
**d3pm (Discrete Denoising Diffusion Probabilistic Models)**:
- Foundation framework for discrete diffusion
- Handles categorical distributions properly
- Good starting point for SEDD implementation

**Diffusion-LM**:
- Continuous diffusion applied to text embeddings
- Less direct but relevant techniques
- Shows embedding-space approaches

### Key Dependencies
- PyTorch (core framework)
- transformers (HuggingFace, for backbone)
- einops (tensor manipulation)
- wandb (experiment tracking)

## Bijective Layer Implementation Strategy

### Coupling Layers for Text
**Challenge**: Adapting continuous coupling layers to discrete tokens
**Solution**: Work in embedding space, maintain discrete outputs

### Invertible Residual Connections
**Standard**: `output = input + transformation(input)`
**Bijective**: Use invertible residual networks (i-ResNets)

### Memory Considerations
**Bijective models**: Must store intermediate activations for backward pass
**Optimization**: Gradient checkpointing with invertible functions

---

