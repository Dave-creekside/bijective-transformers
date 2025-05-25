# Phase 2 Plan: Bijective Components Integration
**Date**: 2025-05-24  
**Status**: 📋 PLANNING  
**Estimated Duration**: 3-4 weeks  
**Goal**: Integrate bijective (invertible) layers into discrete diffusion model  

## Overview
Phase 2 is the core innovation of this project - replacing standard transformer components with bijective equivalents to enable exact likelihood computation and perfect information preservation in discrete diffusion.

## Theoretical Foundation

### Why Bijective + Discrete Diffusion?
1. **Exact Likelihood**: Replace variational bounds with exact calculations
2. **Information Preservation**: No data loss through transformations
3. **Parallel Processing**: Bidirectional attention compatible with bijective constraints
4. **Reversible Denoising**: Mathematically guaranteed invertibility

### Mathematical Requirements
- Each layer f: X → Y must have exact inverse f⁻¹: Y → X
- Jacobian determinant must be computable and non-zero
- Composed transformations: F = f_n ∘ f_{n-1} ∘ ... ∘ f_1
- Exact likelihood: log p(x) = log p(F(x)) + log|det(J_F(x))|

## Implementation Strategy

### Week 1: Coupling Layer Foundation
**Goal**: Implement core bijective building blocks

#### Day 1-2: Basic Coupling Layers
- **File**: `src/layers/coupling.py`
- **Components**:
  - `AdditiveCouplingLayer`: y₁ = x₁, y₂ = x₂ + f(x₁)
  - `AffineCouplingLayer`: y₁ = x₁, y₂ = x₂ ⊙ exp(s(x₁)) + t(x₁)
  - `NeuralSplineCouplingLayer`: More expressive transformations
- **Features**:
  - Exact invertibility by construction
  - Configurable split dimensions
  - Multiple coupling network architectures

#### Day 3-4: Invertibility Framework
- **File**: `src/utils/invertibility.py`
- **Components**:
  - `InvertibilityTester`: Rigorous testing of f⁻¹(f(x)) = x
  - `JacobianComputer`: Efficient determinant calculation
  - `NumericalStabilityChecker`: Condition number analysis
- **Features**:
  - Automated testing with various input distributions
  - Tolerance checking and error reporting
  - Performance profiling

#### Day 5-7: Memory Management
- **File**: `src/utils/memory.py`
- **Components**:
  - `BijectiveCheckpoint`: Gradient checkpointing for invertible functions
  - `ActivationRecomputation`: Trade computation for memory
  - `MemoryProfiler`: Track memory usage patterns
- **Features**:
  - Recompute activations using inverse functions
  - Strategic checkpointing for memory efficiency
  - Memory usage optimization

### Week 2: Integration Architecture
**Goal**: Replace transformer components with bijective equivalents

#### Day 8-10: Bijective Transformer Blocks
- **File**: `src/models/bijective_transformer.py`
- **Components**:
  - `BijectiveTransformerBlock`: Invertible transformer layer
  - `InvertibleResidualConnection`: Exact reversible residuals
  - `BijectiveMultiHeadAttention`: Attention with invertible projections
- **Features**:
  - Drop-in replacement for standard blocks
  - Configurable bijective vs standard layers
  - Hybrid architectures support

#### Day 11-12: Likelihood Computation
- **File**: `src/utils/likelihood.py`
- **Components**:
  - `ExactLikelihoodComputer`: log p(x) calculation
  - `JacobianAccumulator`: Efficient determinant tracking
  - `LogDetEstimator`: Approximation methods for large models
- **Features**:
  - Exact likelihood for bijective components
  - Efficient computation through layer composition
  - Numerical stability safeguards

#### Day 13-14: Configuration & Testing
- **Files**: `configs/model/bijective_transformer.yaml`, `test_phase2.py`
- **Components**:
  - Bijective model configurations
  - Comprehensive test suite for invertibility
  - Performance benchmarking framework
- **Features**:
  - Easy switching between standard/bijective modes
  - Automated invertibility validation
  - Memory and speed profiling

### Week 3: Optimization & Scaling
**Goal**: Optimize performance and handle scaling challenges

#### Day 15-17: Performance Optimization
- **Focus Areas**:
  - Memory usage optimization
  - Gradient computation efficiency
  - Numerical stability improvements
  - Device-specific optimizations (MPS)
- **Deliverables**:
  - Optimized coupling layer implementations
  - Efficient Jacobian computation
  - Memory-efficient training procedures

#### Day 18-21: Architecture Variants
- **Hybrid Approaches**:
  - Partial bijective replacement (some layers only)
  - Selective coupling (certain dimensions only)
  - Multi-scale bijective processing
- **Evaluation**:
  - Performance vs accuracy trade-offs
  - Memory usage analysis
  - Training stability assessment

### Week 4: Evaluation & Refinement
**Goal**: Comprehensive testing and performance analysis

#### Day 22-24: Comprehensive Testing
- **Test Suites**:
  - Invertibility validation across all components
  - Memory usage profiling
  - Training stability analysis
  - Generation quality assessment
- **Benchmarking**:
  - Bijective vs standard model comparison
  - Likelihood computation accuracy
  - Training speed and convergence

#### Day 25-28: Analysis & Documentation
- **Performance Analysis**:
  - Detailed metrics comparison
  - Memory and computational overhead
  - Training dynamics analysis
- **Documentation**:
  - Complete API documentation
  - Usage examples and tutorials
  - Performance optimization guide

## Key Components to Implement

### 1. Coupling Layers (`src/layers/`)
```
coupling.py
├── AdditiveCouplingLayer
├── AffineCouplingLayer
├── NeuralSplineCouplingLayer
└── CouplingLayerConfig
```

### 2. Bijective Utilities (`src/utils/`)
```
invertibility.py
├── InvertibilityTester
├── JacobianComputer
└── NumericalStabilityChecker

memory.py
├── BijectiveCheckpoint
├── ActivationRecomputation
└── MemoryProfiler

likelihood.py
├── ExactLikelihoodComputer
├── JacobianAccumulator
└── LogDetEstimator
```

### 3. Bijective Models (`src/models/`)
```
bijective_transformer.py
├── BijectiveTransformerBlock
├── InvertibleResidualConnection
├── BijectiveMultiHeadAttention
└── BijectiveTransformerConfig

bijective_diffusion.py
├── BijectiveDiffusionModel
├── ExactLikelihoodDiffusion
└── HybridDiffusionModel
```

### 4. Testing Framework (`tests/`)
```
test_coupling_layers.py
test_invertibility.py
test_bijective_transformer.py
test_memory_efficiency.py
test_likelihood_computation.py
```

## Risk Assessment & Mitigation

### High-Risk Areas
1. **Memory Explosion**: Bijective models require storing all activations
   - *Mitigation*: Gradient checkpointing, activation recomputation
2. **Numerical Instability**: Inverse computations can be unstable
   - *Mitigation*: Spectral normalization, condition number monitoring
3. **Training Divergence**: Bijective constraints may hurt optimization
   - *Mitigation*: Careful initialization, learning rate scheduling
4. **Performance Degradation**: Computational overhead vs accuracy
   - *Mitigation*: Hybrid architectures, selective bijective layers

### Medium-Risk Areas
1. **Implementation Complexity**: Bijective layers are complex
   - *Mitigation*: Incremental implementation, extensive testing
2. **Device Compatibility**: MPS backend edge cases
   - *Mitigation*: Comprehensive device testing, fallback options

## Success Criteria

### Technical Milestones
- [ ] All coupling layers pass invertibility tests (error < 1e-6)
- [ ] Bijective transformer blocks integrate without errors
- [ ] Memory usage remains reasonable (< 2x baseline)
- [ ] Training converges stably
- [ ] Exact likelihood computation works correctly

### Performance Targets
- **Accuracy**: Match or exceed baseline discrete diffusion
- **Memory**: < 2x memory usage vs standard model
- **Speed**: < 3x training time vs standard model
- **Stability**: Consistent training without divergence

### Quality Gates
1. **Unit Tests**: All bijective components pass invertibility tests
2. **Integration Tests**: Full model trains without errors
3. **Performance Tests**: Memory and speed within acceptable bounds
4. **Accuracy Tests**: Generation quality maintained or improved

## Monitoring & Debugging

### Key Metrics to Track
- Invertibility error: ||x - f⁻¹(f(x))||₂
- Jacobian condition number
- Memory usage per layer
- Training loss stability
- Generation quality (BLEU, perplexity)

### Debugging Tools
- Invertibility test suite
- Memory profiler
- Gradient flow analyzer
- Numerical stability checker

## Fallback Plans

### If Bijective Integration Fails
1. **Partial Integration**: Use bijective layers selectively
2. **Hybrid Approach**: Mix standard and bijective components
3. **Approximation Methods**: Use approximate invertibility
4. **Baseline Fallback**: Return to standard discrete diffusion

### If Performance is Poor
1. **Architecture Optimization**: Reduce bijective layer count
2. **Memory Optimization**: More aggressive checkpointing
3. **Approximation**: Use approximate likelihood computation
4. **Hardware Scaling**: Consider larger memory systems

## Expected Outcomes

### Best Case Scenario
- Bijective model matches baseline accuracy
- Exact likelihood provides better training signals
- Memory usage acceptable for development
- Clear path to scaling and optimization

### Realistic Scenario
- Bijective model works with some performance trade-offs
- Hybrid architecture provides good balance
- Memory usage manageable with optimization
- Proof of concept demonstrates feasibility

### Worst Case Scenario
- Bijective constraints too restrictive for text
- Memory requirements prohibitive
- Training instability issues
- Fall back to baseline with lessons learned

---

**Next Action**: Begin Week 1 implementation with coupling layers
**Dependencies**: Phase 1 completion (✅ Done)
**Resources**: M3 Mac development environment, comprehensive test suite
