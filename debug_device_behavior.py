#!/usr/bin/env python3
"""
Debug script to understand actual device behavior on this system.
"""

import torch
from src.data.corruption_truly_fixed import NoiseScheduler, TextCorruptor, CorruptionConfig

def debug_device_behavior():
    print("🔍 Debugging Device Behavior")
    print("=" * 40)
    
    # Check MPS availability
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Test basic tensor device behavior
    print("\n📱 Basic Tensor Device Tests:")
    cpu_tensor = torch.tensor([1, 2, 3])
    print(f"CPU tensor device: {cpu_tensor.device}")
    
    if torch.backends.mps.is_available():
        mps_tensor = cpu_tensor.to('mps')
        print(f"MPS tensor device: {mps_tensor.device}")
        print(f"MPS tensor device type: {type(mps_tensor.device)}")
        print(f"torch.device('mps'): {torch.device('mps')}")
        print(f"Device equality: {mps_tensor.device == torch.device('mps')}")
        print(f"Device str comparison: {str(mps_tensor.device) == 'mps'}")
    
    # Test NoiseScheduler
    print("\n🔧 NoiseScheduler Device Tests:")
    scheduler = NoiseScheduler(num_timesteps=10, device='cpu')
    print(f"Initial scheduler device: {scheduler.device}")
    print(f"Initial noise_schedule device: {scheduler.noise_schedule.device}")
    
    if torch.backends.mps.is_available():
        scheduler_mps = scheduler.to('mps')
        print(f"After .to('mps') scheduler device: {scheduler_mps.device}")
        print(f"After .to('mps') noise_schedule device: {scheduler_mps.noise_schedule.device}")
        print(f"Device type: {type(scheduler_mps.noise_schedule.device)}")
        print(f"Expected device: {torch.device('mps')}")
        print(f"Device equality: {scheduler_mps.noise_schedule.device == torch.device('mps')}")
    
    # Test TextCorruptor
    print("\n🔧 TextCorruptor Device Tests:")
    config = CorruptionConfig(vocab_size=100)
    scheduler = NoiseScheduler(num_timesteps=10, device='cpu')
    corruptor = TextCorruptor(config, scheduler)
    
    print(f"Initial corruptor device: {corruptor.device}")
    
    if torch.backends.mps.is_available():
        corruptor_mps = corruptor.to('mps')
        print(f"After .to('mps') corruptor device: {corruptor_mps.device}")
        print(f"Scheduler device: {corruptor_mps.noise_scheduler.device}")
        
        # Test corruption
        input_ids = torch.randint(0, 100, (1, 5))
        timesteps = torch.randint(0, 10, (1,))
        print(f"Input device: {input_ids.device}")
        
        corrupted, mask = corruptor_mps.corrupt_sequence(input_ids, timesteps)
        print(f"Output device: {corrupted.device}")
        print(f"Mask device: {mask.device}")

if __name__ == "__main__":
    debug_device_behavior()
