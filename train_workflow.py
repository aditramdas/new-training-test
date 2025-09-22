#!/usr/bin/env python3
"""
Complete training workflow: SFT -> DPO for Qwen2.5-Instruct
"""

import os
import sys
import subprocess
import argparse

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STARTING: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("SUCCESS:", description)
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {description}:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[-1000:])  # Last 1000 chars
        if e.stderr:
            print("STDERR:", e.stderr[-1000:])  # Last 1000 chars
        return False

def main():
    parser = argparse.ArgumentParser(description="Run SFT -> DPO training workflow")
    parser.add_argument("--skip-sft", action="store_true", 
                       help="Skip SFT training (use if already completed)")
    parser.add_argument("--skip-dpo", action="store_true", 
                       help="Skip DPO training (only run SFT)")
    parser.add_argument("--data-file", default="preference_pairs_with_placeholders.jsonl",
                       help="Path to preference dataset file")
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        print(f"ERROR: Data file '{args.data_file}' not found!")
        print("Please ensure your preference dataset is in the current directory.")
        sys.exit(1)
    
    print("🎯 Qwen2.5-Instruct Training Workflow")
    print("=====================================")
    print(f"Data file: {args.data_file}")
    print(f"Skip SFT: {args.skip_sft}")
    print(f"Skip DPO: {args.skip_dpo}")
    
    # Step 1: Supervised Fine-Tuning
    if not args.skip_sft:
        print("\\n🔧 STEP 1: Supervised Fine-Tuning (SFT)")
        print("This will adapt Qwen2.5 to your coding domain using chosen responses...")
        
        if not run_command("python src/fine_tuning.py", "Supervised Fine-Tuning"):
            print("❌ SFT failed! Check the error messages above.")
            sys.exit(1)
        
        # Check if SFT model was created
        if not os.path.exists("Qwen2.5-7B-Instruct-SFT"):
            print("❌ SFT model not found after training!")
            sys.exit(1)
        
        print("✅ SFT completed successfully!")
    else:
        print("\\n⏩ STEP 1: Skipping SFT (as requested)")
        if not os.path.exists("Qwen2.5-7B-Instruct-SFT"):
            print("⚠️  Warning: SFT model not found, DPO will use base model")
    
    # Step 2: Direct Preference Optimization
    if not args.skip_dpo:
        print("\\n🎯 STEP 2: Direct Preference Optimization (DPO)")
        print("This will align the model with your preferences using preference pairs...")
        
        if not run_command("python src/dpo_training.py", "Direct Preference Optimization"):
            print("❌ DPO failed! Check the error messages above.")
            sys.exit(1)
        
        print("✅ DPO completed successfully!")
    else:
        print("\\n⏩ STEP 2: Skipping DPO (as requested)")
    
    print("\\n🎉 TRAINING WORKFLOW COMPLETED!")
    print("================================")
    
    if os.path.exists("Qwen2.5-7B-Instruct-DPO-Finetuned"):
        print("✅ Final model saved as: Qwen2.5-7B-Instruct-DPO-Finetuned")
        print("\\n📝 Next steps:")
        print("1. Test your model with: python src/app.py")
        print("2. Update app.py model path to use your fine-tuned model")
        print("3. Consider pushing to HuggingFace Hub for sharing")
    elif os.path.exists("Qwen2.5-7B-Instruct-SFT"):
        print("✅ SFT model available: Qwen2.5-7B-Instruct-SFT")
        print("Run with --skip-sft to continue with DPO training")
    
    print("\\n📊 Check outputs/ directory for training logs and TensorBoard files")

if __name__ == "__main__":
    main()
