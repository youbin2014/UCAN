#!/bin/bash

# UCAN: Towards Strong Certified Defense with Asymmetric Randomization
# Script to reproduce paper results

set -e  # Exit on error

echo "🔬 UCAN Paper Results Reproduction Script"
echo "=========================================="

# Configuration
DATASETS=("mnist" "cifar10")
SIGMAS=(0.25 0.5 1.0)
METHODS=("PersNoise_isoR" "UniversalNoise" "PreassignedNoise")
GPU_ID=${1:-"0"}  # Use first argument as GPU ID, default to "0"
EPOCHS=${2:-200}   # Use second argument as epochs, default to 200

echo "Configuration:"
echo "- GPU ID: $GPU_ID"
echo "- Epochs: $EPOCHS"
echo "- Datasets: ${DATASETS[*]}"
echo "- Sigma values: ${SIGMAS[*]}"
echo ""

# Create necessary directories
mkdir -p model_saved results logs

echo "📁 Created output directories"

# Function to run training
train_model() {
    local dataset=$1
    local arch=$2
    local method=$3
    local sigma=$4
    
    echo "🏋️ Training $method on $dataset (σ=$sigma)..."
    
    case $method in
        "PersNoise_isoR")
            python train_certification_noise.py $dataset $arch ./model_saved/ \
                --method="$method" \
                --lr=0.01 \
                --batch=100 \
                --sigma=$sigma \
                --epochs=$EPOCHS \
                --workers=16 \
                --lr_step_size=50 \
                --gpu="$GPU_ID" \
                --noise_name="Gaussian" \
                --IsoMeasure=True \
                > logs/train_${method}_${dataset}_sigma${sigma}.log 2>&1
            ;;
        "UniversalNoise")
            python train_dataset_noise.py $dataset $arch ./model_saved/ \
                --method="$method" \
                --lr=0.01 \
                --batch=100 \
                --sigma=$sigma \
                --epochs=$EPOCHS \
                --gpu="$GPU_ID" \
                > logs/train_${method}_${dataset}_sigma${sigma}.log 2>&1
            ;;
        "PreassignedNoise")
            python train_pattern_noise.py $dataset $arch ./model_saved/ \
                --method="$method" \
                --pattern_type="center_focus" \
                --lr=0.01 \
                --batch=100 \
                --sigma=$sigma \
                --epochs=$EPOCHS \
                --gpu="$GPU_ID" \
                > logs/train_${method}_${dataset}_sigma${sigma}.log 2>&1
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        echo "✅ Training completed: $method on $dataset (σ=$sigma)"
    else
        echo "❌ Training failed: $method on $dataset (σ=$sigma)"
        echo "Check logs/train_${method}_${dataset}_sigma${sigma}.log for details"
    fi
}

# Function to run certification
certify_model() {
    local dataset=$1
    local arch=$2
    local method=$3
    local sigma=$4
    
    echo "🔒 Certifying $method on $dataset (σ=$sigma)..."
    
    case $method in
        "PersNoise_isoR")
            python certification_certification_noise.py $dataset $arch \
                --method="$method" \
                --batch=1000 \
                --sigma=$sigma \
                --workers=16 \
                --gpu="$GPU_ID" \
                --norm=2 \
                --noise_name="Gaussian" \
                --IsoMeasure=True \
                > results/cert_${method}_${dataset}_sigma${sigma}.txt 2>&1
            ;;
        "UniversalNoise")
            python certification_dataset_noise.py $dataset $arch \
                --method="$method" \
                --batch=1000 \
                --sigma=$sigma \
                --workers=16 \
                --gpu="$GPU_ID" \
                --norm=2 \
                > results/cert_${method}_${dataset}_sigma${sigma}.txt 2>&1
            ;;
        "PreassignedNoise")
            python certification_pattern_noise.py $dataset $arch \
                --method="$method" \
                --batch=1000 \
                --sigma=$sigma \
                --workers=16 \
                --gpu="$GPU_ID" \
                --norm=2 \
                > results/cert_${method}_${dataset}_sigma${sigma}.txt 2>&1
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        echo "✅ Certification completed: $method on $dataset (σ=$sigma)"
    else
        echo "❌ Certification failed: $method on $dataset (σ=$sigma)"
        echo "Check results/cert_${method}_${dataset}_sigma${sigma}.txt for details"
    fi
}

# Function to run baseline
run_baseline() {
    local dataset=$1
    local arch=$2
    local sigma=$3
    
    echo "📊 Running baseline (Cohen et al.) on $dataset (σ=$sigma)..."
    
    # Train baseline
    python train_baselines.py $dataset $arch ./model_saved/ \
        --lr=0.01 \
        --batch=100 \
        --sigma=$sigma \
        --epochs=$EPOCHS \
        --gpu="$GPU_ID" \
        > logs/train_baseline_${dataset}_sigma${sigma}.log 2>&1
    
    # Certify baseline
    python certification_baseline.py $dataset $arch \
        --sigma=$sigma \
        --batch=1000 \
        --gpu="$GPU_ID" \
        --norm=2 \
        > results/cert_baseline_${dataset}_sigma${sigma}.txt 2>&1
    
    echo "✅ Baseline completed: $dataset (σ=$sigma)"
}

# Main execution
echo "🚀 Starting experiments..."

for dataset in "${DATASETS[@]}"; do
    # Determine architecture based on dataset
    if [ "$dataset" == "mnist" ]; then
        arch="mnist_cnn"
    elif [ "$dataset" == "cifar10" ]; then
        arch="cifar_resnet110"
    else
        arch="resnet50"  # For ImageNet
    fi
    
    echo ""
    echo "🎯 Processing dataset: $dataset (architecture: $arch)"
    echo "----------------------------------------"
    
    for sigma in "${SIGMAS[@]}"; do
        echo ""
        echo "📋 Sigma = $sigma"
        
        # Run baseline first
        run_baseline $dataset $arch $sigma
        
        # Run our methods
        for method in "${METHODS[@]}"; do
            train_model $dataset $arch $method $sigma
            certify_model $dataset $arch $method $sigma
        done
    done
done

echo ""
echo "🎉 All experiments completed!"
echo "📊 Results are saved in the 'results/' directory"
echo "📝 Training logs are saved in the 'logs/' directory"
echo ""
echo "To analyze results:"
echo "  python scripts/analyze_results.py results/"
echo ""
echo "To generate paper figures:"
echo "  python scripts/generate_figures.py results/ figures/"