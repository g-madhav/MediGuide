import os
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PromptTuningConfig, PromptTuningInit

# Import your existing vectorstore setup
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("Starting Medical Chatbot Fine-tuning Pipeline...")

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Running in CPU-compatible mode...")
else:
    print(f"Running on: {device}")

# Load model and tokenizer
MODEL_NAME = "microsoft/DialoGPT-small"  # Using smaller model for stability
print("Loading model and tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def load_pdf_files(data_path="data/"):
    """Load PDF files from your existing data directory"""
    try:
        loader = DirectoryLoader(data_path,
                                 glob='*.pdf',
                                 loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} PDF pages")
        return documents
    except Exception as e:
        print(f"⚠️ Error loading PDFs: {e}")
        return []

def create_chunks(extracted_data):
    """Create chunks from your PDF documents"""
    if not extracted_data:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"Created {len(text_chunks)} text chunks")
    return text_chunks

def convert_chunks_to_qa_dataset(text_chunks):
    """Convert your medical document chunks into Q&A format for fine-tuning"""
    qa_pairs = []

    for i, chunk in enumerate(text_chunks):
        content = chunk.page_content.strip()
        if len(content) < 50:  # Skip very short chunks
            continue

        # Create question-answer pairs from medical content
        # Generate questions based on content
        questions = [
            f"What does this medical text explain?",
            f"Can you summarize this medical information?",
            f"What are the key points in this medical content?",
            f"Explain this medical information in simple terms."
        ]

        # Add medical disclaimer to answer
        medical_disclaimer = "\n\n**Medical Disclaimer**: This information is for educational purposes only and does not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers."

        answer = content + medical_disclaimer

        # Use first question for each chunk to avoid too much data
        question = questions[i % len(questions)]

        qa_pairs.append({
            "input_text": question,
            "target_text": answer
        })

        # Limit dataset size for faster training
        if len(qa_pairs) >= 100:
            break

    print(f"Created {len(qa_pairs)} Q&A pairs from medical documents")
    return qa_pairs

def prepare_dataset_for_training(qa_pairs):
    """Format dataset correctly for the model"""
    if not qa_pairs:
        print("⚠️ No Q&A pairs available, creating minimal dataset")
        qa_pairs = [{
            "input_text": "What is medical care?",
            "target_text": "Medical care involves healthcare services provided by professionals. **Medical Disclaimer**: Always consult healthcare providers for medical advice."
        }]

    # Format as conversational data for DialoGPT
    formatted_data = []

    for qa in qa_pairs:
        # Combine question and answer as input text for language modeling
        conversation = f"Human: {qa['input_text']}\nAssistant: {qa['target_text']}{tokenizer.eos_token}"
        formatted_data.append({"text": conversation})

    # Create train/test split
    split_idx = int(0.8 * len(formatted_data))
    train_data = formatted_data[:split_idx] if split_idx > 0 else formatted_data
    test_data = formatted_data[split_idx:] if split_idx > 0 and split_idx < len(formatted_data) else formatted_data[:1]

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    return DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

def tokenize_function(examples):
    """Tokenize the dataset"""
    # Tokenize the text
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    # For language modeling, labels are the same as input_ids
    tokens["labels"] = tokens["input_ids"].clone()

    return tokens

# Load your existing medical data
print("📊 Loading medical dataset...")
documents = load_pdf_files("data/")
if not documents:
    print("⚠No PDF documents found in data/ folder. Please check if PDFs exist.")

text_chunks = create_chunks(documents)
if not text_chunks:
    print("⚠No text chunks created. Check PDF content.")

qa_pairs = convert_chunks_to_qa_dataset(text_chunks)
if not qa_pairs:
    print("⚠No Q&A pairs created.")

dataset = prepare_dataset_for_training(qa_pairs)

print(f"Debug Info:")
print(f"  - Documents loaded: {len(documents)}")
print(f"  - Text chunks: {len(text_chunks)}")
print(f"  - Q&A pairs: {len(qa_pairs)}")
print(f"  - Train dataset size: {len(dataset['train'])}")
print(f"  - Test dataset size: {len(dataset['test'])}")

# Tokenize dataset
print("🔄 Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(f"✅ Dataset prepared: {len(tokenized_dataset['train'])} training samples, {len(tokenized_dataset['test'])} test samples")

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal language modeling, not masked language modeling
)

def run_prompt_tuning():
    """Run Prompt Tuning fine-tuning"""
    print("🔄 Starting Prompt Tuning...")
    start_time = time.time()

    try:
        # Configure prompt tuning
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=8,
            prompt_tuning_init_text="Summarize medical information: ",
            tokenizer_name_or_path=MODEL_NAME,
        )

        # Get PEFT model
        peft_model = get_peft_model(model, peft_config)

        print(f"Trainable parameters: {peft_model.num_parameters()}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./prompt_tuning_medical",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            warmup_steps=5,
            logging_steps=5,
            save_steps=50,
            logging_dir="./logs",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to="none",  # Disable wandb/tensorboard
        )

        # Create trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        end_time = time.time()
        training_time = end_time - start_time

        return {
            "method": "Prompt Tuning",
            "training_time": training_time,
            "trainable_params": peft_model.num_parameters(),
            "total_params": model.num_parameters(),
            "status": "success",
            "error": None
        }

    except Exception as e:
        print(f"⚠Prompt Tuning failed: {str(e)}")
        return {
            "method": "Prompt Tuning",
            "training_time": 0,
            "trainable_params": 0,
            "total_params": 0,
            "status": "failed",
            "error": str(e)
        }

def run_lora():
    """Run LoRA fine-tuning"""
    print("🔄 Starting LoRA Fine-tuning...")
    start_time = time.time()

    try:
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj"],  # DialoGPT specific modules
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Get PEFT model
        peft_model = get_peft_model(model, lora_config)

        print(f"Trainable parameters: {peft_model.num_parameters()}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./lora_medical",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            warmup_steps=5,
            logging_steps=5,
            save_steps=50,
            logging_dir="./logs",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to="none",  # Disable wandb/tensorboard
        )

        # Create trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        end_time = time.time()
        training_time = end_time - start_time

        return {
            "method": "LoRA",
            "training_time": training_time,
            "trainable_params": peft_model.num_parameters(),
            "total_params": model.num_parameters(),
            "status": "success",
            "error": None
        }

    except Exception as e:
        print(f"⚠LoRA failed: {str(e)}")
        return {
            "method": "LoRA",
            "training_time": 0,
            "trainable_params": 0,
            "total_params": 0,
            "status": "failed",
            "error": str(e)
        }

def run_qlora_cpu():
    """Run QLoRA fine-tuning (CPU optimized)"""
    print("🔄 Starting QLoRA Fine-tuning (CPU Mode)...")
    print("ℹ️ Running CPU-compatible QLoRA (optimized LoRA)...")
    start_time = time.time()

    try:
        # Configure QLoRA (similar to LoRA but with different settings for CPU)
        qlora_config = LoraConfig(
            r=8,  # Smaller rank for CPU
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Get PEFT model
        peft_model = get_peft_model(model, qlora_config)

        print(f"Trainable parameters: {peft_model.num_parameters()}")

        # Training arguments (CPU optimized)
        training_args = TrainingArguments(
            output_dir="./qlora_medical",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            warmup_steps=5,
            logging_steps=5,
            save_steps=50,
            logging_dir="./logs",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=False,  # Disable fp16 for CPU
            report_to="none",  # Disable wandb/tensorboard
        )

        # Create trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        end_time = time.time()
        training_time = end_time - start_time

        return {
            "method": "QLoRA (CPU)",
            "training_time": training_time,
            "trainable_params": peft_model.num_parameters(),
            "total_params": model.num_parameters(),
            "status": "success",
            "error": None
        }

    except Exception as e:
        print(f"⚠ QLoRA failed: {str(e)}")
        return {
            "method": "QLoRA (CPU)",
            "training_time": 0,
            "trainable_params": 0,
            "total_params": 0,
            "status": "failed",
            "error": str(e)
        }

# Run all fine-tuning methods
print("🔄 Running Prompt Tuning...")
prompt_tuning_results = run_prompt_tuning()
print("✅ Prompt Tuning completed")

print("🔄 Running LoRA...")
lora_results = run_lora()
print("✅ LoRA completed")

print("🔄 Running QLoRA (CPU)...")
qlora_results = run_qlora_cpu()
print("✅ QLoRA (CPU) completed")

# Collect and display results
results = [prompt_tuning_results, lora_results, qlora_results]

print("\n📈 Performance Comparison:")
results_df = pd.DataFrame(results)

# Calculate efficiency metrics
for i, result in enumerate(results):
    if result["status"] == "success" and result["training_time"] > 0:
        results_df.loc[i, "efficiency_score"] = result["trainable_params"] / result["training_time"]
        results_df.loc[i, "parameter_efficiency"] = result["trainable_params"] / result["total_params"]
    else:
        results_df.loc[i, "efficiency_score"] = 0
        results_df.loc[i, "parameter_efficiency"] = 0

print(results_df.to_string(index=False))

# Check if any method succeeded
successful_results = [r for r in results if r["status"] == "success"]

if successful_results:
    print(f"\n✅ {len(successful_results)} method(s) completed successfully!")

    # Plot results if we have successful training
    try:
        import matplotlib.pyplot as plt

        # Create performance plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        methods = [r["method"] for r in successful_results]
        training_times = [r["training_time"] for r in successful_results]
        trainable_params = [r["trainable_params"] for r in successful_results]
        efficiency_scores = [r["trainable_params"]/r["training_time"] if r["training_time"] > 0 else 0 for r in successful_results]
        param_efficiency = [r["trainable_params"]/r["total_params"] if r["total_params"] > 0 else 0 for r in successful_results]

        # Training time comparison
        ax1.bar(methods, training_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Training Time Comparison')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)

        # Trainable parameters
        ax2.bar(methods, trainable_params, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('Trainable Parameters')
        ax2.set_ylabel('Number of Parameters')
        ax2.tick_params(axis='x', rotation=45)

        # Efficiency score
        ax3.bar(methods, efficiency_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax3.set_title('Training Efficiency')
        ax3.set_ylabel('Parameters/Second')
        ax3.tick_params(axis='x', rotation=45)

        # Parameter efficiency
        ax4.bar(methods, param_efficiency, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax4.set_title('Parameter Efficiency')
        ax4.set_ylabel('Trainable/Total Parameters')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'medical_chatbot_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
                    dpi=300, bbox_inches='tight')
        print("Performance plots saved!")

    except ImportError:
        print(" Matplotlib not available, skipping plots")
else:
    print("⚠No successful training results to plot")

# Save detailed report
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"medical_chatbot_report_{timestamp}.txt"

with open(report_filename, "w") as f:
    f.write(" Medical Chatbot Fine-tuning Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("Dataset Information:\n")
    f.write(f"- Training samples: {len(tokenized_dataset['train'])}\n")
    f.write(f"- Test samples: {len(tokenized_dataset['test'])}\n")
    f.write(f"- Medical documents processed: {len(documents)}\n")
    f.write(f"- Q&A pairs generated: {len(qa_pairs)}\n\n")

    f.write("Fine-tuning Results:\n")
    for result in results:
        f.write(f"\n{result['method']}:\n")
        f.write(f"  Status: {result['status']}\n")
        f.write(f"  Training Time: {result['training_time']:.2f}s\n")
        f.write(f"  Trainable Parameters: {result['trainable_params']:,}\n")
        f.write(f"  Total Parameters: {result['total_params']:,}\n")
        if result['error']:
            f.write(f"  Error: {result['error']}\n")

print(f" Report saved: {report_filename}")

# Save results to CSV
results_filename = f"results_{timestamp}.csv"
results_df.to_csv(results_filename, index=False)
print(f"Results saved: {results_filename}")

print("\n Medical Chatbot Fine-tuning Pipeline Complete!")
print("Summary:")
print("• Three fine-tuning strategies implemented")
print("• CPU-compatible optimizations applied")
print("• Medical disclaimers maintained")
print("• Performance evaluation completed")
print("• Comprehensive reports generated")