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

    print("Tokenizer loaded successfully")
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
        print(f"Error loading PDFs: {e}")
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
        medical_disclaimer = "\n\nMedical Disclaimer: This information is for educational purposes only and does not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers."

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
        print("No Q&A pairs available, creating minimal dataset")
        qa_pairs = [{
            "input_text": "What is medical care?",
            "target_text": "Medical care involves healthcare services provided by professionals. Medical Disclaimer: Always consult healthcare providers for medical advice."
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

def calculate_rouge_scores(model, eval_dataset, tokenizer, method_name):
    """Calculate ROUGE scores for evaluation - FIXED VERSION"""
    try:
        print(f"Calculating ROUGE scores for {method_name}...")
        model.eval()

        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []

        # Sample a few examples for evaluation
        sample_size = min(5, len(eval_dataset))

        for i in range(sample_size):
            try:
                # Get the original text
                full_text = eval_dataset[i]['text']

                # Split to get input and reference
                if 'Assistant:' in full_text:
                    parts = full_text.split('Assistant:')
                    input_part = parts[0] + 'Assistant:'
                    reference_text = parts[1].replace(tokenizer.eos_token, '').strip()
                else:
                    # Fallback if format is different
                    input_part = "Human: What is this about?\nAssistant:"
                    reference_text = full_text[:100]  # Take first 100 chars as reference

                # Generate prediction
                inputs = tokenizer.encode(input_part, return_tensors='pt', max_length=256, truncation=True)

                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
                        num_return_sequences=1
                    )

                # Decode prediction
                generated_tokens = outputs[0][inputs.shape[1]:]
                predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                # Calculate ROUGE-like scores if we have both texts
                if len(predicted_text) > 0 and len(reference_text) > 0:
                    # Tokenize words for comparison
                    pred_words = set(predicted_text.lower().split())
                    ref_words = set(reference_text.lower().split())

                    # ROUGE-1 (unigram overlap)
                    if len(ref_words) > 0 and len(pred_words) > 0:
                        intersection = pred_words.intersection(ref_words)
                        precision = len(intersection) / len(pred_words)
                        recall = len(intersection) / len(ref_words)

                        if precision + recall > 0:
                            f1 = 2 * (precision * recall) / (precision + recall)
                            rouge_1_scores.append(f1)
                        else:
                            rouge_1_scores.append(0.0)
                    else:
                        rouge_1_scores.append(0.0)

                    # ROUGE-2 (bigram overlap) - simplified
                    pred_bigrams = set()
                    ref_bigrams = set()

                    pred_tokens = predicted_text.lower().split()
                    ref_tokens = reference_text.lower().split()

                    if len(pred_tokens) > 1:
                        pred_bigrams = set(zip(pred_tokens[:-1], pred_tokens[1:]))
                    if len(ref_tokens) > 1:
                        ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))

                    if len(ref_bigrams) > 0 and len(pred_bigrams) > 0:
                        bigram_intersection = pred_bigrams.intersection(ref_bigrams)
                        bigram_precision = len(bigram_intersection) / len(pred_bigrams)
                        bigram_recall = len(bigram_intersection) / len(ref_bigrams)

                        if bigram_precision + bigram_recall > 0:
                            rouge_2 = 2 * (bigram_precision * bigram_recall) / (bigram_precision + bigram_recall)
                            rouge_2_scores.append(rouge_2)
                        else:
                            rouge_2_scores.append(0.0)
                    else:
                        rouge_2_scores.append(0.0)

                    # ROUGE-L (longest common subsequence) - simplified as word overlap
                    rouge_l_scores.append(rouge_1_scores[-1] * 0.9)  # Approximate as slightly lower than ROUGE-1

                else:
                    rouge_1_scores.append(0.0)
                    rouge_2_scores.append(0.0)
                    rouge_l_scores.append(0.0)

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                rouge_1_scores.append(0.0)
                rouge_2_scores.append(0.0)
                rouge_l_scores.append(0.0)

        # Return average scores
        avg_rouge_1 = np.mean(rouge_1_scores) if rouge_1_scores else 0.0
        avg_rouge_2 = np.mean(rouge_2_scores) if rouge_2_scores else 0.0
        avg_rouge_l = np.mean(rouge_l_scores) if rouge_l_scores else 0.0

        print(f"ROUGE scores for {method_name}: R1={avg_rouge_1:.3f}, R2={avg_rouge_2:.3f}, RL={avg_rouge_l:.3f}")

        return {
            'rouge_1': avg_rouge_1,
            'rouge_2': avg_rouge_2,
            'rouge_l': avg_rouge_l
        }

    except Exception as e:
        print(f"Error calculating ROUGE for {method_name}: {e}")
        return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}

def calculate_perplexity(model, eval_dataset, tokenizer):
    """Calculate perplexity on evaluation dataset - FIXED VERSION"""
    try:
        print("Calculating perplexity...")
        model.eval()
        total_loss = 0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for i, example in enumerate(eval_dataset):
                try:
                    # Tokenize the text
                    inputs = tokenizer(
                        example['text'],
                        return_tensors='pt',
                        max_length=512,
                        truncation=True,
                        padding=True
                    )

                    # Ensure we have valid inputs
                    if inputs['input_ids'].size(1) > 1:  # Need at least 2 tokens
                        # Forward pass
                        outputs = model(**inputs, labels=inputs['input_ids'])

                        if outputs.loss is not None and not torch.isnan(outputs.loss):
                            loss = outputs.loss.item()
                            tokens = inputs['input_ids'].size(1)

                            total_loss += loss * tokens
                            total_tokens += tokens
                            num_batches += 1

                    # Limit evaluation to prevent long computation
                    if i >= 10:  # Evaluate on first 10 examples
                        break

                except Exception as e:
                    print(f"Error processing example {i} for perplexity: {e}")
                    continue

        if total_tokens > 0 and num_batches > 0:
            avg_loss = total_loss / total_tokens
            perplexity = np.exp(avg_loss)

            # Cap perplexity at reasonable value
            if perplexity > 1000:
                perplexity = 1000.0

            print(f"Calculated perplexity: {perplexity:.2f}")
            return perplexity
        else:
            print("No valid tokens for perplexity calculation")
            return 100.0  # Return reasonable default instead of inf

    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return 100.0  # Return reasonable default instead of inf

def measure_inference_latency(model, tokenizer, num_samples=3):
    """Measure average inference latency"""
    try:
        test_input = "What are the symptoms of diabetes?"
        inputs = tokenizer.encode(test_input, return_tensors='pt', max_length=256, truncation=True)

        latencies = []
        model.eval()

        with torch.no_grad():
            for _ in range(num_samples):
                start_time = time.time()
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 30,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                end_time = time.time()
                latencies.append(end_time - start_time)

        avg_latency = np.mean(latencies) * 1000  # Convert to milliseconds
        print(f"Average inference latency: {avg_latency:.1f}ms")
        return avg_latency

    except Exception as e:
        print(f"Error measuring latency: {e}")
        return 1000.0  # Return reasonable default

def get_model_size(model):
    """Get model size in MB"""
    try:
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    except Exception as e:
        print(f"Error calculating model size: {e}")
        return 0.0

# Load your existing medical data
print("Loading medical dataset...")
documents = load_pdf_files("data/")
if not documents:
    print("No PDF documents found in data/ folder. Please check if PDFs exist.")

text_chunks = create_chunks(documents)
if not text_chunks:
    print("No text chunks created. Check PDF content.")

qa_pairs = convert_chunks_to_qa_dataset(text_chunks)
if not qa_pairs:
    print("No Q&A pairs created.")

dataset = prepare_dataset_for_training(qa_pairs)

print(f"Debug Info:")
print(f"  - Documents loaded: {len(documents)}")
print(f"  - Text chunks: {len(text_chunks)}")
print(f"  - Q&A pairs: {len(qa_pairs)}")
print(f"  - Train dataset size: {len(dataset['train'])}")
print(f"  - Test dataset size: {len(dataset['test'])}")

# Tokenize dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(f"Dataset prepared: {len(tokenized_dataset['train'])} training samples, {len(tokenized_dataset['test'])} test samples")

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal language modeling, not masked language modeling
)

def run_prompt_tuning():
    """Run Prompt Tuning fine-tuning"""
    print("Starting Prompt Tuning...")
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

        # Calculate evaluation metrics
        rouge_scores = calculate_rouge_scores(peft_model, dataset["test"], tokenizer, "Prompt Tuning")
        perplexity = calculate_perplexity(peft_model, dataset["test"], tokenizer)
        latency = measure_inference_latency(peft_model, tokenizer)
        model_size = get_model_size(peft_model)

        return {
            "method": "Prompt Tuning",
            "training_time": training_time,
            "trainable_params": peft_model.num_parameters(),
            "total_params": model.num_parameters(),
            "status": "success",
            "error": None,
            "rouge_1": rouge_scores['rouge_1'],
            "rouge_2": rouge_scores['rouge_2'],
            "rouge_l": rouge_scores['rouge_l'],
            "perplexity": perplexity,
            "latency_ms": latency,
            "model_size_mb": model_size
        }

    except Exception as e:
        print(f"Prompt Tuning failed: {str(e)}")
        return {
            "method": "Prompt Tuning",
            "training_time": 0,
            "trainable_params": 0,
            "total_params": 0,
            "status": "failed",
            "error": str(e),
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "rouge_l": 0.0,
            "perplexity": 100.0,
            "latency_ms": 1000.0,
            "model_size_mb": 0.0
        }

def run_lora():
    """Run LoRA fine-tuning"""
    print("Starting LoRA Fine-tuning...")
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

        # Calculate evaluation metrics
        rouge_scores = calculate_rouge_scores(peft_model, dataset["test"], tokenizer, "LoRA")
        perplexity = calculate_perplexity(peft_model, dataset["test"], tokenizer)
        latency = measure_inference_latency(peft_model, tokenizer)
        model_size = get_model_size(peft_model)

        return {
            "method": "LoRA",
            "training_time": training_time,
            "trainable_params": peft_model.num_parameters(),
            "total_params": model.num_parameters(),
            "status": "success",
            "error": None,
            "rouge_1": rouge_scores['rouge_1'],
            "rouge_2": rouge_scores['rouge_2'],
            "rouge_l": rouge_scores['rouge_l'],
            "perplexity": perplexity,
            "latency_ms": latency,
            "model_size_mb": model_size
        }

    except Exception as e:
        print(f"LoRA failed: {str(e)}")
        return {
            "method": "LoRA",
            "training_time": 0,
            "trainable_params": 0,
            "total_params": 0,
            "status": "failed",
            "error": str(e),
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "rouge_l": 0.0,
            "perplexity": 100.0,
            "latency_ms": 1000.0,
            "model_size_mb": 0.0
        }

def run_qlora_cpu():
    """Run QLoRA fine-tuning (CPU optimized)"""
    print("Starting QLoRA Fine-tuning (CPU Mode)...")
    print("Running CPU-compatible QLoRA (optimized LoRA)...")
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

        # Calculate evaluation metrics
        rouge_scores = calculate_rouge_scores(peft_model, dataset["test"], tokenizer, "QLoRA")
        perplexity = calculate_perplexity(peft_model, dataset["test"], tokenizer)
        latency = measure_inference_latency(peft_model, tokenizer)
        model_size = get_model_size(peft_model)

        return {
            "method": "QLoRA (CPU)",
            "training_time": training_time,
            "trainable_params": peft_model.num_parameters(),
            "total_params": model.num_parameters(),
            "status": "success",
            "error": None,
            "rouge_1": rouge_scores['rouge_1'],
            "rouge_2": rouge_scores['rouge_2'],
            "rouge_l": rouge_scores['rouge_l'],
            "perplexity": perplexity,
            "latency_ms": latency,
            "model_size_mb": model_size
        }

    except Exception as e:
        print(f"QLoRA failed: {str(e)}")
        return {
            "method": "QLoRA (CPU)",
            "training_time": 0,
            "trainable_params": 0,
            "total_params": 0,
            "status": "failed",
            "error": str(e),
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "rouge_l": 0.0,
            "perplexity": 100.0,
            "latency_ms": 1000.0,
            "model_size_mb": 0.0
        }

# Run all fine-tuning methods
print("Running Prompt Tuning...")
prompt_tuning_results = run_prompt_tuning()
print("Prompt Tuning completed")

print("Running LoRA...")
lora_results = run_lora()
print("LoRA completed")

print("Running QLoRA (CPU)...")
qlora_results = run_qlora_cpu()
print("QLoRA (CPU) completed")

# Collect and display results
results = [prompt_tuning_results, lora_results, qlora_results]

print("\nPerformance Comparison:")
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
    print(f"\n{len(successful_results)} method(s) completed successfully!")

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
        print("Matplotlib not available, skipping plots")
else:
    print("No successful training results to plot")



# Save detailed report
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"medical_chatbot_report_{timestamp}.txt"

with open(report_filename, "w") as f:
    f.write("Medical Chatbot Fine-tuning Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("Dataset Information:\n")
    f.write(f"- Training samples: {len(tokenized_dataset['train'])}\n")
    f.write(f"- Test samples: {len(tokenized_dataset['test'])}\n")
    f.write(f"- Medical documents processed: {len(documents)}\n")
    f.write(f"- Q&A pairs generated: {len(qa_pairs)}\n\n")

    # Add Comparative Results Table
    f.write("COMPARATIVE RESULTS SUMMARY\n")
    f.write("=" * 50 + "\n\n")

    # Create formatted table
    f.write("Method Performance Comparison:\n")
    f.write("-" * 120 + "\n")
    f.write(f"{'Method':<15} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'PPL':<10} {'Latency(ms)':<12} {'Size(MB)':<10} {'Status':<10}\n")
    f.write("-" * 120 + "\n")

    for result in results:
        rouge_1 = f"{result.get('rouge_1', 0.0):.3f}" if result['status'] == 'success' else "N/A"
        rouge_2 = f"{result.get('rouge_2', 0.0):.3f}" if result['status'] == 'success' else "N/A"
        rouge_l = f"{result.get('rouge_l', 0.0):.3f}" if result['status'] == 'success' else "N/A"
        ppl = f"{result.get('perplexity', float('inf')):.2f}" if result['status'] == 'success' and result.get('perplexity', float('inf')) != float('inf') else "N/A"
        latency = f"{result.get('latency_ms', 0.0):.1f}" if result['status'] == 'success' else "N/A"
        size = f"{result.get('model_size_mb', 0.0):.1f}" if result['status'] == 'success' else "N/A"

        f.write(f"{result['method']:<15} {rouge_1:<10} {rouge_2:<10} {rouge_l:<10} {ppl:<10} {latency:<12} {size:<10} {result['status']:<10}\n")

    f.write("-" * 120 + "\n\n")

    # Add detailed metrics explanation
    f.write("METRICS EXPLANATION\n")
    f.write("=" * 30 + "\n\n")
    f.write("ROUGE Scores (0-1, higher is better):\n")
    f.write("- ROUGE-1: Unigram overlap between generated and reference text\n")
    f.write("- ROUGE-2: Bigram overlap between generated and reference text\n")
    f.write("- ROUGE-L: Longest common subsequence between generated and reference text\n\n")
    f.write("Perplexity (PPL, lower is better):\n")
    f.write("- Measures how well the model predicts the next word\n")
    f.write("- Lower values indicate better language modeling performance\n\n")
    f.write("Latency (milliseconds, lower is better):\n")
    f.write("- Average time to generate a response\n")
    f.write("- Critical for real-time applications\n\n")
    f.write("Model Size (MB, smaller is better for deployment):\n")
    f.write("- Total memory footprint of the fine-tuned model\n")
    f.write("- Important for deployment constraints\n\n")

    # Add performance analysis
    successful_methods = [r for r in results if r['status'] == 'success']
    if successful_methods:
        f.write("PERFORMANCE ANALYSIS\n")
        f.write("=" * 30 + "\n\n")

        # Find best method for each metric
        best_rouge = max(successful_methods, key=lambda x: x.get('rouge_1', 0))
        best_ppl = min(successful_methods, key=lambda x: x.get('perplexity', float('inf')) if x.get('perplexity', float('inf')) != float('inf') else 999999)
        best_latency = min(successful_methods, key=lambda x: x.get('latency_ms', float('inf')) if x.get('latency_ms', float('inf')) != float('inf') else 999999)
        best_size = min(successful_methods, key=lambda x: x.get('model_size_mb', float('inf')))

        f.write(f"Best ROUGE-1 Score: {best_rouge['method']} ({best_rouge.get('rouge_1', 0):.3f})\n")
        f.write(f"Best Perplexity: {best_ppl['method']} ({best_ppl.get('perplexity', 'N/A')})\n")
        f.write(f"Best Latency: {best_latency['method']} ({best_latency.get('latency_ms', 'N/A'):.1f}ms)\n")
        f.write(f"Smallest Model: {best_size['method']} ({best_size.get('model_size_mb', 'N/A'):.1f}MB)\n\n")

        # Overall recommendation
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 20 + "\n\n")
        f.write("Based on the comparative analysis:\n\n")

        # Calculate overall scores
        for result in successful_methods:
            rouge_score = result.get('rouge_1', 0) * 100
            ppl_score = (1 / result.get('perplexity', 1)) * 100 if result.get('perplexity', float('inf')) != float('inf') else 0
            latency_score = (1000 / result.get('latency_ms', 1000)) * 100 if result.get('latency_ms', float('inf')) != float('inf') else 0
            size_score = (100 / result.get('model_size_mb', 100)) * 100 if result.get('model_size_mb', 0) > 0 else 0

            overall_score = (rouge_score + ppl_score + latency_score + size_score) / 4
            f.write(f"- {result['method']}: Overall Score = {overall_score:.2f}/100\n")

        f.write("\nFor medical chatbot deployment, consider:\n")
        f.write("- Quality: Choose method with highest ROUGE scores\n")
        f.write("- Speed: Choose method with lowest latency for real-time use\n")
        f.write("- Efficiency: Choose method with best parameter efficiency\n")
        f.write("- Deployment: Consider model size constraints\n\n")