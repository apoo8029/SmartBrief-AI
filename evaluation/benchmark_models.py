import os
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.makedirs("evaluation", exist_ok=True)

MODELS_TO_BENCHMARK = [
    "facebook/bart-large-cnn",
    "sshleifer/distilbart-cnn-12-6"
]

SAMPLE_TEXTS = [
    (
        "Artificial Intelligence and Machine Learning engineering demand a robust understanding "
        "of modern deep learning architectures and production deployment pipelines. Utilizing "
        "open-source frameworks such as Hugging Face Transformers allows developers to integrate "
        "state-of-the-art models efficiently, accelerating research prototypes and scalable "
        "enterprise applications alike."
    ),
    (
        "Cloud computing has fundamentally transformed how modern enterprises scale infrastructure, "
        "manage data storage, and deploy microservices. By leveraging containerization tools like "
        "Docker and Kubernetes alongside automated CI/CD pipelines, engineering teams can achieve "
        "unprecedented deployment velocity and high system reliability."
    )
]

def run_model_benchmark():
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation benchmark on device: {device.upper()}\n" + "="*60)
    
    for model_name in MODELS_TO_BENCHMARK:
        print(f"\nEvaluating model: {model_name}...")
        
        load_start = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            model.eval()
            load_time = round(time.time() - load_start, 2)
            print(f" -> Loaded successfully in {load_time} seconds.")
        except Exception as e:
            print(f" -> Failed to load {model_name}: {e}")
            continue
            
        model_latencies = []
        generated_summaries = []
        
        for text in SAMPLE_TEXTS:
            inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(device)
            
            inf_start = time.time()
            with torch.inference_mode():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=80,
                    min_length=25,
                    do_sample=False
                )
            inf_time = time.time() - inf_start
            model_latencies.append(inf_time)
            
            decoded = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            generated_summaries.append(decoded)
            
        avg_latency = round(sum(model_latencies) / len(model_latencies), 2)
        avg_orig_words = round(sum([len(t.split()) for t in SAMPLE_TEXTS]) / len(SAMPLE_TEXTS), 2)
        avg_sum_words = round(sum([len(s.split()) for s in generated_summaries]) / len(generated_summaries), 2)
        compression_pct = round((avg_sum_words / avg_orig_words) * 100, 2)
        
        results.append({
            "Model Name": model_name,
            "Load Time (s)": load_time,
            "Avg Inference Latency (s)": avg_latency,
            "Avg Original Words": avg_orig_words,
            "Avg Summary Words": avg_sum_words,
            "Compression Ratio (%)": compression_pct,
            "Hardware": device.upper()
        })
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
    df_results = pd.DataFrame(results)
    csv_path = "evaluation/model_benchmark_results.csv"
    df_results.to_csv(csv_path, index=False)
    print("\n" + "="*60)
    print("           MODEL BENCHMARK RESULTS SUMMARY            ")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60)
    print(f"Results successfully exported to '{csv_path}'")

if __name__ == "__main__":
    run_model_benchmark()
