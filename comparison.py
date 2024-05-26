import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from preprocess import preprocessing_function
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import utils
import pandas as pd
import baseline_methods
import os
# Load the model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Resize token embeddings if pad_token is not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Load the fine-tuned model
model_path1 = '/content/drive/MyDrive/AIfinal/gpt2_finetuned.pth'
model_path2 = '/content/drive/MyDrive/AI-2024-final-project/gpt2_finetuned.pth'
if os.path.exists(model_path1):
    model_path = model_path1
elif os.path.exists(model_path2):
    model_path = model_path2
else:
    raise FileNotFoundError("Both model paths do not exist.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = utils.load_model(model, model_path, device)

def generate_summary(article, tokenizer, model, device, max_new_tokens=20, window_size=512, step_size=256):
    # Preprocess the article
    processed_article = preprocessing_function(article)
    
    # Ensure processed_article is not empty
    if not processed_article.strip():
        raise ValueError("Processed article is empty.")
    
    # Tokenize the article
    article_tokens = tokenizer.encode(processed_article, return_tensors='pt')
    
    # Use sliding window to handle long articles
    generated_summaries = []
    for start in range(0, article_tokens.size(1), step_size):
        end = min(start + window_size, article_tokens.size(1))
        input_tokens = article_tokens[:, start:end].to(device)
        
        # Check the size of the input tokens
        if input_tokens.size(1) > model.config.n_positions:
            print(f"Skipping window from {start} to {end} as it exceeds model's max position embeddings.")
            continue
        
        # Generate a summary with the model
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                input_tokens, 
                attention_mask=(input_tokens != tokenizer.pad_token_id).float().to(device), 
                max_new_tokens=max_new_tokens
            )
            generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_summaries.append(generated_summary)
        
        if end == article_tokens.size(1):
            break
    
    # Concatenate all generated summaries
    final_summary = " ".join(generated_summaries).strip()
    
    # Print the processed article and generated summaries for comparison
    print(f"Processed Article: {processed_article}")
    print(f"Generated Summaries: {final_summary}")
    
    return final_summary

def calculate_bleu(reference, hypothesis):
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    return sentence_bleu(reference, hypothesis)

def calculate_rouge(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    return scores['rouge-l']['f']

def is_ai_better(baseline_summary, ai_summary):
    rouge = Rouge()
    rouge_scores_baseline = rouge.get_scores(baseline_summary, baseline_summary, avg=True)
    rouge_scores_ai = rouge.get_scores(baseline_summary, ai_summary, avg=True)
    
    rouge_baseline = rouge_scores_baseline['rouge-l']['f']
    rouge_ai = rouge_scores_ai['rouge-l']['f']
    
    return rouge_ai > rouge_baseline

def compare_results(data, baseline_results, num_samples):
    selected_articles = data.sample(n=num_samples)
    ai_better_count = 0
    total_count = len(selected_articles)
    bleu_scores = []
    rouge_scores = []

    for i, row in enumerate(selected_articles.iterrows()):
        index, row = row
        article = row['article']
        baseline_summary = baseline_results[i][2]
        try:
            ai_summary = generate_summary(article, tokenizer, model, device)
            print(f"Article {i + 1}: {article[:250]}...")
            print(f"Baseline Summary: {baseline_summary}")
            print(f"AI Summary: {ai_summary}")
            print("-" * 80)

            bleu_score = calculate_bleu(baseline_summary, ai_summary)
            rouge_score = calculate_rouge(baseline_summary, ai_summary)
            bleu_scores.append(bleu_score)
            rouge_scores.append(rouge_score)
            
            if is_ai_better(baseline_summary, ai_summary):
                ai_better_count += 1
        except Exception as e:
            print(f"Error generating summary for article {i + 1}: {article[:250]}...\n{e}")

    better_percentage = (ai_better_count / total_count) * 100
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge_score = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    
    print(f"AI generated better summaries in {better_percentage}% of cases.")
    print(f"Average BLEU score: {avg_bleu_score}")
    print(f"Average ROUGE score: {avg_rouge_score}")

def main():
    # Load data
    data_path1 = '/content/drive/MyDrive/AIfinal/validation.csv/validation.csv'
    data_path2 = '/content/drive/MyDrive/AI-2024-final-project/validation.csv'
    if os.path.exists(data_path1):
        data_path = data_path1
    elif os.path.exists(data_path2):
        data_path = data_path2
    else:
        raise FileNotFoundError("Both data paths do not exist.")
    data = pd.read_csv(data_path)
    
    # Generate baseline results
    num_articles = 30
    show_results = False
    baseline_results = baseline_methods.process_articles(data, num_articles, show_results)
    
    # Compare AI-generated summaries with baseline
    num_samples = 30  # Set the number of samples to compare
    compare_results(data, baseline_results, num_samples)

if __name__ == "__main__":
    main()
