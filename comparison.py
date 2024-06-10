import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from preprocess import preprocessing_function
import utils
import pandas as pd
import baseline_methods
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

model_path = '/content/drive/MyDrive/AIfinal/gpt2_finetuned.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = utils.load_model(model, model_path, device)

def generate_summary(article, tokenizer, model, device, max_length=200, window_size=512, step_size=256):
    processed_article = preprocessing_function(article) + " TL;DR:"
    if not processed_article.strip():
        raise ValueError("Processed article is empty.")
    article_tokens = tokenizer.encode(processed_article, return_tensors='pt')
    generated_summaries = []

    for start in range(0, article_tokens.size(1), step_size):
        end = min(start + window_size, article_tokens.size(1))
        input_tokens = article_tokens[:, start:end].to(device)
        if input_tokens.size(1) > model.config.n_positions:
            print(f"Skipping window from {start} to {end} as it exceeds model's max position embeddings.")
            continue

        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                input_tokens,
                attention_mask=(input_tokens != tokenizer.pad_token_id).float().to(device),
                max_new_tokens=max_length,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
                min_length=30
            )
            generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_summaries.append(generated_summary.split(' TL;DR:')[-1].strip())

        if end == article_tokens.size(1):
            break

    final_summary = " ".join(generated_summaries).strip()
    return final_summary

def calculate_similarity(reference, hypothesis):
    return SequenceMatcher(None, reference, hypothesis).ratio()

def calculate_bleu(reference, hypothesis):
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)

def calculate_rouge(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    return scores['rouge-l']['f']

def compare_summaries(baseline_summary, ai_summary, original_article):
    similarity_score = calculate_similarity(original_article, ai_summary)
    baseline_similarity_score = calculate_similarity(original_article, baseline_summary)
    
    bleu_score = calculate_bleu(original_article, ai_summary)
    baseline_bleu_score = calculate_bleu(original_article, baseline_summary)
    
    rouge_score = calculate_rouge(original_article, ai_summary)
    baseline_rouge_score = calculate_rouge(original_article, baseline_summary)
    
    return similarity_score, baseline_similarity_score, bleu_score, baseline_bleu_score, rouge_score, baseline_rouge_score

def compare_results(data, baseline_results, num_samples):
    selected_articles = data.sample(n=num_samples)
    ai_better_similarity_count = 0
    baseline_better_similarity_count = 0
    ai_better_bleu_count = 0
    baseline_better_bleu_count = 0
    ai_better_rouge_count = 0
    baseline_better_rouge_count = 0
    total_count = len(selected_articles)
    ai_similarity_scores = []
    baseline_similarity_scores = []
    ai_bleu_scores = []
    baseline_bleu_scores = []
    ai_rouge_scores = []
    baseline_rouge_scores = []

    for i, row in enumerate(selected_articles.iterrows()):
        index, row = row
        article = row['article']
        baseline_summary = baseline_results[i][2]
        try:
            ai_summary = generate_summary(article, tokenizer, model, device)
            similarity_score, baseline_similarity_score, bleu_score, baseline_bleu_score, rouge_score, baseline_rouge_score = compare_summaries(
                baseline_summary, ai_summary, article)
            ai_similarity_scores.append(similarity_score)
            baseline_similarity_scores.append(baseline_similarity_score)
            ai_bleu_scores.append(bleu_score)
            baseline_bleu_scores.append(baseline_bleu_score)
            ai_rouge_scores.append(rouge_score)
            baseline_rouge_scores.append(baseline_rouge_score)

            print(f"Article {i + 1}: {article}")
            print(f"Baseline Summary: {baseline_summary}")
            print(f"AI Summary: {ai_summary}")
            print(f"Similarity Score: {similarity_score}")
            print(f"Baseline Similarity Score: {baseline_similarity_score}")
            print(f"BLEU Score: {bleu_score}")
            print(f"Baseline BLEU Score: {baseline_bleu_score}")
            print(f"ROUGE Score: {rouge_score}")
            print(f"Baseline ROUGE Score: {baseline_rouge_score}")
            print("-" * 80)

            if similarity_score > baseline_similarity_score:
                ai_better_similarity_count += 1
            else:
                baseline_better_similarity_count += 1

            if bleu_score > baseline_bleu_score:
                ai_better_bleu_count += 1
            else:
                baseline_better_bleu_count += 1

            if rouge_score > baseline_rouge_score:
                ai_better_rouge_count += 1
            else:
                baseline_better_rouge_count += 1

        except Exception as e:
            print(f"Error generating summary for article {i + 1}: {article}\n{e}")

    ai_better_similarity_percentage = (ai_better_similarity_count / total_count) * 100
    baseline_better_similarity_percentage = (baseline_better_similarity_count / total_count) * 100
    ai_better_bleu_percentage = (ai_better_bleu_count / total_count) * 100
    baseline_better_bleu_percentage = (baseline_better_bleu_count / total_count) * 100
    ai_better_rouge_percentage = (ai_better_rouge_count / total_count) * 100
    baseline_better_rouge_percentage = (baseline_better_rouge_count / total_count) * 100

    avg_ai_similarity_score = sum(ai_similarity_scores) / len(ai_similarity_scores) if ai_similarity_scores else 0
    avg_baseline_similarity_score = sum(baseline_similarity_scores) / len(baseline_similarity_scores) if baseline_similarity_scores else 0
    avg_ai_bleu_score = sum(ai_bleu_scores) / len(ai_bleu_scores) if ai_bleu_scores else 0
    avg_baseline_bleu_score = sum(baseline_bleu_scores) / len(baseline_bleu_scores) if baseline_bleu_scores else 0
    avg_ai_rouge_score = sum(ai_rouge_scores) / len(ai_rouge_scores) if ai_rouge_scores else 0
    avg_baseline_rouge_score = sum(baseline_rouge_scores) / len(baseline_rouge_scores) if baseline_rouge_scores else 0

    print(f"AI generated better summaries in {ai_better_similarity_percentage}% of cases (Similarity).")
    print(f"Baseline generated better summaries in {baseline_better_similarity_percentage}% of cases (Similarity).")
    print(f"AI generated better summaries in {ai_better_bleu_percentage}% of cases (BLEU).")
    print(f"Baseline generated better summaries in {baseline_better_bleu_percentage}% of cases (BLEU).")
    print(f"AI generated better summaries in {ai_better_rouge_percentage}% of cases (ROUGE).")
    print(f"Baseline generated better summaries in {baseline_better_rouge_percentage}% of cases (ROUGE).")
    print(f"Average AI Summary Similarity Score: {avg_ai_similarity_score}")
    print(f"Average Baseline Summary Similarity Score: {avg_baseline_similarity_score}")
    print(f"Average AI Summary BLEU Score: {avg_ai_bleu_score}")
    print(f"Average Baseline Summary BLEU Score: {avg_baseline_bleu_score}")
    print(f"Average AI Summary ROUGE Score: {avg_ai_rouge_score}")
    print(f"Average Baseline Summary ROUGE Score: {avg_baseline_rouge_score}")

def main():
    data_path = '/content/drive/MyDrive/AIfinal/validation.csv/validation.csv'
    data = pd.read_csv(data_path)

    num_samples = 30
    baseline_results = baseline_methods.process_articles(data, num_samples, show=False)

    compare_results(data, baseline_results, num_samples)

if __name__ == "__main__":
    main()