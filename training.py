import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from preprocess import preprocessing_function  # Assuming this function exists
from utils import save_model  # Assuming this function exists
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler

class ArticleDataset(Dataset):
    def __init__(self, articles, summaries, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.articles = articles
        self.summaries = summaries
        self.max_len = max_len

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        summary = self.summaries[idx]
        processed_article = preprocessing_function(article)
        article_tokens = self.tokenizer.encode(processed_article + " TL;DR:", return_tensors='pt').squeeze(0)

        if len(article_tokens) > self.max_len:
            article_tokens = article_tokens[:self.max_len]

        summary_tokens = self.tokenizer.encode(summary, return_tensors='pt').squeeze(0)
        return article_tokens, summary_tokens

def collate_fn(batch):
    articles, summaries = zip(*batch)
    articles = pad_sequence(articles, batch_first=True, padding_value=50256)
    summaries = pad_sequence(summaries, batch_first=True, padding_value=50256)
    return articles, summaries

def generate_summary(article_tokens, model, tokenizer, device, max_length=50, num_beams=3):
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            article_tokens.to(device),
            attention_mask=(article_tokens != tokenizer.pad_token_id).float().to(device),
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
            min_length=30
        )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.split(' TL;DR:')[-1].strip()

def sliding_window_processing(article_tokens, window_size=256, step_size=128):
    chunks = []
    for start in range(0, article_tokens.size(1), step_size):
        end = min(start + window_size, article_tokens.size(1))
        chunks.append(article_tokens[:, start:end])
        if end == article_tokens.size(1):
            break
    return chunks

def train_model(data_loader, tokenizer, model, device, optimizer, criterion, scaler, max_length=50, clip_value=1.0, window_size=256, step_size=128):
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 50256
    model.train()
    article_count = 0
    for epoch in range(4):
        for article_tokens, summary_tokens in data_loader:
            article_count += len(article_tokens)
            article_tokens = article_tokens.to(device)
            summary_tokens = summary_tokens.to(device)

            if article_tokens.size(1) > model.config.n_positions:
                article_chunks = sliding_window_processing(article_tokens, window_size, step_size)
            else:
                article_chunks = [article_tokens]

            for chunk in article_chunks:
                if chunk.size(1) > model.config.n_positions:
                    chunk = chunk[:, :model.config.n_positions]

                inputs = torch.cat((chunk, summary_tokens[:, :-1]), dim=1)
                labels = torch.cat((torch.full_like(chunk, -100), summary_tokens[:, 1:]), dim=1)

                attention_mask = (inputs != pad_token_id).float()
                max_length = min(inputs.size(1), model.config.n_positions)
                inputs = inputs[:, :max_length]
                labels = labels[:, :max_length]
                attention_mask = attention_mask[:, :max_length]

                with autocast():
                    outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                    loss = criterion(outputs.logits.reshape(-1, model.config.vocab_size), labels.reshape(-1))

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                print(f"Epoch {epoch}, Loss: {loss.item()}, Articles Processed: {article_count}")

                try:
                    generated_summary = generate_summary(chunk, model, tokenizer, device, max_length)
                    print(f"Generated Summary: {generated_summary}")
                except ValueError as e:
                    print(f"Skipping summary generation: {e}")
                print("-" * 80)

                # 显存释放
                del inputs, labels, attention_mask, outputs, loss
                torch.cuda.empty_cache()

def main():
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    data_path = '/content/drive/MyDrive/AIfinal/validation.csv/validation.csv'
    data = pd.read_csv(data_path)

    articles = data['article'].tolist()
    summaries = data['highlights'].tolist()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = ArticleDataset(articles, summaries, tokenizer, max_len=1024)
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)  # Reduced batch size

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    criterion = CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler()

    train_model(data_loader, tokenizer, model, device, optimizer, criterion, scaler, max_length=50)
    save_model(model, '/content/drive/MyDrive/AIfinal/gpt2_finetuned.pth')

if __name__ == "__main__":
    main()
