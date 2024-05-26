import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from preprocess import preprocessing_function
from utils import save_model
from torch.nn import CrossEntropyLoss

class ArticleDataset(Dataset):
    def __init__(self, articles, summaries, tokenizer, max_len=512, step=256):
        self.tokenizer = tokenizer
        self.articles = articles
        self.summaries = summaries
        self.max_len = max_len
        self.step = step

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        summary = self.summaries[idx]
        processed_article = preprocessing_function(article)
        processed_summary = preprocessing_function(summary)
        article_tokens = self.tokenizer.encode(processed_article, return_tensors='pt').squeeze(0)
        summary_tokens = self.tokenizer.encode(processed_summary, return_tensors='pt').squeeze(0)
        return article_tokens, summary_tokens

def collate_fn(batch):
    articles, summaries = zip(*batch)
    articles = pad_sequence(articles, batch_first=True, padding_value=50256)  # 50256 is GPT-2's pad token ID
    summaries = pad_sequence(summaries, batch_first=True, padding_value=50256)
    return articles, summaries

def generate_summary(article_tokens, model, tokenizer, device, max_new_tokens=20):
    model.eval()
    input_length = article_tokens.size(1)
    max_length = input_length + max_new_tokens
    
    if max_length > model.config.n_positions:
        raise ValueError(f"Input length {input_length} plus max_new_tokens {max_new_tokens} exceeds model's maximum length {model.config.n_positions}")
    
    with torch.no_grad():
        outputs = model.generate(
            article_tokens.to(device), 
            attention_mask=(article_tokens != tokenizer.pad_token_id).float().to(device), 
            max_length=max_length
        )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def train_model(data_loader, tokenizer, model, device, optimizer, criterion, max_new_tokens=20):
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 50256
    model.train()
    article_count = 0
    for epoch in range(4):  # Adjust epochs as necessary
        for article_tokens, summary_tokens in data_loader:
            article_tokens = article_tokens.to(device)
            summary_tokens = summary_tokens.to(device)
            article_count += article_tokens.size(0)  # Increase the count by the batch size

            # Prepare the inputs and labels
            inputs = torch.cat((article_tokens, summary_tokens[:, :-1]), dim=1)
            labels = torch.cat((torch.full_like(article_tokens, -100), summary_tokens[:, 1:]), dim=1)  # Mask the article part

            # Compute attention mask
            attention_mask = (inputs != pad_token_id).float()

            # Adjust the length to be within the model's limits
            max_length = min(inputs.size(1), model.config.n_positions)
            inputs = inputs[:, :max_length]
            labels = labels[:, :max_length]
            attention_mask = attention_mask[:, :max_length]

            # Forward pass
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits.view(-1, model.config.vocab_size), labels.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Print loss and article count
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            print(f"Articles Processed: {article_count}")

            try:
                # Generate a summary for the current batch
                generated_summary = generate_summary(article_tokens, model, tokenizer, device, max_new_tokens)
                print(f"Generated Summary: {generated_summary}")
            except ValueError as e:
                print(f"Skipping summary generation: {e}")
            print("-" * 80)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def main():
    # Set CUDA_LAUNCH_BLOCKING environment variable
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
    
    articles = data['article'].tolist()
    summaries = data['highlights'].tolist()

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Resize token embeddings
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare data loader
    dataset = ArticleDataset(articles, summaries, tokenizer)
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Criterion
    criterion = CrossEntropyLoss(ignore_index=-100)

    # Train the model
    train_model(data_loader, tokenizer, model, device, optimizer, criterion, max_new_tokens=20)

    # Save the model
    save_model(model, '/content/drive/MyDrive/AIfinal/gpt2_finetuned.pth')

if __name__ == "__main__":
    main()