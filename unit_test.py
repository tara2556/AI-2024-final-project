import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from preprocess import preprocessing_function

def generate_summary(article, tokenizer, model, device, max_new_tokens=50):
    # Preprocess the article
    processed_article = preprocessing_function(article)
    article_tokens = tokenizer.encode(processed_article, return_tensors='pt').to(device)

    # Generate a summary with the model
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            article_tokens, 
            attention_mask=(article_tokens != tokenizer.pad_token_id).to(torch.float), 
            max_new_tokens=max_new_tokens
        )
        generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_summary

def unit_test(tokenizer, model, device):
    # Define a small article and its summary for the unit test
    small_article = "A quick brown fox jumps over the lazy dog."
    small_summary = "A fox jumps over a dog."

    # Generate a summary with the model
    generated_summary = generate_summary(small_article, tokenizer, model, device)

    # Print the results
    print("Original Article:", small_article)
    print("Original Summary:", small_summary)
    print("Generated Summary:", generated_summary)

def main():
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Resize token embeddings if pad_token is not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Run unit test
    unit_test(tokenizer, model, device)

if __name__ == "__main__":
    main()
