import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def summary_truncation(text, num_words=5):
    return ' '.join(text.split()[:num_words])

def extract_keywords(text, num_keywords=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
    X = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(X.toarray().flatten())[::-1]
    top_n = feature_array[tfidf_sorting][:num_keywords]
    return ' '.join(top_n)

def process_articles(data, num, show):
    selected_articles = data.sample(n=num)
    results = []
    for index, row in selected_articles.iterrows():
        original_article = row['article']
        summary = row['highlights']
        truncated_title = summary_truncation(summary)
        keyword_title = extract_keywords(original_article)
        results.append((original_article, summary, truncated_title, keyword_title))
        if show:
            print(f"Original Article: {original_article[:250]}...")
            print(f"Original Summary: {summary}")
            print(f"Truncated Title: {truncated_title}")
            print(f"Keyword Title: {keyword_title}")
            print("-" * 80)
    return results
