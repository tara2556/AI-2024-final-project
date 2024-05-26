import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def remove_stopwords(text: str) -> str:
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    return ' '.join(filtered_tokens)

def preprocessing_function(text: str) -> str:
    text = text.lower().replace('<br />', '')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    preprocessed_text = remove_stopwords(text)
    lemmatizer = WordNetLemmatizer()
    tokens = preprocessed_text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)
