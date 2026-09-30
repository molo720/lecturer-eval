import re
import string
import nltk

try:
    from nltk.corpus import stopwords
    nltk_stopwords = set(stopwords.words('english'))
except Exception:
    nltk.download('stopwords', quiet=True)
    try:
        from nltk.corpus import stopwords
        nltk_stopwords = set(stopwords.words('english'))
    except Exception:
        nltk_stopwords = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
            "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
            "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
            "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
            "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my", "myself",
            "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
            "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
            "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
            "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
            "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
            "with", "would", "you", "your", "yours", "yourself", "yourselves"
        }

try:
    from nltk.tokenize import word_tokenize
    # quick test
    word_tokenize("test sentence")
except Exception:
    nltk.download('punkt', quiet=True)
    try:
        from nltk.tokenize import word_tokenize
    except Exception:
        word_tokenize = None

try:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
except Exception:
    stemmer = None

# Common English contractions expansion dictionary (Section 3.5.2)
CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "mightn't": "might not", "mustn't": "must not",
    "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have",
    "wasn't": "was not", "we'd": "we would", "we'll": "we will", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what's": "what is", "where's": "where is",
    "who's": "who is", "won't": "will not", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have"
}

# Negation words and sentiment intensifiers to keep in the vocabulary
negation_words = {
    "not", "no", "nor", "never", "n't", "none", "nobody", "nothing",
    "neither", "nowhere", "hardly", "scarcely", "barely", "without", "rarely", "seldom"
}
intensifiers = {"very", "too", "so", "really", "extremely", "quite", "most", "more", "less", "least"}

custom_stopwords = nltk_stopwords - negation_words - intensifiers
domain_stop_words = {"lecturer", "class", "course", "topic", "topics", "session", "sessions", "semester"}
all_stop_words = custom_stopwords.union(domain_stop_words)

def expand_contractions(text):
    """Expands contractions in text (e.g. don't -> do not)."""
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS.keys()) + r')\b', re.IGNORECASE)
    def replace(match):
        contracted = match.group(0).lower()
        return CONTRACTIONS.get(contracted, contracted)
    return pattern.sub(replace, text)

def clean_text(text):
    """
    Cleans raw comment text:
    1. Removes HTML tags and URLs
    2. Expands contractions
    3. Lowercases text
    4. Strips non-printable characters & excess whitespace
    """
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Strip URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # Expand contractions
    text = expand_contractions(text)
    # Lowercase
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """Fast, safe tokenization with fallback."""
    if word_tokenize:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"\b[a-zA-Z0-9']+\b", text)

def preprocess_comment(text, stem=True):
    """
    Complete preprocessing pipeline as specified in Section 3.5.2:
    - Text cleaning
    - Normalisation (lowercase, contraction expansion)
    - Tokenisation
    - Stop-word removal (sentiment-preserving)
    - Optional Porter stemming
    Returns a list of token strings.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    # Remove punctuation
    tokens = [t for t in tokens if t not in string.punctuation and not all(c in string.punctuation for c in t)]
    # Stop-word removal preserving negations and sentiment modifiers
    tokens = [t for t in tokens if t not in all_stop_words]
    # Optional stemming
    if stem and stemmer:
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def preprocess_for_vectorizer(text):
    """Returns preprocessed tokens rejoined into a single string for TF-IDF."""
    tokens = preprocess_comment(text, stem=True)
    return " ".join(tokens)