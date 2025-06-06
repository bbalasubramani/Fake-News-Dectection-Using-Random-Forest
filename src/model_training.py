import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

def train_model(df):
    tfidf = TfidfVectorizer()
    x = tfidf.fit_transform(df['clean_text'])
    y = df['target']

    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x_resampled, y_resampled,
        test_size=0.2, random_state=42, stratify=y_resampled
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # ✅ Ensure models folder exists before saving
    os.makedirs('models', exist_ok=True)
    pickle.dump(tfidf, open('models/vectorizer.pkl', 'wb'))
    pickle.dump(model, open('models/model.pkl', 'wb'))

    return model, x_test, y_test  # Return test data for evaluation
