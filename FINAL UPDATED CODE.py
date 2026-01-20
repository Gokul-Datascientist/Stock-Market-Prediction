#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import nltk, re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
from IPython.display import display
# --- FIX: Missing Imports ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


# In[19]:


import requests
import pandas as pd
import random
from datetime import datetime
import time

# 🔑 API KEYS
NEWS_API_KEY = "1d3b7e6b933246fc917eabfbe7d4b6d4"
OPENAI_API_KEY = "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"

# 🧍 Random user generator
def random_username():
    users = [
        "morrismitchell", "anna80", "angiegarcia", "carloscook", "robertsrobert",
        "ericaburns", "henry88", "richard54", "christopherleonard", "sheri53",
        "josephbarajas", "kentmoore", "christopherkim", "shelbyespinoza",
        "amandamitchell", "williamsolivia", "yward", "jasonlogan", "faguirre"
    ]
    return "@" + random.choice(users)

# 🕒 Timestamp
def now_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 🌐 Fetch political/economic news
def fetch_political_news(limit=10):
    keywords = "politics OR economy OR market OR government OR trade OR Russia OR India OR policy OR summit OR inflation"
    url = f"https://newsapi.org/v2/everything?q={keywords}&language=en&sortBy=publishedAt&pageSize={limit}&apiKey={NEWS_API_KEY}"

    response = requests.get(url)
    data = response.json()

    print(f"🔍 API Status: {data.get('status', 'Unknown')}")
    print(f"📰 Articles Found: {data.get('totalResults', 0)}")

    if data.get("status") != "ok" or not data.get("articles"):
        print("⚠️ No articles found or API key invalid.")
        return []

    articles = []
    for article in data["articles"]:
        title = article.get("title", "")
        description = article.get("description", "")
        if not title:
            continue
        articles.append({
            "title": title,
            "description": description,
            "publishedAt": article.get("publishedAt", "")
        })
    return articles

# 🤖 Generate tweet text with realistic sentiment tone
def generate_political_tweet(title, description):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = (
        f"Convert this political or economic headline into a short tweet-style update (max 280 characters). "
        f"The tone should naturally vary — some positive, some negative, some neutral — just like real social media. "
        f"Do NOT mention sentiment labels, just write the tweet text. Include one relevant hashtag.\n\n"
        f"Headline: {title}\nDescription: {description}"
    )

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 120,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ GPT Error: {e}"

# 🚀 Main Script
if __name__ == "__main__":
    limit = int(input("How many political tweets to generate?: "))

    print(f"\n📰 Fetching top {limit} political/economic news articles...")
    news_list = fetch_political_news(limit)

    if not news_list:
        print("❌ No news data available.")
        exit()

    print("🧠 Generating realistic tweet-style posts with mixed sentiment tones...\n")

    rows = []
    for item in news_list:
        tweet = generate_political_tweet(item['title'], item['description'])
        rows.append({
            "Date_Time": now_timestamp(),
            "User": random_username(),
            "Tweet_Text": tweet
        })
        time.sleep(1)

    # 💾 Save to Excel (without date in filename)
    df = pd.DataFrame(rows)
    filename = "/Users/apple/Desktop/Political_News.xlsx"
    df.to_excel(filename, index=False)

    print(f"\n✅ {len(rows)} Political news tweets saved successfully!")
    print(f"📁 File saved at: {filename}")


# In[20]:


# ============================================================================
# 1. DATA PREPROCESSING MODULE (JUPYTER-FRIENDLY FINAL VERSION)
# ============================================================================

class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing with AI-based generation for missing data"""
    
    def __init__(self):
        print("🧠 Initializing DataPreprocessor...", flush=True)
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('wordnet')
            nltk.download('vader_lexicon')
            self.stop_words = set(stopwords.words('english'))
        print("✅ NLTK resources ready.\n", flush=True)
    
    def load_data(self, filepath='/Users/apple/Desktop/TWEET_DATASET.xlsx'):
        """Load dataset from CSV/Excel with enhanced error handling"""
        print(f"📂 Loading dataset from: {filepath}", flush=True)
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            else:
                raise ValueError("File must be CSV or Excel format.")
            
            print(f"✅ File loaded successfully — Rows: {len(df)} Columns: {list(df.columns)}", flush=True)
            
            if 'Date_Time' in df.columns:
                df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')
                invalid = df['Date_Time'].isna().sum()
                if invalid > 0:
                    print(f"⚠️ Removed {invalid} invalid Date_Time entries.", flush=True)
                    df = df.dropna(subset=['Date_Time'])
            else:
                raise ValueError("Missing required column: 'Date_Time'")
            
            if 'Tweet_Text' not in df.columns:
                raise ValueError("Missing required column: 'Tweet_Text'")
            
            print("📊 Dataset sample:")
            display(df.head(3))
            
            return df
        
        except Exception as e:
            print(f"❌ Error loading dataset: {e}", flush=True)
            return pd.DataFrame(columns=['Date_Time', 'Tweet_Text'])
    
    def clean_text(self, text):
        """Clean and preprocess tweet text"""
        try:
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            text = re.sub(r'@\w+|#\w+', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens
                      if word not in self.stop_words and len(word) > 2]
            return ' '.join(tokens)
        except Exception:
            return ""
    
    def aggregate_by_datetime(self, df):
        """Aggregate tweets by Date_Time and generate AI-based PnL & Market_Type"""
        try:
            if df.empty:
                print("⚠️ Empty dataset provided. Skipping aggregation.", flush=True)
                return df
            
            print("\n🔄 Aggregating tweets and generating AI-based market data...", flush=True)
            df['Cleaned_Text'] = df['Tweet_Text'].apply(self.clean_text)
            
            sia = SentimentIntensityAnalyzer()
            df['Sentiment_Score'] = df['Cleaned_Text'].apply(lambda x: sia.polarity_scores(x)['compound'])
            
            if 'PnL' not in df.columns or df['PnL'].isna().all():
                df['PnL'] = df['Sentiment_Score'] * np.random.uniform(0.8, 1.2, len(df)) * 5
                print("💰 Generated AI-based PnL based on sentiment.", flush=True)
            
            if 'Market_Type' not in df.columns or df['Market_Type'].isna().all():
                def infer_market_type(text, score):
                    text = text.lower()
                    if 'oil' in text or 'energy' in text:
                        topic = 'Energy'
                    elif 'inflation' in text or 'price' in text:
                        topic = 'Inflation'
                    elif 'trade' in text or 'export' in text:
                        topic = 'Trade'
                    elif 'stock' in text or 'market' in text:
                        topic = 'Equity'
                    else:
                        topic = 'General'
                    
                    if score >= 0.3:
                        sentiment = 'Bullish'
                    elif score <= -0.3:
                        sentiment = 'Bearish'
                    else:
                        sentiment = 'Stable'
                    
                    return f"{sentiment}-{topic}"
                
                df['Market_Type'] = df.apply(lambda x: infer_market_type(x['Cleaned_Text'], x['Sentiment_Score']), axis=1)
                print("📈 Generated AI-based Market_Type using sentiment + topics.", flush=True)
            
            aggregated = df.groupby('Date_Time').agg({
                'Tweet_Text': 'count',
                'Cleaned_Text': lambda x: ' '.join(x),
                'PnL': 'mean',
                'Market_Type': lambda x: x.value_counts().index[0]
            }).reset_index()
            
            aggregated.rename(columns={'Tweet_Text': 'Total_Tweets'}, inplace=True)
            
            print(f"\n✅ Aggregation successful! Total time periods: {len(aggregated)}", flush=True)
            display(aggregated.head(5))
            
            # Optional auto-save
            save_path = '/Users/apple/Desktop/aggregated_output.csv'
            aggregated.to_csv(save_path, index=False)
            print(f"💾 Aggregated dataset saved to: {save_path}\n", flush=True)
            
            return aggregated
        
        except Exception as e:
            print(f"\n❌ Error during aggregation: {e}", flush=True)
            if 'Date_Time' in df.columns:
                display(df.head(3))
            return df


# In[10]:


# Test if class is available
'DataPreprocessor' in globals()


# In[21]:


pre = DataPreprocessor()


# In[22]:


df = pre.load_data('/Users/apple/Desktop/Political_News.xlsx')


# In[23]:


agg = pre.aggregate_by_datetime(df)


# In[24]:


# ============================================================================
# 2. SENTIMENT ANALYSIS MODULE (JUPYTER-FRIENDLY FINAL VERSION)
# ============================================================================

class SentimentAnalyzer:
    """Performs sentiment analysis using VADER"""
    
    def __init__(self):
        print("🧠 Initializing Sentiment Analyzer...", flush=True)
        try:
            self.sia = SentimentIntensityAnalyzer()
            print("✅ VADER Sentiment Analyzer loaded successfully.\n", flush=True)
        except:
            print("⚙️ Downloading required NLTK resources...", flush=True)
            nltk.download('vader_lexicon')
            self.sia = SentimentIntensityAnalyzer()
            print("✅ VADER resources downloaded and loaded.\n", flush=True)

    def analyze_sentiment(self, text):
        """Get sentiment scores using VADER"""
        if not isinstance(text, str):
            text = str(text)
        return self.sia.polarity_scores(text)

    def classify_sentiment(self, compound_score):
        """Classify sentiment based on compound score"""
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    def batch_analyze(self, df):
        """Analyze sentiment for all records"""
        try:
            if 'Cleaned_Text' not in df.columns:
                raise ValueError("⚠️ Missing column: 'Cleaned_Text' — please clean text first.")
            
            print("\n🔍 Performing batch sentiment analysis...", flush=True)
            sentiments = df['Cleaned_Text'].apply(self.analyze_sentiment)
            
            df['Sentiment_Score'] = sentiments.apply(lambda x: x['compound'])
            df['Positive_Score'] = sentiments.apply(lambda x: x['pos'])
            df['Negative_Score'] = sentiments.apply(lambda x: x['neg'])
            df['Neutral_Score'] = sentiments.apply(lambda x: x['neu'])
            df['Sentiment_Label'] = df['Sentiment_Score'].apply(self.classify_sentiment)

            print("✅ Sentiment analysis completed successfully!")
            display(df[['Cleaned_Text', 'Sentiment_Label', 'Sentiment_Score']].head(5))
            return df
        
        except Exception as e:
            print(f"❌ Error during sentiment analysis: {e}", flush=True)
            return df


# In[26]:


sent = SentimentAnalyzer()


# In[27]:


df_sent = sent.batch_analyze(agg)


# In[28]:


df_sent.head()


# In[29]:


class FeatureEngineer:
    """Creates advanced engineered features for machine learning models."""
    
    def __init__(self):
        print("🧩 Initializing Feature Engineer...", flush=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        print("✅ Feature Engineer ready.\n", flush=True)
    
    def create_temporal_features(self, df):
        """Generate lag, rolling, ratio, and time-based features."""
        try:
            print("⚙️ Creating temporal and statistical features...", flush=True)
            
            df = df.sort_values('Date_Time').reset_index(drop=True)
            
            # Lagged features
            df['Lagged_Sentiment_Score'] = df['Sentiment_Score'].shift(1)
            df['Lagged_PnL'] = df['PnL'].shift(1)
            
            # Rolling statistics (3-period window)
            df['Rolling_Sentiment_Mean_3'] = df['Sentiment_Score'].rolling(window=3, min_periods=1).mean()
            df['Rolling_Sentiment_Std_3'] = df['Sentiment_Score'].rolling(window=3, min_periods=1).std()
            df['Rolling_PnL_Mean_3'] = df['PnL'].rolling(window=3, min_periods=1).mean()
            
            # Sentiment momentum
            df['Sentiment_Momentum'] = df['Sentiment_Score'] - df['Lagged_Sentiment_Score']
            
            # Tweet volume change
            df['Tweet_Volume_Change'] = df['Total_Tweets'].pct_change()
            
            # Sentiment ratios
            df['Positive_Ratio'] = df['Positive_Score'] / (df['Total_Tweets'] + 1)
            df['Negative_Ratio'] = df['Negative_Score'] / (df['Total_Tweets'] + 1)
            df['Net_Sentiment_Index'] = df['Positive_Score'] - df['Negative_Score']
            
            # Time-based features
            df['Hour'] = df['Date_Time'].dt.hour
            df['DayOfWeek'] = df['Date_Time'].dt.dayofweek
            df['Month'] = df['Date_Time'].dt.month
            
            # Handle NaN values
            df = df.fillna(method='bfill').fillna(0)
            
            print("✅ Temporal features created successfully!")
            display(df.head(5))
            
            return df
        
        except Exception as e:
            print(f"❌ Error during feature creation: {e}", flush=True)
            display(df.head(3))
            return df
    
    def encode_market_type(self, df, fit=True):
        """Encode Market_Type categorical variable."""
        try:
            print("\n🔠 Encoding Market_Type categories...", flush=True)
            
            if 'Market_Type' not in df.columns:
                raise ValueError("Missing column: 'Market_Type'")
            
            if fit:
                df['Market_Type_Encoded'] = self.label_encoder.fit_transform(df['Market_Type'])
            else:
                df['Market_Type_Encoded'] = self.label_encoder.transform(df['Market_Type'])
            
            print("✅ Market_Type encoding complete.")
            display(df[['Market_Type', 'Market_Type_Encoded']].head(5))
            
            return df
        
        except Exception as e:
            print(f"❌ Error during encoding: {e}", flush=True)
            return df

    def prepare_features(self, df, target_col='PnL'):
        """Prepare feature matrix (X) and target vector (y) for ML models."""
        try:
            print("\n📊 Preparing feature matrix and target vector...", flush=True)
            
            feature_cols = [
                'Sentiment_Score', 'Positive_Score', 'Negative_Score', 'Neutral_Score',
                'Total_Tweets', 'Lagged_Sentiment_Score', 'Lagged_PnL',
                'Rolling_Sentiment_Mean_3', 'Rolling_Sentiment_Std_3', 'Rolling_PnL_Mean_3',
                'Sentiment_Momentum', 'Tweet_Volume_Change', 'Positive_Ratio',
                'Negative_Ratio', 'Net_Sentiment_Index', 'Hour', 'DayOfWeek', 'Month'
            ]
            
            # Check that all required columns exist
            missing = [col for col in feature_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            X = df[feature_cols].values
            y = df[target_col].values
            
            print(f"✅ Feature preparation complete! Total samples: {len(X)}")
            print(f"   Input features: {len(feature_cols)}, Target: '{target_col}'")
            
            return X, y, feature_cols
        
        except Exception as e:
            print(f"❌ Error preparing features: {e}", flush=True)
            return None, None, []


# In[30]:


feat = FeatureEngineer()


# In[31]:


# ============================================================================
# 4. PNL PREDICTION MODULE (JUPYTER-FRIENDLY FINAL VERSION)
# ============================================================================

class PnLPredictor:
    """Predicts Profit & Loss (PnL) using ensemble and deep learning models."""
    
    def __init__(self):
        print("💹 Initializing PnL Predictor...", flush=True)
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        print("✅ PnL Predictor ready.\n", flush=True)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model for time-series prediction."""
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_models(self, X_train, y_train, X_val, y_val, show_progress=True):
        """Train RandomForest, XGBoost, LSTM, and SGD models."""
        try:
            print("\n🚀 Starting model training pipeline...", flush=True)
            
            # Check for empty or invalid data
            if len(X_train) == 0 or len(y_train) == 0:
                raise ValueError("Training data is empty.")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # ------------------ Random Forest ------------------
            print("🌲 Training Random Forest...", flush=True)
            rf = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            self.models['random_forest'] = rf
            print("✅ Random Forest trained.")
            
            # ------------------ XGBoost ------------------
            print("🔥 Training XGBoost...", flush=True)
            xgb_model = xgb.XGBRegressor(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train_scaled, y_train)
            self.models['xgboost'] = xgb_model
            print("✅ XGBoost trained.")
            
            # ------------------ LSTM ------------------
            print("🧠 Training LSTM (Deep Learning)...", flush=True)
            X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
            
            lstm_model = self.build_lstm_model((1, X_train_scaled.shape[1]))
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            lstm_model.fit(
                X_train_lstm, y_train,
                validation_data=(X_val_lstm, y_val),
                epochs=40, batch_size=32,
                verbose=1 if show_progress else 0,
                callbacks=[early_stop]
            )
            self.models['lstm'] = lstm_model
            print("✅ LSTM trained.")
            
            # ------------------ SGD ------------------
            print("⚡ Training SGD Regressor (for incremental learning)...", flush=True)
            sgd = SGDRegressor(max_iter=1000, random_state=42)
            sgd.fit(X_train_scaled, y_train)
            self.models['sgd'] = sgd
            print("✅ SGD trained.")
            
            self.is_trained = True
            print("\n🎯 All models trained successfully!\n")
        
        except Exception as e:
            print(f"❌ Error during model training: {e}", flush=True)
    
    def predict(self, X):
        """Generate ensemble predictions using all trained models."""
        try:
            if not self.is_trained:
                raise RuntimeError("Models are not trained yet.")
            
            X_scaled = self.scaler.transform(X)
            predictions = []
            
            # Predict using each model
            if 'random_forest' in self.models:
                predictions.append(self.models['random_forest'].predict(X_scaled))
            if 'xgboost' in self.models:
                predictions.append(self.models['xgboost'].predict(X_scaled))
            if 'lstm' in self.models:
                X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                predictions.append(self.models['lstm'].predict(X_lstm, verbose=0).flatten())
            if 'sgd' in self.models:
                predictions.append(self.models['sgd'].predict(X_scaled))
            
            # Ensemble average prediction
            ensemble_pred = np.mean(predictions, axis=0)
            print(f"📈 Ensemble prediction generated for {len(ensemble_pred)} samples.")
            
            return ensemble_pred
        
        except Exception as e:
            print(f"❌ Error during prediction: {e}", flush=True)
            return np.zeros(len(X))
    
    def incremental_update(self, X_new, y_new):
        """Continuously update model with new data."""
        try:
            print("\n🔁 Incremental model update with new data...", flush=True)
            X_new_scaled = self.scaler.transform(X_new)
            
            if 'sgd' in self.models:
                self.models['sgd'].partial_fit(X_new_scaled, y_new)
                print("✅ SGD model updated incrementally.")
            
            if 'random_forest' in self.models:
                self.models['random_forest'].fit(X_new_scaled, y_new)
                print("✅ Random Forest retrained with new data.")
            
            if 'xgboost' in self.models:
                self.models['xgboost'].fit(X_new_scaled, y_new)
                print("✅ XGBoost retrained with new data.")
            
            print("🎯 Models updated successfully!\n")
        
        except Exception as e:
            print(f"❌ Error during incremental update: {e}", flush=True)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        try:
            print("\n📊 Evaluating ensemble performance...", flush=True)
            y_pred = self.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
            
            print(f"✅ Evaluation complete:\n  RMSE: {rmse:.4f}\n  MAE: {mae:.4f}\n  R²: {r2:.4f}\n")
            return metrics, y_pred
        
        except Exception as e:
            print(f"❌ Error during evaluation: {e}", flush=True)
            return {}, np.zeros(len(X_test))


# In[32]:


pnl_model = PnLPredictor()


# In[39]:


pnl_model = PnLPredictor()


# In[33]:


# ============================================================================
# 5. MARKET TYPE CLASSIFIER (JUPYTER-FRIENDLY FINAL VERSION)
# ============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class MarketTypeClassifier:
    """Classifies market type (e.g., Bullish, Bearish, Stable) based on sentiment and features."""
    
    def __init__(self):
        print("📊 Initializing Market Type Classifier...", flush=True)
        self.model = None
        self.scaler = StandardScaler()
        print("✅ Market Type Classifier ready.\n", flush=True)
    
    def train(self, X_train, y_train):
        """Train Random Forest classifier for market type prediction."""
        try:
            print("🚀 Training Market Type Classifier...", flush=True)
            
            if len(X_train) == 0:
                raise ValueError("Empty training data provided.")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=120,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
            
            print("✅ Market Type Classifier trained successfully!\n", flush=True)
        
        except Exception as e:
            print(f"❌ Error during training: {e}", flush=True)
    
    def predict(self, X):
        """Predict market type labels."""
        try:
            if self.model is None:
                raise RuntimeError("Model not trained yet.")
            
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            print(f"📈 Predicted {len(predictions)} market type labels.", flush=True)
            return predictions
        
        except Exception as e:
            print(f"❌ Error during prediction: {e}", flush=True)
            return np.array([])
    
    def evaluate(self, X_test, y_test):
        """Evaluate classifier performance with accuracy and classification report."""
        try:
            print("📊 Evaluating Market Type Classifier...", flush=True)
            
            y_pred = self.predict(X_test)
            
            if len(y_pred) == 0:
                raise ValueError("No predictions made. Possibly untrained model.")
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            print(f"✅ Evaluation Complete! Accuracy: {accuracy:.4f}\n", flush=True)
            print("Classification Report:\n", report)
            print("Confusion Matrix:\n", conf_matrix)
            
            return accuracy, report, conf_matrix
        
        except Exception as e:
            print(f"❌ Error during evaluation: {e}", flush=True)
            return 0.0, "", np.array([])


# In[34]:


market_clf = MarketTypeClassifier()


# In[35]:


'MarketTypeClassifier' in globals()


# In[36]:


# ============================================================================
# 6. MAIN PIPELINE (JUPYTER-FRIENDLY FINAL VERSION)
# ============================================================================

import matplotlib.pyplot as plt
import joblib
from pathlib import Path

class SentimentPnLPipeline:
    """Complete pipeline for sentiment analysis and PnL prediction"""
    
    def __init__(self):
        print("🚀 Initializing SentimentPnLPipeline...\n", flush=True)
        self.preprocessor = DataPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.feature_engineer = FeatureEngineer()
        self.pnl_predictor = PnLPredictor()
        self.market_classifier = MarketTypeClassifier()
        self.results_df = None
        print("✅ All modules initialized successfully.\n", flush=True)
    
    def run_pipeline(self, filepath, test_size=0.2):
        """Execute the complete AI-driven sentiment + PnL prediction pipeline"""
        print("=" * 70)
        print("AI-DRIVEN TEMPORAL SENTIMENT ANALYSIS & PnL PREDICTION")
        print("=" * 70)
        
        # Step 1: Load and preprocess data
        print("\n[1/7] Loading and preprocessing data...")
        df = self.preprocessor.load_data(filepath)
        df_agg = self.preprocessor.aggregate_by_datetime(df)
        print(f"✅ Loaded {len(df_agg)} time periods with {df['Tweet_Text'].count()} total tweets.\n")
        
        # Step 2: Sentiment analysis
        print("[2/7] Performing sentiment analysis...")
        df_agg = self.sentiment_analyzer.batch_analyze(df_agg)
        sentiment_dist = df_agg['Sentiment_Label'].value_counts()
        print(f"✅ Sentiment Distribution:\n{sentiment_dist}\n")
        
        # Step 3: Feature engineering
        print("[3/7] Engineering features...")
        df_agg = self.feature_engineer.create_temporal_features(df_agg)
        df_agg = self.feature_engineer.encode_market_type(df_agg)
        print(f"✅ Feature engineering complete — {df_agg.shape[1]} total features.\n")
        
        # Step 4: Prepare train/test split
        print("[4/7] Preparing train/test split...")
        X, y_pnl, feature_cols = self.feature_engineer.prepare_features(df_agg, 'PnL')
        y_market = df_agg['Market_Type'].values
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_pnl_train, y_pnl_test = y_pnl[:split_idx], y_pnl[split_idx:]
        y_market_train, y_market_test = y_market[:split_idx], y_market[split_idx:]
        
        val_size = int(len(X_train) * 0.2)
        X_val, y_pnl_val = X_train[-val_size:], y_pnl_train[-val_size:]
        X_train, y_pnl_train = X_train[:-val_size], y_pnl_train[:-val_size]
        print(f"✅ Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")
        
        # Step 5: Train PnL predictor
        print("[5/7] Training PnL prediction models...")
        self.pnl_predictor.train_models(X_train, y_pnl_train, X_val, y_pnl_val)
        
        # Step 6: Train Market Type classifier
        print("\n[6/7] Training Market Type classifier...")
        self.market_classifier.train(X_train, y_market_train[:len(X_train)])
        
        # Step 7: Evaluate and predict
        print("\n[7/7] Evaluating models...")
        pnl_metrics, y_pnl_pred = self.pnl_predictor.evaluate(X_test, y_pnl_test)
        print(f"\n✅ PnL Prediction Metrics:")
        print(f"  RMSE: {pnl_metrics['RMSE']:.4f}")
        print(f"  MAE:  {pnl_metrics['MAE']:.4f}")
        print(f"  R²:   {pnl_metrics['R2']:.4f}")
        
        market_acc, market_report, conf_matrix = self.market_classifier.evaluate(X_test, y_market_test)
        print(f"\n✅ Market Type Classification Accuracy: {market_acc:.4f}\n")
        
        # Generate predictions for all data
        y_pnl_all = self.pnl_predictor.predict(X)
        y_market_all = self.market_classifier.predict(X)
        
        # Combine results
        self.results_df = df_agg[['Date_Time', 'Sentiment_Label', 'Sentiment_Score']].copy()
        self.results_df['Actual_PnL'] = df_agg['PnL']
        self.results_df['Predicted_PnL'] = y_pnl_all
        self.results_df['Actual_Market_Type'] = df_agg['Market_Type']
        self.results_df['Predicted_Market_Type'] = y_market_all
        self.results_df['Total_Tweets'] = df_agg['Total_Tweets']
        
        print("=" * 70)
        print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return self.results_df
    
    # ------------------------------------------------------------------------
    # CONTINUOUS LEARNING MODULE
    # ------------------------------------------------------------------------
    def continuous_learning(self, new_data_filepath):
        """Update models with new real-time data."""
        print("\n[CONTINUOUS LEARNING] Processing new data...")
        df_new = self.preprocessor.load_data(new_data_filepath)
        df_new_agg = self.preprocessor.aggregate_by_datetime(df_new)
        df_new_agg = self.sentiment_analyzer.batch_analyze(df_new_agg)
        df_new_agg = self.feature_engineer.create_temporal_features(df_new_agg)
        df_new_agg = self.feature_engineer.encode_market_type(df_new_agg, fit=False)
        
        X_new, y_new, _ = self.feature_engineer.prepare_features(df_new_agg, 'PnL')
        self.pnl_predictor.incremental_update(X_new, y_new)
        print("✅ Models updated successfully with new data.\n")
    
    # ------------------------------------------------------------------------
    # MODEL SAVE/LOAD
    # ------------------------------------------------------------------------
    def save_models(self, save_dir='models'):
        Path(save_dir).mkdir(exist_ok=True)
        joblib.dump(self.pnl_predictor, f'{save_dir}/pnl_predictor.pkl')
        joblib.dump(self.market_classifier, f'{save_dir}/market_classifier.pkl')
        joblib.dump(self.feature_engineer, f'{save_dir}/feature_engineer.pkl')
        print(f"💾 Models saved successfully to: {save_dir}/\n")
    
    def load_models(self, save_dir='models'):
        self.pnl_predictor = joblib.load(f'{save_dir}/pnl_predictor.pkl')
        self.market_classifier = joblib.load(f'{save_dir}/market_classifier.pkl')
        self.feature_engineer = joblib.load(f'{save_dir}/feature_engineer.pkl')
        print(f"📦 Models loaded from: {save_dir}/\n")
    
    # ------------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------------
    def visualize_results(self):
        if self.results_df is None:
            print("⚠️ No results to visualize. Run pipeline first!\n")
            return
        
        print("📊 Generating visualizations...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sentiment Score timeline
        axes[0, 0].plot(self.results_df['Date_Time'], self.results_df['Sentiment_Score'], color='blue')
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_title("Sentiment Score Over Time")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Actual vs Predicted PnL
        axes[0, 1].scatter(self.results_df['Actual_PnL'], self.results_df['Predicted_PnL'], color='green', alpha=0.6)
        axes[0, 1].plot(
            [self.results_df['Actual_PnL'].min(), self.results_df['Actual_PnL'].max()],
            [self.results_df['Actual_PnL'].min(), self.results_df['Actual_PnL'].max()],
            'r--'
        )
        axes[0, 1].set_title("Actual vs Predicted PnL")
        axes[0, 1].grid(True, alpha=0.3)
        
        # PnL Timeline
        axes[1, 0].plot(self.results_df['Date_Time'], self.results_df['Actual_PnL'], label='Actual', color='blue')
        axes[1, 0].plot(self.results_df['Date_Time'], self.results_df['Predicted_PnL'], label='Predicted', color='orange')
        axes[1, 0].legend()
        axes[1, 0].set_title("PnL Timeline (Actual vs Predicted)")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sentiment Distribution
        sentiment_counts = self.results_df['Sentiment_Label'].value_counts()
        axes[1, 1].bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'gray', 'red'])
        axes[1, 1].set_title("Sentiment Distribution")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        print("✅ Visualization completed.\n")


# In[37]:


pipeline = SentimentPnLPipeline()
results = pipeline.run_pipeline('/Users/apple/Desktop/tweets_dataset_final.csv')


# In[38]:


# ============================================================================
# 7. USAGE EXAMPLE (JUPYTER-FRIENDLY)
# ============================================================================

# Initialize pipeline
pipeline = SentimentPnLPipeline()

# Your dataset path
dataset_path = '/Users/apple/Desktop/tweets_dataset_final.csv'

print(f"\n📂 Loading dataset from: {dataset_path}")

try:
    # Run complete pipeline with your data
    results = pipeline.run_pipeline(dataset_path, test_size=0.2)

except Exception as e:
    print(f"\n⚠️ Error loading file: {e}")
    print("\nPlease ensure your CSV has these columns:")
    print("  - Date_Time (datetime format)")
    print("  - Tweet_Text (text)")
    print("  - PnL (numeric)")
    print("  - Market_Type (categorical)")
    print("\n⚙️ Creating sample dataset for demonstration...")

    # Generate demo dataset
    dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
    sample_data = {
        'Date_Time': dates,
        'Tweet_Text': [
            f"Market {np.random.choice(['bullish', 'bearish', 'stable'])} today! "
            f"{'Great news!' if np.random.rand() > 0.5 else 'Concerns rising.'}"
            for _ in range(100)
        ],
        'Category': np.random.choice(['Finance', 'Economics', 'Trading'], 100),
        'PnL': np.random.normal(0, 1.5, 100),
        'Market_Type': np.random.choice(['CPI', 'Trade', 'Inflation', 'Stable'], 100)
    }

    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('/Users/apple/Desktop/sample_market_data.csv', index=False)
    print("✅ Sample dataset created: /Users/apple/Desktop/sample_market_data.csv")

    results = pipeline.run_pipeline('/Users/apple/Desktop/sample_market_data.csv', test_size=0.2)

# ============================================================================
#  RESULTS & VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("📊 SAMPLE PREDICTIONS")
print("=" * 70)
display(results[['Date_Time', 'Sentiment_Label', 'Sentiment_Score',
                 'Predicted_PnL', 'Predicted_Market_Type']].head(10))

# Save trained models
pipeline.save_models()

# Generate visualization
pipeline.visualize_results()

print("\n✅ Complete AI/ML pipeline executed successfully!")
print("✅ Models trained and saved")
print("✅ Results visualized and saved")

print("\n💡 To use with your own new data:")
print("   results = pipeline.run_pipeline('your_new_data.csv')")

print("\n🔁 For continuous learning:")
print("   pipeline.continuous_learning('new_data.csv')")


# In[ ]:




