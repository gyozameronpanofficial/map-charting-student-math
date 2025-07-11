#!/usr/bin/env python3
"""
MAP Competition - Advanced Baseline
Strategy 2 (Mathematical Features) + Strategy 4 (MAP@3 Optimization)
Target: Beat current #1 position (Public LB: 0.841)
"""

import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import lightgbm as lgb
from scipy import sparse

# Text Processing
import nltk
try:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
except:
    lemmatizer = None
    print("NLTK WordNetLemmatizer not available, using basic processing")

import time
import os


class MathematicalFeatureExtractor:
    """Extract mathematical features from text with robust error handling"""
    
    def __init__(self):
        # Mathematical patterns (fixed regex escaping)
        self.fraction_pattern = re.compile(r'\\frac\{([^}]+)\}\{([^}]+)\}')
        self.simple_fraction_pattern = re.compile(r'(\d+)\s*/\s*(\d+)')
        self.decimal_pattern = re.compile(r'\d+\.\d+')
        self.percentage_pattern = re.compile(r'\d+%')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.operation_pattern = re.compile(r'[+\-*/=]')
        
        # Mathematical concepts
        self.math_concepts = {
            'fraction': ['fraction', 'numerator', 'denominator', 'over', 'divided'],
            'decimal': ['decimal', 'point', 'place', 'tenths', 'hundredths'],
            'percentage': ['percent', 'percentage', '%', 'out of 100'],
            'addition': ['add', 'plus', 'sum', 'total', 'altogether'],
            'subtraction': ['subtract', 'minus', 'difference', 'take away'],
            'multiplication': ['multiply', 'times', 'product', 'of'],
            'division': ['divide', 'quotient', 'split', 'share'],
            'comparison': ['greater', 'less', 'equal', 'bigger', 'smaller', 'same']
        }
        
    def safe_extract_numbers(self, text):
        """Safely extract numbers with bounds checking"""
        try:
            numbers = []
            for match in self.number_pattern.findall(str(text)):
                try:
                    num = float(match)
                    if 0 <= num <= 1e6:  # Reasonable bounds
                        numbers.append(num)
                except (ValueError, OverflowError):
                    continue
            return numbers
        except:
            return []
    
    def extract_numerical_features(self, text):
        """Extract numerical features safely"""
        text = str(text)
        features = {}
        
        # Count patterns
        features['fraction_count'] = len(self.fraction_pattern.findall(text))
        features['simple_fraction_count'] = len(self.simple_fraction_pattern.findall(text))
        features['decimal_count'] = len(self.decimal_pattern.findall(text))
        features['percentage_count'] = len(self.percentage_pattern.findall(text))
        features['operation_count'] = len(self.operation_pattern.findall(text))
        
        # Number analysis
        numbers = self.safe_extract_numbers(text)
        features['number_count'] = len(numbers)
        
        if numbers:
            features['max_number'] = max(numbers)
            features['min_number'] = min(numbers)
            features['number_range'] = features['max_number'] - features['min_number']
            features['avg_number'] = np.mean(numbers)
        else:
            features['max_number'] = 0
            features['min_number'] = 0
            features['number_range'] = 0
            features['avg_number'] = 0
        
        return features
    
    def extract_concept_features(self, text):
        """Extract mathematical concept features"""
        text_lower = str(text).lower()
        features = {}
        
        for concept, keywords in self.math_concepts.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            features[f'{concept}_concept'] = min(count, 10)  # Cap at 10
            features[f'has_{concept}'] = 1 if count > 0 else 0
            
        return features
    
    def extract_complexity_features(self, text):
        """Extract text complexity features"""
        text = str(text)
        features = {}
        
        # Basic text stats
        features['text_length'] = min(len(text), 5000)
        words = text.split()
        features['word_count'] = len(words)
        
        if words:
            features['avg_word_length'] = min(np.mean([len(w) for w in words]), 20)
        else:
            features['avg_word_length'] = 0
            
        features['sentence_count'] = min(len(re.split(r'[.!?]', text)), 50)
        features['has_latex'] = 1 if '\\' in text else 0
        features['parentheses_count'] = min(text.count('(') + text.count(')'), 20)
        
        return features
    
    def extract_all_features(self, text):
        """Extract all features with error handling"""
        all_features = {}
        
        try:
            all_features.update(self.extract_numerical_features(text))
        except Exception as e:
            print(f"Error in numerical features: {e}")
            
        try:
            all_features.update(self.extract_concept_features(text))
        except Exception as e:
            print(f"Error in concept features: {e}")
            
        try:
            all_features.update(self.extract_complexity_features(text))
        except Exception as e:
            print(f"Error in complexity features: {e}")
        
        # Ensure all values are finite
        for key, value in all_features.items():
            if not np.isfinite(value):
                all_features[key] = 0
                
        return all_features


class SemanticFeatureExtractor:
    """Extract semantic relationship features"""
    
    @staticmethod
    def word_overlap_similarity(text1, text2):
        """Calculate word overlap similarity"""
        try:
            words1 = set(str(text1).lower().split())
            words2 = set(str(text2).lower().split())
            if not words1 or not words2:
                return 0
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0
        except:
            return 0
    
    def extract_features(self, df):
        """Extract semantic features from dataframe"""
        features = {}
        
        # Similarity features
        features['question_answer_similarity'] = df.apply(
            lambda x: self.word_overlap_similarity(x['QuestionText'], x['MC_Answer']), axis=1
        )
        
        features['question_explanation_similarity'] = df.apply(
            lambda x: self.word_overlap_similarity(x['QuestionText'], x['StudentExplanation']), axis=1
        )
        
        features['answer_explanation_similarity'] = df.apply(
            lambda x: self.word_overlap_similarity(x['MC_Answer'], x['StudentExplanation']), axis=1
        )
        
        # Length features
        q_len = df['QuestionText'].str.len().fillna(1)
        e_len = df['StudentExplanation'].str.len().fillna(0)
        
        features['explanation_question_length_ratio'] = np.clip(e_len / q_len, 0, 10)
        features['explanation_length'] = np.clip(e_len, 0, 1000)
        features['question_length'] = np.clip(q_len, 0, 1000)
        
        return pd.DataFrame(features)


class TextProcessor:
    """Handle text preprocessing"""
    
    def __init__(self):
        self.lemmatizer = lemmatizer
    
    def clean_text(self, text):
        """Clean text for processing"""
        text = str(text)
        
        # Basic cleaning
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        
        # Keep alphanumeric, basic math, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s+\-*/=().]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def lemmatize_text(self, text):
        """Lemmatize text if lemmatizer available"""
        if self.lemmatizer:
            try:
                words = text.split()
                lemmatized = [self.lemmatizer.lemmatize(word) for word in words]
                return ' '.join(lemmatized)
            except:
                return text
        return text
    
    def process_text(self, text):
        """Full text processing pipeline"""
        text = self.clean_text(text)
        text = self.lemmatize_text(text)
        return text


class MAP3Optimizer:
    """MAP@3 specific optimization and evaluation"""
    
    @staticmethod
    def map3_score(y_true, y_pred_proba, class_names):
        """Calculate MAP@3 score"""
        scores = []
        
        for i, true_label in enumerate(y_true):
            # Get top 3 predictions
            top_3_indices = np.argsort(y_pred_proba[i])[::-1][:3]
            
            # Find rank of true label
            score = 0.0
            for rank, pred_idx in enumerate(top_3_indices, 1):
                pred_label = class_names[pred_idx]
                if pred_label == true_label:
                    score = 1.0 / rank
                    break
            
            scores.append(score)
        
        return np.mean(scores)
    
    @staticmethod
    def generate_combined_predictions(cat_probs, misc_probs, category_classes, misconception_classes, top_k=3):
        """Generate combined Category:Misconception predictions"""
        predictions = []
        
        for i in range(len(cat_probs)):
            pred_combos = []
            
            # Get top categories and misconceptions
            top_cats = np.argsort(cat_probs[i])[::-1][:top_k]
            top_miscs = np.argsort(misc_probs[i])[::-1][:top_k]
            
            # Generate combinations
            for cat_idx in top_cats:
                cat_name = category_classes[cat_idx]
                cat_prob = cat_probs[i][cat_idx]
                
                if 'Misconception' in cat_name:
                    # Add top misconceptions for misconception categories
                    for misc_idx in top_miscs:
                        misc_name = misconception_classes[misc_idx]
                        misc_prob = misc_probs[i][misc_idx]
                        
                        if misc_name != 'NA':
                            combined_label = f"{cat_name}:{misc_name}"
                            combined_prob = cat_prob * misc_prob
                            pred_combos.append((combined_label, combined_prob))
                else:
                    # Non-misconception categories always use NA
                    combined_label = f"{cat_name}:NA"
                    pred_combos.append((combined_label, cat_prob))
            
            # Sort by probability and take top 3
            pred_combos.sort(key=lambda x: x[1], reverse=True)
            top_3 = [combo[0] for combo in pred_combos[:3]]
            
            # Ensure exactly 3 predictions
            while len(top_3) < 3:
                top_3.append("True_Correct:NA")
            
            predictions.append(top_3)
        
        return predictions


def main():
    print("=== MAP Competition - Advanced Baseline ===")
    print("Strategies: Mathematical Features + MAP@3 Optimization")
    print()
    
    # Load data
    print("Loading data...")
    # Local paths
    train_path = "/Users/osawa/kaggle/map-charting-student-math-misunderstandings/data/raw/train.csv"
    test_path = "/Users/osawa/kaggle/map-charting-student-math-misunderstandings/data/raw/test.csv"
    
    # For Kaggle submission, use these paths:
    # train_path = "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
    # test_path = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Prepare targets
    train['Misconception'] = train['Misconception'].fillna('NA')
    train['target_combined'] = train['Category'] + ':' + train['Misconception']
    
    # Create text features
    print("\nCreating combined text...")
    train['combined_text'] = ("Question: " + train['QuestionText'].astype(str) + 
                             " Answer: " + train['MC_Answer'].astype(str) + 
                             " Explanation: " + train['StudentExplanation'].astype(str))
    
    test['combined_text'] = ("Question: " + test['QuestionText'].astype(str) + 
                            " Answer: " + test['MC_Answer'].astype(str) + 
                            " Explanation: " + test['StudentExplanation'].astype(str))
    
    # Process text
    print("Processing text...")
    processor = TextProcessor()
    train['processed_text'] = train['combined_text'].apply(processor.process_text)
    test['processed_text'] = test['combined_text'].apply(processor.process_text)
    
    # Extract mathematical features
    print("Extracting mathematical features...")
    math_extractor = MathematicalFeatureExtractor()
    
    train_math_features = []
    for text in train['combined_text']:
        features = math_extractor.extract_all_features(text)
        train_math_features.append(features)
    
    test_math_features = []
    for text in test['combined_text']:
        features = math_extractor.extract_all_features(text)
        test_math_features.append(features)
    
    train_math_df = pd.DataFrame(train_math_features).fillna(0)
    test_math_df = pd.DataFrame(test_math_features).fillna(0)
    
    print(f"Mathematical features shape: {train_math_df.shape}")
    
    # Extract semantic features
    print("Extracting semantic features...")
    semantic_extractor = SemanticFeatureExtractor()
    train_semantic_df = semantic_extractor.extract_features(train)
    test_semantic_df = semantic_extractor.extract_features(test)
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),
        max_df=0.95,
        min_df=2,
        max_features=20000
    )
    
    all_text = pd.concat([train['processed_text'], test['processed_text']])
    tfidf.fit(all_text)
    
    train_tfidf = tfidf.transform(train['processed_text'])
    test_tfidf = tfidf.transform(test['processed_text'])
    
    print(f"TF-IDF shape: {train_tfidf.shape}")
    
    # Combine features
    print("Combining features...")
    train_math_sparse = sparse.csr_matrix(train_math_df.values)
    test_math_sparse = sparse.csr_matrix(test_math_df.values)
    
    train_semantic_sparse = sparse.csr_matrix(train_semantic_df.values)
    test_semantic_sparse = sparse.csr_matrix(test_semantic_df.values)
    
    train_features = sparse.hstack([train_tfidf, train_math_sparse, train_semantic_sparse])
    test_features = sparse.hstack([test_tfidf, test_math_sparse, test_semantic_sparse])
    
    print(f"Combined features shape: {train_features.shape}")
    
    # Prepare targets
    categories = sorted(train['Category'].unique())
    misconceptions = sorted(train['Misconception'].unique())
    
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    misc_to_idx = {misc: idx for idx, misc in enumerate(misconceptions)}
    
    train['cat_target'] = train['Category'].map(cat_to_idx)
    train['misc_target'] = train['Misconception'].map(misc_to_idx)
    
    print(f"Categories: {len(categories)}")
    print(f"Misconceptions: {len(misconceptions)}")
    
    # Cross-validation training
    print("\nStarting cross-validation...")
    n_folds = 5  # Reduced for faster execution
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_cat_preds = np.zeros((len(train), len(categories)))
    oof_misc_preds = np.zeros((len(train), len(misconceptions)))
    
    test_cat_preds = np.zeros((len(test), len(categories)))
    test_misc_preds = np.zeros((len(test), len(misconceptions)))
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, train['cat_target'])):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        # Split data
        X_train, X_val = train_features[train_idx], train_features[val_idx]
        y_cat_train, y_cat_val = train['cat_target'].iloc[train_idx], train['cat_target'].iloc[val_idx]
        y_misc_train, y_misc_val = train['misc_target'].iloc[train_idx], train['misc_target'].iloc[val_idx]
        
        # Train category model
        print("  Training category model...")
        cat_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        cat_model.fit(X_train, y_cat_train)
        
        # Train misconception model  
        print("  Training misconception model...")
        misc_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            objective='multiclass',
            metric='multi_logloss',
            verbosity=-1
        )
        misc_model.fit(X_train, y_misc_train)
        
        # Predictions
        oof_cat_preds[val_idx] = cat_model.predict_proba(X_val)
        oof_misc_preds[val_idx] = misc_model.predict_proba(X_val)
        
        test_cat_preds += cat_model.predict_proba(test_features) / n_folds
        test_misc_preds += misc_model.predict_proba(test_features) / n_folds
        
        # Calculate MAP@3 for this fold
        val_predictions = MAP3Optimizer.generate_combined_predictions(
            oof_cat_preds[val_idx], oof_misc_preds[val_idx], 
            categories, misconceptions
        )
        
        val_true = train['target_combined'].iloc[val_idx].tolist()
        val_pred_first = [pred[0] for pred in val_predictions]
        
        # Simple accuracy for fold evaluation
        fold_acc = np.mean([true == pred for true, pred in zip(val_true, val_pred_first)])
        fold_scores.append(fold_acc)
        
        print(f"  Fold {fold + 1} accuracy: {fold_acc:.4f}")
    
    print(f"\nCV Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    # Generate final predictions
    print("\nGenerating test predictions...")
    test_predictions = MAP3Optimizer.generate_combined_predictions(
        test_cat_preds, test_misc_preds, categories, misconceptions
    )
    
    # Create submission
    submission_data = []
    for i, preds in enumerate(test_predictions):
        row_id = test.iloc[i]['row_id']
        pred_str = ' '.join(preds)
        submission_data.append({'row_id': row_id, 'Category:Misconception': pred_str})
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('submission.csv', index=False)
    
    print(f"\nSubmission created with {len(submission_df)} rows")
    print("Sample predictions:")
    for i in range(min(3, len(test_predictions))):
        print(f"  {i}: {' '.join(test_predictions[i])}")
    
    print("\n=== Advanced Baseline Complete ===")


if __name__ == "__main__":
    main()