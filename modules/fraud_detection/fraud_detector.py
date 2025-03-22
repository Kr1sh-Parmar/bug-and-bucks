import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime, timedelta
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetector:
    def __init__(self):
        # Define the possible values first
        self.merchants = ['Amazon', 'Flipkart', 'Local Store', 'Foreign Merchant', 'Unknown Vendor']
        self.locations = ['Mumbai', 'Delhi', 'Bangalore', 'International', 'Unknown']
        self.categories = ['Shopping', 'Food', 'Transport', 'Entertainment', 'Bills', 'Transfer']
        
        # Define transaction patterns
        self.patterns = {
            'rapid_transactions': {'window_minutes': 60, 'threshold': 3},
            'late_night': {'start_hour': 23, 'end_hour': 6},
            'high_amount': {'threshold': 15000},
            'foreign_transaction': {'locations': ['International', 'Unknown']},
            'unusual_merchant': {'merchants': ['Foreign Merchant', 'Unknown Vendor']},
            'velocity_check': {'window_hours': 24, 'amount_threshold': 50000}
        }
        
        # Initialize model components
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Generate data and train model
        self.synthetic_data = self._generate_synthetic_data()
        
        # Prepare features and store columns
        features = self._encode_features(self.synthetic_data)
        self.feature_columns = features.columns
        
        # Train the model
        self._train_model(features)
        
        logger.info("FraudDetector initialized successfully")
    
    def _generate_synthetic_data(self, num_transactions=1000):
        """Generate synthetic transaction data with more features"""
        data = []
        
        # Generate normal transactions
        for _ in range(int(num_transactions * 0.9)):
            amount = np.random.normal(5000, 2000)
            hour = np.random.normal(14, 4)
            location = np.random.choice(self.locations[:3])
            merchant = np.random.choice(self.merchants[:3])
            category = np.random.choice(self.categories)
            frequency = np.random.normal(5, 2)  # transactions per week
            avg_amount = np.random.normal(3000, 1000)
            data.append([
                amount, hour, location, merchant, category,
                frequency, avg_amount, 0  # 0 for normal
            ])
        
        # Generate fraudulent transactions
        for _ in range(int(num_transactions * 0.1)):
            amount = np.random.normal(20000, 5000)
            hour = np.random.normal(3, 2)
            location = np.random.choice(self.locations[3:])
            merchant = np.random.choice(self.merchants[3:])
            category = np.random.choice(self.categories)
            frequency = np.random.normal(15, 5)  # higher frequency
            avg_amount = np.random.normal(15000, 5000)  # higher amounts
            data.append([
                amount, hour, location, merchant, category,
                frequency, avg_amount, 1  # 1 for fraudulent
            ])
        
        df = pd.DataFrame(data, columns=[
            'amount', 'hour', 'location', 'merchant', 'category',
            'frequency', 'avg_amount', 'is_fraud'
        ])
        return df
    
    def _encode_features(self, df):
        """Encode features with additional derived features"""
        # Create numeric features
        features = df[['amount', 'hour', 'frequency', 'avg_amount']].copy()
        
        # Add derived features
        features['amount_to_avg_ratio'] = df['amount'] / df['avg_amount']
        features['hour_risk'] = df['hour'].apply(self._calculate_hour_risk)
        
        # One-hot encode location
        for loc in self.locations:
            features[f'loc_{loc}'] = (df['location'] == loc).astype(int)
            
        # One-hot encode merchant
        for merch in self.merchants:
            features[f'merchant_{merch}'] = (df['merchant'] == merch).astype(int)
            
        # One-hot encode category
        for cat in self.categories:
            features[f'category_{cat}'] = (df['category'] == cat).astype(int)
        
        return features
    
    def _calculate_hour_risk(self, hour):
        """Calculate risk score based on transaction hour"""
        if 23 <= hour or hour <= 5:
            return 1.0  # Highest risk (late night)
        elif 6 <= hour <= 8 or 21 <= hour <= 22:
            return 0.7  # High risk (early morning/late evening)
        elif 9 <= hour <= 17:
            return 0.2  # Low risk (business hours)
        else:
            return 0.5  # Medium risk (evening)
    
    def _validate_transaction(self, transaction):
        """Validate transaction data"""
        required_fields = ['amount', 'timestamp', 'merchant', 'location', 'category']
        for field in required_fields:
            if field not in transaction:
                raise ValueError(f"Missing required field: {field}")
        
        try:
            float(transaction['amount'])
        except (ValueError, TypeError):
            raise ValueError("Invalid amount value")
        
        if not isinstance(transaction['timestamp'], datetime):
            raise ValueError("Invalid timestamp format")
        
        if transaction['merchant'] not in self.merchants:
            logger.warning(f"Unknown merchant: {transaction['merchant']}")
            transaction['merchant'] = 'Unknown Vendor'
            
        if transaction['location'] not in self.locations:
            logger.warning(f"Unknown location: {transaction['location']}")
            transaction['location'] = 'Unknown'
            
        if transaction['category'] not in self.categories:
            logger.warning(f"Unknown category: {transaction['category']}")
            transaction['category'] = 'Shopping'
        
        return transaction
    
    def _analyze_patterns(self, transaction, user_history):
        """Analyze transaction patterns"""
        patterns_detected = []
        
        # Check for rapid transactions
        recent_transactions = [
            t for t in user_history 
            if (transaction['timestamp'] - t['timestamp']).total_seconds() <= self.patterns['rapid_transactions']['window_minutes'] * 60
        ]
        if len(recent_transactions) >= self.patterns['rapid_transactions']['threshold']:
            patterns_detected.append({
                'type': 'frequency',
                'description': 'Multiple rapid transactions detected',
                'confidence': min(90, len(recent_transactions) * 20)
            })
        
        # Check transaction velocity (amount over time)
        day_transactions = [
            t for t in user_history 
            if (transaction['timestamp'] - t['timestamp']).total_seconds() <= self.patterns['velocity_check']['window_hours'] * 3600
        ]
        total_amount = sum(float(t['amount']) for t in day_transactions)
        if total_amount > self.patterns['velocity_check']['amount_threshold']:
            patterns_detected.append({
                'type': 'velocity',
                'description': 'Unusual transaction velocity detected',
                'confidence': min(90, int(total_amount / 1000))
            })
        
        return patterns_detected
    
    def analyze_transaction(self, transaction, user_history=None):
        """Analyze a single transaction with enhanced pattern detection"""
        try:
            # Validate and preprocess transaction
            transaction = self._validate_transaction(transaction)
            
            # Create feature vector
            df = pd.DataFrame([{
                'amount': float(transaction['amount']),
                'hour': transaction['timestamp'].hour,
                'location': transaction['location'],
                'merchant': transaction['merchant'],
                'category': transaction['category'],
                'frequency': len(user_history) if user_history else 5,  # default to average
                'avg_amount': np.mean([float(t['amount']) for t in user_history]) if user_history else 5000  # default to average
            }])
            
            # Encode features
            features = self._encode_features(df)
            
            # Ensure all columns exist in the same order
            missing_cols = set(self.feature_columns) - set(features.columns)
            for col in missing_cols:
                features[col] = 0
            features = features[self.feature_columns]
            
            # Get model prediction
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)
            score = self.model.score_samples(features_scaled)
            
            # Calculate fraud probability
            baseline_features = self._encode_features(self.synthetic_data)
            baseline_scaled = self.scaler.transform(baseline_features)
            baseline_scores = self.model.score_samples(baseline_scaled)
            
            fraud_probability = int((1 - (score - baseline_scores.min()) / 
                               (baseline_scores.max() - baseline_scores.min())) * 100)
            fraud_probability = max(0, min(100, fraud_probability))
            
            # Get risk factors
            risk_factors = self._get_risk_factors(transaction, fraud_probability)
            
            # Add pattern-based risk factors
            if user_history:
                pattern_risks = self._analyze_patterns(transaction, user_history)
                risk_factors.extend(pattern_risks)
            
            return {
                'is_fraud': prediction[0] == -1,
                'fraud_probability': fraud_probability,
                'risk_factors': risk_factors,
                'transaction_details': {
                    'amount': float(transaction['amount']),
                    'timestamp': transaction['timestamp'],
                    'merchant': transaction['merchant'],
                    'location': transaction['location'],
                    'category': transaction['category']
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transaction: {str(e)}", exc_info=True)
            logger.error(f"Transaction data: {transaction}")
            raise

    def get_fraud_summary(self, transactions):
        """Get enhanced fraud summary with pattern analysis"""
        total = len(transactions)
        if total == 0:
            return self._get_empty_summary()
        
        try:
            # Sort transactions by timestamp
            sorted_transactions = sorted(transactions, key=lambda x: x['timestamp'])
            
            # Analyze each transaction with history
            analyzed = []
            for i, transaction in enumerate(sorted_transactions):
                history = sorted_transactions[:i]  # Use previous transactions as history
                analysis = self.analyze_transaction(transaction, history)
                analyzed.append(analysis)
            
            # Calculate statistics
            flagged = sum(1 for a in analyzed if a['is_fraud'])
            high_risk = sum(1 for a in analyzed if a['fraud_probability'] > 80)
            medium_risk = sum(1 for a in analyzed if 50 < a['fraud_probability'] <= 80)
            
            # Calculate pattern statistics
            pattern_stats = self._calculate_pattern_stats(analyzed)
            
            return {
                'flagged_count': flagged,
                'suspicious_percentage': (flagged / total) * 100,
                'risk_score': sum(a['fraud_probability'] for a in analyzed) / total / 10,
                'chart_data': {
                    'safe': ((total - medium_risk - high_risk) / total) * 100,
                    'suspicious': (medium_risk / total) * 100,
                    'fraudulent': (high_risk / total) * 100
                },
                'pattern_summary': pattern_stats
            }
            
        except Exception as e:
            logger.error(f"Error generating fraud summary: {str(e)}", exc_info=True)
            logger.error(f"Transactions data: {transactions}")
            raise
    
    def _get_empty_summary(self):
        """Return empty summary structure"""
        return {
            'flagged_count': 0,
            'suspicious_percentage': 0,
            'risk_score': 0,
            'chart_data': {'safe': 100, 'suspicious': 0, 'fraudulent': 0},
            'pattern_summary': {
                'rapid_transactions': 0,
                'late_night': 0,
                'high_amount': 0,
                'foreign_transaction': 0,
                'unusual_merchant': 0,
                'velocity_alerts': 0
            }
        }
    
    def _calculate_pattern_stats(self, analyzed_transactions):
        """Calculate statistics about detected patterns"""
        pattern_counts = {
            'rapid_transactions': 0,
            'late_night': 0,
            'high_amount': 0,
            'foreign_transaction': 0,
            'unusual_merchant': 0,
            'velocity_alerts': 0
        }
        
        for analysis in analyzed_transactions:
            for risk in analysis['risk_factors']:
                if risk['type'] == 'frequency':
                    pattern_counts['rapid_transactions'] += 1
                elif risk['type'] == 'time':
                    pattern_counts['late_night'] += 1
                elif risk['type'] == 'amount':
                    pattern_counts['high_amount'] += 1
                elif risk['type'] == 'location':
                    pattern_counts['foreign_transaction'] += 1
                elif risk['type'] == 'velocity':
                    pattern_counts['velocity_alerts'] += 1
                    
        return pattern_counts

    def _train_model(self, features):
        """Train the isolation forest model"""
        # Scale features
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        
        # Train model
        self.model.fit(features_scaled)
        
    def _get_risk_factors(self, transaction, fraud_probability):
        """Analyze risk factors for a transaction"""
        risk_factors = []
        
        # Check amount
        if transaction['amount'] > 15000:
            risk_factors.append({
                'type': 'amount',
                'description': 'Unusual spending amount compared to history',
                'confidence': min(90, int(float(transaction['amount']) / 200))
            })
        
        # Check time
        hour = transaction['timestamp'].hour
        if hour < 6 or hour > 23:
            risk_factors.append({
                'type': 'time',
                'description': 'Atypical transaction timing pattern',
                'confidence': 75
            })
        
        # Check location
        if transaction['location'] in ['International', 'Unknown']:
            risk_factors.append({
                'type': 'location',
                'description': 'Unusual location detected for transactions',
                'confidence': 85
            })
        
        return risk_factors 