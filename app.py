from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import time
import random
import datetime
from datetime import timedelta
import logging

# Import our custom modules
from modules.face_recognition.face_detector import FaceRecognition
from modules.blockchain.blockchain_auth import BlockchainAuth
from modules.doc_verification.document_verifier import DocumentVerifier
from modules.fraud_detection.fraud_detector import FraudDetector

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize our modules - FORCE FALLBACK MODE
face_system = FaceRecognition(model_path=None)  # Explicitly use no model
blockchain = BlockchainAuth()
doc_verifier = DocumentVerifier()
fraud_detector = FraudDetector()  # Initialize our new fraud detector

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# This would be executed after the DocumentVerifier is created
if not os.path.exists("models"):
    os.makedirs("models")

# Define regions of interest for Aadhar card verification
aadhar_card_roi = {
    "number_region": (100, 300, 400, 50),  # example coordinates (x, y, width, height)
    "name_region": (100, 200, 400, 50),
    "dob_region": (100, 250, 200, 50),
    "qr_code_region": (500, 400, 200, 200)
}

# In a real implementation, you would load an actual template image
# For now, we're just defining the regions of interest
doc_verifier.add_document_template("aadhar_card", None, aadhar_card_roi)

# Initialize some storage
if not hasattr(app, 'users'):
    app.users = {}
    
if not hasattr(app, 'transactions'):
    app.transactions = {}

# Add this at the top of your app.py file (replace any existing debug lines)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add this to app.py - a debugging flag to disable camera
app.config['DISABLE_CAMERA'] = True  # Set to False to enable camera

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        user_id = request.form['user_id']
        username = request.form['username']
        # Add new user data
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        
        # Add user data to a dictionary to be stored
        user_data = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "phone": phone,
            "address": address,
            "registration_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Store this in a users dictionary (add this to the top of your app.py)
        if not hasattr(app, 'users'):
            app.users = {}
        app.users[user_id] = user_data
        
        # Handle face image upload
        if 'face_image' not in request.files:
            return render_template('register.html', error="No face image provided")
        
        face_file = request.files['face_image']
        if face_file.filename == '':
            return render_template('register.html', error="No face image selected")
        
        # Save and process face image
        face_filename = secure_filename(face_file.filename)
        face_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
        face_file.save(face_path)
        
        face_image = cv2.imread(face_path)
        if face_image is None:
            return render_template('register.html', error="Could not read face image")
        
        # Register face
        success, face_msg = face_system.register_face(user_id, face_image)
        if not success:
            return render_template('register.html', error=face_msg)
        
        # Handle document upload
        if 'document_image' not in request.files:
            return render_template('register.html', error="No document image provided")
        
        doc_file = request.files['document_image']
        if doc_file.filename == '':
            return render_template('register.html', error="No document image selected")
        
        # Save and process document image
        doc_filename = secure_filename(doc_file.filename)
        doc_path = os.path.join(app.config['UPLOAD_FOLDER'], doc_filename)
        doc_file.save(doc_path)
        
        doc_image = cv2.imread(doc_path)
        if doc_image is None:
            return render_template('register.html', error="Could not read document image")
        
        # Verify document
        doc_type = request.form['document_type']
        
        # Bypass document verification if needed
        bypass_doc_verification = True  # Set to False when Tesseract is installed

        if bypass_doc_verification:
            doc_data = {"doc_type": doc_type, "simulated": True}
        else:
            is_verified, doc_result = doc_verifier.verify_document(doc_image, doc_type)
            if not is_verified:
                os.remove(doc_path)
                return render_template('register.html', error=f"Document verification failed: {doc_result}")
            
            doc_data = doc_result["extracted_data"] if isinstance(doc_result, dict) and "extracted_data" in doc_result else {}
        
        # Register on blockchain
        try:
            # Register identity on blockchain
            face_data = None
            
            # Check if we're running in fallback mode
            if face_system.embedding_model is None:
                # Use the face database entry which already stores the face in fallback mode
                if user_id in face_system.face_database:
                    face_data = face_system.face_database[user_id]
                else:
                    return render_template('register.html', error="Face data not found in database")
            else:
                # Get the face embedding for blockchain registration
                _, faces = face_system.detect_face(face_image)
                if len(faces) == 0:
                    return render_template('register.html', error="No face detected for blockchain registration")
                
                face_data = face_system.get_face_embedding(face_image[
                    faces[0][1]:faces[0][1]+faces[0][3], 
                    faces[0][0]:faces[0][0]+faces[0][2]
                ])
            
            # Register identity on blockchain
            blockchain_result = blockchain.register_identity(user_id, face_data, doc_data, user_data)
            
            # Delete temporary files
            os.remove(face_path)
            os.remove(doc_path)
            
            return render_template('register_success.html', 
                                   user_id=user_id, 
                                   identity_hash=blockchain_result['identity_hash'])
            
        except Exception as e:
            return render_template('register.html', error=f"Blockchain registration error: {str(e)}")
        
    # Pass the disable_camera flag to the template
    return render_template('register.html', disable_camera=app.config['DISABLE_CAMERA'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get form data
        user_id = request.form['user_id']
        logger.info(f"Login attempt for user_id: {user_id}")
        
        # Check if user exists
        if user_id not in face_system.face_database:
            logger.warning(f"User {user_id} not found in face database")
            return render_template('login.html', error=f"User ID {user_id} not registered")
        
        # Handle face image upload
        if 'face_image' not in request.files:
            return render_template('login.html', error="No face image provided")
        
        face_file = request.files['face_image']
        if face_file.filename == '':
            return render_template('login.html', error="No face image selected")
        
        # Save and process face image
        face_filename = secure_filename(face_file.filename)
        face_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
        face_file.save(face_path)
        
        face_image = cv2.imread(face_path)
        if face_image is None:
            return render_template('login.html', error="Could not read face image")
        
        # Verify face
        logger.info(f"Verifying face for user {user_id}")
        success, face_msg = face_system.verify_face(user_id, face_image)
        if not success:
            os.remove(face_path)
            logger.warning(f"Face verification failed: {face_msg}")
            return render_template('login.html', error=f"Face verification failed: {face_msg}")
        
        # Get face data for blockchain verification
        if face_system.embedding_model is None:
            # In fallback mode, we use the face database entry directly
            logger.info("Using fallback mode for blockchain verification")
            if user_id in face_system.face_database:
                face_data = face_system.face_database[user_id]
            else:
                os.remove(face_path)
                logger.error(f"User {user_id} found for face verification but not in database for blockchain")
                return render_template('login.html', error="Face data not found in database")
        else:
            # Normal mode with embeddings
            logger.info("Using embedding mode for blockchain verification")
            face_images, faces = face_system.detect_face(face_image)
            if len(face_images) == 0:
                os.remove(face_path)
                return render_template('login.html', error="No face detected for blockchain verification")
            
            face_data = face_system.get_face_embedding(face_images[0])

        # Verify on blockchain
        logger.info(f"Verifying identity on blockchain for user {user_id}")
        blockchain_success, blockchain_msg = blockchain.verify_identity(user_id, face_data)
        
        os.remove(face_path)
        
        if not blockchain_success:
            logger.warning(f"Blockchain verification failed: {blockchain_msg}")
            return render_template('login.html', error=f"Blockchain verification failed: {blockchain_msg}")
        
        # Set session
        session['user_id'] = user_id
        session['logged_in'] = True
        
        logger.info(f"User {user_id} logged in successfully")
        return redirect(url_for('dashboard'))
        
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    
    # Get or create user data
    if user_id not in app.users:
        # This should only happen if they registered before we added the additional fields
        app.users[user_id] = {
            'user_id': user_id,
            'username': f"User {user_id}",
            'email': f"user{user_id}@example.com",
            'phone': f"+91 98765{random.randint(10000, 99999)}",
            'address': "123 Main Street, Mumbai, India"
        }
    
    # Get some data for the dashboard
    identity_hash = blockchain.user_identities.get(user_id, {}).get('identity_hash', 'Unknown')
    last_login = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    recent_login_time = last_login
    
    return render_template('dashboard.html',
                          user_id=user_id,
                          user_data=app.users[user_id],
                          identity_hash=identity_hash,
                          last_login=last_login,
                          recent_login_time=recent_login_time)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/transactions', methods=['GET', 'POST'])
def transactions():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    
    # Generate transactions if none exist for this user
    if user_id not in app.transactions:
        app.transactions[user_id] = generate_transactions(user_id)
    
    return render_template('transactions.html', 
                          user_id=user_id,
                          user_data=app.users.get(user_id, {}),
                          transactions=app.transactions[user_id])

@app.route('/flag_transaction/<transaction_id>', methods=['POST'])
def flag_transaction(transaction_id):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    
    # Find and toggle the transaction flag
    for transaction in app.transactions.get(user_id, []):
        if transaction['id'] == transaction_id:
            transaction['flagged'] = not transaction.get('flagged', False)
            break
    
    return redirect(url_for('transactions'))

@app.route('/fraud_detection')
def fraud_detection():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    
    # Get user's transactions
    transactions = app.transactions.get(user_id, [])
    
    # Convert transactions to the format expected by fraud detector
    formatted_transactions = []
    for t in transactions:
        # Add some variability to fraud probability based on amount and merchant
        base_prob = t.get('fraud_probability', 0)
        amount = abs(float(t['amount']))
        merchant = t['merchant']
        
        # Adjust probability based on amount
        if amount > 40000:
            base_prob = min(base_prob + 15, 100)
        elif amount > 20000:
            base_prob = min(base_prob + 10, 100)
        elif amount > 10000:
            base_prob = min(base_prob + 5, 100)
            
        # Adjust probability based on merchant type
        if 'Foreign' in merchant:
            base_prob = min(base_prob + 10, 100)
        elif 'Unknown' in merchant:
            base_prob = min(base_prob + 8, 100)
        elif 'Crypto' in merchant:
            base_prob = min(base_prob + 12, 100)
            
        # Add some random variation (±5%)
        variation = random.uniform(-5, 5)
        final_prob = max(0, min(100, base_prob + variation))
        
        # Update the transaction's fraud probability
        t['fraud_probability'] = round(final_prob, 1)
        
        formatted_transactions.append({
            'id': t.get('id', ''),
            'amount': amount,
            'timestamp': datetime.datetime.strptime(t['date'], '%Y-%m-%d'),
            'merchant': merchant,
            'location': 'International' if 'International' in merchant else 'Mumbai',
            'category': t['category'],
            'fraud_probability': t['fraud_probability']
        })
    
    # Calculate summary statistics
    total_transactions = len(formatted_transactions)
    flagged_count = sum(1 for t in transactions if t.get('flagged', False))
    suspicious_count = sum(1 for t in transactions if 30 <= t.get('fraud_probability', 0) < 70)
    fraud_count = sum(1 for t in transactions if t.get('fraud_probability', 0) >= 70)
    
    # Calculate risk score (weighted average of fraud probabilities)
    total_risk = sum(t.get('fraud_probability', 0) for t in transactions)
    risk_score = int((total_risk / total_transactions) / 10) if total_transactions > 0 else 0
    
    # Calculate suspicious percentage
    suspicious_percentage = (suspicious_count + fraud_count) / total_transactions * 100 if total_transactions > 0 else 0
    
    # Prepare chart data in the format expected by the template
    chart_data = {
        'safe': total_transactions - suspicious_count - fraud_count,
        'suspicious': suspicious_count,
        'fraudulent': fraud_count
    }
    
    # Analyze each transaction
    ml_flagged_transactions = []
    for t in transactions:
        if t.get('fraud_status') != 'dismiss' and t.get('fraud_probability', 0) > 30:
            ml_flagged_transactions.append({
                'id': t.get('id', ''),
                'date': t['date'],
                'merchant': t['merchant'],
                'category': t['category'],
                'amount': t['amount'],
                'fraud_probability': t['fraud_probability'],
                'flag_reason': get_flag_reason(t),
                'fraud_status': t.get('fraud_status', None)
            })
    
    # Sort by fraud probability (highest first)
    ml_flagged_transactions.sort(key=lambda x: x['fraud_probability'], reverse=True)
    
    # Get fraud patterns
    fraud_patterns = analyze_fraud_patterns(transactions)
    
    return render_template('fraud_detection.html',
                          user_id=user_id,
                          user_data=app.users.get(user_id, {}),
                          flagged_count=flagged_count,
                          suspicious_percentage=int(suspicious_percentage),
                          risk_score=risk_score,
                          fraud_patterns=fraud_patterns,
                          chart_data=chart_data,
                          ml_flagged_transactions=ml_flagged_transactions)

def get_flag_reason(transaction):
    """Helper function to determine flag reason based on transaction characteristics"""
    fraud_prob = transaction.get('fraud_probability', 0)
    amount = abs(transaction.get('amount', 0))
    merchant = transaction.get('merchant', '')
    
    reasons = []
    
    # Check amount
    if amount > 40000:
        reasons.append("Very large transaction amount")
    elif amount > 20000:
        reasons.append("Large transaction amount")
    elif amount > 10000:
        reasons.append("Above average transaction amount")
    
    # Check merchant
    if 'Foreign' in merchant:
        reasons.append("International merchant")
    elif 'Unknown' in merchant:
        reasons.append("Unverified merchant")
    elif 'Crypto' in merchant:
        reasons.append("Cryptocurrency transaction")
    
    # Combine reasons based on probability
    if fraud_prob >= 85:
        return "High-risk: " + " and ".join(reasons)
    elif fraud_prob >= 70:
        return "Suspicious: " + " and ".join(reasons)
    elif fraud_prob >= 30:
        return "Unusual: " + " and ".join(reasons)
    else:
        return "Normal transaction"

def analyze_fraud_patterns(transactions):
    """Helper function to analyze and categorize fraud patterns"""
    patterns = {
        'amount': {'count': 0, 'confidence_sum': 0},
        'merchant': {'count': 0, 'confidence_sum': 0},
        'location': {'count': 0, 'confidence_sum': 0}
    }
    
    for t in transactions:
        fraud_prob = t.get('fraud_probability', 0)
        if fraud_prob > 50:
            # Amount-based pattern
            if abs(t.get('amount', 0)) > 15000:
                patterns['amount']['count'] += 1
                patterns['amount']['confidence_sum'] += fraud_prob
            
            # Merchant-based pattern
            if any(x in t.get('merchant', '') for x in ['Foreign', 'Crypto', 'Unknown']):
                patterns['merchant']['count'] += 1
                patterns['merchant']['confidence_sum'] += fraud_prob
            
            # Location-based pattern
            if 'International' in t.get('merchant', ''):
                patterns['location']['count'] += 1
                patterns['location']['confidence_sum'] += fraud_prob
    
    # Convert patterns to list format
    fraud_patterns = []
    for pattern_type, data in patterns.items():
        if data['count'] > 0:
            avg_confidence = int(data['confidence_sum'] / data['count'])
            icon = {
                'location': 'location_off',
                'merchant': 'store',
                'amount': 'attach_money'
            }.get(pattern_type, 'warning')
            
            description = {
                'location': 'International transactions detected',
                'merchant': 'Suspicious merchants identified',
                'amount': 'Unusual transaction amounts'
            }.get(pattern_type, 'Unknown pattern')
            
            fraud_patterns.append({
                'icon': icon,
                'description': description,
                'confidence': avg_confidence
            })
    
    # Sort patterns by confidence
    fraud_patterns.sort(key=lambda x: x['confidence'], reverse=True)
    return fraud_patterns

@app.route('/handle_flagged_transaction/<transaction_id>', methods=['POST'])
def handle_flagged_transaction(transaction_id):
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    user_id = session.get('user_id')
    action = request.form.get('action')
    
    if action not in ['confirm', 'dismiss']:
        return jsonify({'success': False, 'error': 'Invalid action'})
    
    # Get user's transactions
    transactions = app.transactions.get(user_id, [])
    
    # Find the transaction
    target_transaction = None
    for t in transactions:
        if str(t.get('id', '')) == str(transaction_id):
            target_transaction = t
            t['fraud_status'] = action
            t['reviewed_at'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # For dismiss action, set fraud probability to 0
            if action == 'dismiss':
                t['fraud_probability'] = 0
                t['flagged'] = False
            
            # For confirm action, ensure it's marked as fraudulent
            if action == 'confirm':
                t['fraud_probability'] = max(t.get('fraud_probability', 0), 85)
                t['flagged'] = True
            break
    
    if not target_transaction:
        return jsonify({'success': True})  # Return success even if not found to avoid errors
    
    # Calculate summary statistics
    total_transactions = len(transactions)
    flagged_count = sum(1 for t in transactions if t.get('flagged', False))
    suspicious_count = sum(1 for t in transactions if 30 <= t.get('fraud_probability', 0) < 70)
    fraud_count = sum(1 for t in transactions if t.get('fraud_probability', 0) >= 70)
    
    # Calculate risk score
    total_risk = sum(t.get('fraud_probability', 0) for t in transactions)
    risk_score = int((total_risk / total_transactions) / 10) if total_transactions > 0 else 0
    
    # Calculate suspicious percentage
    suspicious_percentage = (suspicious_count + fraud_count) / total_transactions * 100 if total_transactions > 0 else 0
    
    # Prepare chart data
    chart_data = {
        'safe': total_transactions - suspicious_count - fraud_count,
        'suspicious': suspicious_count,
        'fraudulent': fraud_count
    }
    
    return jsonify({
        'success': True,
        'summary': {
            'flagged_count': flagged_count,
            'suspicious_percentage': int(suspicious_percentage),
            'risk_score': risk_score,
            'chart_data': chart_data
        }
    })

@app.route('/refresh_fraud_analysis')
def refresh_fraud_analysis():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    user_id = session.get('user_id')
    
    # Get user's transactions
    transactions = app.transactions.get(user_id, [])
    
    # Convert transactions
    formatted_transactions = []
    for t in transactions:
        formatted_transactions.append({
            'id': t.get('id', ''),
            'amount': float(t['amount']),
            'timestamp': datetime.datetime.strptime(t['date'], '%Y-%m-%d'),
            'merchant': t['merchant'],
            'location': 'International' if 'International' in t['merchant'] else 'Mumbai',
            'category': t['category'],
            'fraud_status': t.get('fraud_status', None),
            'fraud_probability': t.get('fraud_probability', 0)
        })
    
    # Calculate summary statistics
    total_transactions = len(formatted_transactions)
    flagged_count = sum(1 for t in transactions if t.get('flagged', False))
    suspicious_count = sum(1 for t in transactions if 0 < t.get('fraud_probability', 0) < 50)
    fraud_count = sum(1 for t in transactions if t.get('fraud_probability', 0) >= 50)
    
    # Calculate risk score
    total_risk = sum(t.get('fraud_probability', 0) for t in transactions)
    risk_score = int((total_risk / total_transactions) / 10) if total_transactions > 0 else 0
    
    # Calculate suspicious percentage
    suspicious_percentage = (suspicious_count + fraud_count) / total_transactions * 100 if total_transactions > 0 else 0
    
    # Prepare chart data
    chart_data = {
        'safe': total_transactions - suspicious_count - fraud_count,
        'suspicious': suspicious_count,
        'fraudulent': fraud_count
    }
    
    # Analyze each transaction
    ml_flagged_transactions = []
    for t in transactions:
        # Skip transactions that have been dismissed
        if t.get('fraud_status') != 'dismiss' and t.get('fraud_probability', 0) > 30:
            ml_flagged_transactions.append({
                'id': t.get('id', ''),
                'date': t['date'],
                'merchant': t['merchant'],
                'category': t['category'],
                'amount': t['amount'],
                'fraud_probability': t.get('fraud_probability', 0),
                'flag_reason': get_flag_reason(t),
                'fraud_status': t.get('fraud_status', None)
            })
    
    # Sort by fraud probability (highest first)
    ml_flagged_transactions.sort(key=lambda x: x['fraud_probability'], reverse=True)
    
    # Get fraud patterns
    fraud_patterns = analyze_fraud_patterns(transactions)
    
    return jsonify({
        'success': True,
        'summary': {
            'flagged_count': flagged_count,
            'suspicious_percentage': int(suspicious_percentage),
            'risk_score': risk_score,
            'chart_data': chart_data
        },
        'patterns': fraud_patterns,
        'transactions': ml_flagged_transactions
    })

# Helper function to generate transactions
def generate_transactions(user_id):
    transactions = []
    merchants = ['Paytm', 'Amazon Pay', 'Flipkart', 'Swiggy', 'Zomato', 'Uber', 'Ola', 'BigBasket', 'Myntra', 'IRCTC']
    categories = ['Shopping', 'Food', 'Transport', 'Groceries', 'Entertainment', 'Bills', 'Transfer']
    
    # Generate 50 transactions for better distribution
    total_transactions = 50
    
    # Define transaction types and their percentages
    transaction_types = {
        'safe': 0.50,        # 50% safe transactions
        'suspicious': 0.10,  # 10% slightly suspicious
        'fraud': 0.10,       # 10% fraudulent
        'normal': 0.30       # 30% normal transactions
    }
    
    # Calculate number of transactions for each type
    type_counts = {
        'safe': int(total_transactions * transaction_types['safe']),
        'suspicious': int(total_transactions * transaction_types['suspicious']),
        'fraud': int(total_transactions * transaction_types['fraud']),
        'normal': total_transactions - sum([int(total_transactions * v) for v in transaction_types.values()])
    }
    
    # Define transaction characteristics for each type
    transaction_profiles = {
        'safe': {
            'merchants': ['Paytm', 'Amazon Pay', 'Flipkart', 'Swiggy', 'Zomato', 'Uber', 'Ola', 'BigBasket', 'Myntra', 'IRCTC'],
            'categories': ['Shopping', 'Food', 'Transport', 'Groceries', 'Entertainment', 'Bills'],
            'amount_range': (100, 5000),
            'fraud_probability': 0,
            'flagged': False
        },
        'suspicious': {
            'merchants': ['Unknown Vendor', 'New Online Store', 'Late Night Shop', 'Foreign Exchange'],
            'categories': ['Shopping', 'Transfer', 'Entertainment'],
            'amount_range': (5000, 15000),
            'fraud_probability': 30,
            'flagged': True
        },
        'fraud': {
            'merchants': ['Crypto Exchange', 'Unknown Vendor', 'Foreign Exchange', 'New Online Store'],
            'categories': ['Transfer', 'Shopping', 'Entertainment'],
            'amount_range': (15000, 50000),
            'fraud_probability': 85,
            'flagged': True
        },
        'normal': {
            'merchants': ['Paytm', 'Amazon Pay', 'Flipkart', 'Swiggy', 'Zomato', 'Uber', 'Ola', 'BigBasket', 'Myntra', 'IRCTC'],
            'categories': ['Shopping', 'Food', 'Transport', 'Groceries', 'Entertainment', 'Bills'],
            'amount_range': (100, 10000),
            'fraud_probability': 5,
            'flagged': False
        }
    }
    
    # Generate transactions for each type
    for trans_type, count in type_counts.items():
        profile = transaction_profiles[trans_type]
        for _ in range(count):
            # Random date within the last 30 days
            days_ago = random.randint(0, 30)
            transaction_date = (datetime.datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            # Generate amount based on type
            min_amount, max_amount = profile['amount_range']
            amount = random.randint(min_amount, max_amount) * -1  # Negative for spending
            
            transaction = {
                'id': f"{user_id}-{len(transactions)}",
                'date': transaction_date,
                'merchant': random.choice(profile['merchants']),
                'category': random.choice(profile['categories']),
                'amount': amount,
                'fraud_probability': profile['fraud_probability'],
                'flagged': profile['flagged']
            }
            
            transactions.append(transaction)
    
    # Shuffle transactions to randomize order
    random.shuffle(transactions)
    
    # Sort by date (newest first)
    transactions.sort(key=lambda x: x['date'], reverse=True)
    
    return transactions

# Helper function to generate ML-flagged transactions
def generate_ml_flagged_transactions(user_id):
    ml_transactions = []
    merchants = ['Unknown Vendor', 'Foreign Exchange', 'Late Night Shop', 'New Online Store', 'Crypto Exchange']
    categories = ['Unknown', 'International', 'High Risk', 'Unusual Activity']
    reasons = [
        'Unusual location',
        'Amount exceeds typical spending',
        'Suspicious merchant',
        'Similar to known fraud patterns',
        'Multiple rapid transactions'
    ]
    
    # Generate 3-5 flagged transactions
    for i in range(random.randint(3, 5)):
        # Random date within the last 14 days
        days_ago = random.randint(0, 14)
        transaction_date = (datetime.datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        # Always negative amounts for suspicious transactions
        amount = random.randint(5000, 50000) * -1
        
        transaction = {
            'id': f"{user_id}-ml-{i}",
            'date': transaction_date,
            'merchant': random.choice(merchants),
            'category': random.choice(categories),
            'amount': amount,
            'fraud_probability': random.randint(60, 98),
            'flag_reason': random.choice(reasons)
        }
        
        ml_transactions.append(transaction)
    
    # Sort by fraud probability (highest first)
    ml_transactions.sort(key=lambda x: x['fraud_probability'], reverse=True)
    
    return ml_transactions

@app.route('/test-camera')
def test_camera():
    """A simple route to test camera functionality"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Camera Test</title>
        <style>
            body { background-color: #121212; color: white; font-family: Arial, sans-serif; text-align: center; }
            video { max-width: 100%; background-color: #000; }
            .container { max-width: 800px; margin: 0 auto; padding: 20px; }
            button { padding: 10px 20px; background-color: #bb86fc; border: none; color: #000; cursor: pointer; margin: 5px; }
            .controls { margin: 20px 0; }
            select { padding: 5px; margin: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Camera Test Page</h1>
            <p>If the camera is working, you should see your video feed below:</p>
            <video id="video" width="640" height="480" autoplay playsinline></video>
            <p>FPS: <span id="fps">0</span></p>
            <div class="controls">
                <button id="toggle">Toggle Camera</button>
                <select id="resolution">
                    <option value="640x480">640x480</option>
                    <option value="1280x720">1280x720</option>
                    <option value="1920x1080">1920x1080</option>
                </select>
                <select id="fps">
                    <option value="30">30 FPS</option>
                    <option value="60">60 FPS</option>
                    <option value="120">120 FPS</option>
                </select>
            </div>
        </div>
        
        <script>
            const video = document.getElementById('video');
            const fpsElement = document.getElementById('fps');
            const toggleBtn = document.getElementById('toggle');
            const resolutionSelect = document.getElementById('resolution');
            const fpsSelect = document.getElementById('fps');
            let stream = null;
            let frameCount = 0;
            let lastTime = 0;
            
            // FPS counter
            function updateFPS() {
                const now = performance.now();
                if (now - lastTime >= 1000) {
                    fpsElement.textContent = frameCount;
                    frameCount = 0;
                    lastTime = now;
                }
                frameCount++;
                requestAnimationFrame(updateFPS);
            }
            
            // Parse resolution string
            function parseResolution(resolution) {
                const [width, height] = resolution.split('x').map(Number);
                return { width, height };
            }
            
            // Start camera with high quality settings
            async function startCamera() {
                try {
                    const resolution = parseResolution(resolutionSelect.value);
                    const targetFPS = parseInt(fpsSelect.value);
                    
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: { 
                            width: { ideal: resolution.width },
                            height: { ideal: resolution.height },
                            frameRate: { ideal: targetFPS },
                            facingMode: "user",
                            aspectRatio: resolution.width / resolution.height,
                            // Additional quality settings
                            brightness: { ideal: 100 },
                            contrast: { ideal: 100 },
                            saturation: { ideal: 100 },
                            sharpness: { ideal: 100 }
                        }
                    });
                    
                    video.srcObject = stream;
                    updateFPS();
                } catch (err) {
                    console.error('Error:', err);
                    alert('Camera error: ' + err.message);
                }
            }
            
            // Stop camera
            function stopCamera() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    video.srcObject = null;
                }
            }
            
            // Toggle camera
            toggleBtn.addEventListener('click', function() {
                if (stream && stream.active) {
                    stopCamera();
                    toggleBtn.textContent = 'Start Camera';
                } else {
                    startCamera();
                    toggleBtn.textContent = 'Stop Camera';
                }
            });
            
            // Handle resolution change
            resolutionSelect.addEventListener('change', function() {
                if (stream && stream.active) {
                    stopCamera();
                    startCamera();
                }
            });
            
            // Handle FPS change
            fpsSelect.addEventListener('change', function() {
                if (stream && stream.active) {
                    stopCamera();
                    startCamera();
                }
            });
            
            // Start on load
            startCamera();
        </script>
    </body>
    </html>
    """

@app.route('/register-simple', methods=['GET', 'POST'])
def register_simple():
    if request.method == 'POST':
        # Same logic as the regular register route
        # Just use the simpler template
        # Copy the code from your register route
        user_id = request.form['user_id']
        # ... rest of your registration logic ...
    
    return render_template('register_simple.html')

# Add this right before app.run()
if __name__ == '__main__':
    print("Starting Facial Blockchain Authentication System")
    print(f"Check if uploads folder exists: {os.path.exists(UPLOAD_FOLDER)}")
    app.run(debug=True, host='0.0.0.0') 