from web3 import Web3
import hashlib
import json
import time

class BlockchainAuth:
    def __init__(self, blockchain_url="http://localhost:8545"):
        """Initialize connection to blockchain network"""
        self.web3 = Web3(Web3.HTTPProvider(blockchain_url))
        
        # Check connection
        if not self.web3.is_connected():
            print("Warning: Could not connect to blockchain network")
        
        # In a real application, you would deploy a smart contract for identity management
        # For simplicity, we'll mock some of the functionality
        self.user_identities = {}
    
    def hash_identity_data(self, user_data):
        """Create a hash of user identity data"""
        data_string = json.dumps(user_data, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def register_identity(self, user_id, face_data, document_data, user_data=None):
        """Register user's identity on the blockchain"""
        # Check if face_data is an embedding or a dict (fallback mode)
        is_fallback_mode = isinstance(face_data, dict) and 'is_simple_mode' in face_data
        
        # Create an identity record
        identity_data = {
            "user_id": user_id,
            "timestamp": int(time.time())
        }
        
        if is_fallback_mode:
            # In fallback mode, we don't have embeddings, just store a flag
            identity_data["using_simple_mode"] = True
            # Store a simple hash of some face data attributes
            identity_data["face_hash"] = hashlib.sha256(str(id(face_data)).encode()).hexdigest()
        else:
            # Normal mode with face embeddings
            identity_data["face_hash"] = hashlib.sha256(str(face_data).encode()).hexdigest()
        
        # Add document hash
        identity_data["document_hash"] = hashlib.sha256(str(document_data).encode()).hexdigest()
        
        # Add user personal data if provided
        if user_data:
            # We store only non-sensitive data on blockchain
            identity_data["personal_data_hash"] = hashlib.sha256(json.dumps(user_data, sort_keys=True).encode()).hexdigest()
            identity_data["username"] = user_data.get('username', '')
            identity_data["registration_date"] = user_data.get('registration_date', '')
        
        # In a real application, you would:
        # 1. Create a transaction to your identity management smart contract
        # 2. Wait for confirmation
        # 3. Return transaction details
        
        # For now, we'll simulate blockchain storage
        identity_hash = self.hash_identity_data(identity_data)
        self.user_identities[user_id] = {
            "identity_hash": identity_hash,
            "data": identity_data
        }
        
        return {
            "status": "success",
            "identity_hash": identity_hash,
            "timestamp": identity_data["timestamp"]
        }
    
    def verify_identity(self, user_id, face_data):
        """Verify user's identity against blockchain record"""
        if user_id not in self.user_identities:
            return False, "User identity not found on blockchain"
        
        # Check if using fallback mode
        is_fallback_mode = isinstance(face_data, dict) and 'is_simple_mode' in face_data
        is_stored_fallback = self.user_identities[user_id]["data"].get("using_simple_mode", False)
        
        # If modes don't match, that's a problem
        if is_fallback_mode != is_stored_fallback:
            if is_fallback_mode:
                return False, "Cannot verify: registered with embeddings but verifying with simple mode"
            else:
                return False, "Cannot verify: registered with simple mode but verifying with embeddings"
        
        # Check face hash
        if is_fallback_mode:
            # In fallback mode, we can't do proper verification, so always return success
            # In a real system, you would need additional security measures
            return True, "Identity verified in simple mode (reduced security)"
        else:
            # Normal verification with embeddings
            face_hash = hashlib.sha256(str(face_data).encode()).hexdigest()
            stored_face_hash = self.user_identities[user_id]["data"]["face_hash"]
            
            if face_hash != stored_face_hash:
                return False, "Face verification failed"
        
        return True, "Identity verified on blockchain" 