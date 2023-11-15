from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import utils
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

def generate_key_pair():
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    public_key = private_key.public_key()
    return private_key, public_key

def encrypt(message, recipient_public_key):
    ciphertext = recipient_public_key.encrypt(
        message,
        ec.ECDH(),
        backend=default_backend()
    )
    return ciphertext

def decrypt(ciphertext, private_key):
    plaintext = private_key.decrypt(
        ciphertext,
        ec.ECDH(),
        backend=default_backend()
    )
    return plaintext

# Exemple d'utilisation
private_key, public_key = generate_key_pair()

# À ce stade, vous avez la paire de clés (private_key, public_key) pour le destinataire.

# Pour chiffrer un message en utilisant la clé publique du destinataire
message = "السلام عليكم"  # Message en arabe
ciphertext = encrypt(message.encode('utf-8'), public_key)

# Pour déchiffrer le message en utilisant la clé privée du destinataire
decrypted_message = decrypt(ciphertext, private_key)
print(decrypted_message.decode('utf-8'))
