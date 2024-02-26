import hashlib

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from django.db import models


class Email(models.Model):
    sender = models.ForeignKey('User', on_delete=models.CASCADE, related_name='sender')
    receiver = models.ForeignKey('User', on_delete=models.CASCADE, related_name='receiver')
    subject = models.CharField(max_length=100)
    content = models.TextField()
    date = models.DateTimeField(auto_now_add=True)
    read = models.DateTimeField(null=True, default=None)
    state = models.IntegerField(default=0)
    # 0 = non classé, 10 = normal, 15 = normal confirmé, 20 = spam, 25 = spam confirmé, 30 = deleted
    parent = models.ForeignKey('self', on_delete=models.CASCADE)  # default=self

    def __str__(self):
        return f'{self.subject} {self.sender.email}->{self.receiver.email}'

    def to_json(self):
        return {
            'id': self.id,
            'sender_id': self.sender.id,
            'sender_email': self.sender.email,
            'sender_name': self.sender.username,
            'sender_photo': self.sender.photo.url,
            'receiver_email': self.receiver.email,
            'receiver_name': self.receiver.username,
            'receiver_photo': self.receiver.photo.url,
            'subject': self.subject,
            'content': self.get_content(),
            'date': self.date.strftime('%Y-%m-%d %H:%M:%S'),
            'read': self.read.strftime('%Y-%m-%d %H:%M:%S') if self.read else None,
            'parent_id': self.parent.id,
        }

    def get_first(self):
        return Email.objects.filter(id=self.parent.id).first()

    @staticmethod
    def pad_pkcs7(data):
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding

    @staticmethod
    def unpad_pkcs7(padded_data):
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]

    def set_content(self, email):
        key = "2024"
        # Convert the key to bytes and pad it to 16 bytes
        key = hashlib.sha256(key.encode()).digest()[:16]

        # Generate a random initialization vector of 16 bytes
        iv = get_random_bytes(16)

        # Create the AES cipher object with CBC mode
        cipher = AES.new(key, AES.MODE_CBC, iv)

        # Pad the plaintext to a multiple of the block size
        padded_plaintext = Email.pad_pkcs7(email.encode())

        # Perform encryption by prepending the initialization vector to the encrypted message
        ciphertext = iv + cipher.encrypt(padded_plaintext)

        self.content = ciphertext.hex()

    def get_content(self):
        key = "2024"
        # Convert the key to bytes and pad it to 16 bytes
        key = hashlib.sha256(key.encode()).digest()[:16]

        # Convert the ciphertext to bytes
        ciphertext = bytes.fromhex(self.content)

        # Extract the initialization vector from the beginning of the ciphertext
        iv = ciphertext[:16]

        # Create the AES cipher object with CBC mode
        cipher = AES.new(key, AES.MODE_CBC, iv)

        # Perform decryption
        padded_plaintext = cipher.decrypt(ciphertext[16:])

        # Remove the PKCS#7 padding
        plaintext = Email.unpad_pkcs7(padded_plaintext)

        # Return the decrypted text (decoded bytes to str)
        return plaintext.decode()
