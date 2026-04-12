import json
import struct
import socket
import hashlib
import hmac
from typing import List, Optional, Callable, Tuple


# === chunk: target="onion.py", operation=insert, after="import hashlib" ===
# Encryption Helper (Simulated AES)
def derive_keystream(key: bytes, nonce: int, length: int) -> bytes:
    """Generate a deterministic keystream using HMAC-SHA256."""
    keystream = b""
    counter = 0
    while len(keystream) < length:
        data = hmac.new(key, nonce.to_bytes(4, 'big') + counter.to_bytes(4, 'big'), 'sha256').digest()
        keystream += data
        counter += 1
    return keystream[:length]


def encrypt_layer(plaintext: bytes, key: bytes, nonce: int) -> bytes:
    """Encrypt a plaintext layer using XOR with derived keystream."""
    keystream = derive_keystream(key, nonce, len(plaintext))
    return bytes(p ^ k for p, k in zip(plaintext, keystream))


def decrypt_layer(ciphertext: bytes, key: bytes, nonce: int) -> bytes:
    """Decrypt a ciphertext layer using XOR with derived keystream (symmetric)."""
    return encrypt_layer(ciphertext, key, nonce)

# === Network Functions ===
def send_packet_over_udp(payload: bytes, server_address: Tuple[str, int]) -> bytes:
    """Send an onion packet over UDP and receive a response."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(5.0)
        sock.sendto(payload, server_address)
        response, _ = sock.recvfrom(65535)
        return response


def create_udp_socket(bind_address: Tuple[str, int] = ("0.0.0.0", 0)):
    """Create and bind a UDP socket for receiving onion packets."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(bind_address)
    sock.settimeout(10.0)
    return sock


def receive_packet_over_udp(sock, buffer_size: int = 65535) -> Tuple[bytes, Tuple[str, int]]:
    """Receive an onion packet over UDP."""
    packet, addr = sock.recvfrom(buffer_size)
    return packet, addr


def encrypt_and_send(payload: bytes, nodes: List["Node"], server_address: Tuple[str, int]) -> bytes:
    """Build an onion packet and send it over UDP."""
    if not nodes:
        raise ValueError("Circuit must have at least one node")
    temp_circuit = Circuit(nodes)
    onion_packet = temp_circuit.build_onion_packet(payload)
    return send_packet_over_udp(onion_packet, server_address)


def start_hidden_service(host: str = "0.0.0.0", port: int = 8443) -> socket.socket:
    """Start a hidden service listener on a UDP socket."""
    sock = create_udp_socket((host, port))
    print(f"Hidden service listening on {host}:{port}")
    return sock


def process_incoming_packet(packet: bytes, service: "HiddenService", nonce: int = 0) -> bytes:
    """Process an incoming onion packet as a hidden service."""
    return service.handle_packet(packet, nonce)

# Node class
class Node:
    def __init__(self, node_id: str, key: Optional[bytes] = None):
        """Create a node with deterministic key derivation if not provided."""
        self.address = self._generate_address(node_id)
        self.key = key if key is not None else self._derive_key(node_id)
        self.next_hop = None

    def _generate_address(self, node_id: str) -> str:
        """Generate .onion-style address from node_id."""
        return hashlib.sha256(node_id.encode()).hexdigest()[:8] + ".onion"

    def _derive_key(self, node_id: str, secret: bytes = b"global_secret") -> bytes:
        """Derive symmetric key from node_id using HMAC-SHA256."""
        return hmac.new(secret, node_id.encode(), "sha256").digest()

    def handle_packet(self, packet: bytes, nonce: int) -> Tuple[str, bytes]:
        """Process an onion packet: decrypt outer layer, extract next hop and payload."""
        decrypted = decrypt_layer(packet, self.key, nonce)
        next_hop_address = decrypted[:16].rstrip(b'\x00').decode('utf-8', errors='ignore')
        next_layer_payload = decrypted[16:]
        return next_hop_address, next_layer_payload

# Circuit class
class Circuit:
    def __init__(self, path: List[Node]):
        """Initialize circuit with a path of nodes."""
        self.path = path
        self.keys = [node.key for node in path]
        self.nonce = 0

    def build_onion_packet(self, payload: bytes) -> bytes:
        """Build a layered onion packet starting from the exit node."""
        layer = payload
        for node in reversed(self.path):
            next_hop_address = b'\x00' * 16 if node == self.path[-1] else self._get_next_node_address(node)
            layer = encrypt_layer(next_hop_address + layer, node.key, self.nonce)
        self.nonce += 1
        return layer

    def _get_next_node_address(self, current_node: Node) -> bytes:
        """Get the address of the next node in the path (as bytes, 16-char padded)."""
        if current_node not in self.path:
            return b'\x00' * 16
        current_idx = self.path.index(current_node)
        if current_idx + 1 < len(self.path):
            next_node = self.path[current_idx + 1]
            return next_node.address[:16].encode().ljust(16, b'\x00')
        else:
            return b'\x00' * 16

    def process_response(self, packet: bytes) -> bytes:
        """Process a response packet, stripping layers from entry to exit."""
        for node in self.path:
            packet = decrypt_layer(packet, node.key, self.nonce - 1)
        return packet

# Hidden Service class
class HiddenService:
    def __init__(self, service_id: str, response_handler: Callable[[bytes], bytes]):
        """Initialize a hidden service with deterministic key derivation."""
        self.address = self._generate_address(service_id)
        self.service_key = self._derive_key(service_id)
        self.response_handler = response_handler

    def _generate_address(self, service_id: str) -> str:
        """Generate .onion-style address from service_id."""
        return hashlib.sha256(service_id.encode()).hexdigest()[:8] + ".onion"

    def _derive_key(self, service_id: str, secret: bytes = b"global_secret") -> bytes:
        """Derive symmetric key from service_id using HMAC-SHA256."""
        return hmac.new(secret, service_id.encode(), "sha256").digest()

    def handle_packet(self, packet: bytes, nonce: int) -> bytes:
        """Process an onion packet as a hidden service (final destination)."""
        decrypted = decrypt_layer(packet, self.service_key, nonce)
        return decrypted

# Utility functions
def derive_key(node_id: str, secret: bytes = b"global_secret") -> bytes:
    """Convenience function to derive a key from a node/service ID."""
    return hmac.new(secret, node_id.encode(), "sha256").digest()