import pytest
import sys
import os

# Add artifacts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "artifacts"))

from onion import (
    derive_keystream,
    encrypt_layer,
    decrypt_layer,
    Node,
    Circuit,
    HiddenService,
    derive_key,
)


class TestEncryption:
    """Test encryption helper functions."""

    def test_encrypt_roundtrip(self):
        """Test that encrypt then decrypt returns original data."""
        key = b"test_key_12345"
        nonce = 42
        plaintext = b"Hello, onion routing!"

        ciphertext = encrypt_layer(plaintext, key, nonce)
        decrypted = decrypt_layer(ciphertext, key, nonce)

        assert decrypted == plaintext

    def test_encrypt_empty(self):
        """Test encryption with empty plaintext."""
        key = b"test_key_12345"
        nonce = 42
        plaintext = b""

        ciphertext = encrypt_layer(plaintext, key, nonce)
        decrypted = decrypt_layer(ciphertext, key, nonce)

        assert decrypted == plaintext

    def test_different_nonces_produce_different_ciphertexts(self):
        """Test that same plaintext with different nonces produces different ciphertexts."""
        key = b"test_key_12345"
        plaintext = b"Same message"

        ciphertext1 = encrypt_layer(plaintext, key, 1)
        ciphertext2 = encrypt_layer(plaintext, key, 2)

        assert ciphertext1 != ciphertext2

    def test_different_keys_produce_different_ciphertexts(self):
        """Test that same plaintext with different keys produces different ciphertexts."""
        nonce = 42
        plaintext = b"Same message"

        ciphertext1 = encrypt_layer(plaintext, b"key1", nonce)
        ciphertext2 = encrypt_layer(plaintext, b"key2", nonce)

        assert ciphertext1 != ciphertext2


class TestNode:
    """Test Node class."""

    def test_node_address_generation(self):
        """Test that node addresses are generated correctly."""
        node = Node("test_node")
        assert node.address.endswith(".onion")
        assert len(node.address) == 14  # 8 hex chars + ".onion"

    def test_node_key_derivation(self):
        """Test that node keys are derived deterministically."""
        node1 = Node("same_id")
        node2 = Node("same_id")

        assert node1.key == node2.key

    def test_node_custom_key(self):
        """Test that node can use custom key."""
        custom_key = b"custom_key_12345"
        node = Node("test_node", key=custom_key)

        assert node.key == custom_key

    def test_node_handle_packet(self):
        """Test packet handling."""
        node = Node("test_node")

        plaintext = b"Hello"
        nonce = 42

        # Build the full inner message: 16-byte next_hop + payload
        next_hop = b"next_hop_addr\x00\x00\x00"[:16]
        inner = next_hop + plaintext

        # Encrypt the entire thing (handle_packet decrypts then splits)
        packet = encrypt_layer(inner, node.key, nonce)

        next_hop_addr, payload = node.handle_packet(packet, nonce)

        assert next_hop_addr == "next_hop_addr"
        assert payload == plaintext


class TestCircuit:
    """Test Circuit class."""

    def test_circuit_creation(self):
        """Test circuit creation with multiple nodes."""
        nodes = [
            Node("entry_node"),
            Node("middle_node"),
            Node("exit_node"),
        ]
        circuit = Circuit(nodes)

        assert len(circuit.path) == 3
        assert circuit.nonce == 0

    def test_onion_packet_layering(self):
        """Test that onion packets have correct layering."""
        nodes = [
            Node("entry_node"),
            Node("middle_node"),
            Node("exit_node"),
        ]
        circuit = Circuit(nodes)

        payload = b"Secret message"
        packet = circuit.build_onion_packet(payload)

        # Entry node should only see next_hop + encrypted data
        entry_node = nodes[0]
        next_hop, inner_payload = entry_node.handle_packet(packet, 0)

        # Entry node should see middle_node's address as next hop
        assert next_hop == nodes[1].address

    def test_end_to_end_communication(self):
        """Test complete round-trip communication through circuit."""
        nodes = [
            Node("entry_node"),
            Node("middle_node"),
            Node("exit_node"),
        ]
        circuit = Circuit(nodes)

        payload = b"Hello, service!"
        packet = circuit.build_onion_packet(payload)

        # Peel all circuit layers (entry -> middle -> exit)
        current_packet = packet
        for node in nodes:
            next_hop, current_packet = node.handle_packet(current_packet, 0)

        # Exit node delivers the original payload
        assert current_packet == payload

        # Simulate response going back: each relay encrypts with its key
        response = b"Echo: " + payload
        response_packet = encrypt_layer(response, nodes[-1].key, 0)
        for node in reversed(nodes[:-1]):
            response_packet = encrypt_layer(response_packet, node.key, 0)

        # Client peels response layers
        final_response = circuit.process_response(response_packet)
        assert final_response == response


class TestHiddenService:
    """Test HiddenService class."""

    def test_service_address_generation(self):
        """Test that service addresses are generated correctly."""
        service = HiddenService("test_service", lambda x: x)

        assert service.address.endswith(".onion")
        assert len(service.address) == 14

    def test_service_handle_packet(self):
        """Test service packet handling."""
        def echo_handler(message):
            return b"Echo: " + message

        service = HiddenService("test_service", echo_handler)

        payload = b"Test message"
        nonce = 42

        # Encrypt payload with service key
        ciphertext = encrypt_layer(payload, service.service_key, nonce)

        # Handle packet
        response = service.handle_packet(ciphertext, nonce)

        assert response == payload


class TestUtilityFunctions:
    """Test utility functions."""

    def test_derive_key_deterministic(self):
        """Test that derive_key is deterministic."""
        key1 = derive_key("test_id")
        key2 = derive_key("test_id")

        assert key1 == key2

    def test_derive_key_different_ids(self):
        """Test that different IDs produce different keys."""
        key1 = derive_key("id1")
        key2 = derive_key("id2")

        assert key1 != key2
