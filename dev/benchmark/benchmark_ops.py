#!/usr/bin/env python3
"""Benchmark key operations used in transformer models."""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def time_operation(name, func, warmup=3, iterations=100):
    """Time an operation with warmup runs."""
    # Warmup
    for _ in range(warmup):
        func()
    
    # Time
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    end = time.perf_counter()
    
    avg_time = (end - start) / iterations * 1000  # ms
    print(f"{name}: {avg_time:.3f}ms")
    return avg_time

def benchmark_matmul(batch_size=256, seq_len=16, hidden_dim=64):
    """Benchmark matrix multiplication (core of attention and FFN)."""
    print(f"\n=== Matrix Multiplication (batch={batch_size}, seq={seq_len}, hidden={hidden_dim}) ===")
    
    # Input tensors
    x = torch.randn(batch_size, seq_len, hidden_dim)
    w = torch.randn(hidden_dim, hidden_dim)
    
    def matmul():
        return torch.matmul(x, w)
    
    time_operation("matmul", matmul)

def benchmark_attention(batch_size=256, seq_len=16, hidden_dim=64, num_heads=4):
    """Benchmark multi-head attention operation."""
    print(f"\n=== Multi-Head Attention (batch={batch_size}, seq={seq_len}, hidden={hidden_dim}, heads={num_heads}) ===")
    
    head_dim = hidden_dim // num_heads
    
    # Input
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Projection weights
    wq = torch.randn(hidden_dim, hidden_dim)
    wk = torch.randn(hidden_dim, hidden_dim)
    wv = torch.randn(hidden_dim, hidden_dim)
    wo = torch.randn(hidden_dim, hidden_dim)
    
    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    
    def attention():
        # Project to Q, K, V
        q = torch.matmul(x, wq).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = torch.matmul(x, wk).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = torch.matmul(x, wv).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        scores = scores + mask
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        # Output projection
        return torch.matmul(out, wo)
    
    time_operation("attention", attention)

def benchmark_gru_cell(batch_size=256, hidden_dim=64):
    """Benchmark a single GRU cell computation."""
    print(f"\n=== GRU Cell (batch={batch_size}, hidden={hidden_dim}) ===")
    
    # Input and hidden state
    x = torch.randn(batch_size, hidden_dim)
    h = torch.randn(batch_size, hidden_dim)
    
    # GRU weights
    w_ir = torch.randn(hidden_dim, hidden_dim)
    w_hr = torch.randn(hidden_dim, hidden_dim)
    b_r = torch.randn(hidden_dim)
    
    w_iz = torch.randn(hidden_dim, hidden_dim)
    w_hz = torch.randn(hidden_dim, hidden_dim)
    b_z = torch.randn(hidden_dim)
    
    w_in = torch.randn(hidden_dim, hidden_dim)
    w_hn = torch.randn(hidden_dim, hidden_dim)
    b_n = torch.randn(hidden_dim)
    
    def gru_cell():
        # Reset gate
        r = torch.sigmoid(torch.matmul(x, w_ir) + torch.matmul(h, w_hr) + b_r)
        
        # Update gate
        z = torch.sigmoid(torch.matmul(x, w_iz) + torch.matmul(h, w_hz) + b_z)
        
        # New gate
        n = torch.tanh(torch.matmul(x, w_in) + torch.matmul(r * h, w_hn) + b_n)
        
        # Update hidden state
        h_new = (1 - z) * n + z * h
        return h_new
    
    time_operation("gru_cell", gru_cell)

def benchmark_gru_sequence(batch_size=256, seq_len=16, hidden_dim=64):
    """Benchmark GRU over a sequence."""
    print(f"\n=== GRU Sequence (batch={batch_size}, seq={seq_len}, hidden={hidden_dim}) ===")
    
    # Input sequence
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Use PyTorch's GRU
    gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
    
    def gru_sequence():
        return gru(x)
    
    time_operation("gru_sequence", gru_sequence)

def benchmark_layernorm(batch_size=256, seq_len=16, hidden_dim=64):
    """Benchmark layer normalization."""
    print(f"\n=== Layer Normalization (batch={batch_size}, seq={seq_len}, hidden={hidden_dim}) ===")
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    ln = nn.LayerNorm(hidden_dim)
    
    def layernorm():
        return ln(x)
    
    time_operation("layernorm", layernorm)

def benchmark_embedding(batch_size=256, seq_len=16, vocab_size=27, hidden_dim=64):
    """Benchmark embedding lookup."""
    print(f"\n=== Embedding (batch={batch_size}, seq={seq_len}, vocab={vocab_size}, hidden={hidden_dim}) ===")
    
    # Input indices
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Embedding layer
    emb = nn.Embedding(vocab_size, hidden_dim)
    
    def embedding():
        return emb(x)
    
    time_operation("embedding", embedding)

def main():
    print("=" * 60)
    print("PyTorch Benchmark - Transformer Operations")
    print("=" * 60)
    
    # Common dimensions from makemore
    batch_size = 256
    seq_len = 16
    hidden_dim = 64
    vocab_size = 27
    
    benchmark_embedding(batch_size, seq_len, vocab_size, hidden_dim)
    benchmark_matmul(batch_size, seq_len, hidden_dim)
    benchmark_layernorm(batch_size, seq_len, hidden_dim)
    benchmark_attention(batch_size, seq_len, hidden_dim, num_heads=4)
    benchmark_gru_cell(batch_size, hidden_dim)
    benchmark_gru_sequence(batch_size, seq_len, hidden_dim)

if __name__ == "__main__":
    main()