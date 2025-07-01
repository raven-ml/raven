# kaun ᚲ Documentation

Kaun is our PyTorch. It's the high-level deep learning framework built on top of Rune.

## What kaun Does

Kaun gives you the building blocks for neural networks: layers, optimizers, training loops. If you've used PyTorch or Keras, you'll feel at home. Define your model, specify your loss, call train, kaun handles the rest.

The name comes from the rune ᚲ meaning "torch" or "fire." Fitting for a deep learning library.

## Current Status

Kaun is in early development. The goal for alpha is training MNIST, a simple but complete workflow that proves the concept.

What's planned:
- Essential layers (dense, conv2d, dropout)
- Common optimizers (SGD, Adam)
- Training utilities
- Model serialization

This is enough to train real models. Everything else comes after we prove it works.

## Design Philosophy

Kaun aims for PyTorch's flexibility, not Keras's high-level abstractions. You get building blocks, not black boxes. This means more code to write, but you understand exactly what's happening.

## Learn More

- [Getting Started](/docs/kaun/getting-started/) - Build your first neural network
- [MNIST Tutorial](/docs/kaun/mnist-tutorial/) - Train a CNN on handwritten digits
- Examples - (coming soon)