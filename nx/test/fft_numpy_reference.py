#!/usr/bin/env python3
"""
NumPy reference implementation for FFT tests to verify expected results.
"""

import numpy as np

def print_array(name, arr):
    """Print array in OCaml format"""
    if arr.dtype == np.complex64 or arr.dtype == np.complex128:
        print(f"{name} = [|")
        for val in arr.flat:
            print(f"  Complex.{{ re = {val.real}; im = {val.imag} }};")
        print("|]")
    else:
        print(f"{name} = [| {'; '.join(str(x) for x in arr.flat)} |]")
    print()

def test_fft_axes():
    """Test FFT with specific axes on 2D array"""
    print("=== Test FFT Axes ===")
    shape = (4, 6)
    size = 4 * 6
    # Create the same input as OCaml test
    input_data = np.array([i + 1j * ((i % 7) * 0.1) for i in range(size)], dtype=np.complex64)
    input_arr = input_data.reshape(shape)
    
    print(f"Input shape: {shape}")
    print(f"Input data (first few): {input_data[:4]}")
    
    # Test axis 0
    fft_axis0 = np.fft.fft(input_arr, axis=0)
    print(f"\nFFT axis=0 shape: {fft_axis0.shape}")
    # Verify round-trip
    ifft_axis0 = np.fft.ifft(fft_axis0, axis=0)
    print(f"Round-trip error axis=0: {np.max(np.abs(ifft_axis0 - input_arr))}")
    
    # Test axis 1
    fft_axis1 = np.fft.fft(input_arr, axis=1)
    print(f"\nFFT axis=1 shape: {fft_axis1.shape}")
    ifft_axis1 = np.fft.ifft(fft_axis1, axis=1)
    print(f"Round-trip error axis=1: {np.max(np.abs(ifft_axis1 - input_arr))}")
    
    # Test negative axis
    fft_neg2 = np.fft.fft(input_arr, axis=-2)
    print(f"\nFFT axis=-2 shape: {fft_neg2.shape}")
    print(f"axis=-2 equivalent to axis=0: {np.allclose(fft_neg2, fft_axis0)}")

def test_hfft_ihfft():
    """Test Hermitian FFT"""
    print("\n=== Test HFFT/IHFFT ===")
    n = 8
    # Create real signal
    signal = np.array([np.sin(2*np.pi*i/n) for i in range(n)])
    print(f"Input signal shape: {signal.shape}")
    print(f"Input signal: {signal}")
    
    # ihfft expects real input and outputs complex with hermitian symmetry
    # For a real input of length n, ihfft outputs (n//2 + 1) complex values
    ihfft_out = np.fft.ihfft(signal, n=n)
    print(f"\nihfft output shape: {ihfft_out.shape}")
    print(f"ihfft output: {ihfft_out}")
    
    # hfft expects complex hermitian input and outputs real
    hfft_out = np.fft.hfft(ihfft_out, n=n)
    print(f"\nhfft output shape: {hfft_out.shape}")
    print(f"hfft output: {hfft_out}")
    print(f"Round-trip error: {np.max(np.abs(hfft_out - signal))}")

def test_fftfreq():
    """Test FFT frequency generation"""
    print("\n=== Test FFT Frequencies ===")
    
    # Odd length
    n = 5
    freq = np.fft.fftfreq(n)
    print(f"fftfreq(n={n}): {freq}")
    print_array(f"fftfreq_{n}", freq)
    
    # Even length with custom spacing
    n = 4
    d = 0.5
    freq = np.fft.fftfreq(n, d=d)
    print(f"fftfreq(n={n}, d={d}): {freq}")
    print_array(f"fftfreq_{n}_d", freq)

def test_rfftfreq():
    """Test real FFT frequency generation"""
    print("\n=== Test RFFT Frequencies ===")
    
    # Even length
    n = 8
    freq = np.fft.rfftfreq(n)
    print(f"rfftfreq(n={n}): {freq}")
    print_array(f"rfftfreq_{n}", freq)
    
    # Odd length with custom spacing
    n = 9
    d = 2.0
    freq = np.fft.rfftfreq(n, d=d)
    print(f"rfftfreq(n={n}, d={d}): {freq}")
    print_array(f"rfftfreq_{n}_d", freq)

def test_fftshift():
    """Test FFT shift operations"""
    print("\n=== Test FFT Shift ===")
    
    # 1D case
    x = np.array([0.0, 1.0, 2.0, 3.0])
    shifted = np.fft.fftshift(x)
    print(f"Original: {x}")
    print(f"fftshift: {shifted}")
    print_array("fftshift_1d", shifted)
    
    # 2D case
    x2d = np.arange(9, dtype=float).reshape(3, 3)
    print(f"\n2D input:\n{x2d}")
    shifted2d = np.fft.fftshift(x2d, axes=(0, 1))
    print(f"fftshift 2D:\n{shifted2d}")
    print_array("fftshift_2d", shifted2d)
    
    # Test ifftshift
    unshifted = np.fft.ifftshift(shifted)
    print(f"\nifftshift of 1D: {unshifted}")
    print(f"Matches original: {np.array_equal(unshifted, x)}")

def test_rfft_sizes():
    """Test RFFT output sizes"""
    print("\n=== Test RFFT Sizes ===")
    
    # Test with different input sizes
    for n in [8, 7, 4, 6]:
        signal = np.arange(n, dtype=float)
        rfft_out = np.fft.rfft(signal)
        print(f"n={n}: rfft output shape = {rfft_out.shape[0]} (expected {n//2 + 1})")
        
        # Test round-trip
        irfft_out = np.fft.irfft(rfft_out, n=n)
        print(f"  Round-trip error: {np.max(np.abs(irfft_out - signal))}")

def test_fft_edge_cases():
    """Test edge cases for FFT"""
    print("\n=== Test FFT Edge Cases ===")
    
    # Empty array - NumPy doesn't support FFT of empty arrays
    print("Note: NumPy raises ValueError for FFT of empty array")
    
    # Size 1
    single = np.array([5.0 - 3.0j])
    fft_single = np.fft.fft(single)
    print(f"FFT of single element: {fft_single}")
    print(f"Should equal input: {np.array_equal(fft_single, single)}")

def test_rfft_edge_cases():
    """Test edge cases for RFFT"""
    print("\n=== Test RFFT Edge Cases ===")
    
    # Empty array - NumPy doesn't support RFFT of empty arrays
    print("Note: NumPy raises ValueError for RFFT of empty array")
    
    # Size 1
    single = np.array([5.0])
    rfft_single = np.fft.rfft(single)
    print(f"RFFT of single element: {rfft_single}")
    print(f"Shape: {rfft_single.shape}")

def test_2d_rfft():
    """Test 2D RFFT behavior"""
    print("\n=== Test 2D RFFT ===")
    
    m, n = 4, 6
    signal = np.arange(m * n, dtype=float).reshape(m, n)
    
    # rfft2 applies rfft to last axis
    rfft2_out = np.fft.rfft2(signal)
    print(f"Input shape: {signal.shape}")
    print(f"rfft2 output shape: {rfft2_out.shape} (expected ({m}, {n//2 + 1}))")
    
    # Test round-trip
    irfft2_out = np.fft.irfft2(rfft2_out, s=(m, n))
    print(f"Round-trip error: {np.max(np.abs(irfft2_out - signal))}")

def test_rfftn_axes():
    """Test RFFTN with specific axes"""
    print("\n=== Test RFFTN Axes ===")
    
    shape = (4, 6, 8)
    signal = np.arange(np.prod(shape), dtype=float).reshape(shape)
    
    # Single axis
    rfft_axis1 = np.fft.rfftn(signal, axes=[1])
    print(f"rfftn axes=[1] shape: {rfft_axis1.shape} (expected (4, 4, 8))")
    
    # Multiple axes - last specified axis is halved
    rfft_axes_01 = np.fft.rfftn(signal, axes=[0, 1])
    print(f"rfftn axes=[0,1] shape: {rfft_axes_01.shape} (expected (4, 4, 8))")
    
    # Negative axis
    rfft_neg1 = np.fft.rfftn(signal, axes=[-1])
    print(f"rfftn axes=[-1] shape: {rfft_neg1.shape} (expected (4, 6, 5))")
    
    # Last axis transform for ND
    shape_nd = (2, 3, 8)
    signal_nd = np.arange(np.prod(shape_nd), dtype=float).reshape(shape_nd)
    rfft_nd = np.fft.rfftn(signal_nd, axes=[2])
    print(f"rfftn last axis shape: {rfft_nd.shape} (expected (2, 3, 5))")

if __name__ == "__main__":
    test_fft_axes()
    test_hfft_ihfft()
    test_fftfreq()
    test_rfftfreq()
    test_fftshift()
    test_rfft_sizes()
    test_fft_edge_cases()
    test_rfft_edge_cases()
    test_2d_rfft()
    test_rfftn_axes()