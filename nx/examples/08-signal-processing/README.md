# `08-signal-processing`

Analyze frequencies with FFT — decompose signals and filter noise. This example
builds a signal from two sine waves plus noise, identifies the component
frequencies, and filters the noise in the frequency domain.

```bash
dune exec nx/examples/08-signal-processing/main.exe
```

## What You'll Learn

- Constructing synthetic signals from sine waves and noise
- Transforming to the frequency domain with `rfft`
- Mapping frequency bins to Hz with `rfftfreq`
- Identifying dominant frequency components by magnitude
- Filtering noise by zeroing small-magnitude frequency bins
- Reconstructing a clean signal with `irfft`

## Key Functions

| Function                      | Purpose                                           |
| ----------------------------- | ------------------------------------------------- |
| `rfft t`                      | Real-valued FFT (time domain to frequency domain) |
| `irfft ~n t`                  | Inverse real FFT (frequency domain back to time)  |
| `rfftfreq ~d n`               | Frequency bin labels for `rfft` output            |
| `linspace dtype start stop n` | Evenly spaced time samples                        |
| `sin t`                       | Element-wise sine                                 |
| `Rng.normal ~key dtype shape` | Gaussian noise                                    |
| `Nx.Infix` (`+`, `*`, `*$`)   | Clean arithmetic on arrays                        |

## Output Walkthrough

### Signal construction

A 256-sample signal at 256 Hz composed of two sine waves plus noise:

```
Signal: 256 samples at 256 Hz
Components: 5 Hz (amplitude 1.0) + 20 Hz (amplitude 0.5) + noise
```

### Frequency analysis

After `rfft`, compute magnitudes scaled by 2/N to get amplitudes:

```
Dominant frequencies:
  5.0 Hz  (magnitude 1.002)
  20.0 Hz  (magnitude 0.501)
```

The FFT correctly recovers both sine components. Noise spreads across many
bins with small magnitudes.

### Noise filtering

Zero all frequency bins below a threshold, then reconstruct with `irfft`:

```
After filtering (threshold=0.2):
  Original first 8:  [1.29, 1.16, 0.81, 0.44, ...]
  Filtered first 8:  [1.00, 1.16, 0.79, 0.36, ...]
```

The filtered signal retains the two sine waves while removing noise.

### Frequency bins

For an N-sample real signal, `rfft` produces N/2 + 1 complex bins from 0 Hz
to the Nyquist frequency (sample_rate / 2):

```
Total bins: 129 (for 256-sample signal)
```

## Try It

1. Add a third sine wave at 40 Hz and verify it appears in the dominant
   frequencies.
2. Raise the filter threshold to 0.5 and observe how the 20 Hz component
   gets removed (its amplitude is only 0.5).
3. Double the sample rate to 512 Hz and check how the frequency resolution
   changes.

## Next Steps

Continue to [09-image-processing](../09-image-processing/) to apply
convolution and pooling to 2D image data.
