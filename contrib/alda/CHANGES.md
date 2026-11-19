# Changelog

All notable changes to Alda are documented in this file.

## Unreleased

- New library `alda`, discrete-time signals over nx: windows, frame grids
  and their streams, `stft` and its least-squares inverse `istft`, `psd`
  and `power_spectrum`, mel filterbanks, FIR and IIR filters (`fir`, `sos`,
  `zero_phase`, `Biquad`) with Butterworth, Chebyshev, elliptic and Bessel
  designs (`Iir`), least-squares, equiripple and Savitzky-Golay taps
  (`Taps`), `fft_convolve` and band-limited `resample`. Each is a
  composition of nx operations that rune differentiates, batches and
  compiles, and one with memory across samples also runs over a signal given
  in chunks.
