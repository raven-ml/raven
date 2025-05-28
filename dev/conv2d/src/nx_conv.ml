(* Optimized convolution implementation *)
(* This module provides drop-in replacements for Nx convolution functions *)

open Nx_core.Make_frontend (Nx_native)

(* For now, just re-export the original functions *)
(* We'll gradually add optimizations here *)

let convolve2d = convolve2d
let convolve1d = convolve1d
let correlate2d = correlate2d
let correlate1d = correlate1d

(* Optimization status: - [ ] Reduce contiguous calls - [ ] Optimize pool
   function for convolution workloads - [ ] Special case for 3x3 stride=1 - [ ]
   Minimize intermediate tensor creation *)
