(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

(* ---------------------------------------------------------------------------
   BLIS-style GEMM implementation
   ---------------------------------------------------------------------------

   We use a BLIS-style blocked GEMM with three levels of tiling (jc, pc, ic)
   and explicit packing of A and B panels into contiguous buffers (pack_a,
   pack_b) so that the microkernel streams over cache-friendly memory.

   Microkernel design (ARM64 NEON, 128-bit vectors):
   - f64: MR=4, NR=4 → 8 Float64x2 accumulators (4×2 tile = 4×4 scalars)
   - f32: MR=6, NR=8 → 12 Float32x4 accumulators (6×2 tile = 6×8 scalars)

   Blocking parameters (tuned for Apple Silicon L1/L2):
   - f64: KC=128, MC=384, NC=256
   - f32: KC=256, MC=240, NC=640

   The microkernel is a recursive kloop (f64_kloop / f32_kloop) defined at
   module level, with all SIMD accumulators passed as function arguments so
   they stay in registers across the entire k-iteration. kloop must be at
   module level — not nested inside kernel_zero/kernel_accum — to avoid
   per-call closure allocations.

   Threading: the ic-loop is parallelized via Parallel.parallel_for. Each
   domain gets its own ap/bp scratch buffers allocated inside the closure.

   Known limitations
   ~~~~~~~~~~~~~~~~~

   - No FMA: mul_add compiles to fmul + fadd. NEON fmla exists in simd_neon
     but is not a [@@builtin] external, so it hits the same cross-module
     inlining issue (see SIMD wrappers comment below). Needs upstream OxCaml.

   - 128-bit SIMD only: ARM64 NEON is limited to 128-bit vectors.
     AVX2/AVX-512 on x86 would be a large win.

   - Pack B is redundantly done per domain. Restructuring to pack once per
     (jc, pc) block regressed performance — the extra parallel_execute calls
     and effect-handler overhead outweigh the redundant packing.

   Remaining gap vs. C backend (~8–31× on Apple Silicon) is dominated by
   Apple Accelerate's AMX coprocessor, plus the above.
   --------------------------------------------------------------------------- *)

(* ---------------------------- Helpers ------------------------------------ *)

let[@inline] min_int a b = if a < b then a else b
let[@inline] round_up x m = ((x + m - 1) / m) * m

(* Local wrappers that call [@@builtin] externals directly.
   Wrappers defined in other modules (e.g. mul_add, set1 in Simd) are not
   inlined into this compilation unit — even when both modules are in the
   same library. One hypothesis is dune's -opaque flag preventing flambda2
   from exporting function bodies, but moving Simd into the same library
   did not help, so the root cause may lie elsewhere (flambda2 inlining
   heuristics, or how [@@builtin] externals bypass the optimizer while
   regular wrappers do not). Defining them here works around the issue.

   TODO: mul_add uses separate mul+add instead of a true FMA instruction.
   OxCaml has NEON FMA via simd_neon, but the emulated fma is not a
   [@@builtin] external and suffers from the same inlining issue described
   above. Upstreaming NEON fmla/fmls as [@@builtin] in OxCaml would let
   us replace these with single-instruction FMA. *)
let[@inline always] f64_mul_add a b c =
  Float64x2.add (Float64x2.mul a b) c

let[@inline always] f64_set1 a =
  Float64x2.of_int64x2 (Int64x2.dup (Int64x2.of_float64x2 (Float64x2.low_of a)))

let[@inline always] f32_mul_add a b c =
  Float32x4.add (Float32x4.mul a b) c

let[@inline always] f32_set1 a =
  Float32x4.of_int32x4 (Int32x4.dup (Int32x4.of_float32x4 (Float32x4.low_of a)))

module Gemm_f64 = struct
  let mr = 4
  let nr = 4
  let kc_blk = 128
  let mc_blk = 384
  let nc_blk = 256

  let pack_a a ~a_off ~lda ~ic ~pc ~mc ~kc ap =
    let dst = ref 0 in
    let i = ref 0 in
    while !i + mr <= mc do
      for p = 0 to kc - 1 do
        let src_base = a_off + (ic + !i) * lda + pc + p in
        for ii = 0 to mr - 1 do
          Array.unsafe_set ap (!dst + ii)
            (Array.unsafe_get a (src_base + ii * lda))
        done;
        dst := !dst + mr
      done;
      i := !i + mr
    done;
    if !i < mc then begin
      let mr_rem = mc - !i in
      for p = 0 to kc - 1 do
        let src_base = a_off + (ic + !i) * lda + pc + p in
        for ii = 0 to mr_rem - 1 do
          Array.unsafe_set ap (!dst + ii)
            (Array.unsafe_get a (src_base + ii * lda))
        done;
        for ii = mr_rem to mr - 1 do
          Array.unsafe_set ap (!dst + ii) #0.
        done;
        dst := !dst + mr
      done
    end

  let pack_b b ~b_off ~ldb ~pc ~jc ~kc ~nc bp =
    let dst = ref 0 in
    let j = ref 0 in
    while !j + nr <= nc do
      for p = 0 to kc - 1 do
        let src = b_off + (pc + p) * ldb + jc + !j in
        for jj = 0 to nr - 1 do
          Array.unsafe_set bp (!dst + jj) (Array.unsafe_get b (src + jj))
        done;
        dst := !dst + nr
      done;
      j := !j + nr
    done;
    if !j < nc then begin
      let nr_rem = nc - !j in
      for p = 0 to kc - 1 do
        let src = b_off + (pc + p) * ldb + jc + !j in
        for jj = 0 to nr_rem - 1 do
          Array.unsafe_set bp (!dst + jj) (Array.unsafe_get b (src + jj))
        done;
        for jj = nr_rem to nr - 1 do
          Array.unsafe_set bp (!dst + jj) #0.
        done;
        dst := !dst + nr
      done
    end

  let rec f64_kloop ap ap_off bp bp_off c_buf c_off ldc kc p
      c00 c01 c10 c11 c20 c21 c30 c31 =
    if p = kc then begin
      Float64x2.Array.unsafe_set c_buf ~idx:c_off c00;
      Float64x2.Array.unsafe_set c_buf ~idx:(c_off + 2) c01;
      let r1 = c_off + ldc in
      Float64x2.Array.unsafe_set c_buf ~idx:r1 c10;
      Float64x2.Array.unsafe_set c_buf ~idx:(r1 + 2) c11;
      let r2 = c_off + 2 * ldc in
      Float64x2.Array.unsafe_set c_buf ~idx:r2 c20;
      Float64x2.Array.unsafe_set c_buf ~idx:(r2 + 2) c21;
      let r3 = c_off + 3 * ldc in
      Float64x2.Array.unsafe_set c_buf ~idx:r3 c30;
      Float64x2.Array.unsafe_set c_buf ~idx:(r3 + 2) c31
    end
    else
      let ab = ap_off + p * 4 in
      let bb = bp_off + p * 4 in
      let a0 = f64_set1 (Array.unsafe_get ap ab) in
      let a1 = f64_set1 (Array.unsafe_get ap (ab + 1)) in
      let a2 = f64_set1 (Array.unsafe_get ap (ab + 2)) in
      let a3 = f64_set1 (Array.unsafe_get ap (ab + 3)) in
      let b0 = Float64x2.Array.unsafe_get bp ~idx:bb in
      let b1 = Float64x2.Array.unsafe_get bp ~idx:(bb + 2) in
      f64_kloop ap ap_off bp bp_off c_buf c_off ldc kc (p + 1)
        (f64_mul_add a0 b0 c00) (f64_mul_add a0 b1 c01)
        (f64_mul_add a1 b0 c10) (f64_mul_add a1 b1 c11)
        (f64_mul_add a2 b0 c20) (f64_mul_add a2 b1 c21)
        (f64_mul_add a3 b0 c30) (f64_mul_add a3 b1 c31)

  let kernel_zero ap ~ap_off bp ~bp_off c_buf ~c_off ~ldc ~kc =
    let z = f64_set1 #0. in
    f64_kloop ap ap_off bp bp_off c_buf c_off ldc kc 0
      z z z z z z z z

  let kernel_accum ap ~ap_off bp ~bp_off c_buf ~c_off ~ldc ~kc =
    let r1 = c_off + ldc in
    let r2 = c_off + 2 * ldc in
    let r3 = c_off + 3 * ldc in
    f64_kloop ap ap_off bp bp_off c_buf c_off ldc kc 0
      (Float64x2.Array.unsafe_get c_buf ~idx:c_off)
      (Float64x2.Array.unsafe_get c_buf ~idx:(c_off + 2))
      (Float64x2.Array.unsafe_get c_buf ~idx:r1)
      (Float64x2.Array.unsafe_get c_buf ~idx:(r1 + 2))
      (Float64x2.Array.unsafe_get c_buf ~idx:r2)
      (Float64x2.Array.unsafe_get c_buf ~idx:(r2 + 2))
      (Float64x2.Array.unsafe_get c_buf ~idx:r3)
      (Float64x2.Array.unsafe_get c_buf ~idx:(r3 + 2))

  let edge_scalar ap ~ap_off bp ~bp_off c_buf ~c_off ~ldc ~mr_eff ~nr_eff
      ~kc ~first =
    for i = 0 to mr_eff - 1 do
      for j = 0 to nr_eff - 1 do
        let c_idx = c_off + i * ldc + j in
        let init = if first then #0. else Array.unsafe_get c_buf c_idx in
        let rec loop p acc =
          if p = kc then acc
          else
            let a = Array.unsafe_get ap (ap_off + p * 4 + i) in
            let b = Array.unsafe_get bp (bp_off + p * 4 + j) in
            loop (p + 1) (Float_u.fma a b acc)
        in
        Array.unsafe_set c_buf c_idx (loop 0 init)
      done
    done

  let macro_kernel ap bp c_buf ~c_off ~ldc ~mc ~nc ~kc ~first =
    let ir = ref 0 in
    while !ir < mc do
      let mr_eff = min_int mr (mc - !ir) in
      let ap_off = (!ir / mr) * mr * kc in
      let jr = ref 0 in
      while !jr < nc do
        let nr_eff = min_int nr (nc - !jr) in
        let bp_off = (!jr / nr) * nr * kc in
        let c_tile = c_off + (!ir * ldc) + !jr in
        if mr_eff = mr && nr_eff = nr then begin
          if first then
            kernel_zero ap ~ap_off bp ~bp_off c_buf
              ~c_off:c_tile ~ldc ~kc
          else
            kernel_accum ap ~ap_off bp ~bp_off c_buf
              ~c_off:c_tile ~ldc ~kc
        end
        else
          edge_scalar ap ~ap_off bp ~bp_off c_buf
            ~c_off:c_tile ~ldc ~mr_eff ~nr_eff ~kc ~first;
        jr := !jr + nr
      done;
      ir := !ir + mr
    done

  let gemm ~pool a_buf b_buf c_buf ~m ~n ~k ~a_off ~b_off ~c_off ~ldc () =
    let lda = k and ldb = n in
    let mc = mc_blk and nc = nc_blk and kc = kc_blk in
    let rec jc_loop jc =
      if jc >= n then ()
      else
        let nc' = min_int nc (n - jc) in
        Parallel.parallel_for pool 0 (m - 1) (fun start_row end_row ->
            let bp = Array.make_float64 (round_up nc' nr * kc) in
            let ap = Array.make_float64 (round_up mc mr * kc) in
            let rec pc_loop pc =
              if pc >= k then ()
              else
                let kc' = min_int kc (k - pc) in
                let first = pc = 0 in
                pack_b b_buf ~b_off ~ldb ~pc ~jc ~kc:kc' ~nc:nc' bp;
                let rec ic_loop ic =
                  if ic >= end_row then ()
                  else
                    let mc' = min_int mc (end_row - ic) in
                    pack_a a_buf ~a_off ~lda ~ic ~pc ~mc:mc' ~kc:kc' ap;
                    macro_kernel ap bp c_buf
                      ~c_off:(c_off + ic * ldc + jc)
                      ~ldc ~mc:mc' ~nc:nc' ~kc:kc' ~first;
                    ic_loop (ic + mc')
                in
                ic_loop start_row;
                pc_loop (pc + kc')
            in
            pc_loop 0);
        jc_loop (jc + nc')
    in
    jc_loop 0
end

let matmul_float64_slow a_buf b_buf c_buf va vb vout start_idx end_idx =
  let nd_a = Array.length (shape va)
  and nd_b = Array.length (shape vb)
  and nd_out = Array.length (shape vout) in

  let rank = nd_out in
  let m = (shape vout).(rank - 2) in
  let n = (shape vout).(rank - 1) in
  let k = (shape va).(rank - 1) in

  let a_idx0 = Array.make nd_a 0
  and a_idx1 = Array.make nd_a 0
  and b_idx = Array.make nd_b 0
  and out_idx0 = Array.make nd_out 0
  and out_idx1 = Array.make nd_out 0 in

  let a_str = View.strides va
  and b_str = View.strides vb
  and out_str = View.strides vout in

  let batch_shape = Array.sub (shape vout) 0 (max 0 (nd_out - 2)) in
  let batch_sz =
    if Array.length batch_shape = 0 then 1 else Shape.numel batch_shape
  in

  let work = ref start_idx in
  while !work < end_idx do
    let i0 = !work mod m in
    let batch = !work / m in
    let has_row1 = (i0 + 1 < m) && (!work + 1 < end_idx) in

    if has_row1 then 
      begin
        if batch_sz <> 1 then begin
          Shape.unravel_index_into batch batch_shape out_idx0;
          Shape.unravel_index_into batch batch_shape out_idx1;
        end;
    
        Shape.broadcast_index_into out_idx0 (shape va) a_idx0;
        Shape.broadcast_index_into out_idx0 (shape vb) b_idx;
        Shape.broadcast_index_into out_idx0 (shape va) a_idx1;
    
        out_idx0.(nd_out - 2) <- i0;
        a_idx0.(nd_a - 2) <- i0;
    
        if has_row1 then begin
          out_idx1.(nd_out - 2) <- i0 + 1;
          a_idx1.(nd_a - 2) <- i0 + 1
        end;
    
        let j = ref 0 in
        while !j + 1 < n do
          out_idx0.(nd_out - 1) <- !j;
          b_idx.(nd_b - 1) <- !j;
    
          out_idx1.(nd_out - 1) <- !j;
    
          let rec kloop l acc0 acc1=
            if l = k then #(acc0, acc1)
            else begin
              a_idx1.(nd_a - 1) <- l;
              b_idx.(nd_b - 2) <- l;
              let bv =
                Float64x2.Array.unsafe_get b_buf
                  ~idx:(View.offset vb + Shape.ravel_index b_idx b_str)
              in
              let av0 =
                Array.unsafe_get a_buf
                  (View.offset va + Shape.ravel_index a_idx0 a_str)
              in
              let a0v = f64_set1 av0 in
              let av1 =
                Array.unsafe_get a_buf
                  (View.offset va + Shape.ravel_index a_idx1 a_str)
              in
              let a1v = f64_set1 av1 in
              kloop (l + 1)
              (f64_mul_add a0v bv acc0)
              (f64_mul_add a1v bv acc1)
            end
          in
          let #(acc0, acc1) = kloop 0 (f64_set1 #0.0) (f64_set1 #0.0) in
          let out_off0 =
            View.offset vout + Shape.ravel_index out_idx0 out_str
          in
          Float64x2.Array.unsafe_set c_buf ~idx:out_off0 acc0;
          let out_off1 =
            View.offset vout + Shape.ravel_index out_idx1 out_str
          in
          Float64x2.Array.unsafe_set c_buf ~idx:out_off1 acc1;
    
          j := !j + 2
        done;
    
        while !j < n do
          out_idx0.(nd_out - 1) <- !j;
          b_idx.(nd_b - 1) <- !j;
    
          let rec scalar l acc0 acc1 =
            if l = k then #(acc0, acc1)
            else begin
              a_idx0.(nd_a - 1) <- l;
              b_idx.(nd_b - 2) <- l;
              let av0 =
                Array.unsafe_get a_buf
                  (View.offset va + Shape.ravel_index a_idx0 a_str)
              in
              let av1 =
                Array.unsafe_get a_buf
                  (View.offset va + Shape.ravel_index a_idx1 a_str)
              in
              let bv =
                Array.unsafe_get b_buf
                  (View.offset vb + Shape.ravel_index b_idx b_str)
              in
              scalar (l + 1) (Float_u.fma av0 bv acc0) (Float_u.fma av1 bv acc1)
            end
          in
    
          let #(sum0, sum1) = scalar 0 #0.0 #0.0 in
          let out0 =
            View.offset vout + Shape.ravel_index out_idx0 out_str
          in
          Array.unsafe_set c_buf out0 sum0;

          out_idx1.(nd_out - 1) <- !j;
          let out1 =
            View.offset vout + Shape.ravel_index out_idx1 out_str
          in
          Array.unsafe_set c_buf out1 sum1;

          j := !j + 1
        done;
    
        work := !work + 2
    end else 
      begin
        if batch_sz <> 1 then begin
          Shape.unravel_index_into batch batch_shape out_idx0;
        end;
    
        Shape.broadcast_index_into out_idx0 (shape va) a_idx0;
        Shape.broadcast_index_into out_idx0 (shape vb) b_idx;
    
        out_idx0.(nd_out - 2) <- i0;
        a_idx0.(nd_a - 2) <- i0;
    
    
        let j = ref 0 in
        while !j + 1 < n do
          out_idx0.(nd_out - 1) <- !j;
          b_idx.(nd_b - 1) <- !j;
    
    
          let rec kloop l acc0 =
            if l = k then acc0
            else begin
              a_idx0.(nd_a - 1) <- l;
              b_idx.(nd_b - 2) <- l;
              let av0 =
                Array.unsafe_get a_buf
                  (View.offset va + Shape.ravel_index a_idx0 a_str)
              in
              let bv =
                Float64x2.Array.unsafe_get b_buf
                  ~idx:(View.offset vb + Shape.ravel_index b_idx b_str)
              in
              let a0v = f64_set1 av0 in
              kloop (l + 1) (f64_mul_add a0v bv acc0)
            end
          in
          let acc0 = kloop 0 (f64_set1 #0.0) in
          let out_off0 =
            View.offset vout + Shape.ravel_index out_idx0 out_str
          in
          Float64x2.Array.unsafe_set c_buf ~idx:out_off0 acc0;
    
          j := !j + 2
        done;
    
        while !j < n do
          out_idx0.(nd_out - 1) <- !j;
          b_idx.(nd_b - 1) <- !j;
    
          let rec scalar l acc =
            if l = k then acc
            else begin
              a_idx0.(nd_a - 1) <- l;
              b_idx.(nd_b - 2) <- l;
    
              let av =
                Array.unsafe_get a_buf
                  (View.offset va + Shape.ravel_index a_idx0 a_str)
              in
              let bv =
                Array.unsafe_get b_buf
                  (View.offset vb + Shape.ravel_index b_idx b_str)
              in
              scalar (l + 1) (Float_u.fma av bv acc)
            end
          in
    
          let sum0 = scalar 0 #0.0 in
          let out0 =
            View.offset vout + Shape.ravel_index out_idx0 out_str
          in
          Array.unsafe_set c_buf out0 sum0;
    
          j := !j + 1
        done;
    
        work := !work + 1
    end;

  done

module Gemm_f32 = struct
  let mr = 6
  let nr = 8
  let kc_blk = 256
  let mc_blk = 240
  let nc_blk = 640

  let pack_a a ~a_off ~lda ~ic ~pc ~mc ~kc ap =
    let dst = ref 0 in
    let i = ref 0 in
    while !i + mr <= mc do
      for p = 0 to kc - 1 do
        let src_base = a_off + (ic + !i) * lda + pc + p in
        for ii = 0 to mr - 1 do
          Array.unsafe_set ap (!dst + ii)
            (Array.unsafe_get a (src_base + ii * lda))
        done;
        dst := !dst + mr
      done;
      i := !i + mr
    done;
    if !i < mc then begin
      let mr_rem = mc - !i in
      for p = 0 to kc - 1 do
        let src_base = a_off + (ic + !i) * lda + pc + p in
        for ii = 0 to mr_rem - 1 do
          Array.unsafe_set ap (!dst + ii)
            (Array.unsafe_get a (src_base + ii * lda))
        done;
        for ii = mr_rem to mr - 1 do
          Array.unsafe_set ap (!dst + ii) #0.0s
        done;
        dst := !dst + mr
      done
    end

  let pack_b b ~b_off ~ldb ~pc ~jc ~kc ~nc bp =
    let dst = ref 0 in
    let j = ref 0 in
    while !j + nr <= nc do
      for p = 0 to kc - 1 do
        let src = b_off + (pc + p) * ldb + jc + !j in
        for jj = 0 to nr - 1 do
          Array.unsafe_set bp (!dst + jj) (Array.unsafe_get b (src + jj))
        done;
        dst := !dst + nr
      done;
      j := !j + nr
    done;
    if !j < nc then begin
      let nr_rem = nc - !j in
      for p = 0 to kc - 1 do
        let src = b_off + (pc + p) * ldb + jc + !j in
        for jj = 0 to nr_rem - 1 do
          Array.unsafe_set bp (!dst + jj) (Array.unsafe_get b (src + jj))
        done;
        for jj = nr_rem to nr - 1 do
          Array.unsafe_set bp (!dst + jj) #0.0s
        done;
        dst := !dst + nr
      done
    end

  let rec f32_kloop ap ap_off bp bp_off c_buf c_off ldc kc p
      c00 c01 c10 c11 c20 c21 c30 c31 c40 c41 c50 c51 =
    if p = kc then begin
      Float32x4.Array.unsafe_set c_buf ~idx:c_off c00;
      Float32x4.Array.unsafe_set c_buf ~idx:(c_off + 4) c01;
      let r1 = c_off + ldc in
      Float32x4.Array.unsafe_set c_buf ~idx:r1 c10;
      Float32x4.Array.unsafe_set c_buf ~idx:(r1 + 4) c11;
      let r2 = c_off + 2 * ldc in
      Float32x4.Array.unsafe_set c_buf ~idx:r2 c20;
      Float32x4.Array.unsafe_set c_buf ~idx:(r2 + 4) c21;
      let r3 = c_off + 3 * ldc in
      Float32x4.Array.unsafe_set c_buf ~idx:r3 c30;
      Float32x4.Array.unsafe_set c_buf ~idx:(r3 + 4) c31;
      let r4 = c_off + 4 * ldc in
      Float32x4.Array.unsafe_set c_buf ~idx:r4 c40;
      Float32x4.Array.unsafe_set c_buf ~idx:(r4 + 4) c41;
      let r5 = c_off + 5 * ldc in
      Float32x4.Array.unsafe_set c_buf ~idx:r5 c50;
      Float32x4.Array.unsafe_set c_buf ~idx:(r5 + 4) c51
    end
    else
      let ab = ap_off + p * 6 in
      let bb = bp_off + p * 8 in
      let a0 = f32_set1 (Array.unsafe_get ap ab) in
      let a1 = f32_set1 (Array.unsafe_get ap (ab + 1)) in
      let a2 = f32_set1 (Array.unsafe_get ap (ab + 2)) in
      let a3 = f32_set1 (Array.unsafe_get ap (ab + 3)) in
      let a4 = f32_set1 (Array.unsafe_get ap (ab + 4)) in
      let a5 = f32_set1 (Array.unsafe_get ap (ab + 5)) in
      let b0 = Float32x4.Array.unsafe_get bp ~idx:bb in
      let b1 = Float32x4.Array.unsafe_get bp ~idx:(bb + 4) in
      f32_kloop ap ap_off bp bp_off c_buf c_off ldc kc (p + 1)
        (f32_mul_add a0 b0 c00) (f32_mul_add a0 b1 c01)
        (f32_mul_add a1 b0 c10) (f32_mul_add a1 b1 c11)
        (f32_mul_add a2 b0 c20) (f32_mul_add a2 b1 c21)
        (f32_mul_add a3 b0 c30) (f32_mul_add a3 b1 c31)
        (f32_mul_add a4 b0 c40) (f32_mul_add a4 b1 c41)
        (f32_mul_add a5 b0 c50) (f32_mul_add a5 b1 c51)

  let kernel_zero ap ~ap_off bp ~bp_off c_buf ~c_off ~ldc ~kc =
    let z = f32_set1 #0.0s in
    f32_kloop ap ap_off bp bp_off c_buf c_off ldc kc 0
      z z z z z z z z z z z z

  let kernel_accum ap ~ap_off bp ~bp_off c_buf ~c_off ~ldc ~kc =
    let r1 = c_off + ldc in
    let r2 = c_off + 2 * ldc in
    let r3 = c_off + 3 * ldc in
    let r4 = c_off + 4 * ldc in
    let r5 = c_off + 5 * ldc in
    f32_kloop ap ap_off bp bp_off c_buf c_off ldc kc 0
      (Float32x4.Array.unsafe_get c_buf ~idx:c_off)
      (Float32x4.Array.unsafe_get c_buf ~idx:(c_off + 4))
      (Float32x4.Array.unsafe_get c_buf ~idx:r1)
      (Float32x4.Array.unsafe_get c_buf ~idx:(r1 + 4))
      (Float32x4.Array.unsafe_get c_buf ~idx:r2)
      (Float32x4.Array.unsafe_get c_buf ~idx:(r2 + 4))
      (Float32x4.Array.unsafe_get c_buf ~idx:r3)
      (Float32x4.Array.unsafe_get c_buf ~idx:(r3 + 4))
      (Float32x4.Array.unsafe_get c_buf ~idx:r4)
      (Float32x4.Array.unsafe_get c_buf ~idx:(r4 + 4))
      (Float32x4.Array.unsafe_get c_buf ~idx:r5)
      (Float32x4.Array.unsafe_get c_buf ~idx:(r5 + 4))

  let edge_scalar ap ~ap_off bp ~bp_off c_buf ~c_off ~ldc ~mr_eff ~nr_eff
      ~kc ~first =
    for i = 0 to mr_eff - 1 do
      for j = 0 to nr_eff - 1 do
        let c_idx = c_off + i * ldc + j in
        let init = if first then #0.0s else Array.unsafe_get c_buf c_idx in
        let rec loop p acc =
          if p = kc then acc
          else
            let a = Array.unsafe_get ap (ap_off + p * 6 + i) in
            let b = Array.unsafe_get bp (bp_off + p * 8 + j) in
            loop (p + 1) (Float32_u.fma a b acc)
        in
        Array.unsafe_set c_buf c_idx (loop 0 init)
      done
    done

  let macro_kernel ap bp c_buf ~c_off ~ldc ~mc ~nc ~kc ~first =
    let ir = ref 0 in
    while !ir < mc do
      let mr_eff = min_int mr (mc - !ir) in
      let ap_off = (!ir / mr) * mr * kc in
      let jr = ref 0 in
      while !jr < nc do
        let nr_eff = min_int nr (nc - !jr) in
        let bp_off = (!jr / nr) * nr * kc in
        let c_tile = c_off + (!ir * ldc) + !jr in
        if mr_eff = mr && nr_eff = nr then begin
          if first then
            kernel_zero ap ~ap_off bp ~bp_off c_buf
              ~c_off:c_tile ~ldc ~kc
          else
            kernel_accum ap ~ap_off bp ~bp_off c_buf
              ~c_off:c_tile ~ldc ~kc
        end
        else
          edge_scalar ap ~ap_off bp ~bp_off c_buf
            ~c_off:c_tile ~ldc ~mr_eff ~nr_eff ~kc ~first;
        jr := !jr + nr
      done;
      ir := !ir + mr
    done

  let gemm ~pool a_buf b_buf c_buf ~m ~n ~k ~a_off ~b_off ~c_off ~ldc () =
    let lda = k and ldb = n in
    let mc = mc_blk and nc = nc_blk and kc = kc_blk in
    let rec jc_loop jc =
      if jc >= n then ()
      else
        let nc' = min_int nc (n - jc) in
        Parallel.parallel_for pool 0 (m - 1) (fun start_row end_row ->
            let bp = Array.make_float32 (round_up nc' nr * kc) in
            let ap = Array.make_float32 (round_up mc mr * kc) in
            let rec pc_loop pc =
              if pc >= k then ()
              else
                let kc' = min_int kc (k - pc) in
                let first = pc = 0 in
                pack_b b_buf ~b_off ~ldb ~pc ~jc ~kc:kc' ~nc:nc' bp;
                let rec ic_loop ic =
                  if ic >= end_row then ()
                  else
                    let mc' = min_int mc (end_row - ic) in
                    pack_a a_buf ~a_off ~lda ~ic ~pc ~mc:mc' ~kc:kc' ap;
                    macro_kernel ap bp c_buf
                      ~c_off:(c_off + ic * ldc + jc)
                      ~ldc ~mc:mc' ~nc:nc' ~kc:kc' ~first;
                    ic_loop (ic + mc')
                in
                ic_loop start_row;
                pc_loop (pc + kc')
            in
            pc_loop 0);
        jc_loop (jc + nc')
    in
    jc_loop 0
end

let matmul_float32_slow a_buf b_buf c_buf va vb vout start_idx end_idx =
  let nd_a = Array.length (shape va)
  and nd_b = Array.length (shape vb)
  and nd_out = Array.length (shape vout) in

  let rank = nd_out in
  let m = (shape vout).(rank - 2) in
  let n = (shape vout).(rank - 1) in
  let k = (shape va).(rank - 1) in

  let a_idx0 = Array.make nd_a 0
  and a_idx1 = Array.make nd_a 0
  and b_idx = Array.make nd_b 0
  and out_idx0 = Array.make nd_out 0
  and out_idx1 = Array.make nd_out 0 in

  let a_str = View.strides va
  and b_str = View.strides vb
  and out_str = View.strides vout in

  let batch_shape = Array.sub (shape vout) 0 (max 0 (nd_out - 2)) in
  let batch_sz =
    if Array.length batch_shape = 0 then 1 else Shape.numel batch_shape
  in

  let work = ref start_idx in
  while !work < end_idx do
    let i0 = !work mod m in
    let batch = !work / m in
    let has_row1 = (i0 + 1 < m) && (!work + 1 < end_idx) in

    if has_row1 then 
      begin
        if batch_sz <> 1 then begin
          Shape.unravel_index_into batch batch_shape out_idx0;
          Shape.unravel_index_into batch batch_shape out_idx1;
        end;
    
        Shape.broadcast_index_into out_idx0 (shape va) a_idx0;
        Shape.broadcast_index_into out_idx0 (shape vb) b_idx;
        Shape.broadcast_index_into out_idx0 (shape va) a_idx1;
    
        out_idx0.(nd_out - 2) <- i0;
        a_idx0.(nd_a - 2) <- i0;
    
        if has_row1 then begin
          out_idx1.(nd_out - 2) <- i0 + 1;
          a_idx1.(nd_a - 2) <- i0 + 1
        end;
    
        let j = ref 0 in
        while !j + 3 < n do
          out_idx0.(nd_out - 1) <- !j;
          b_idx.(nd_b - 1) <- !j;
    
          out_idx1.(nd_out - 1) <- !j;
    
    
          let rec kloop l acc0 acc1 =
            if l = k then #(acc0, acc1)
            else begin
              a_idx0.(nd_a - 1) <- l;
              b_idx.(nd_b - 2) <- l;
              let av0 =
                Array.unsafe_get a_buf
                  (View.offset va + Shape.ravel_index a_idx0 a_str)
              in
              let av1 =
                Array.unsafe_get a_buf
                  (View.offset va + Shape.ravel_index a_idx1 a_str)
              in
              let bv =
                Float32x4.Array.unsafe_get b_buf
                  ~idx:(View.offset vb + Shape.ravel_index b_idx b_str)
              in
              let a0v = f32_set1 av0 in
              let a1v = f32_set1 av1 in
              kloop (l + 1)
              (f32_mul_add a0v bv acc0)
              (f32_mul_add a1v bv acc1)
            end

          in
          let #(acc0, acc1) = kloop 0 (f32_set1 #0.0s) (f32_set1 #0.0s) in
          let out_off0 =
            View.offset vout + Shape.ravel_index out_idx0 out_str
          in
          Float32x4.Array.unsafe_set c_buf ~idx:out_off0 acc0;
          let out_off1 =
            View.offset vout + Shape.ravel_index out_idx1 out_str
          in
          Float32x4.Array.unsafe_set c_buf ~idx:out_off1 acc1;
    
          j := !j + 4
        done;
    
        while !j < n do
          out_idx0.(nd_out - 1) <- !j;
          b_idx.(nd_b - 1) <- !j;
    
          let rec scalar l acc0 acc1 =
            if l = k then #(acc0, acc1)
            else begin
              a_idx0.(nd_a - 1) <- l;
              b_idx.(nd_b - 2) <- l;
    
              let av =
                Array.unsafe_get a_buf
                  (View.offset va + Shape.ravel_index a_idx0 a_str)
              in
              let bv =
                Array.unsafe_get b_buf
                  (View.offset vb + Shape.ravel_index b_idx b_str)
              in
              scalar (l + 1) (Float32_u.fma av bv acc0) (Float32_u.fma av bv acc1)
            end
          in
    
          let #(sum0, sum1) = scalar 0 #0.0s #0.0s in
          let out0 =
            View.offset vout + Shape.ravel_index out_idx0 out_str
          in
          Array.unsafe_set c_buf out0 sum0;

          out_idx1.(nd_out - 1) <- !j;
          let out1 =
            View.offset vout + Shape.ravel_index out_idx1 out_str
          in
          Array.unsafe_set c_buf out1 sum1;

          j := !j + 1
        done;
    
        work := !work + 2
    end else 
      begin
        if batch_sz <> 1 then begin
          Shape.unravel_index_into batch batch_shape out_idx0;
        end;
    
        Shape.broadcast_index_into out_idx0 (shape va) a_idx0;
        Shape.broadcast_index_into out_idx0 (shape vb) b_idx;
    
        out_idx0.(nd_out - 2) <- i0;
        a_idx0.(nd_a - 2) <- i0;
    
    
        let j = ref 0 in
        while !j + 7 < n do
          out_idx0.(nd_out - 1) <- !j;
          b_idx.(nd_b - 1) <- !j;
    
    
          let rec kloop_r0 l acc0 =
            if l = k then acc0
            else begin
              a_idx0.(nd_a - 1) <- l;
              b_idx.(nd_b - 2) <- l;
              let av0 =
                Array.unsafe_get a_buf
                  (View.offset va + Shape.ravel_index a_idx0 a_str)
              in
              let bv =
                Float32x4.Array.unsafe_get b_buf
                  ~idx:(View.offset vb + Shape.ravel_index b_idx b_str)
              in
              let a0v = f32_set1 av0 in
              kloop_r0 (l + 1) (f32_mul_add a0v bv acc0)
            end
          in
          let acc0 = kloop_r0 0 (f32_set1 #0.0s) in
          let out_off0 =
            View.offset vout + Shape.ravel_index out_idx0 out_str
          in
          Float32x4.Array.unsafe_set c_buf ~idx:out_off0 acc0;
    
          j := !j + 8
        done;

        while !j + 3 < n do
          out_idx0.(nd_out - 1) <- !j;
          b_idx.(nd_b - 1) <- !j;
    
    
          let rec kloop l acc0 =
            if l = k then acc0
            else begin
              a_idx0.(nd_a - 1) <- l;
              b_idx.(nd_b - 2) <- l;
              let av0 =
                Array.unsafe_get a_buf
                  (View.offset va + Shape.ravel_index a_idx0 a_str)
              in
              let bv =
                Float32x4.Array.unsafe_get b_buf
                  ~idx:(View.offset vb + Shape.ravel_index b_idx b_str)
              in
              let a0v = f32_set1 av0 in
              kloop (l + 1) (f32_mul_add a0v bv acc0)
            end
          in
          let acc0 = kloop 0 (f32_set1 #0.0s) in
          let out_off0 =
            View.offset vout + Shape.ravel_index out_idx0 out_str
          in
          Float32x4.Array.unsafe_set c_buf ~idx:out_off0 acc0;
    
          j := !j + 4
        done;
    
        while !j < n do
          out_idx0.(nd_out - 1) <- !j;
          b_idx.(nd_b - 1) <- !j;
    
          let rec scalar l acc =
            if l = k then acc
            else begin
              a_idx0.(nd_a - 1) <- l;
              b_idx.(nd_b - 2) <- l;
    
              let av =
                Array.unsafe_get a_buf
                  (View.offset va + Shape.ravel_index a_idx0 a_str)
              in
              let bv =
                Array.unsafe_get b_buf
                  (View.offset vb + Shape.ravel_index b_idx b_str)
              in
              scalar (l + 1) (Float32_u.fma av bv acc)
            end
          in
    
          let sum0 = scalar 0 #0.0s in
          let out0 =
            View.offset vout + Shape.ravel_index out_idx0 out_str
          in
          Array.unsafe_set c_buf out0 sum0;
    
          j := !j + 1
        done;
    
        work := !work + 1
    end;

  done

let matmul_int64_fast a_buf b_buf c_buf va vb vout start_idx end_idx = 
  let mc = 128 in
  let nc = 128 in
  let kc = 64 in
  let rank = Array.length (shape vout) in
  let n = (shape vout).(rank - 1) in
  let k = (shape va).(rank - 1) in

  let a_rs = k and b_rs = n and c_rs = n in
    let a0 = View.offset va and b0 = View.offset vb and c0 = View.offset vout in
  

  let rec jc_loop jc =
    if jc >= n then ()
    else
      let nc' = min nc (n - jc) in
      let rec pc_loop pc =
        if pc >= k then ()
        else
          let kc' = min kc (k - pc) in
          let rec ic_loop ic =
            if ic >= end_idx then ()
            else
              let mc' = min mc (end_idx - ic) in
                for i = ic to ic + mc' - 1 do
                  let a_row = a0 + (i * a_rs) + pc
                  and c_row = c0 + (i * c_rs) + jc in
                  for j = jc to jc + nc' - 1 do
                    let a_idx0 = a_row in
                    let b_idx0 = b0 + (pc * b_rs) + j in
                    
                    let rec loop p a_idx b_idx acc =
                      if p = kc' then
                        acc
                    else
                        let av = Array.unsafe_get a_buf a_idx in
                        let bv = Array.unsafe_get b_buf b_idx in
                        loop (p + 1) (a_idx + 1) (b_idx + b_rs) (Int64_u.add (Int64_u.mul av bv) acc)
                  in
                    let sum = loop 0 a_idx0 b_idx0 #0L in
                    Array.unsafe_set c_buf (c_row + j - jc) sum
                  done
                done;
              ic_loop (ic + mc')
          in
          ic_loop start_idx;
          pc_loop (pc + kc')
      in
      pc_loop 0;
      jc_loop (jc + nc')
  in
  jc_loop 0
    
let matmul_int64_slow a_buf b_buf c_buf va vb vout start_idx end_idx = 
  let nd_a, nd_b, nd_out =
  (Array.length (shape va), Array.length (shape vb), Array.length (shape vout))
  in
  let rank = Array.length (shape vout) in
  let m = (shape vout).(rank - 2) in
  let n = (shape vout).(rank - 1) in
  let k = (shape va).(rank - 1) in
  let a_idx = Array.make nd_a 0
  and b_idx = Array.make nd_b 0
  and out_idx = Array.make nd_out 0 in
  let a_str = View.strides va
  and b_str = View.strides vb
  and out_str = View.strides vout in
  let nd_out = Array.length (shape vout) in
  let batch_shape = Array.sub (shape vout) 0 (max 0 (nd_out - 2)) in
  let batch_sz =
    if Array.length batch_shape = 0 then 1 else Shape.numel batch_shape
  in

  for work = start_idx to end_idx - 1 do
    let batch = work / m and i = work mod m in
    (* unravel batch index into leading dims of C *)
    if batch_sz <> 1 then Shape.unravel_index_into batch batch_shape out_idx;
    (* broadcast batch into a_idx / b_idx *)
    Shape.broadcast_index_into out_idx (shape va) a_idx;
    Shape.broadcast_index_into out_idx (shape vb) b_idx;
    (* set row index *)
    out_idx.(nd_out - 2) <- i;
    a_idx.(nd_a - 2) <- i;
    for j = 0 to n - 1 do
      out_idx.(nd_out - 1) <- j;
      b_idx.(nd_b - 1) <- j;
      let rec loop l acc =
        if l = k then
          acc
        else (
          a_idx.(nd_a - 1) <- l;
          b_idx.(nd_b - 2) <- l;
      
          let av =
            Array.unsafe_get a_buf (View.offset va + Shape.ravel_index a_idx a_str)
          in
          let bv =
            Array.unsafe_get b_buf (View.offset vb + Shape.ravel_index b_idx b_str)
          in
      
          loop (l + 1) (Int64_u.add (Int64_u.mul av bv) acc)
        )
      in
      let sum = loop 0 #0L in
      
      let out_off = View.offset vout + Shape.ravel_index out_idx out_str in
      Array.unsafe_set c_buf out_off sum
    done
  done

let matmul_int32_fast a_buf b_buf c_buf va vb vout start_idx end_idx = 
  let mc = 128 in
  let nc = 128 in
  let kc = 64 in
  let rank = Array.length (shape vout) in
  let n = (shape vout).(rank - 1) in
  let k = (shape va).(rank - 1) in

  let a_rs = k and b_rs = n and c_rs = n in
  let a0 = View.offset va and b0 = View.offset vb and c0 = View.offset vout in


  let rec jc_loop jc =
    if jc >= n then ()
    else
      let nc' = min nc (n - jc) in
      let rec pc_loop pc =
        if pc >= k then ()
        else
          let kc' = min kc (k - pc) in
          let rec ic_loop ic =
            if ic >= end_idx then ()
            else
              let mc' = min mc (end_idx - ic) in
              for i = ic to ic + mc' - 1 do
                let a_row = a0 + (i * a_rs) + pc
                and c_row = c0 + (i * c_rs) + jc in
                for j = jc to jc + nc' - 1 do
                  let a_idx0 = a_row in
                  let b_idx0 = b0 + (pc * b_rs) + j in
                  
                  let rec loop p a_idx b_idx acc =
                    if p = kc' then
                      acc
                    else
                      let av = Array.unsafe_get a_buf a_idx in
                      let bv = Array.unsafe_get b_buf b_idx in
                      loop (p + 1) (a_idx + 1) (b_idx + b_rs) (Int32_u.add (Int32_u.mul av bv) acc)
                  in
                  let sum = loop 0 a_idx0 b_idx0 #0l in
                  Array.unsafe_set c_buf (c_row + j - jc) sum
                done
              done;
              ic_loop (ic + mc')
          in
          ic_loop start_idx;
          pc_loop (pc + kc')
      in
      pc_loop 0;
      jc_loop (jc + nc')
  in
  jc_loop 0

let matmul_int32_slow a_buf b_buf c_buf va vb vout start_idx end_idx = 
  let nd_a, nd_b, nd_out =
  (Array.length (shape va), Array.length (shape vb), Array.length (shape vout))
in
let rank = Array.length (shape vout) in
let m = (shape vout).(rank - 2) in
let n = (shape vout).(rank - 1) in
let k = (shape va).(rank - 1) in
  let a_idx = Array.make nd_a 0
  and b_idx = Array.make nd_b 0
  and out_idx = Array.make nd_out 0 in
  let a_str = View.strides va
  and b_str = View.strides vb
  and out_str = View.strides vout in
  let nd_out = Array.length (shape vout) in
  let batch_shape = Array.sub (shape vout) 0 (max 0 (nd_out - 2)) in
  let batch_sz =
    if Array.length batch_shape = 0 then 1 else Shape.numel batch_shape
  in

  for work = start_idx to end_idx - 1 do
    let batch = work / m and i = work mod m in
    (* unravel batch index into leading dims of C *)
    if batch_sz <> 1 then Shape.unravel_index_into batch batch_shape out_idx;
    (* broadcast batch into a_idx / b_idx *)
    Shape.broadcast_index_into out_idx (shape va) a_idx;
    Shape.broadcast_index_into out_idx (shape vb) b_idx;
    (* set row index *)
    out_idx.(nd_out - 2) <- i;
    a_idx.(nd_a - 2) <- i;
    for j = 0 to n - 1 do
      out_idx.(nd_out - 1) <- j;
      b_idx.(nd_b - 1) <- j;
      let rec loop l acc =
        if l = k then
          acc
        else (
          a_idx.(nd_a - 1) <- l;
          b_idx.(nd_b - 2) <- l;
      
          let av =
            Array.unsafe_get a_buf (View.offset va + Shape.ravel_index a_idx a_str)
          in
          let bv =
            Array.unsafe_get b_buf (View.offset vb + Shape.ravel_index b_idx b_str)
          in
      
          loop (l + 1) (Int32_u.add (Int32_u.mul av bv) acc)
        )
      in
      let sum = loop 0 #0l in
      
      let out_off = View.offset vout + Shape.ravel_index out_idx out_str in
      Array.unsafe_set c_buf out_off sum
    done
  done
