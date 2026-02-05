(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let matmul_float64_fast a_buf b_buf c_buf va vb vout start_idx end_idx =
  let mc = 128 in
  let nc = 128 in
  let kc = 64 in

  let rank = Array.length (shape vout) in
  let n = (shape vout).(rank - 1) in
  let k = (shape va).(rank - 1) in

  let a_rs = k and b_rs = n and c_rs = n in
  let a0 = View.offset va
  and b0 = View.offset vb
  and c0 = View.offset vout in

  let rec jc_loop jc =
    if jc >= n then ()
    else
      let nc' = min nc (n - jc) in

      let rec pc_loop pc =
        if pc >= k then ()
        else if pc == 0 then begin

          let kc' = min kc k in
          let bp = Array.make_float64 (kc' * nc') in
          for p = 0 to kc' - 1 do
            let kk = pc + p in
            let b_row = b0 + kk * b_rs + jc in
            let bp_row = p * nc' in
            for j = 0 to nc' - 1 do
              Array.unsafe_set bp (bp_row + j) (Array.unsafe_get b_buf (b_row + j))
            done
          done;

          let rec ic_loop ic =
            if ic >= end_idx then ()
            else
              let mc' = min mc (end_idx - ic) in
              let i_end = ic + mc' in
              let i = ref ic in

              (* ===== 2×2 SIMD ROW LOOP ===== *)
              while !i + 1 < i_end do
                let i0 = !i in
                let i1 = i0 + 1 in

                let row0 = a0 + i0 * a_rs in
                let row1 = a0 + i1 * a_rs in
                let crow0 = c0 + i0 * c_rs in
                let crow1 = c0 + i1 * c_rs in

                let j = ref 0 in
                let j4 = nc'-3 in
                while !j < j4 do
                  let col = jc + !j in

                  let c_idx0 = crow0 + col in
                  let c_idx1 = crow1 + col in

                  let acc0 = Float64x2.set1 #0.0
                  in
                  let acc1 = Float64x2.set1 #0.0
                  in

                  let rec kloop p acc0 acc1 =
                    if p = kc' then #(acc0, acc1)
                    else
                      let kk = pc + p in
                      let bv = Float64x2.Array.unsafe_get b_buf ~idx:(p * nc' + !j) in  (* load once *)
                      let a0v = Float64x2.set1 (Array.unsafe_get a_buf (row0 + kk)) in
                      let a1v = Float64x2.set1 (Array.unsafe_get a_buf (row1 + kk)) in
                      kloop (p + 1)
                        (Float64x2.mul_add a0v bv acc0)
                        (Float64x2.mul_add a1v bv acc1)
                  in
                  let #(acc0, acc1) = kloop 0 acc0 acc1 in
                  Float64x2.Array.unsafe_set c_buf ~idx:c_idx0 acc0;
                  Float64x2.Array.unsafe_set c_buf ~idx:c_idx1 acc1;

                  j := !j + 4
                done;

                while !j < nc' do
                  let col = jc + !j in
                  let rec scalar p acc0 acc1 =
                    if p = kc' then #(acc0, acc1)
                    else
                      let a0 = Array.unsafe_get a_buf (row0 + p) in
                      let a1 = Array.unsafe_get a_buf (row1 + p) in
                      let b  = Array.unsafe_get bp (p * nc' + !j) in
                      scalar
                        (p + 1)
                        (Float_u.fma a0 b acc0)
                        (Float_u.fma a1 b acc1)
                  in
                    let #(acc0, acc1) = scalar 0 #0.0 #0.0 in
                  Array.unsafe_set c_buf (crow0 + col) acc0;
                  Array.unsafe_set c_buf (crow1 + col) acc1;
                    j := !j + 1
                    done;
                i := !i + 2
              done;

              if !i < i_end then begin
                let row = !i in
                let arow = a0 + row * a_rs in
                let crow = c0 + row * c_rs in

                for j = 0 to nc' - 1 do
                  let col = jc + j in
                  let rec scalar p acc =
                    if p = kc' then acc
                    else
                      let a = Array.unsafe_get a_buf (arow + p) in
                      let b = Array.unsafe_get bp (p * nc' + j) in
                      scalar (p + 1) (Float_u.fma a b acc)
                  in
                  let acc = scalar 0 #0.0
                  in
                  Array.unsafe_set c_buf (crow + col) acc
                done
              end;

              ic_loop (ic + mc')
          in
          ic_loop start_idx;
          pc_loop (kc')
        end
        else
          let kc' = min kc (k - pc) in
          let bp = Array.make_float64 (kc' * nc') in
          for p = 0 to kc' - 1 do
            let kk = pc + p in
            let b_row = b0 + kk * b_rs + jc in
            let bp_row = p * nc' in
            for j = 0 to nc' - 1 do
              Array.unsafe_set bp (bp_row + j) (Array.unsafe_get b_buf (b_row + j))
            done
          done;

          let rec ic_loop ic =
            if ic >= end_idx then ()
            else
              let mc' = min mc (end_idx - ic) in
              let i_end = ic + mc' in
              let i = ref ic in

              (* ===== 2×2 SIMD ROW LOOP ===== *)
              while !i + 1 < i_end do
                let i0 = !i in
                let i1 = i0 + 1 in

                let row0 = a0 + i0 * a_rs in
                let row1 = a0 + i1 * a_rs in
                let crow0 = c0 + i0 * c_rs in
                let crow1 = c0 + i1 * c_rs in

                let j = ref 0 in
                let j4 = nc'-3 in
                while !j < j4 do
                  let col = jc + !j in

                  let c_idx0 = crow0 + col in
                  let c_idx1 = crow1 + col in

                  let acc0 = Float64x2.Array.unsafe_get c_buf ~idx:c_idx0
                  in
                  let acc1 = Float64x2.Array.unsafe_get c_buf ~idx:c_idx1
                  in

                  let rec kloop p acc0 acc1 =
                    if p = kc' then #(acc0, acc1)
                    else
                      let kk = pc + p in
                      let a0v =
                        Float64x2.set1 (Array.unsafe_get a_buf (row0 + kk))
                      in
                      let a1v =
                        Float64x2.set1 (Array.unsafe_get a_buf (row0 + kk))
                      in
                      let bv =
                        Float64x2.Array.unsafe_get bp
                          ~idx:(p * nc' + !j)
                      in
                      kloop
                        (p + 1)
                        (Float64x2.mul_add a0v bv acc0)
                        (Float64x2.mul_add a1v bv acc1)
                    in
                  let #(acc0, acc1) = kloop 0 acc0 acc1 in
                  Float64x2.Array.unsafe_set c_buf ~idx:c_idx0 acc0;
                  Float64x2.Array.unsafe_set c_buf ~idx:c_idx1 acc1;

                  j := !j + 4
                done;

                while !j < nc' do
                  let col = jc + !j in
                  let rec scalar p acc0 acc1 =
                    if p = kc' then #(acc0, acc1)
                    else
                      let kk = pc + p in
                      let a0 = Array.unsafe_get a_buf (row0 + kk) in
                      let a1 = Array.unsafe_get a_buf (row1 + kk) in
                      let b  = Array.unsafe_get bp (p * nc' + !j) in
                      scalar
                        (p + 1)
                        (Float_u.fma a0 b acc0)
                        (Float_u.fma a1 b acc1)
                  in
                  let #(acc0, acc1) = scalar 0
                      (Array.unsafe_get c_buf (crow0 + col))
                      (Array.unsafe_get c_buf (crow1 + col))
                    in
                    Array.unsafe_set c_buf (crow0 + col) acc0;
                    Array.unsafe_set c_buf (crow1 + col) acc1;
                    j := !j + 1
                    done;
                i := !i + 2
              done;

              if !i < i_end then begin
                let row = !i in
                let arow = a0 + row * a_rs in
                let crow = c0 + row * c_rs in

                for j = 0 to nc' - 1 do
                  let col = jc + j in
                  let rec scalar p acc =
                    if p = kc' then acc
                    else
                      let kk = pc + p in
                      let a = Array.unsafe_get a_buf (arow + kk) in
                      let b = Array.unsafe_get bp (p * nc' + j) in
                      scalar (p + 1) (Float_u.fma a b acc)
                  in
                  let acc = scalar 0 (Array.unsafe_get c_buf (crow + col))
                  in
                  Array.unsafe_set c_buf (crow + col) acc
                done
              end;

              ic_loop (ic + mc')
          in
          ic_loop start_idx;
          pc_loop (pc + kc')
      in
      pc_loop 0;
      jc_loop (jc + nc')
  in
  jc_loop 0

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

    if batch_sz <> 1 then begin
      Shape.unravel_index_into batch batch_shape out_idx0;
      if has_row1 then Shape.unravel_index_into batch batch_shape out_idx1
    end;

    Shape.broadcast_index_into out_idx0 (shape va) a_idx0;
    Shape.broadcast_index_into out_idx0 (shape vb) b_idx;
    if has_row1 then Shape.broadcast_index_into out_idx0 (shape va) a_idx1;

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

      if has_row1 then out_idx1.(nd_out - 1) <- !j;


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
            Float64x2.Array.unsafe_get b_buf
              ~idx:(View.offset vb + Shape.ravel_index b_idx b_str)
          in
          let a0v = Float64x2.set1 av0 in
          kloop_r0 (l + 1) (Float64x2.add (Float64x2.mul a0v bv) acc0)
        end
      in
      let rec kloop_r1 l acc1=
        if l = k then acc1
        else begin
          a_idx1.(nd_a - 1) <- l;
          b_idx.(nd_b - 2) <- l;
          let bv =
            Float64x2.Array.unsafe_get b_buf
              ~idx:(View.offset vb + Shape.ravel_index b_idx b_str)
          in
          let av1 =
            Array.unsafe_get a_buf
              (View.offset va + Shape.ravel_index a_idx1 a_str)
          in
          let a1v = Float64x2.set1 av1 in
          kloop_r1 (l + 1) (Float64x2.add (Float64x2.mul a1v bv) acc1)
        end
      in
      let acc0 = kloop_r0 0 (Float64x2.set1 #0.0) in
      let out_off0 =
        View.offset vout + Shape.ravel_index out_idx0 out_str
      in
      Float64x2.Array.unsafe_set c_buf ~idx:out_off0 acc0;

      if has_row1 then begin
        let acc1 = kloop_r1 0 (Float64x2.set1 #0.0) in
        let out_off1 =
          View.offset vout + Shape.ravel_index out_idx1 out_str
        in
        Float64x2.Array.unsafe_set c_buf ~idx:out_off1 acc1
      end;

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

      if has_row1 then begin
        out_idx1.(nd_out - 1) <- !j;
        let sum1 = scalar 0 #0.0 in
        let out1 =
          View.offset vout + Shape.ravel_index out_idx1 out_str
        in
        Array.unsafe_set c_buf out1 sum1
      end;
      j := !j + 1
    done;

    work := !work + (if has_row1 then 2 else 1)
  done

let matmul_float32_fast a_buf b_buf c_buf va vb vout start_idx end_idx =
  let mc = 128 in
  let nc = 128 in
  let kc = 64 in

  let rank = Array.length (shape vout) in
  let n = (shape vout).(rank - 1) in
  let k = (shape va).(rank - 1) in

  let a_rs = k and b_rs = n and c_rs = n in
  let a0 = View.offset va
  and b0 = View.offset vb
  and c0 = View.offset vout in

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
              let i_end = ic + mc' in
              let i = ref ic in

              (* ===== 2×2 SIMD ROW LOOP ===== *)
              while !i + 1 < i_end do
                let i0 = !i in
                let i1 = i0 + 1 in

                let row0 = a0 + i0 * a_rs in
                let row1 = a0 + i1 * a_rs in
                let crow0 = c0 + i0 * c_rs in
                let crow1 = c0 + i1 * c_rs in

                let j = ref 0 in
                let j8 = nc'-7 in
                while !j < j8 do
                  let col = jc + !j in

                  let c_idx0 = crow0 + col in
                  let c_idx1 = crow1 + col in

                  let acc0 =
                    if pc = 0 then Float32x4.set1 #0.0s
                    else Float32x4.Array.unsafe_get c_buf ~idx:c_idx0
                  in
                  let acc1 =
                    if pc = 0 then Float32x4.set1 #0.0s
                    else Float32x4.Array.unsafe_get c_buf ~idx:c_idx1
                  in

                  let rec kloop_r0 p acc0 =
                    if p = kc' then acc0
                    else
                      let kk = pc + p in
                      let a0v =
                        Float32x4.set1 (Array.unsafe_get a_buf (row0 + kk))
                      in
                      let bv =
                        Float32x4.Array.unsafe_get b_buf
                          ~idx:(b0 + kk * b_rs + col)
                      in
                      kloop_r0
                        (p + 1)
                        (Float32x4.mul_add a0v bv acc0)
                    in
                  let rec kloop_r1 p acc1 =
                    if p = kc' then acc1
                    else
                      let kk = pc + p in
                      let a1v =
                        Float32x4.set1 (Array.unsafe_get a_buf (row1 + kk))
                      in
                      let bv =
                        Float32x4.Array.unsafe_get b_buf
                          ~idx:(b0 + kk * b_rs + col)
                      in
                      kloop_r1
                        (p + 1)
                        (Float32x4.mul_add a1v bv acc1)
                  in

                  let acc0 = kloop_r0 0 acc0 in
                  let acc1 = kloop_r1 0 acc1 in
                  Float32x4.Array.unsafe_set c_buf ~idx:c_idx0 acc0;
                  Float32x4.Array.unsafe_set c_buf ~idx:c_idx1 acc1;

                  j := !j + 8
                done;
                let j4 = nc'-3 in
                while !j < j4 do
                  let col = jc + !j in

                  let c_idx0 = crow0 + col in
                  let c_idx1 = crow1 + col in

                  let acc0 =
                    if pc = 0 then Float32x4.set1 #0.0s
                    else Float32x4.Array.unsafe_get c_buf ~idx:c_idx0
                  in
                  let acc1 =
                    if pc = 0 then Float32x4.set1 #0.0s
                    else Float32x4.Array.unsafe_get c_buf ~idx:c_idx1
                  in

                  let rec kloop_r0 p acc0 =
                    if p = kc' then acc0
                    else
                      let kk = pc + p in
                      let a0v =
                        Float32x4.set1 (Array.unsafe_get a_buf (row0 + kk))
                      in
                      let bv =
                        Float32x4.Array.unsafe_get b_buf
                          ~idx:(b0 + kk * b_rs + col)
                      in
                      kloop_r0
                        (p + 1)
                        (Float32x4.add (Float32x4.mul a0v bv) acc0)
                    in
                  let rec kloop_r1 p acc1 =
                    if p = kc' then acc1
                    else
                      let kk = pc + p in
                      let a1v =
                        Float32x4.set1 (Array.unsafe_get a_buf (row1 + kk))
                      in
                      let bv =
                        Float32x4.Array.unsafe_get b_buf
                          ~idx:(b0 + kk * b_rs + col)
                      in
                      kloop_r1
                        (p + 1)
                        (Float32x4.add (Float32x4.mul a1v bv) acc1)
                  in

                  let acc0 = kloop_r0 0 acc0 in
                  let acc1 = kloop_r1 0 acc1 in
                  Float32x4.Array.unsafe_set c_buf ~idx:c_idx0 acc0;
                  Float32x4.Array.unsafe_set c_buf ~idx:c_idx1 acc1;

                  j := !j + 4
                done;

                while !j < nc' do
                  let col = jc + !j in
                  let rec scalar_r0 p acc0  =
                    if p = kc' then acc0
                    else
                      let kk = pc + p in
                      let a0 = Array.unsafe_get a_buf (row0 + kk) in
                      let b  = Array.unsafe_get b_buf (b0 + kk * b_rs + col) in
                      scalar_r0
                        (p + 1)
                        (Float32_u.fma a0 b acc0)
                  in
                  let rec scalar_r1 p acc1 =
                    if p = kc' then acc1
                    else
                      let kk = pc + p in
                      let a1 = Array.unsafe_get a_buf (row1 + kk) in
                      let b  = Array.unsafe_get b_buf (b0 + kk * b_rs + col) in
                      scalar_r1
                        (p + 1)
                        (Float32_u.fma a1 b acc1)
                  in
                  if pc = 0 then 
                    let acc0 = scalar_r0 0 #0.0s in
                    let acc1 = scalar_r1 0 #0.0s 
                  in
                  Array.unsafe_set c_buf (crow0 + col) acc0;
                  Array.unsafe_set c_buf (crow1 + col) acc1
                  else
                    let acc0 = scalar_r0 0
                        (Array.unsafe_get c_buf (crow0 + col)) in
                    let acc1 = scalar_r1 0
                        (Array.unsafe_get c_buf (crow1 + col))
                      in
                      Array.unsafe_set c_buf (crow0 + col) acc0;
                      Array.unsafe_set c_buf (crow1 + col) acc1;
                    j := !j + 1
                    done;
                i := !i + 2
              done;

              if !i < i_end then begin
                let row = !i in
                let arow = a0 + row * a_rs in
                let crow = c0 + row * c_rs in

                for j = 0 to nc' - 1 do
                  let col = jc + j in
                  let rec scalar p acc =
                    if p = kc' then acc
                    else
                      let kk = pc + p in
                      let a = Array.unsafe_get a_buf (arow + kk) in
                      let b = Array.unsafe_get b_buf (b0 + kk * b_rs + col) in
                      scalar (p + 1) (Float32_u.fma a b acc)
                  in
                  let acc =
                    if pc = 0 then scalar 0 #0.0s
                    else scalar 0 (Array.unsafe_get c_buf (crow + col))
                  in
                  Array.unsafe_set c_buf (crow + col) acc
                done
              end;

              ic_loop (ic + mc')
          in
          ic_loop start_idx;
          pc_loop (pc + kc')
      in
      pc_loop 0;
      jc_loop (jc + nc')
  in
  jc_loop 0

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

    if batch_sz <> 1 then begin
      Shape.unravel_index_into batch batch_shape out_idx0;
      if has_row1 then Shape.unravel_index_into batch batch_shape out_idx1
    end;

    Shape.broadcast_index_into out_idx0 (shape va) a_idx0;
    Shape.broadcast_index_into out_idx0 (shape vb) b_idx;
    if has_row1 then Shape.broadcast_index_into out_idx0 (shape va) a_idx1;

    out_idx0.(nd_out - 2) <- i0;
    a_idx0.(nd_a - 2) <- i0;

    if has_row1 then begin
      out_idx1.(nd_out - 2) <- i0 + 1;
      a_idx1.(nd_a - 2) <- i0 + 1
    end;

    let j = ref 0 in
    while !j < n - 3 do
      out_idx0.(nd_out - 1) <- !j;
      b_idx.(nd_b - 1) <- !j;

      if has_row1 then out_idx1.(nd_out - 1) <- !j;


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
          let a0v = Float32x4.set1 av0 in
          kloop_r0 (l + 1) (Float32x4.add (Float32x4.mul a0v bv) acc0)
        end
      in
        let rec kloop_r1 l acc1=
          if l = k then acc1
          else begin
            a_idx1.(nd_a - 1) <- l;
            b_idx.(nd_b - 2) <- l;
            let bv =
              Float32x4.Array.unsafe_get b_buf
                ~idx:(View.offset vb + Shape.ravel_index b_idx b_str)
            in
            let av1 =
              Array.unsafe_get a_buf
                (View.offset va + Shape.ravel_index a_idx1 a_str)
            in
            let a1v = Float32x4.set1 av1 in
            kloop_r1 (l + 1) (Float32x4.add (Float32x4.mul a1v bv) acc1)
          end
      in
      let acc0 = kloop_r0 0 (Float32x4.set1 #0.0s) in
      let out_off0 =
        View.offset vout + Shape.ravel_index out_idx0 out_str
      in
      Float32x4.Array.unsafe_set c_buf ~idx:out_off0 acc0;

      if has_row1 then begin
        let acc1 = kloop_r1 0 (Float32x4.set1 #0.0s) in
        let out_off1 =
          View.offset vout + Shape.ravel_index out_idx1 out_str
        in
        Float32x4.Array.unsafe_set c_buf ~idx:out_off1 acc1
      end;

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

      if has_row1 then begin
        out_idx1.(nd_out - 1) <- !j;
        let sum1 = scalar 0 #0.0s in
        let out1 =
          View.offset vout + Shape.ravel_index out_idx1 out_str
        in
        Array.unsafe_set c_buf out1 sum1
      end;
      j := !j + 1
    done;

    work := !work + (if has_row1 then 2 else 1)
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