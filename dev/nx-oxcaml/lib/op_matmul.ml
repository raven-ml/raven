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

                let j_simd_end = nc' land lnot 1 in
                let j = ref 0 in
                while !j < j_simd_end do
                  let col = jc + !j in

                  let c_idx0 = crow0 + col in
                  let c_idx1 = crow1 + col in

                  let acc0 =
                    if pc = 0 then Float64x2.set1 #0.0
                    else Float64x2.Array.unsafe_get c_buf ~idx:c_idx0
                  in
                  let acc1 =
                    if pc = 0 then Float64x2.set1 #0.0
                    else Float64x2.Array.unsafe_get c_buf ~idx:c_idx1
                  in

                  let rec kloop_r0 p acc0 =
                    if p = kc' then acc0
                    else
                      let kk = pc + p in
                      let a0v =
                        Float64x2.set1 (Array.unsafe_get a_buf (row0 + kk))
                      in
                      let bv =
                        Float64x2.Array.unsafe_get b_buf
                          ~idx:(b0 + kk * b_rs + col)
                      in
                      kloop_r0
                        (p + 1)
                        (Float64x2.add (Float64x2.mul a0v bv) acc0)
                    in
                  let rec kloop_r1 p acc1 =
                    if p = kc' then acc1
                    else
                      let kk = pc + p in
                      let a1v =
                        Float64x2.set1 (Array.unsafe_get a_buf (row1 + kk))
                      in
                      let bv =
                        Float64x2.Array.unsafe_get b_buf
                          ~idx:(b0 + kk * b_rs + col)
                      in
                      kloop_r1
                        (p + 1)
                        (Float64x2.add (Float64x2.mul a1v bv) acc1)
                  in

                  let acc0 = kloop_r0 0 acc0 in
                  let acc1 = kloop_r1 0 acc1 in
                  Float64x2.Array.unsafe_set c_buf ~idx:c_idx0 acc0;
                  Float64x2.Array.unsafe_set c_buf ~idx:c_idx1 acc1;

                  j := !j + 2
                done;

                (* ===== ODD COLUMN CLEANUP (2 ROWS) ===== *)
                if (nc' land 1) <> 0 then begin
                  let col = jc + nc' - 1 in

                  let rec scalar_r0 p acc0  =
                    if p = kc' then acc0
                    else
                      let kk = pc + p in
                      let a0 = Array.unsafe_get a_buf (row0 + kk) in
                      let b  = Array.unsafe_get b_buf (b0 + kk * b_rs + col) in
                      scalar_r0
                        (p + 1)
                        (Float_u.fma a0 b acc0)
                  in
                  let rec scalar_r1 p acc1 =
                    if p = kc' then acc1
                    else
                      let kk = pc + p in
                      let a1 = Array.unsafe_get a_buf (row1 + kk) in
                      let b  = Array.unsafe_get b_buf (b0 + kk * b_rs + col) in
                      scalar_r1
                        (p + 1)
                        (Float_u.fma a1 b acc1)
                  in
                  if pc = 0 then 
                    let acc0 = scalar_r0 0 #0.0 in
                    let acc1 = scalar_r1 0 #0.0 
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
                      Array.unsafe_set c_buf (crow1 + col) acc1


                end;

                i := !i + 2
              done;

              (* ===== ODD ROW CLEANUP ===== *)
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
                      scalar (p + 1) (Float_u.fma a b acc)
                  in
                  let acc =
                    if pc = 0 then scalar 0 #0.0
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

let matmul_float64_slow a_buf b_buf c_buf va vb vout start_idx end_idx = 
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
      
          loop (l + 1) (Float_u.fma av bv acc)
        )
      in
      let sum = loop 0 #0.0 in
      
      let out_off = View.offset vout + Shape.ravel_index out_idx out_str in
      Array.unsafe_set c_buf out_off sum
    done
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
  
                for i = ic to ic + mc' - 1 do
                  let a_row = a0 + (i * a_rs) + pc in
                  let c_row = c0 + (i * c_rs) + jc in
  
                  (* SIMD over j, 2 columns at a time *)
                  let j_simd_end = nc' - (nc' land 1) in
                  let j = ref 0 in
                  while !j < j_simd_end do
                    let a_idx0 = a_row in
                    let b_idx0 = b0 + (pc * b_rs) + (jc + !j) in
                    let c_idx = c_row + !j in
  
                    let acc =
                      if pc = 0 then Float32x4.set1 #0.0s
                      else Float32x4.Array.unsafe_get c_buf ~idx:c_idx
                    in
  
                    let rec loop p a_idx b_idx acc =
                      if p = kc' then acc
                      else
                        let av = Array.unsafe_get a_buf a_idx in
                        let a_v =Float32x4.set1 av in
                        let b_v =Float32x4.Array.unsafe_get b_buf ~idx:b_idx in
                        loop
                          (p + 1)
                          (a_idx + 1)
                          (b_idx + b_rs)
                          (Float32x4.add (Float32x4.mul a_v b_v) acc)
                    in
  
                    let acc = loop 0 a_idx0 b_idx0 acc in
                    Float32x4.Array.unsafe_set c_buf ~idx:c_idx acc;
  
                    j := !j + 4
                  done;
  
                  (* scalar cleanup for odd column *)
                  if (nc' land 1) <> 0 then begin
                    let j = nc' - 1 in
                    let a_idx0 = a_row in
                    let b_idx0 = b0 + (pc * b_rs) + (jc + j) in
                    let c_idx = c_row + j in
  
                    let rec loop p a_idx b_idx acc =
                      if p = kc' then acc
                      else
                        let av = Array.unsafe_get a_buf a_idx in
                        let bv = Array.unsafe_get b_buf b_idx in
                        loop
                          (p + 1)
                          (a_idx + 1)
                          (b_idx + b_rs)
                          (Float32_u.fma av bv acc)
                    in
  
                    let partial = loop 0 a_idx0 b_idx0 (#0.0s) in
                    let acc =
                      if pc = 0 then partial
                      else Float32_u.add (Array.unsafe_get c_buf c_idx) partial
                    in
                    Array.unsafe_set c_buf c_idx acc
                  end
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
  
  let matmul_float32_slow a_buf b_buf c_buf va vb vout start_idx end_idx = 
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
        
            loop (l + 1) (Float32_u.fma av bv acc)
          )
        in
        let sum = loop 0 #0.0s in
        
        let out_off = View.offset vout + Shape.ravel_index out_idx out_str in
        Array.unsafe_set c_buf out_off sum
      done
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