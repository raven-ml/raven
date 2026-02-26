(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let threefry_parity = Int32_u.of_int32 0x1BD11BDAl
let s1 = Int32_u.of_int32 1l
let s2 = Int32_u.of_int32 2l
let s3 = Int32_u.of_int32 3l
let s4 = Int32_u.of_int32 4l
let s5 = Int32_u.of_int32 5l

let[@inline] rotl32 x r =
  let xi = Int32_u.to_int32 x in
  Int32_u.of_int32
    (Int32.logor (Int32.shift_left xi r) (Int32.shift_right_logical xi (32 - r)))

let[@inline] mix x0 x1 rot k =
  let s = Int32_u.add x0 x1 in
  k s (Int32_u.logxor (rotl32 x1 rot) s)

let[@inline] threefry2x32 ks0 ks1 ks2 c0 c1 k =
  let rec round r x0 x1 =
    if r = 20 then k x0 x1
    else
      let rot =
        match r land 3 with
        | 0 -> 13
        | 1 -> 15
        | 2 -> 26
        | _ -> 6
      in
      mix x0 x1 rot (fun x0 x1 ->
          let r' = r + 1 in
          if r' mod 4 = 0 then
            (match r' / 4 with
            | 1 -> round r' (Int32_u.add x0 ks0) (Int32_u.add x1 (Int32_u.add ks1 s1))
            | 2 -> round r' (Int32_u.add x0 ks1) (Int32_u.add x1 (Int32_u.add ks2 s2))
            | 3 -> round r' (Int32_u.add x0 ks2) (Int32_u.add x1 (Int32_u.add ks0 s3))
            | 4 -> round r' (Int32_u.add x0 ks0) (Int32_u.add x1 (Int32_u.add ks1 s4))
            | _ -> round r' (Int32_u.add x0 ks1) (Int32_u.add x1 (Int32_u.add ks2 s5)))
          else round r' x0 x1)
  in
  round 0 (Int32_u.add c0 ks0) (Int32_u.add c1 ks1)

let[@inline] lane1 v = Int32x4.low_to (Int32x4.dup_lane 1 v)
let[@inline] lane2 v = Int32x4.low_to (Int32x4.dup_lane 2 v)
let[@inline] lane3 v = Int32x4.low_to (Int32x4.dup_lane 3 v)

let[@inline] threefry_pair ~(key_arr : int32# array) ~(ctr_arr : int32# array)
    ~(out_arr : int32# array) ~kb ~cb ~ob ~kl ~cl ~ol =
  let ks0 = Array.unsafe_get key_arr kb in
  let ks1 = Array.unsafe_get key_arr (kb + kl) in
  let ks2 = Int32_u.logxor threefry_parity (Int32_u.logxor ks0 ks1) in
  let c0 = Array.unsafe_get ctr_arr cb in
  let c1 = Array.unsafe_get ctr_arr (cb + cl) in
  threefry2x32 ks0 ks1 ks2 c0 c1 (fun r0 r1 ->
      Array.unsafe_set out_arr ob r0;
      Array.unsafe_set out_arr (ob + ol) r1)

let threefry_int32 pool ~(out_arr : int32# array) ~(key_arr : int32# array)
    ~(ctr_arr : int32# array) ~shape ~key_view ~ctr_view ~out_view =
  let rank = Array.length shape in
  let last_dim = rank - 1 in
  let total_vectors =
    let p = ref 1 in
    for i = 0 to last_dim - 1 do
      p := !p * shape.(i)
    done;
    !p
  in
  if total_vectors = 0 then ()
  else
    let key_strides = View.strides key_view in
    let ctr_strides = View.strides ctr_view in
    let out_strides = View.strides out_view in
    let key_offset = View.offset key_view in
    let ctr_offset = View.offset ctr_view in
    let out_offset = View.offset out_view in
    let contiguous =
      View.is_c_contiguous key_view
      && View.is_c_contiguous ctr_view
      && View.is_c_contiguous out_view
    in
    let process_chunk start_idx end_idx =
      if contiguous then (
        let kb = ref (key_offset + (start_idx lsl 1)) in
        let cb = ref (ctr_offset + (start_idx lsl 1)) in
        let ob = ref (out_offset + (start_idx lsl 1)) in
        let stop = key_offset + (end_idx lsl 1) in
        let stop_simd = stop - (((stop - !kb) land 3)) in
        while !kb < stop_simd do
          let key_v = Int32x4.Array.unsafe_get key_arr ~idx:!kb in
          let ctr_v = Int32x4.Array.unsafe_get ctr_arr ~idx:!cb in
          let k0a = Int32x4.low_to key_v in
          let k1a = lane1 key_v in
          let k0b = lane2 key_v in
          let k1b = lane3 key_v in
          let c0a = Int32x4.low_to ctr_v in
          let c1a = lane1 ctr_v in
          let c0b = lane2 ctr_v in
          let c1b = lane3 ctr_v in
          let ks2a = Int32_u.logxor threefry_parity (Int32_u.logxor k0a k1a) in
          let ks2b = Int32_u.logxor threefry_parity (Int32_u.logxor k0b k1b) in
          threefry2x32 k0a k1a ks2a c0a c1a (fun r0a r1a ->
              threefry2x32 k0b k1b ks2b c0b c1b (fun r0b r1b ->
                  let out_v = Int32x4.set r0a r1a r0b r1b in
                  Int32x4.Array.unsafe_set out_arr ~idx:!ob out_v));
          kb := !kb + 4;
          cb := !cb + 4;
          ob := !ob + 4
        done;
        while !kb < stop do
          threefry_pair ~key_arr ~ctr_arr ~out_arr ~kb:!kb ~cb:!cb ~ob:!ob ~kl:1
            ~cl:1 ~ol:1;
          kb := !kb + 2;
          cb := !cb + 2;
          ob := !ob + 2
        done)
      else (
        let key_last = key_strides.(last_dim) in
        let ctr_last = ctr_strides.(last_dim) in
        let out_last = out_strides.(last_dim) in
        let slice_rank = rank - 1 in
        let dims = Array.make slice_rank 0 in
        let key_str = Array.make slice_rank 0 in
        let ctr_str = Array.make slice_rank 0 in
        let out_str = Array.make slice_rank 0 in
        let j = ref 0 in
        for d = 0 to rank - 1 do
          if d <> last_dim then (
            dims.(!j) <- shape.(d);
            key_str.(!j) <- key_strides.(d);
            ctr_str.(!j) <- ctr_strides.(d);
            out_str.(!j) <- out_strides.(d);
            incr j)
        done;
        let coords = Array.make slice_rank 0 in
        let kb = ref key_offset in
        let cb = ref ctr_offset in
        let ob = ref out_offset in
        let rem = ref start_idx in
        for d = 0 to slice_rank - 1 do
          let block = ref 1 in
          for d' = d + 1 to slice_rank - 1 do
            block := !block * dims.(d')
          done;
          let c = !rem / !block in
          rem := !rem mod !block;
          coords.(d) <- c;
          kb := !kb + (c * key_str.(d));
          cb := !cb + (c * ctr_str.(d));
          ob := !ob + (c * out_str.(d))
        done;
        let rec carry d =
          if d >= 0 then
            let next = coords.(d) + 1 in
            if next < dims.(d) then (
              coords.(d) <- next;
              kb := !kb + key_str.(d);
              cb := !cb + ctr_str.(d);
              ob := !ob + out_str.(d))
            else (
              coords.(d) <- 0;
              kb := !kb - ((dims.(d) - 1) * key_str.(d));
              cb := !cb - ((dims.(d) - 1) * ctr_str.(d));
              ob := !ob - ((dims.(d) - 1) * out_str.(d));
              carry (d - 1))
        in
        for _ = start_idx to end_idx - 1 do
          threefry_pair ~key_arr ~ctr_arr ~out_arr ~kb:!kb ~cb:!cb ~ob:!ob
            ~kl:key_last ~cl:ctr_last ~ol:out_last;
          carry (slice_rank - 1)
        done)
    in
    let parallel_threshold = 62500 in
    if total_vectors > parallel_threshold then
      Parallel.parallel_for pool 0 (total_vectors - 1) process_chunk
    else process_chunk 0 total_vectors
