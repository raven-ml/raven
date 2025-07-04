open Nx_core.Dtype
module Shape = Nx_core.Shape
open Internal

(* Threefry 2x32 Core Implementation *)
module Threefry_impl = struct
  let ks_parity_32 = 0x1BD11BDA_l
  let r_2x32 = [| 13; 15; 26; 6; 17; 29; 16; 24 |]

  let rotl32 x n =
    let n = n land 31 in
    Int32.(logor (shift_left x n) (shift_right_logical x (32 - n)))

  let threefry2x32_20_rounds (c0 : int32) (c1 : int32) (k0 : int32) (k1 : int32)
      : int32 * int32 =
    let x0 = ref c0 in
    let x1 = ref c1 in
    let keys = [| k0; k1; Int32.logxor ks_parity_32 (Int32.logxor k0 k1) |] in

    for r = 0 to 19 do
      if r mod 4 = 0 then (
        let s_div_4 = r / 4 in
        x0 := Int32.add !x0 keys.(s_div_4 mod 3);
        x1 := Int32.add !x1 keys.((s_div_4 + 1) mod 3);
        x1 := Int32.add !x1 (Int32.of_int s_div_4));
      x0 := Int32.add !x0 !x1;
      x1 := rotl32 !x1 r_2x32.(r mod 8);
      x1 := Int32.logxor !x1 !x0
    done;

    let s_div_4_final = 20 / 4 in
    x0 := Int32.add !x0 keys.(s_div_4_final mod 3);
    x1 := Int32.add !x1 keys.((s_div_4_final + 1) mod 3);
    x1 := Int32.add !x1 (Int32.of_int s_div_4_final);
    (!x0, !x1)
end

let kernel_threefry_int32 (data_t : (int32, int32_elt) t)
    (seed_t : (int32, int32_elt) t) (out_t : (int32, int32_elt) t) start_idx
    end_idx =
  let data_buf = buffer data_t in
  let seed_buf = buffer seed_t in
  let out_buf = buffer out_t in
  let c1_fixed = 0l in
  let k1_fixed = 0xCAFEBABEl in

  if is_c_contiguous data_t && is_c_contiguous seed_t then (
    let data_offset = offset data_t in
    let seed_offset = offset seed_t in
    let out_offset = offset out_t in

    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in

      let d_val0 = Bigarray.Array1.unsafe_get data_buf (data_offset + i0) in
      let s_val0 = Bigarray.Array1.unsafe_get seed_buf (seed_offset + i0) in
      let res0_0, _ =
        Threefry_impl.threefry2x32_20_rounds d_val0 c1_fixed s_val0 k1_fixed
      in
      Bigarray.Array1.unsafe_set out_buf (out_offset + i0) res0_0;

      let d_val1 = Bigarray.Array1.unsafe_get data_buf (data_offset + i1) in
      let s_val1 = Bigarray.Array1.unsafe_get seed_buf (seed_offset + i1) in
      let res0_1, _ =
        Threefry_impl.threefry2x32_20_rounds d_val1 c1_fixed s_val1 k1_fixed
      in
      Bigarray.Array1.unsafe_set out_buf (out_offset + i1) res0_1;

      let d_val2 = Bigarray.Array1.unsafe_get data_buf (data_offset + i2) in
      let s_val2 = Bigarray.Array1.unsafe_get seed_buf (seed_offset + i2) in
      let res0_2, _ =
        Threefry_impl.threefry2x32_20_rounds d_val2 c1_fixed s_val2 k1_fixed
      in
      Bigarray.Array1.unsafe_set out_buf (out_offset + i2) res0_2;

      let d_val3 = Bigarray.Array1.unsafe_get data_buf (data_offset + i3) in
      let s_val3 = Bigarray.Array1.unsafe_get seed_buf (seed_offset + i3) in
      let res0_3, _ =
        Threefry_impl.threefry2x32_20_rounds d_val3 c1_fixed s_val3 k1_fixed
      in
      Bigarray.Array1.unsafe_set out_buf (out_offset + i3) res0_3;

      i := !i + 4
    done;
    while !i < end_idx do
      let current_idx = !i in
      let d_val =
        Bigarray.Array1.unsafe_get data_buf (data_offset + current_idx)
      in
      let s_val =
        Bigarray.Array1.unsafe_get seed_buf (seed_offset + current_idx)
      in
      let res0, _ =
        Threefry_impl.threefry2x32_20_rounds d_val c1_fixed s_val k1_fixed
      in
      Bigarray.Array1.unsafe_set out_buf current_idx res0;
      incr i
    done)
  else
    let out_shape = shape out_t in
    let data_strides = strides data_t in
    let seed_strides = strides seed_t in
    let data_offset = offset data_t in
    let seed_offset = offset seed_t in

    (* Pre-allocate work array *)
    let md_index = Array.make (Array.length out_shape) 0 in

    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_index;

      let data_lin = Shape.ravel_index md_index data_strides in
      let seed_lin = Shape.ravel_index md_index seed_strides in

      let d_val =
        Bigarray.Array1.unsafe_get data_buf (data_offset + data_lin)
      in
      let s_val =
        Bigarray.Array1.unsafe_get seed_buf (seed_offset + seed_lin)
      in

      let res0, _ =
        Threefry_impl.threefry2x32_20_rounds d_val c1_fixed s_val k1_fixed
      in
      Bigarray.Array1.unsafe_set out_buf k res0
    done

let threefry (context : context) (data_t : (int32, int32_elt) t)
    (seed_t : (int32, int32_elt) t) (out_t : (int32, int32_elt) t) : unit =
  let size = size out_t in
  if size = 0 then ()
  else
    Parallel.parallel_for context.pool 0 (size - 1) (fun start_idx end_idx ->
        kernel_threefry_int32 data_t seed_t out_t start_idx end_idx)
