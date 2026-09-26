(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* On the CPU device a compiled function reads and writes host memory in place,
   at whatever address it has: a slice of a larger buffer, the pages of a mapped
   file. Its kernels are vectorised, so the C they are compiled from must not
   declare its vector types aligned. Where the hardware enforces a declared
   alignment, a wrong declaration kills the process. *)

open Windtrap
open Rune_test_support.Support
module U = Tolk_uop.Uop

let n = 4096

(* A contiguous tensor of [n] elements whose data starts one element into its
   buffer: an address that is 4 modulo 16 when the buffer itself is aligned to
   16 bytes, as allocated memory is. *)
let offset_by_one f =
  let whole = Nx.create f32 [| n + 1 |] (Array.init (n + 1) f) in
  let t = Nx.slice [ Nx.R (1, n + 1) ] whole in
  is_true ~msg:"contiguous" (Nx.is_c_contiguous t);
  equal ~msg:"offset" int 1 (Nx.offset t);
  let address =
    Nativeint.add
      (Nx_buffer.unsafe_data_ptr (Nx.to_buffer t))
      (Nativeint.of_int (Nx.offset t * 4))
  in
  equal ~msg:"address modulo 16" nativeint 4n (Nativeint.rem address 16n);
  t

let test_offset_capture () =
  let w = offset_by_one float_of_int in
  let x =
    Nx.create f32 [| n |] (Array.init n (fun i -> float_of_int (2 * i)))
  in
  let f x = Nx.add (Nx.mul x w) w in
  check_arr ~eps:0. ~msg:"offset capture"
    (to_arr (f x))
    (Rune.jit' ~devices:[ Rune.device "CPU" ] f x)

let test_offset_input () =
  let w = Nx.create f32 [| n |] (Array.init n float_of_int) in
  let x = offset_by_one (fun i -> float_of_int (2 * i)) in
  let f x = Nx.add (Nx.mul x w) w in
  check_arr ~eps:0. ~msg:"offset input"
    (to_arr (f x))
    (Rune.jit' ~devices:[ Rune.device "CPU" ] f x)

(* A load and a store of four float32 lanes. *)
let vector_access () =
  let param slot =
    U.param ~slot ~dtype:Tolk_uop.Dtype.float32 ~shape:(U.const_int 4)
      ~addrspace:Tolk_uop.Dtype.Global ()
  in
  let window p =
    U.shrink ~src:p ~offset:(U.const_int 0) ~size:(U.const_int 4)
  in
  let p0 = param 0 and p1 = param 1 in
  let src = window p0 and dst = window p1 in
  let value = U.load ~src () in
  [ p0; p1; src; dst; value; U.store ~dst ~value () ]

(* The alignments the C source [source] declares. *)
let declared_alignments source =
  let marker = "aligned(" in
  let m = String.length marker in
  let rec scan i acc =
    if i + m > String.length source then List.rev acc
    else if String.sub source i m = marker then
      let stop = String.index_from source (i + m) ')' in
      let alignment =
        int_of_string (String.sub source (i + m) (stop - i - m))
      in
      scan stop (alignment :: acc)
    else scan (i + 1) acc
  in
  scan 0 []

let test_vector_types_unaligned () =
  ignore
    (Rune.jit'
       ~devices:[ Rune.device "CPU" ]
       (fun x -> Nx.neg x)
       (vec32 [| 1.0 |]));
  let renderer = Tolk.Device.renderer (Tolk.Device.get "CPU") in
  let source = Tolk.Renderer.render renderer (vector_access ()) in
  equal ~msg:"declared alignments" (list int) [ 1 ] (declared_alignments source)

let () =
  exit
    (run "rune jit alignment"
       [
         group "host memory in place"
           [
             test "a capture at an address that is 4 modulo 16"
               test_offset_capture;
             test "an input at an address that is 4 modulo 16" test_offset_input;
             test "the CPU device declares no vector alignment"
               test_vector_types_unaligned;
           ];
       ])
