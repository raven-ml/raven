(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Aion

let ints a = Nx.to_array a |> Array.map Int64.to_int
let vec a = Nx.create Nx.float32 [| 1; Array.length a |] a
let codes a = Nx.create Nx.int64 [| Array.length a |] (Array.map Int64.of_int a)

(* The code the preamble states: bit [d - 1 - c] set where number [c] is
   positive. *)
let code z =
  let d = Array.length z in
  Array.fold_left ( + ) 0
    (Array.mapi (fun c v -> if v > 0. then 1 lsl (d - 1 - c) else 0) z)

let number =
  Gen.(
    frequency
      [
        (6, map Bounds.f32 (float_range (-2.) 2.));
        (1, of_list [ 0.; -0.; Float.nan; Float.infinity; Float.neg_infinity ]);
      ])

let vectors = Gen.(array ~size:(int_range 1 12) number)

let encodes =
  group "encode"
    [
      cases "number c of 10 sets bit 9 - c where positive" ~id:string_of_int
        [ 0; 1; 5; 9 ] (fun c ->
          let z = Array.init 10 (fun i -> if i = c then 0.5 else -0.5) in
          equal (array int) [| 1 lsl (9 - c) |] (ints (Lfq.encode (vec z))));
      test "0, -0. and NaN clear their bit" (fun () ->
          let z = [| 0.; -0.; Float.nan; 1.; 1.; 1.; 1.; 1.; 1.; 1. |] in
          equal (array int) [| 127 |] (ints (Lfq.encode (vec z))));
      prop "a code's bits are its numbers' signs" vectors (fun z ->
          cover "a number of 0, -0. or NaN"
            (Array.exists (fun v -> not (v > 0. || v < 0.)) z);
          equal (array int) [| code z |] (ints (Lfq.encode (vec z))));
    ]

let decodes =
  prop "decode gives each bit's sign, which encode reads back"
    Gen.(
      let* bits = frequency [ (1, constant 0); (4, int_range 1 20) ] in
      let+ k = int_range 0 ((1 lsl bits) - 1) in
      (bits, k))
    (fun (bits, k) ->
      cover "no bit" (bits = 0);
      let v = Lfq.decode Nx.float32 ~bits (codes [| k |]) in
      let signs =
        Array.init bits (fun c ->
            if (k lsr (bits - 1 - c)) land 1 = 1 then 1. else -1.)
      in
      equal (array int) [| 1; bits |] (Nx.shape v);
      equal (array float_exact) signs (Nx.to_array v);
      equal (array int) [| k |] (ints (Lfq.encode v)))

let refusals =
  group "refusals"
    [
      cases "decode refuses a code outside [0, 1024)" ~id:string_of_int
        [ -1; 1024 ] (fun k ->
          raises_match
            (Exn.invalid_arg
               ~sub:(Printf.sprintf "code %d at [1] outside [0, 1024)" k))
            (fun () -> Lfq.decode Nx.float32 ~bits:10 (codes [| 0; k |])));
      cases "decode refuses bits outside [0, 62]" ~id:string_of_int [ -1; 63 ]
        (fun bits ->
          raises_match (Exn.invalid_arg ~sub:"bits, [0, 62] expected")
            (fun () -> Lfq.decode Nx.float32 ~bits (codes [| 0 |])));
      cases "encode refuses no last axis or one past 62"
        ~id:(fun s -> string_of_int (Array.length s))
        [ [||]; [| 2; 63 |] ]
        (fun s ->
          raises_match Exn.invalid_arg (fun () ->
              Lfq.encode (Nx.zeros Nx.float32 s)));
    ]

(* Compiled with Rune.jit, encode and decode are comparisons, casts, integer
   arithmetic and selections, which it computes exactly: bit for bit. *)
let encode_c = Rune.jit Nx.Ptree.(tensor @-> returns tensor) Lfq.encode

let decode_c =
  Rune.jit Nx.Ptree.(tensor @-> returns tensor) (Lfq.decode Nx.float32 ~bits:10)

let compiled =
  group "compiled"
    [
      prop "compiled equals eager, bit for bit"
        Gen.(array ~size:(constant 40) number)
        (fun z ->
          let z = Nx.create Nx.float32 [| 4; 10 |] z in
          let k = Lfq.encode z in
          equal (array int) (ints k) (ints (encode_c z));
          let bits x = Array.map Int32.bits_of_float (Nx.to_array x) in
          equal (array int32)
            (bits (Lfq.decode Nx.float32 ~bits:10 k))
            (bits (decode_c k)));
      test "compiled decode refuses a code at the call" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"code 1024 at [1] outside [0, 1024)")
            (fun () -> decode_c (codes [| 0; 1024 |])));
    ]

let () =
  exit
    (run "aion.lfq"
       [ encodes; group "decode" [ decodes ]; refusals; compiled ])
