(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Aion

let ints a = Nx.to_array a |> Array.map Int64.to_int
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

let encodes =
  prop "a code's bits are its numbers' signs, number 0 the most significant"
    Gen.(
      array ~size:(frequency [ (1, constant 10); (2, int_range 1 12) ]) number)
    (fun z ->
      cover "a number of 0, -0. or NaN"
        (Array.exists (fun v -> not (v > 0. || v < 0.)) z);
      cover "AION-1's 10 numbers" (Array.length z = 10);
      let k = Lfq.encode (Nx.create Nx.float32 [| 1; Array.length z |] z) in
      equal (array int) [| 1 |] (Nx.shape k);
      equal (array int) [| code z |] (ints k))

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

(* Compiled, encode and decode are comparisons, casts, integer arithmetic and
   selections, which Rune.jit computes exactly: bit for bit. *)
let encode_c = Rune.jit Nx.Ptree.(tensor @-> returns tensor) Lfq.encode

let decode_c =
  Rune.jit Nx.Ptree.(tensor @-> returns tensor) (Lfq.decode Nx.float32 ~bits:10)

let compiled =
  prop "compiled equals eager, bit for bit"
    Gen.(array ~size:(constant 40) number)
    (fun z ->
      let z = Nx.create Nx.float32 [| 4; 10 |] z in
      let k = Lfq.encode z in
      equal (array int) (ints k) (ints (encode_c z));
      let bits x = Array.map Int32.bits_of_float (Nx.to_array x) in
      equal (array int32)
        (bits (Lfq.decode Nx.float32 ~bits:10 k))
        (bits (decode_c k)))

let refusals =
  let refuses ?sub f = raises_match (Exn.invalid_arg ?sub) f in
  group "refusals"
    [
      cases "decode refuses a code outside [0, 1024), eagerly and compiled"
        ~id:string_of_int [ -1; 1024 ] (fun k ->
          let sub = Printf.sprintf "code %d at [1] outside [0, 1024)" k in
          refuses ~sub (fun () ->
              Lfq.decode Nx.float32 ~bits:10 (codes [| 0; k |]));
          refuses ~sub (fun () -> decode_c (codes [| 0; k |])));
      cases "decode refuses bits outside [0, 62]" ~id:string_of_int [ -1; 63 ]
        (fun bits ->
          refuses ~sub:"bits, [0, 62] expected" (fun () ->
              Lfq.decode Nx.float32 ~bits (codes [| 0 |])));
      cases "encode refuses no last axis or one past 62"
        ~id:(fun s -> string_of_int (Array.length s))
        [ [||]; [| 2; 63 |] ]
        (fun s -> refuses (fun () -> Lfq.encode (Nx.zeros Nx.float32 s)));
    ]

let () =
  exit
    (run "aion.lfq" [ encodes; group "decode" [ decodes ]; refusals; compiled ])
