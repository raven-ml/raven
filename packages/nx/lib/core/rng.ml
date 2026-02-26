(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Keys *)

type key = int

let key seed = Stdlib.abs seed land 0x7FFFFFFF

let hash_int x =
  let open Int32 in
  let x = of_int x in
  let x = logxor x (shift_right_logical x 16) in
  let x = mul x 0x85ebca6bl in
  let x = logxor x (shift_right_logical x 13) in
  let x = mul x 0xc2b2ae35l in
  let x = logxor x (shift_right_logical x 16) in
  to_int (logand x 0x7FFFFFFFl)

let split ?(n = 2) k = Array.init n (fun i -> hash_int ((k * (n + 1)) + i + 1))
let fold_in k data = hash_int (k lxor data)
let to_int k = k

(* Implicit key management *)

type _ Effect.t += E_next_key : key Effect.t

let make_handler root =
  let counter = ref 0 in
  let open Effect.Deep in
  {
    retc = Fun.id;
    exnc = raise;
    effc =
      (fun (type a) (eff : a Effect.t) ->
        match eff with
        | E_next_key ->
            Some
              (fun (k : (a, _) continuation) ->
                let i = !counter in
                incr counter;
                continue k (fold_in root i))
        | _ -> None);
  }

let run ~seed f = Effect.Deep.match_with f () (make_handler (key seed))
let with_key k f = Effect.Deep.match_with f () (make_handler k)
let fallback_key = Domain.DLS.new_key (fun () -> ref (key (Random.bits ())))

let next_key () =
  try Effect.perform E_next_key
  with Effect.Unhandled _ ->
    let state = Domain.DLS.get fallback_key in
    let keys = split !state in
    state := keys.(0);
    keys.(1)
