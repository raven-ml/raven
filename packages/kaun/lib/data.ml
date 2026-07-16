(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let check_batch_size op batch_size =
  if batch_size <= 0 then
    invalid_argf "Data.%s: batch_size must be positive, got %d" op batch_size

let examples op t =
  match Nx.shape t with
  | [||] -> invalid_argf "Data.%s: input must not be a scalar" op
  | shape -> shape.(0)

(* [cut perm t start stop] is examples [start, stop) of [t]: a view without a
   permutation, a gathered copy with one. *)
let cut perm t start stop =
  match perm with
  | None -> Nx.slice [ Nx.R (start, stop) ] t
  | Some perm ->
      Nx.take ~axis:0 ~indices:(Nx.slice [ Nx.R (start, stop) ] perm) t

(* Each traversal forces the outer thunk once, drawing one permutation per epoch
   from the ambient RNG scope; the recursion threads it along. *)
let batch_seq ~shuffle ~drop_last ~batch_size ~n slice =
  let rec from perm start () =
    if start >= n || (drop_last && start + batch_size > n) then Seq.Nil
    else
      let stop = min n (start + batch_size) in
      Seq.Cons (slice perm start stop, from perm stop)
  in
  fun () ->
    if n = 0 then Seq.Nil
    else
      let perm = if shuffle then Some (Nx.permutation n) else None in
      from perm 0 ()

let batches ?(shuffle = false) ?(drop_last = false) ~batch_size x =
  check_batch_size "batches" batch_size;
  let n = examples "batches" x in
  batch_seq ~shuffle ~drop_last ~batch_size ~n (fun perm start stop ->
      cut perm x start stop)

let batches2 ?(shuffle = false) ?(drop_last = false) ~batch_size (x, y) =
  check_batch_size "batches2" batch_size;
  let n = examples "batches2" x in
  let ny = examples "batches2" y in
  if n <> ny then
    invalid_argf "Data.batches2: x has %d examples but y has %d" n ny;
  batch_seq ~shuffle ~drop_last ~batch_size ~n (fun perm start stop ->
      (cut perm x start stop, cut perm y start stop))
