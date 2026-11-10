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

(* The batches of [n] examples, [slice start stop] each. *)
let batch_seq ~drop_last ~batch_size ~n slice =
  let rec from start () =
    if start >= n || (drop_last && start + batch_size > n) then Seq.Nil
    else
      let stop = min n (start + batch_size) in
      Seq.Cons (slice start stop, from stop)
  in
  from 0

let cut t start stop = Nx.slice [ Nx.R (start, stop) ] t

let batches ?(drop_last = false) ~batch_size x =
  check_batch_size "batches" batch_size;
  let n = examples "batches" x in
  batch_seq ~drop_last ~batch_size ~n (cut x)

let batches2 ?(drop_last = false) ~batch_size (x, y) =
  check_batch_size "batches2" batch_size;
  let n = examples "batches2" x in
  let ny = examples "batches2" y in
  if n <> ny then
    invalid_argf "Data.batches2: x has %d examples but y has %d" n ny;
  batch_seq ~drop_last ~batch_size ~n (fun start stop ->
      (cut x start stop, cut y start stop))
