(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { mutable values : int array; mutable count : int }

let create ?(capacity = 64) () =
  { values = Array.make (max 4 capacity) 0; count = 0 }

let clear t = t.count <- 0
let length t = t.count
let[@inline] get t i = Array.unsafe_get t.values i
let to_array t = Array.sub t.values 0 t.count

let grow t needed =
  let cap = ref (Array.length t.values * 2) in
  while !cap < needed do
    cap := !cap * 2
  done;
  let a = Array.make !cap 0 in
  Array.blit t.values 0 a 0 t.count;
  t.values <- a

let[@inline] ensure t extra =
  if t.count + extra > Array.length t.values then grow t (t.count + extra)

let truncate t n = t.count <- n

let[@inline] add t n =
  ensure t 1;
  Array.unsafe_set t.values t.count n;
  t.count <- t.count + 1

let[@inline] add4 t a b c d ~count =
  ensure t 4;
  let n = t.count in
  let values = t.values in
  Array.unsafe_set values n a;
  Array.unsafe_set values (n + 1) b;
  Array.unsafe_set values (n + 2) c;
  Array.unsafe_set values (n + 3) d;
  t.count <- n + count
