(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { mutable top : int; mutable peak : int }

let make () = { top = 0; peak = 0 }

let take a n =
  let at = (a.top + 255) land lnot 255 in
  a.top <- at + n;
  a.peak <- Int.max a.peak a.top;
  at

let mark a = a.top
let release a m = a.top <- m
let peak a = a.peak
