(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* SHA-256, FIPS 180-4 §6.2, on 32-bit words held in OCaml's [int]. *)

let k =
  [|
    0x428a2f98; 0x71374491; 0xb5c0fbcf; 0xe9b5dba5; 0x3956c25b; 0x59f111f1;
    0x923f82a4; 0xab1c5ed5; 0xd807aa98; 0x12835b01; 0x243185be; 0x550c7dc3;
    0x72be5d74; 0x80deb1fe; 0x9bdc06a7; 0xc19bf174; 0xe49b69c1; 0xefbe4786;
    0x0fc19dc6; 0x240ca1cc; 0x2de92c6f; 0x4a7484aa; 0x5cb0a9dc; 0x76f988da;
    0x983e5152; 0xa831c66d; 0xb00327c8; 0xbf597fc7; 0xc6e00bf3; 0xd5a79147;
    0x06ca6351; 0x14292967; 0x27b70a85; 0x2e1b2138; 0x4d2c6dfc; 0x53380d13;
    0x650a7354; 0x766a0abb; 0x81c2c92e; 0x92722c85; 0xa2bfe8a1; 0xa81a664b;
    0xc24b8b70; 0xc76c51a3; 0xd192e819; 0xd6990624; 0xf40e3585; 0x106aa070;
    0x19a4c116; 0x1e376c08; 0x2748774c; 0x34b0bcb5; 0x391c0cb3; 0x4ed8aa4a;
    0x5b9cca4f; 0x682e6ff3; 0x748f82ee; 0x78a5636f; 0x84c87814; 0x8cc70208;
    0x90befffa; 0xa4506ceb; 0xbef9a3f7; 0xc67178f2;
  |]

let mask = 0xffffffff
let ( +: ) a b = (a + b) land mask
let rotr x n = ((x lsr n) lor (x lsl (32 - n))) land mask

type t = {
  h : int array;
  w : int array;
  block : Bytes.t;
  mutable fill : int;
  mutable length : int;
}

let init () =
  {
    h =
      [|
        0x6a09e667; 0xbb67ae85; 0x3c6ef372; 0xa54ff53a; 0x510e527f; 0x9b05688c;
        0x1f83d9ab; 0x5be0cd19;
      |];
    w = Array.make 64 0;
    block = Bytes.create 64;
    fill = 0;
    length = 0;
  }

let compress t =
  let w = t.w and b = t.block in
  for i = 0 to 15 do
    w.(i) <- Int32.to_int (Bytes.get_int32_be b (4 * i)) land mask
  done;
  for i = 16 to 63 do
    let x = w.(i - 15) and y = w.(i - 2) in
    let s0 = rotr x 7 lxor rotr x 18 lxor (x lsr 3) in
    let s1 = rotr y 17 lxor rotr y 19 lxor (y lsr 10) in
    w.(i) <- w.(i - 16) +: s0 +: w.(i - 7) +: s1
  done;
  let h = t.h in
  let a = ref h.(0) and b = ref h.(1) and c = ref h.(2) and d = ref h.(3) in
  let e = ref h.(4) and f = ref h.(5) and g = ref h.(6) and hh = ref h.(7) in
  for i = 0 to 63 do
    let s1 = rotr !e 6 lxor rotr !e 11 lxor rotr !e 25 in
    let ch = !e land !f lxor (lnot !e land mask land !g) in
    let t1 = !hh +: s1 +: ch +: k.(i) +: w.(i) in
    let s0 = rotr !a 2 lxor rotr !a 13 lxor rotr !a 22 in
    let maj = !a land !b lxor (!a land !c) lxor (!b land !c) in
    let t2 = s0 +: maj in
    hh := !g;
    g := !f;
    f := !e;
    e := !d +: t1;
    d := !c;
    c := !b;
    b := !a;
    a := t1 +: t2
  done;
  List.iteri
    (fun i x -> h.(i) <- h.(i) +: x)
    [ !a; !b; !c; !d; !e; !f; !g; !hh ]

let feed t s off len =
  t.length <- t.length + len;
  let off = ref off and len = ref len in
  while !len > 0 do
    let n = min !len (64 - t.fill) in
    Bytes.blit s !off t.block t.fill n;
    t.fill <- t.fill + n;
    off := !off + n;
    len := !len - n;
    if t.fill = 64 then begin
      compress t;
      t.fill <- 0
    end
  done

(* The message, a 1 bit, zeros to 56 bytes mod 64, its length in bits. *)
let hex t =
  let bits = t.length * 8 in
  let pad =
    Bytes.make (if t.fill < 56 then 56 - t.fill else 120 - t.fill) '\x00'
  in
  Bytes.set pad 0 '\x80';
  let size = Bytes.create 8 in
  Bytes.set_int64_be size 0 (Int64.of_int bits);
  feed t pad 0 (Bytes.length pad);
  feed t size 0 8;
  String.concat "" (Array.to_list (Array.map (Printf.sprintf "%08x") t.h))
