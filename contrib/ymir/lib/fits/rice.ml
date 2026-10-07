(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rice coding of integers (FITS 4.0 §10.4.1), as cfitsio writes it: the
   first value as is, then blocks of differences between neighbours, mapped
   to unsigned numbers and Rice-coded with a split chosen per block. A block
   starts with fs + 1 in [fsbits] bits: 0 for a block of equal values, [fsmax
   + 1] for differences written in full. Arithmetic is modulo the width. *)

open Err

let fsbits = function 1 -> 3 | 2 -> 4 | _ -> 5
let fsmax = function 1 -> 6 | 2 -> 14 | _ -> 25

(* Leading bits of a byte, the highest set bit's position plus one. *)
let nonzero_count =
  Array.init 256 (fun b ->
      let rec go n = if b lsr n = 0 then n else go (n + 1) in
      go 0)

(* [signed w v] reads the [8w]-bit pattern [v] as a signed integer, except
   for bytes, which are unsigned. *)
let signed w v =
  if w = 1 then v land 0xFF
  else
    let bits = 8 * w in
    let v = v land ((1 lsl bits) - 1) in
    if v land (1 lsl (bits - 1)) <> 0 then v - (1 lsl bits) else v

(* [decode ~width ~block src off len out n] decodes [n] values of [width]
   bytes from the [len] bytes of [src] at [off] into [out]. Bytes after the
   last value are ignored. *)
let decode ~width ~block (src : Checksum.bigbytes) off len
    (out : (int, Bigarray.int_elt, Bigarray.c_layout) Bigarray.Array1.t) n =
  let fsbits = fsbits width and fsmax = fsmax width in
  let bbits = 8 * width in
  let mask = (1 lsl bbits) - 1 in
  let pos = ref off and stop = off + len in
  let byte () =
    if !pos >= stop then fail "the stream ends before its %d values" n;
    let b = Bigarray.Array1.unsafe_get src !pos in
    incr pos;
    b
  in
  let last = ref 0 in
  for _ = 1 to width do
    last := (!last lsl 8) lor byte ()
  done;
  let b = ref (byte ()) and nbits = ref 8 in
  let i = ref 0 in
  let undo diff =
    let d = if diff land 1 = 0 then diff lsr 1 else lnot (diff lsr 1) in
    let v = (d + !last) land mask in
    last := v;
    v
  in
  while !i < n do
    nbits := !nbits - fsbits;
    while !nbits < 0 do
      b := (!b lsl 8) lor byte ();
      nbits := !nbits + 8
    done;
    let fs = (!b lsr !nbits) - 1 in
    b := !b land ((1 lsl !nbits) - 1);
    let imax = Int.min n (!i + block) in
    if fs < 0 then begin
      while !i < imax do
        Bigarray.Array1.unsafe_set out !i (signed width !last);
        incr i
      done
    end
    else if fs = fsmax then begin
      while !i < imax do
        (* bbits bits written as they are *)
        let k = ref (bbits - !nbits) in
        let diff = ref ((!b lsl !k) land mask) in
        k := !k - 8;
        while !k >= 0 do
          b := byte ();
          diff := !diff lor (!b lsl !k);
          k := !k - 8
        done;
        if !nbits > 0 then begin
          b := byte ();
          diff := !diff lor (!b lsr - !k);
          b := !b land ((1 lsl !nbits) - 1)
        end
        else b := 0;
        Bigarray.Array1.unsafe_set out !i
          (signed width (undo (!diff land mask)));
        incr i
      done
    end
    else if fs > fsmax then
      fail "a block's split %d is past the width's %d" fs fsmax
    else
      while !i < imax do
        while !b = 0 do
          nbits := !nbits + 8;
          b := byte ()
        done;
        let nzero = !nbits - nonzero_count.(!b) in
        nbits := !nbits - (nzero + 1);
        b := !b lxor (1 lsl !nbits);
        nbits := !nbits - fs;
        while !nbits < 0 do
          b := (!b lsl 8) lor byte ();
          nbits := !nbits + 8
        done;
        let diff = (nzero lsl fs) lor (!b lsr !nbits) in
        b := !b land ((1 lsl !nbits) - 1);
        Bigarray.Array1.unsafe_set out !i (signed width (undo diff));
        incr i
      done
  done

(* Encoding *)

(* Output bits, most significant first. *)
type bits = { buf : Buffer.t; mutable acc : int; mutable left : int }

let put b n v =
  (* [n] bits of [v], n <= 32 *)
  let v = v land ((1 lsl n) - 1) in
  b.acc <- (b.acc lsl n) lor v;
  b.left <- b.left - n;
  while b.left <= 0 do
    Buffer.add_char b.buf (Char.unsafe_chr ((b.acc lsr -b.left) land 0xFF));
    b.left <- b.left + 8
  done;
  b.acc <- b.acc land ((1 lsl (8 - b.left)) - 1)

(* [encode ~width ~block v n] codes the [n] values of [v], each wrapped to
   [width] bytes, as cfitsio's fits_rcomp does. *)
let encode ~width ~block
    (v : (int, Bigarray.int_elt, Bigarray.c_layout) Bigarray.Array1.t) n =
  let fsbits = fsbits width and fsmax = fsmax width in
  let bbits = 8 * width in
  let mask = (1 lsl bbits) - 1 in
  let b = { buf = Buffer.create ((n * width / 2) + 16); acc = 0; left = 8 } in
  if n > 0 then begin
    put b bbits (Bigarray.Array1.get v 0);
    let last = ref (Bigarray.Array1.get v 0) in
    let diff = Array.make block 0 in
    let i = ref 0 in
    while !i < n do
      let len = Int.min block (n - !i) in
      let sum = ref 0. in
      for j = 0 to len - 1 do
        let next = Bigarray.Array1.get v (!i + j) in
        (* the difference wrapped to a signed value of the width, then
           mapped to an unsigned one *)
        let d = (next - !last) land mask in
        let d =
          if d land (1 lsl (bbits - 1)) <> 0 then d - (1 lsl bbits) else d
        in
        let m =
          if d < 0 then lnot (d lsl 1) land mask else (d lsl 1) land mask
        in
        diff.(j) <- m;
        sum := !sum +. float_of_int m;
        last := next
      done;
      let dpsum =
        Float.max 0. ((!sum -. float_of_int (len / 2) -. 1.) /. float_of_int len)
      in
      let psum = ref ((int_of_float dpsum land mask) lsr 1) in
      let fs = ref 0 in
      while !psum > 0 do
        psum := !psum lsr 1;
        incr fs
      done;
      let fs = !fs in
      if fs >= fsmax then begin
        put b fsbits (fsmax + 1);
        for j = 0 to len - 1 do
          put b bbits diff.(j)
        done
      end
      else if fs = 0 && !sum = 0. then put b fsbits 0
      else begin
        put b fsbits (fs + 1);
        for j = 0 to len - 1 do
          let m = diff.(j) in
          let top = m lsr fs in
          (* top zeros, then a one *)
          let t = ref top in
          while !t >= 24 do
            put b 24 0;
            t := !t - 24
          done;
          put b (!t + 1) 1;
          if fs > 0 then put b fs (m land ((1 lsl fs) - 1))
        done
      end;
      i := !i + len
    done;
    if b.left < 8 then
      Buffer.add_char b.buf (Char.unsafe_chr ((b.acc lsl b.left) land 0xFF))
  end;
  Buffer.contents b.buf
