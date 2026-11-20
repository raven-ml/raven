(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Image

type t = { height : int; width : int; counts : int array }

let make fn ~height ~width counts =
  if height < 0 || width < 0 then
    invalid_argf "%s: %d by %d pixels" fn height width;
  let sum =
    Array.fold_left
      (fun s c ->
        if c < 0 then invalid_argf "%s: a negative count, %d" fn c;
        s + c)
      0 counts
  in
  if sum <> height * width then
    invalid_argf "%s: counts sum to %d, not %d by %d" fn sum height width;
  { height; width; counts }

let v ~height ~width counts = make "Mynd.Rle.v" ~height ~width counts

(* Runs in column-major order, the first of [false] pixels. *)
let encode m =
  let fn = "Mynd.Rle.encode" in
  if Nx.ndim m <> 3 || Nx.dim (-1) m <> 1 then
    invalid_argf "%s: a mask has axes [h; w; 1]" fn;
  let height = Nx.dim 0 m and width = Nx.dim 1 m in
  let px =
    Nx.to_array
      (Nx.reshape
         [| height * width |]
         (Nx.copy (Nx.transpose ~axes:[ 1; 0; 2 ] m)))
  in
  let runs = ref [] and run = ref 0 and cur = ref false in
  Array.iter
    (fun p ->
      if p = !cur then incr run
      else (
        runs := !run :: !runs;
        run := 1;
        cur := p))
    px;
  { height; width; counts = Array.of_list (List.rev (!run :: !runs)) }

let decode r =
  let px = Array.make (r.height * r.width) false in
  let at = ref 0 in
  Array.iteri
    (fun i c ->
      if i land 1 = 1 then Array.fill px !at c true;
      at := !at + c)
    r.counts;
  Nx.transpose ~axes:[ 1; 0; 2 ]
    (Nx.create Nx.bool [| r.width; r.height; 1 |] px)
  |> Nx.copy

let area r =
  let a = ref 0 in
  Array.iteri (fun i c -> if i land 1 = 1 then a := !a + c) r.counts;
  !a

(* pycocotools' rleToString: each count after the third less the count two
   before, then five bits a character from the low end, [0x20] marking one more,
   [0x10] the sign of the last, plus 48. *)
let to_string r =
  let b = Buffer.create (Array.length r.counts * 2) in
  Array.iteri
    (fun i c ->
      let x = ref (if i > 2 then c - r.counts.(i - 2) else c) in
      let more = ref true in
      while !more do
        let c = !x land 0x1f in
        x := !x asr 5;
        more := if c land 0x10 <> 0 then !x <> -1 else !x <> 0;
        Buffer.add_char b (Char.chr (48 + if !more then c lor 0x20 else c))
      done)
    r.counts;
  Buffer.contents b

(* A count of more than 12 characters passes 60 bits. *)
let max_chars = 12

let of_string ~height ~width s =
  let fn = "Mynd.Rle.of_string" in
  let n = String.length s in
  let counts = ref [] and m = ref 0 and p = ref 0 in
  while !p < n do
    let x = ref 0 and k = ref 0 and more = ref true in
    while !more do
      if !p >= n then invalid_argf "%s: %S ends inside a count" fn s;
      if !k >= max_chars then invalid_argf "%s: a count of %S overflows" fn s;
      let c = Char.code s.[!p] - 48 in
      if c < 0 || c > 63 then
        invalid_argf "%s: %C is not a count's character" fn s.[!p];
      x := !x lor ((c land 0x1f) lsl (5 * !k));
      more := c land 0x20 <> 0;
      incr p;
      incr k;
      if (not !more) && c land 0x10 <> 0 then x := !x lor (-1 lsl (5 * !k))
    done;
    let x = if !m > 2 then !x + List.nth !counts 1 else !x in
    counts := x :: !counts;
    incr m
  done;
  make fn ~height ~width (Array.of_list (List.rev !counts))
