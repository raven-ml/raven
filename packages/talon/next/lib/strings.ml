(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module B = Nx_device.Buffer
module A = Bigarray.Array1

type bytes = (int, Nx.uint8_elt) Nx_ragged.t

(* [reading ~by x f] is [f b] for [b] a host buffer of [x]'s elements in C
   order, under a read claim so that no compiled call lends its memory while [f]
   reads it. *)
let reading ~by x f =
  let b = Nx.Op.eval (Read { by; x }) in
  B.Claim.read b;
  Fun.protect ~finally:(fun () -> B.Claim.release b) (fun () -> f b)

(* UTF-8 *)

(* [within a j stop lo hi] is [true] iff byte [j] lies before [stop] and in [lo,
   hi]. *)
let within a j stop lo hi =
  j < stop
  &&
  let c = A.unsafe_get a j in
  lo <= c && c <= hi

(* [sequence a i stop] is the length of the valid UTF-8 sequence that starts at
   byte [i], or [0]. The ranges of the second byte exclude overlong forms,
   surrogates and values past U+10FFFF (RFC 3629, section 4). *)
let sequence a i stop =
  let b = A.unsafe_get a i in
  let tail j = within a j stop 0x80 0xbf in
  if b < 0x80 then 1
  else if b < 0xc2 then 0
  else if b < 0xe0 then if tail (i + 1) then 2 else 0
  else if b < 0xf0 then
    let lo, hi =
      match b with
      | 0xe0 -> (0xa0, 0xbf)
      | 0xed -> (0x80, 0x9f)
      | _ -> (0x80, 0xbf)
    in
    if within a (i + 1) stop lo hi && tail (i + 2) then 3 else 0
  else if b < 0xf5 then
    let lo, hi =
      match b with
      | 0xf0 -> (0x90, 0xbf)
      | 0xf4 -> (0x80, 0x8f)
      | _ -> (0x80, 0xbf)
    in
    if within a (i + 1) stop lo hi && tail (i + 2) && tail (i + 3) then 4 else 0
  else 0

(* [invalid a i stop] is the first byte of [i, stop) that starts no valid
   sequence, or [-1]. *)
let rec invalid a i stop =
  if i >= stop then -1
  else match sequence a i stop with 0 -> i | n -> invalid a (i + n) stop

(* [rows ~by ?mask r f] calls [f i v first stop] on each row [i] of [r] that
   [mask] holds, [v] the bytes and [\[first, stop)] the row's. *)
let rows ~by ?mask r f =
  let n = Nx_ragged.length r and read x f = reading ~by x f in
  read (Nx_ragged.offsets r) @@ fun o ->
  read (Nx_ragged.values r) @@ fun v ->
  let o = B.bigarray Bigarray.int64 o
  and v = B.bigarray Bigarray.int8_unsigned v in
  let loop skip =
    for i = 0 to n - 1 do
      if not (skip i) then
        f i v
          (Int64.to_int (A.unsafe_get o i))
          (Int64.to_int (A.unsafe_get o (i + 1)))
    done
  in
  match mask with
  | None -> loop (Fun.const false)
  | Some mask ->
      read mask @@ fun m ->
      let m = B.bigarray Bigarray.int8_unsigned m in
      loop (fun i -> A.unsafe_get m i = 0)

exception Invalid_row of int * string

let utf_8 ~by ?mask r =
  let check i v first stop =
    match invalid v first stop with
    | -1 -> ()
    | j ->
        let why = Printf.sprintf "invalid UTF-8 at byte %d" (j - first) in
        raise_notrace (Invalid_row (i, why))
  in
  match rows ~by ?mask r check with
  | () -> None
  | exception Invalid_row (i, why) -> Some (i, why)

(* Scalar values *)

let tensor a = Nx.of_bigarray (Bigarray.genarray_of_array1 a)
let starts_scalar v j = A.unsafe_get v j land 0xc0 <> 0x80

(* [count v i stop] is the number of scalar values in the bytes [i, stop). *)
let count v i stop =
  let n = ref 0 in
  for j = i to stop - 1 do
    if starts_scalar v j then incr n
  done;
  !n

(* [skip v i stop k] is the byte at which scalar value [k] of [i, stop) starts,
   or [stop] past the last. *)
let rec skip v i stop k =
  if i >= stop || (k = 0 && starts_scalar v i) then i
  else skip v (i + 1) stop (if starts_scalar v i then k - 1 else k)

let length ~by ?mask r =
  let ns = A.create Bigarray.int64 Bigarray.c_layout (Nx_ragged.length r) in
  A.fill ns 0L;
  rows ~by ?mask r (fun i v first stop ->
      A.unsafe_set ns i (Int64.of_int (count v first stop)));
  tensor ns

let slice ~by ?mask ~offset ~length r =
  let n = Nx_ragged.length r in
  let lo = Array.make n 0 and hi = Array.make n 0 in
  rows ~by ?mask r (fun i v first stop ->
      let p = if offset >= 0 then offset else count v first stop + offset in
      let p1 = if p > max_int - length then max_int else p + length in
      let p0 = Int.max p 0 and p1 = Int.max p1 0 in
      lo.(i) <- skip v first stop p0;
      hi.(i) <- skip v lo.(i) stop (Int.max 0 (p1 - p0)));
  let offsets = A.create Bigarray.int64 Bigarray.c_layout (n + 1) in
  A.unsafe_set offsets 0 0L;
  for i = 0 to n - 1 do
    A.unsafe_set offsets (i + 1)
      (Int64.add (A.unsafe_get offsets i) (Int64.of_int (hi.(i) - lo.(i))))
  done;
  let bytes =
    A.create Bigarray.int8_unsigned Bigarray.c_layout
      (Int64.to_int (A.unsafe_get offsets n))
  in
  reading ~by (Nx_ragged.values r) (fun v ->
      let v = B.bigarray Bigarray.int8_unsigned v in
      for i = 0 to n - 1 do
        let at = Int64.to_int (A.unsafe_get offsets i) in
        A.blit
          (A.sub v lo.(i) (hi.(i) - lo.(i)))
          (A.sub bytes at (hi.(i) - lo.(i)))
      done);
  Nx_ragged.v ~offsets:(tensor offsets) (tensor bytes)

type pattern =
  | Literal of string
  | Prefix of string
  | Suffix of string
  | Pieces of string list

(* [at v j s] is [true] iff the bytes of [s] start at byte [j]. *)
let at v j s =
  let n = String.length s in
  let rec loop k =
    k = n || (A.unsafe_get v (j + k) = Char.code s.[k] && loop (k + 1))
  in
  loop 0

(* [find v i stop s] is the first byte of [i, stop) at which [s] lies whole, or
   [-1]. *)
let rec find v i stop s =
  if i + String.length s > stop then -1
  else if at v i s then i
  else find v (i + 1) stop s

let matches ~by ?mask p r =
  let hits =
    A.create Bigarray.int8_unsigned Bigarray.c_layout (Nx_ragged.length r)
  in
  A.fill hits 0;
  let rec pieces v i stop = function
    | [] -> true
    | s :: ss ->
        let j = find v i stop s in
        j >= 0 && pieces v (j + String.length s) stop ss
  in
  let matches v first stop =
    match p with
    | Literal s -> find v first stop s >= 0
    | Prefix s -> stop - first >= String.length s && at v first s
    | Suffix s ->
        let j = stop - String.length s in
        j >= first && at v j s
    | Pieces ss -> pieces v first stop ss
  in
  rows ~by ?mask r (fun i v first stop ->
      if matches v first stop then A.unsafe_set hits i 1);
  Nx.cast Nx.bool (tensor hits)

let equal ~by r one =
  let s = ref "" in
  rows ~by one (fun _ v first stop ->
      s :=
        String.init (stop - first) (fun k ->
            Char.unsafe_chr (A.unsafe_get v (first + k))));
  let s = !s in
  let n = String.length s in
  let hits =
    A.create Bigarray.int8_unsigned Bigarray.c_layout (Nx_ragged.length r)
  in
  A.fill hits 0;
  rows ~by r (fun i v first stop ->
      if stop - first = n && at v first s then A.unsafe_set hits i 1);
  Nx.cast Nx.bool (tensor hits)
