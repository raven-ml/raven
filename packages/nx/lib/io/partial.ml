(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Bigarray
module M = Map.Make (Int)

(* A partial file: its name, the length of its file, and the bytes it holds
   as pieces by their first offset. Pieces are disjoint, and none is written
   once made, so values share them. *)
type t = { name : string; length : int; pieces : Storage.bytes M.t }

let v ~name ~length =
  if length < 0 then
    Fail.invalid_argf "Nx_io.Partial.v: length %d is negative" length;
  { name; length; pieces = M.empty }

let name p = p.name
let length p = p.length

(* The pieces that hold bytes [pos] to [pos + n - 1], in order. *)
let overlapping p ~pos n =
  let first =
    match M.find_last_opt (fun k -> k <= pos) p.pieces with
    | Some (k, b) when k + Array1.dim b > pos -> k
    | _ -> pos
  in
  Seq.take_while (fun (k, _) -> k < pos + n) (M.to_seq_from first p.pieces)

let lacking p ~pos n =
  let gaps, upto =
    Seq.fold_left
      (fun (gaps, upto) (k, b) ->
        let gaps = if k > upto then (upto, k - upto) :: gaps else gaps in
        (gaps, Int.max upto (k + Array1.dim b)))
      ([], pos) (overlapping p ~pos n)
  in
  List.rev (if pos + n > upto then (upto, pos + n - upto) :: gaps else gaps)

let bytes p ~pos n =
  match M.find_last_opt (fun k -> k <= pos) p.pieces with
  | Some (k, b) when k + Array1.dim b >= pos + n -> Array1.sub b (pos - k) n
  | _ ->
      let dst = Array1.create int8_unsigned c_layout n in
      Seq.iter
        (fun (k, b) ->
          let lo = Int.max k pos in
          let hi = Int.min (k + Array1.dim b) (pos + n) in
          Array1.blit
            (Array1.sub b (lo - k) (hi - lo))
            (Array1.sub dst (lo - pos) (hi - lo)))
        (overlapping p ~pos n);
      dst

let string p ~pos n =
  let b = bytes p ~pos n in
  String.init n (fun i -> Char.unsafe_chr (Array1.unsafe_get b i))

let add p ~at s =
  let n = String.length s in
  if at < 0 || at > p.length - n then
    Fail.invalid_argf
      "Nx_io.Partial.add: %d bytes at byte %d lie outside the file's %d" n at
      p.length;
  (* Bytes [lo] to [hi - 1] of the file, copied from [s]. *)
  let piece lo hi =
    let b = Array1.create int8_unsigned c_layout (hi - lo) in
    for i = 0 to hi - lo - 1 do
      Array1.unsafe_set b i (Char.code (String.unsafe_get s (lo - at + i)))
    done;
    b
  in
  let pieces, upto =
    Seq.fold_left
      (fun (pieces, upto) (k, b) ->
        let pieces =
          if k > upto then M.add upto (piece upto k) pieces else pieces
        in
        let hi = Int.min (k + Array1.dim b) (at + n) in
        for i = Int.max k at to hi - 1 do
          if Array1.get b (i - k) <> Char.code s.[i - at] then
            Fail.failf "%s: byte %d differs from the byte already held" p.name
              i
        done;
        (pieces, Int.max upto (k + Array1.dim b)))
      (p.pieces, at) (overlapping p ~pos:at n)
  in
  let pieces =
    if at + n > upto then M.add upto (piece upto (at + n)) pieces else pieces
  in
  { p with pieces }

let ranges rs =
  let merge acc (at, n) =
    match acc with
    | (a, m) :: rest when at <= a + m -> (a, Int.max m (at + n - a)) :: rest
    | _ -> (at, n) :: acc
  in
  List.filter (fun (_, n) -> n > 0) rs
  |> List.sort compare |> List.fold_left merge [] |> List.rev
