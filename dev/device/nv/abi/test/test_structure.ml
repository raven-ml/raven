(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Structures as Qmd makes them, against structure.mli's description of holes:
   each fills the low bits of the narrowest 1, 2, 4 or 8-byte word that holds
   its bits, and keeps the word's other bits. *)

open Windtrap
open Device_nv_abi
module S = Device_nv_abi_support

let structure d = Qmd.structure (S.descriptor d)

(* The width in bytes of a hole's word. *)
let width bits = List.find (fun w -> 8 * w >= bits) [ 1; 2; 4; 8 ]
let mask bits = if bits = 64 then -1L else Int64.(sub (shift_left 1L bits) 1L)

let read b at w =
  let n = ref 0L in
  for i = w - 1 downto 0 do
    n :=
      Int64.(
        logor (shift_left !n 8) (of_int (Char.code (Bytes.get b (at + i)))))
  done;
  !n

let write b at w n =
  for i = 0 to w - 1 do
    Bytes.set b (at + i)
      (Char.chr (Int64.to_int (Int64.shift_right_logical n (8 * i)) land 0xff))
  done

let rec eval value : _ Packet.term -> int64 = function
  | Value v -> value v
  | Add (t, n) -> Int64.add (eval value t) n
  | Shift (t, n) -> Int64.shift_right_logical (eval value t) n

(* [s.bytes] with each hole filled as structure.mli describes it. *)
let reference value (s : _ Structure.t) =
  let b = Bytes.of_string s.bytes in
  List.iter
    (fun (h : _ Structure.hole) ->
      let w = width h.bits and m = mask h.bits in
      let word = read b h.at w in
      write b h.at w
        Int64.(logor (logand word (lognot m)) (logand (eval value h.value) m)))
    s.holes;
  Bytes.to_string b

let pp_hole ppf (h : int64 Structure.hole) =
  Format.fprintf ppf "{ at = %d; bits = %d }" h.at h.bits

let holes =
  group ~timeout:10. "holes"
    [
      prop "holes are by increasing offset, their words in the bytes and apart"
        S.drawn (fun d ->
          let s = structure d in
          cover "a hole" (s.holes <> []);
          let _ =
            List.fold_left
              (fun next (h : int64 Structure.hole) ->
                let msg = Format.asprintf "%a" pp_hole h in
                at_least ~msg int ~than:1 h.bits;
                at_most ~msg int ~than:64 h.bits;
                at_least ~msg int ~than:next h.at;
                at_most ~msg int ~than:(String.length s.bytes)
                  (h.at + width h.bits);
                h.at + width h.bits)
              0 s.holes
          in
          ());
      prop "a hole's field is zero in the bytes" S.drawn (fun d ->
          let s = structure d in
          let b = Bytes.of_string s.bytes in
          List.iter
            (fun (h : int64 Structure.hole) ->
              equal
                ~msg:(Format.asprintf "%a" pp_hole h)
                int64 0L
                (Int64.logand (read b h.at (width h.bits)) (mask h.bits)))
            s.holes);
      test "a size patched after it was set has a zero field" (fun () ->
          let s =
            Qmd.make (S.launch (S.gpu ()) (S.kernel ()))
            |> Qmd.set_dim (Block Z) 1 |> Qmd.patch_dim (Block Z) 0L
            |> Qmd.structure
          in
          (* CTA_THREAD_DIMENSION2, MW(639:624) of clc7c0qmd.h. *)
          equal int 0 (S.field s.bytes (639, 624)));
      prop
        "encode fills each hole's field with its term's low bits, and keeps \
         every other bit"
        (Gen.pair S.drawn S.u64) (fun (d, k) ->
          let s = structure d in
          let value v = Int64.logxor v k in
          cover "a hole beside a field it shares bytes with"
            (List.exists
               (fun (h : int64 Structure.hole) -> 8 * width h.bits > h.bits)
               s.holes);
          equal string (reference value s) (Structure.encode value s));
    ]

let () = exit (run "device_nv_abi.structure" [ holes ])
