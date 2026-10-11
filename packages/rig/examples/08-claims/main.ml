(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Claims.

   A function handed buffers may write over an input in place only if
   nothing else reads that memory. Readers claim memory while they read it.
   A function claims its inputs, donates those it no longer needs
   ([donate]), and holds a donated input exclusive only when the buffer
   spans its memory and no other claim holds it. Consuming the memory makes
   every buffer made over it until then dead, so none of them sees the
   write, and the write goes through the buffer [consume] returns. Memory
   exported as an array is outside the claims for good, and never
   exclusive. *)

open Rig

(* [x + y], written over [y], which [add] donates, when [add] holds it
   exclusive, and otherwise into a new buffer that starts as [y]'s copy. In
   place, the sum is written through the consumed buffer, the one live
   buffer over [y]'s memory, once the claims are released. *)
let add ~x ~y =
  let how, out =
    Claim.with_ ~read:[ x ] ~donate:[ y ] @@ fun c ->
    if Claim.exclusive c y then
      ("in place", Claim.consume c ~why:"donated to add" y)
    else
      let o = Buffer.create host (Buffer.length y) in
      Buffer.copy ~src:y ~dst:o;
      ("into a new buffer", o)
  in
  Buffer.read x Bigarray.int32 (fun xs ->
      Buffer.write out Bigarray.int32 (fun o ->
          for i = 0 to Bigarray.Array1.dim o - 1 do
            o.{i} <- Int32.add xs.{i} o.{i}
          done));
  (how, out)

(* A bigarray of [l]. *)
let int32s l = Bigarray.(Array1.of_array int32 c_layout (Array.of_list l))

(* A host buffer that holds [l], filled by a copy. A buffer over an array
   is never exclusive, since its memory is outside the claims. *)
let holding l =
  let b = Buffer.create host (4 * List.length l) in
  Buffer.copy ~src:(Buffer.of_bigarray (int32s l)) ~dst:b;
  b

(* [a]'s elements, joined. *)
let ints a =
  String.concat "; "
    (List.init (Bigarray.Array1.dim a) (fun i -> Int32.to_string a.{i}))

let show name b =
  Buffer.read b Bigarray.int32 (fun a ->
      Printf.printf "%-18s [%s]\n" name (ints a))

let show_add ~x ~y =
  let how, r = add ~x ~y in
  show how r

let () =
  let x = holding [ 1l; 2l; 3l ] in

  (* No other claim holds [y], so it is exclusive and written in place, and
     [y] is dead. Its memory lives on in the result. *)
  let y = holding [ 10l; 20l; 30l ] in
  show_add ~x ~y;
  (match Buffer.read y Bigarray.int32 ignore with
  | () -> ()
  | exception Invalid_argument msg -> print_endline msg);

  (* While a reader claims [y], the donation stays a read claim, and [add]
     allocates. *)
  let y = holding [ 10l; 20l; 30l ] in
  Buffer.read y Bigarray.int32 (fun _ -> show_add ~x ~y);
  show "y" y;

  (* A view of part of an allocation does not span its memory, and a buffer
     over a bigarray has its memory outside the claims. Neither is ever
     exclusive. *)
  let whole = holding [ 0l; 10l; 20l; 30l ] in
  show_add ~x ~y:(Buffer.view whole ~first:4 ~length:12);
  show_add ~x ~y:(Buffer.of_bigarray (int32s [ 0l; 1l; 2l ]));

  (* Nor is a buffer whose memory was exported as an array, which puts the
     memory outside the claims for good. *)
  let y = holding [ 10l; 20l; 30l ] in
  ignore (Buffer.export y Bigarray.int32);
  show "exported" y;
  show_add ~x ~y
