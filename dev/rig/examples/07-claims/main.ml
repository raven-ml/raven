(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Claims.

   A function handed buffers may write over an input in place only if nothing
   else can see that memory. Readers claim memory while they read it; a function
   claims its inputs, offers the ones the caller gave up ([donate]), and gets
   each exclusive only when it spans its memory and nothing else claims it.
   Consuming a buffer kills every older name of its memory, so a stale one
   cannot see the write. *)

open Rig

let int32s b = Buffer.bigarray Bigarray.int32 b

(* [x + y], written over [y] when the caller donated [y] and it is exclusive,
   into a new buffer otherwise. In place, [y]'s bytes are read through the
   consumed buffer: the name [y] is dead. *)
let add ~x ~y =
  Claim.with_ ~read:[ x ] ~donate:[ [ y ] ] @@ fun c ->
  let how, out, ys =
    if Claim.exclusive c y then
      let out = Claim.consume c ~why:"donated to add" y in
      ("in place", out, int32s out)
    else ("into a new buffer", Buffer.create host (Buffer.length y), int32s y)
  in
  let xs = int32s x and o = int32s out in
  for i = 0 to Bigarray.Array1.dim o - 1 do
    o.{i} <- Int32.add xs.{i} ys.{i}
  done;
  (how, out)

let ints xs =
  let b = Buffer.create host (4 * List.length xs) in
  List.iteri (fun i x -> (int32s b).{i} <- x) xs;
  b

let show name b =
  let a = int32s b in
  let xs = List.init (Bigarray.Array1.dim a) (fun i -> Int32.to_string a.{i}) in
  Printf.printf "%-18s [%s]\n" name (String.concat "; " xs)

let show_add ~x ~y =
  let how, r = add ~x ~y in
  show how r

let () =
  let x = ints [ 1l; 2l; 3l ] in

  (* Nothing else claims [y]: it is exclusive, written in place, and the name
     [y] is dead. Its memory lives on in the result. *)
  let y = ints [ 10l; 20l; 30l ] in
  show_add ~x ~y;
  (match int32s y with
  | _ -> ()
  | exception Invalid_argument msg -> print_endline msg);

  (* A reader claims [y]: the donation stays a read, and [add] allocates. *)
  let y = ints [ 10l; 20l; 30l ] in
  Claim.read y;
  show_add ~x ~y;
  Claim.release y;
  show "y" y;

  (* A view does not span its memory, and a bigarray's owner reaches its bytes
     outside the claims: neither is ever exclusive. *)
  let whole = ints [ 0l; 10l; 20l; 30l ] in
  show_add ~x ~y:(Buffer.view whole ~first:4 ~length:12);
  let ba = Bigarray.(Array1.init int32 c_layout 3 (fun i -> Int32.of_int i)) in
  show_add ~x ~y:(Buffer.of_bigarray ba)
