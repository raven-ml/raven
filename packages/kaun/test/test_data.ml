(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Data = Kaun.Data

(* [n] examples of 3 features whose elements count up from 0, so every element
   names its example. *)
let dataset n =
  Nx.create Nx.float32 [| n; 3 |] (Array.init (3 * n) float_of_int)

(* Each example's target, twice its index, at another dtype and width. *)
let targets n =
  Nx.create Nx.int32 [| n |] (Array.init n (fun i -> Int32.of_int (2 * i)))

(* The sizes of the batches of [n] examples: whole batches, then the remainder
   unless it is dropped. *)
let sizes n batch_size drop_last =
  let whole = List.init (n / batch_size) (fun _ -> batch_size) in
  let rest = n mod batch_size in
  if rest = 0 || drop_last = Some true then whole else whole @ [ rest ]

(* Every [n] in 0..9, [batch_size] in 1..10 and [drop_last] given or not: the
   domain is checked whole. *)
let domain = Gen.(triple (int_range 0 9) (int_range 1 10) (option bool))
let count = 10 * 10 * 3

let batches_law (n, batch_size, drop_last) =
  let batches = List.of_seq (Data.batches ?drop_last ~batch_size (dataset n)) in
  let sizes = sizes n batch_size drop_last in
  let kept = List.fold_left ( + ) 0 sizes in
  equal ~msg:"shapes"
    (list (array int))
    (List.map (fun b -> [| b; 3 |]) sizes)
    (List.map Nx.shape batches);
  equal ~msg:"elements" (array float_exact)
    (Array.init (3 * kept) float_of_int)
    (Array.concat (List.map Nx.to_array batches))

let batches2_law (n, batch_size, drop_last) =
  let batches =
    List.of_seq (Data.batches2 ?drop_last ~batch_size (dataset n, targets n))
  in
  let sizes = sizes n batch_size drop_last in
  let kept = List.fold_left ( + ) 0 sizes in
  equal ~msg:"shapes"
    (list (pair (array int) (array int)))
    (List.map (fun b -> ([| b; 3 |], [| b |])) sizes)
    (List.map (fun (x, y) -> (Nx.shape x, Nx.shape y)) batches);
  equal ~msg:"inputs" (array float_exact)
    (Array.init (3 * kept) float_of_int)
    (Array.concat (List.map (fun (x, _) -> Nx.to_array x) batches));
  equal ~msg:"targets" (array int32)
    (Array.init kept (fun i -> Int32.of_int (2 * i)))
    (Array.concat (List.map (fun (_, y) -> Nx.to_array y) batches))

(* What a profile recorded, by kind: a copy moves bytes, and memory a device
   allocates shows as the bytes it then has in use. *)
let recorded f =
  let (), events = Rig.Profile.take f in
  List.filter_map
    (function
      | Rig.Profile.Copy { bytes; _ } ->
          Some (Printf.sprintf "copy of %d bytes" bytes)
      | Rig.Profile.Allocation { allocated; _ } ->
          Some (Printf.sprintf "allocation, %d bytes in use" allocated)
      | _ -> None)
    events

(* A software device counts the memory its values take, where the host's memory
   records nothing. *)
let software = fixture (fun () -> Result.get_ok (Rig.software "kaun-test-data"))

let views () =
  let module S = (val Nx.devices [ software () ]) in
  let x = Nx.place S.on (dataset 10) and y = Nx.place S.on (targets 10) in
  equal ~msg:"batches" (list string) []
    (recorded (fun () -> Seq.iter ignore (Data.batches ~batch_size:4 x)));
  equal ~msg:"batches2" (list string) []
    (recorded (fun () -> Seq.iter ignore (Data.batches2 ~batch_size:4 (x, y))))

(* A scope that refuses keyless draws: a draw would raise. *)
let draws_nothing () =
  let x = dataset 10 and y = targets 10 in
  Nx.Rng.refusing "a traversal" (fun () ->
      equal int 3 (Seq.length (Data.batches ~batch_size:4 x));
      equal int 3 (Seq.length (Data.batches2 ~batch_size:4 (x, y))))

let scalar () = Nx.scalar Nx.float32 1.
let refuses (_, f) = raises_match (fun e -> Exn.invalid_arg e) f

let () =
  exit
    (run "kaun data"
       [
         group "batches"
           [
             prop ~count
               "cuts consecutive slices of batch_size in order, the last \
                smaller unless drop_last"
               domain batches_law;
             test "cuts views of x" views;
             test "draws nothing" draws_nothing;
             cases "raises on a batch_size that is not positive or a scalar x"
               ~id:fst
               [
                 ( "batch_size 0",
                   fun () ->
                     Seq.is_empty (Data.batches ~batch_size:0 (dataset 4)) );
                 ( "batch_size -1",
                   fun () ->
                     Seq.is_empty (Data.batches ~batch_size:(-1) (dataset 4)) );
                 ( "a scalar x",
                   fun () ->
                     Seq.is_empty (Data.batches ~batch_size:1 (scalar ())) );
               ]
               refuses;
           ];
         group "batches2"
           [
             prop ~count "pairs x's slices with y's, as batches cuts them"
               domain batches2_law;
             cases
               "raises on a batch_size that is not positive, a scalar x or y, \
                or unequal example counts"
               ~id:fst
               [
                 ( "batch_size 0",
                   fun () ->
                     Seq.is_empty
                       (Data.batches2 ~batch_size:0 (dataset 4, targets 4)) );
                 ( "a scalar x",
                   fun () ->
                     Seq.is_empty
                       (Data.batches2 ~batch_size:1 (scalar (), targets 4)) );
                 ( "a scalar y",
                   fun () ->
                     Seq.is_empty
                       (Data.batches2 ~batch_size:1 (dataset 4, scalar ())) );
                 ( "4 inputs and 3 targets",
                   fun () ->
                     Seq.is_empty
                       (Data.batches2 ~batch_size:2 (dataset 4, targets 3)) );
               ]
               refuses;
           ];
       ])
