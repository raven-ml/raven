(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Placing values, through Nx, against a model of which window each device
   holds. *)

open Windtrap
module A = Nx_array

let m = Nx_support.memory

module S4 = (val Nx.devices [ m 0; m 1; m 2; m 3 ])

let mesh = Nx.Mesh.v S4.v [ ("a", 2); ("b", 2) ]
let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

(* Placements over S4, and the window each of their devices holds, in order: the
   device, then the first index and extent per axis. *)
type where = On | Split of int | Cuts of int * int | Minor of int

let pp_where ppf = function
  | On -> Format.pp_print_string ppf "on"
  | Split a -> Format.fprintf ppf "split %d" a
  | Cuts (x, y) -> Format.fprintf ppf "cuts %d/a %d/b" x y
  | Minor x -> Format.fprintf ppf "cut %d/b" x

let placement = function
  | On -> S4.on
  | Split a -> S4.split ~axis:a
  | Cuts (x, y) when x = y -> Nx.Placement.mesh mesh [ (x, [ "a"; "b" ]) ]
  | Cuts (x, y) -> Nx.Placement.mesh mesh [ (x, [ "a" ]); (y, [ "b" ]) ]
  | Minor x -> Nx.Placement.mesh mesh [ (x, [ "b" ]) ]

let windows w shape =
  let whole = Array.map (fun n -> (0, n)) shape in
  let cut axis tiles j (b : (int * int) array) =
    let size = shape.(axis) / tiles in
    b.(axis) <- (j * size, size);
    b
  in
  match w with
  | On -> List.init 4 (fun d -> (d, whole))
  | Split a -> List.init 4 (fun d -> (d, cut a 4 d (Array.copy whole)))
  | Cuts (x, y) when x = y ->
      List.init 4 (fun d -> (d, cut x 4 d (Array.copy whole)))
  | Cuts (x, y) ->
      List.init 4 (fun d ->
          (d, cut y 2 (d mod 2) (cut x 2 (d / 2) (Array.copy whole))))
  | Minor x -> List.init 4 (fun d -> (d, cut x 2 (d mod 2) (Array.copy whole)))

let where_of rank =
  let axis = Gen.int_range 0 (rank - 1) in
  Gen.with_pp pp_where
    (Gen.one_of
       [
         Gen.constant On;
         Gen.map (fun a -> Split a) axis;
         Gen.map (fun (x, y) -> Cuts (x, y)) (Gen.pair axis axis);
         Gen.map (fun x -> Minor x) axis;
       ])

let floats shape = Array.init (Array.fold_left ( * ) 1 shape) float_of_int

let host shape =
  Nx.Repr.of_array Nx.host (A.of_array A.Dtype.Float32 shape (floats shape))

(* The elements of [x]'s window [b] in C order, [x] holding [floats shape]. *)
let expected shape b =
  let r = Array.length shape in
  let out = ref [] in
  let rec go a off =
    if a = r then out := float_of_int off :: !out
    else
      let start, n = b.(a) in
      for i = start to start + n - 1 do
        go (a + 1) ((off * shape.(a)) + i)
      done
  in
  go 0 0;
  List.rev !out

(* [x] holds, on each device of [w], that device's window of [floats shape]. *)
let holds shape w x =
  let arrays = require_some (Nx.Repr.shards x) in
  let ws = windows w shape in
  equal int ~msg:"one array per device" (List.length ws) (Array.length arrays);
  List.iteri
    (fun j (d, b) ->
      equal string ~msg:"its device"
        (Rig.name (m d))
        (Rig.name (A.device arrays.(j)));
      equal (list float_exact) ~msg:"its window" (expected shape b)
        (Array.to_list (A.to_array arrays.(j))))
    ws

let case =
  Gen.with_pp
    (fun ppf (s, w, w') ->
      Format.fprintf ppf "[%s] at %a then %a"
        (String.concat "; " (Array.to_list (Array.map string_of_int s)))
        pp_where w pp_where w')
    (Gen.bind (Gen.int_range 1 3) (fun rank ->
         Gen.triple
           (Gen.array ~size:(Gen.constant rank) (Gen.of_list [ 4; 8 ]))
           (where_of rank) (where_of rank)))

let laws =
  group "laws"
    [
      prop "each device holds its window of a host value placed" case
        (fun (s, w, _) ->
          cover "split" (match w with Split _ | Cuts _ -> true | _ -> false);
          holds s w (Nx.place (placement w) (host s)));
      prop ~count:500 "each device holds its window after a second placement"
        case (fun (s, w, w') ->
          cover "a change of arrangement" (w <> w');
          cover "split to split on another axis"
            (match (w, w') with Split a, Split b -> a <> b | _ -> false);
          holds s w' (Nx.place (placement w') (Nx.place (placement w) (host s))));
      prop "a value placed back on the host has its elements" case
        (fun (s, w, _) ->
          let back =
            Nx.place Nx.Placement.host (Nx.place (placement w) (host s))
          in
          equal (list float_exact)
            (Array.to_list (floats s))
            (Array.to_list (A.to_array (require_some (Nx.Repr.array back)))));
    ]

(* A narrow dtype across a change of arrangement, assembled element by element
   for sub-byte formats and by runs otherwise. *)
let narrow (type v s) name (dt : (v, s) A.Dtype.t) (of_int : int -> v) =
  test (name ^ " keeps its elements across a change of arrangement") (fun () ->
      let shape = [| 4; 8 |] in
      let xs = Array.init 32 (fun i -> of_int (i mod 7)) in
      let x = Nx.Repr.of_array Nx.host (A.of_array dt shape xs) in
      let y = Nx.place (S4.split ~axis:1) (Nx.place (S4.split ~axis:0) x) in
      let back = Nx.place Nx.Placement.host y in
      equal bool true (A.to_array (require_some (Nx.Repr.array back)) = xs))

let sharing =
  group "sharing"
    [
      test "a value already at a placement keeps its arrays" (fun () ->
          let x = Nx.place S4.on (host [| 4 |]) in
          let a = require_some (Nx.Repr.shards x) in
          let b = require_some (Nx.Repr.shards (Nx.place S4.on x)) in
          equal bool true (Array.for_all2 ( == ) a b));
      test "a set over the host's device reads the host's array" (fun () ->
          let module Fast =
            (val Nx.devices ~kernels:(module Nx_cpu) [ Rig.host ])
          in
          let x = host [| 4 |] in
          let a = require_some (Nx.Repr.array x) in
          let b = require_some (Nx.Repr.array (Nx.place Fast.on x)) in
          equal bool true (a == b));
      test "a device that maps the host's memory borrows it" (fun () ->
          let x = host [| 1024 |] in
          let a = require_some (Nx.Repr.array x) in
          let b = require_some (Nx.Repr.shards (Nx.place S4.on x)) in
          equal bool true (Rig.Buffer.overlaps (A.buffer a) (A.buffer b.(1))));
      narrow "int4" A.Dtype.Int4 Fun.id;
      narrow "int16" A.Dtype.Int16 Fun.id;
      narrow "bool" A.Dtype.Bool (fun i -> i mod 2 = 0);
    ]

let refusals =
  group "refusals"
    [
      test "a split that does not divide the shape raises" (fun () ->
          invalid ~by:"Nx.place" (fun () ->
              Nx.place (S4.split ~axis:0) (host [| 6 |])));
      test "a lost device raises when placed from, and its facts answer"
        (fun () ->
          let d =
            match Rig.memory_device "nx-place-lost" with
            | Ok d -> d
            | Error e -> failwith e
          in
          let module L = (val Nx.devices [ d ]) in
          let x = Nx.Repr.of_array L.v (A.create d A.Dtype.Float32 [| 4 |]) in
          Rig.close d;
          raises_match
            (function Rig.Lost _ -> true | _ -> false)
            (fun () -> Nx.place Nx.Placement.host x);
          equal (array int) [| 4 |] (Nx.shape x);
          equal string "nx-place-lost"
            (Format.asprintf "%a" Nx.Placement.pp (Nx.placement x)));
    ]

let () = exit (run "nx place" [ laws; sharing; refusals ])
