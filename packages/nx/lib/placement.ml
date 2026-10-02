(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_array

(* Placements: devices over a grid. *)

type t = Device.t Grid.t

let host = Grid.device Device.host
let on d = Grid.device d
let devices = Grid.devices
let cuts = Grid.cuts
let uncut = Grid.uncut
let map_axes = Grid.map_axes
let select = Grid.select

let is_host p =
  match Grid.devices p with [ d ] -> Device.equal d Device.host | _ -> false

(* The disk holds values and computes on none. *)
let on_disk p =
  match Grid.devices p with
  | [ d ] -> Nx_device.equal (Device.memory d) Nx_device.disk
  | _ -> false

let same_devices p q =
  let dp = devices p and dq = devices q in
  List.compare_lengths dp dq = 0
  && List.for_all (fun d -> List.exists (Device.equal d) dq) dp

let equal p q = Grid.equal Device.equal p q

(* Whether [p] and [q] hold each window of a value in one memory, whichever of
   that memory's devices compute on it. *)
let same_memories p q =
  Grid.equal (fun d d' -> Device.memory d == Device.memory d') p q

let pp ppf p = Grid.pp Device.pp ppf p

(* A placement names each memory once: it lowers to the memories a compiler
   addresses, and a memory holds one window of a value. *)
let check what ds =
  let fail fmt =
    Printf.ksprintf invalid_arg ("Nx.Placement.%s: " ^^ fmt) what
  in
  let rec distinct = function
    | [] -> ()
    | d :: rest ->
        if List.exists (Device.equal d) rest then
          fail "%s appears twice" (Device.name d);
        (match
           List.find_opt (fun d' -> Device.memory d' == Device.memory d) rest
         with
        | Some d' ->
            fail "%s and %s share one memory" (Device.name d) (Device.name d')
        | None -> ());
        distinct rest
  in
  match ds with [] -> fail "no device" | _ -> distinct ds

let replicated ds =
  check "replicated" ds;
  Grid.v ds [ List.length ds ] []

let sharded ~axis ds =
  if axis < 0 then
    invalid_arg (Printf.sprintf "Nx.Placement.sharded: axis %d < 0" axis);
  check "sharded" ds;
  Grid.v ds [ List.length ds ] [ (axis, [ 0 ]) ]

(* Raises unless every cut of [p] divides its axis of [shape] evenly. *)
let check_shape what p shape =
  List.iter
    (fun (a, n) ->
      if a >= Array.length shape then
        invalid_arg
          (Printf.sprintf "%s: shape %s has no axis %d to split" what
             (Shape.to_string shape) a);
      if shape.(a) mod n <> 0 then
        invalid_arg
          (Printf.sprintf
             "%s: axis %d of shape %s does not split evenly over %d devices"
             what a (Shape.to_string shape) n))
    (Grid.cuts p)

let window p shape d =
  match List.find_index (Device.equal d) (devices p) with
  | None ->
      invalid_arg
        (Printf.sprintf "Nx.Placement.window: %s holds no window"
           (Device.name d))
  | Some k ->
      check_shape "Nx.Placement.window" p shape;
      let w = Array.map (fun n -> (0, n)) shape in
      List.iter2
        (fun (a, n) (_, j) ->
          let size = shape.(a) / n in
          w.(a) <- (j * size, (j + 1) * size))
        (Grid.cuts p) (Grid.tile_index p k);
      w

(* The placement of a value with a new leading axis, and of one without its
   leading axis: a grid axis that cut it then holds copies. *)
let with_leading_axis p = map_axes succ p

let without_leading_axis p =
  match cuts p with [] -> p | _ -> map_axes pred (uncut p ~axis:0)
