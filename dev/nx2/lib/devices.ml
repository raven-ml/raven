(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type host = |

(* A set at run time. [ones] and [whole] are its placements, written once inside
   [make] before the set escapes and never after. *)
type set = {
  number : int;
  rigs : Rig.t array;
  kernels : (module Nx_kernel.S) option;
  mutable ones : place array;
  mutable whole : place;
}

and place = { set : set; grid : Grid.t }

(* The brand is phantom: every set and placement is one of these records at
   every brand. *)
type 'd t = set
type 'd placement = place
type mesh_rt = { mesh_set : set; names : string array; extents : int array }
type 'd mesh = mesh_rt

let invalid_argf fmt = Format.kasprintf invalid_arg fmt
let positions n = Array.init n Fun.id

let whole_grid n =
  if n = 1 then Grid.device 0
  else
    match Grid.v ~devices:(positions n) ~extents:[| n |] ~cuts:[||] with
    | Ok g -> g
    | Error e -> invalid_arg e

let make number rigs kernels =
  let rec s =
    {
      number;
      rigs;
      kernels;
      ones = [||];
      whole = { set = s; grid = Grid.device 0 };
    }
  in
  s.ones <-
    Array.init (Array.length rigs) (fun k -> { set = s; grid = Grid.device k });
  s.whole <-
    (if Array.length rigs = 1 then s.ones.(0)
     else { set = s; grid = whole_grid (Array.length rigs) });
  s

let cpu : (module Nx_kernel.S) = (module Nx_cpu)
let host = make 0 [| Rig.host |] (Some cpu)
let next = Atomic.make 1

let mint ~by ?kernels ds =
  let rigs = Array.of_list ds in
  if Array.length rigs = 0 then
    invalid_argf "%s: a device set needs a device" by;
  Array.iteri
    (fun i d ->
      for j = 0 to i - 1 do
        if Rig.equal rigs.(j) d then
          invalid_argf "%s: %s appears twice" by (Rig.name d)
      done)
    rigs;
  let kernels =
    match kernels with
    | Some (module K : Nx_kernel.S) as k ->
        Array.iter
          (fun d ->
            if not (K.computes_on d) then
              invalid_argf "%s: %s does not compute on %s" by K.name
                (Rig.name d))
          rigs;
        k
    | None -> if Array.for_all Rig.runs_on_host rigs then Some cpu else None
  in
  make (Atomic.fetch_and_add next 1) rigs kernels

let number s = s.number
let count s = Array.length s.rigs

let rig s k =
  if k < 0 || k >= count s then
    invalid_arg
      (Printf.sprintf "Devices.rig: no device %d in a set of %d" k (count s));
  s.rigs.(k)

let position s d = Array.find_index (Rig.equal d) s.rigs
let kernels s = s.kernels

let pp ppf s =
  if s.number = 0 then Format.pp_print_string ppf "host"
  else
    Format.fprintf ppf "set %d [%a]" s.number
      (Format.pp_print_list
         ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
         (fun ppf d -> Format.pp_print_string ppf (Rig.name d)))
      (Array.to_list s.rigs)

(* Placements *)

let one s k =
  if k < 0 || k >= count s then
    invalid_arg
      (Printf.sprintf "Devices.one: no device %d in a set of %d" k (count s));
  s.ones.(k)

let on s = s.whole

(* [s]'s placement of [g]: its preallocated one where there is one. *)
let normal s g =
  match Grid.one g with
  | Some k -> s.ones.(k)
  | None -> if Grid.equal g s.whole.grid then s.whole else { set = s; grid = g }

let v ~by s g =
  Array.iter
    (fun k ->
      if k >= count s then
        invalid_argf "%s: the grid names device %d of a set of %d" by k
          (count s))
    (Grid.devices g);
  normal s g

let check_axis by axis =
  if axis < 0 || axis >= Nx_array.Layout.max_rank then
    invalid_argf "%s: axis %d is not in [0, %d)" by axis
      Nx_array.Layout.max_rank

let split ~by ~axis s =
  check_axis by axis;
  let n = count s in
  if n = 1 then s.ones.(0)
  else
    match
      Grid.v ~devices:(positions n) ~extents:[| n |] ~cuts:[| (axis, [| 0 |]) |]
    with
    | Ok g -> normal s g
    | Error e -> invalid_argf "%s: %s" by e

let mesh ~by m cuts =
  let s = m.mesh_set in
  let axis_of name =
    match Array.find_index (String.equal name) m.names with
    | Some g -> g
    | None -> invalid_argf "%s: the mesh has no axis %S" by name
  in
  let cuts =
    Array.of_list
      (List.map
         (fun (axis, names) ->
           check_axis by axis;
           (axis, Array.of_list (List.map axis_of names)))
         cuts)
  in
  match Grid.v ~devices:(positions (count s)) ~extents:m.extents ~cuts with
  | Ok g -> normal s g
  | Error e -> invalid_argf "%s: %s" by e

let anywhere = { set = host; grid = Grid.device 0 }
let set p = p.set
let grid p = p.grid
let device p = Grid.one p.grid

let equal p q =
  p == q || (p.set.number = q.set.number && Grid.equal p.grid q.grid)

let window ~by p shape i =
  if i < 0 || i >= Grid.count p.grid then
    invalid_argf "%s: position %d of a placement of %d devices" by i
      (Grid.count p.grid);
  match Grid.window p.grid shape i with
  | Ok w -> w
  | Error e -> invalid_argf "%s: %s" by e

let with_leading_axis p = normal p.set (Grid.map_axes succ p.grid)

let without_leading_axis p =
  normal p.set (Grid.map_axes pred (Grid.uncut p.grid ~axis:0))

let rebrand p = p

let pp_placement ppf p =
  if p == anywhere then Format.pp_print_string ppf "anywhere"
  else
    Grid.pp
      (fun ppf k -> Format.pp_print_string ppf (Rig.name p.set.rigs.(k)))
      ppf p.grid

(* Meshes *)

let mesh_v ~by s axes =
  let names = Array.of_list (List.map fst axes)
  and extents = Array.of_list (List.map snd axes) in
  if Array.exists (fun e -> e < 1) extents then
    invalid_argf "%s: a mesh extent is not positive" by;
  if Array.fold_left ( * ) 1 extents <> count s then
    invalid_argf
      "%s: the mesh's extents do not multiply to the set's %d devices" by
      (count s);
  Array.iteri
    (fun i n ->
      for j = 0 to i - 1 do
        if String.equal names.(j) n then
          invalid_argf "%s: the mesh names %S twice" by n
      done)
    names;
  { mesh_set = s; names; extents }

let mesh_set m = m.mesh_set
