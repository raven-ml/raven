(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Composites: an operation that computes by others refuses naming the
   caller's argument, where it lies, and the place that resolves it, never
   a value it made. Following each refusal's remedy, a composite over
   arguments placed anywhere computes the host's bits. *)

open Windtrap
open Nx_support

let m = Nx_support.memory

module S = (val Nx.devices [ m 0; m 1; m 2; m 3 ])

let mesh = Nx.Mesh.v S.v [ ("a", 2); ("b", 2) ]

(* The placements of a value of rank [r] over four devices: whole, split
   along an axis, cut along an axis over either axis of a 2x2 mesh, and
   along two axes over both. *)
let candidates r =
  let axes = List.init r Fun.id in
  let mesh_cut l = Nx.Placement.mesh mesh l in
  (S.on :: List.map (fun axis -> S.split ~axis) axes)
  @ List.concat_map
      (fun a -> [ mesh_cut [ (a, [ "a" ]) ]; mesh_cut [ (a, [ "b" ]) ] ])
      axes
  @ List.concat_map
      (fun a ->
        List.filter_map
          (fun b ->
            if a = b then None
            else Some (mesh_cut [ (a, [ "a" ]); (b, [ "b" ]) ]))
          axes)
      axes

let pp_combine ppf c =
  Format.pp_print_string ppf
    (match c with Nx.Set -> "Set" | Add -> "Add" | Max -> "Max" | Min -> "Min")

(* A composite over drawn arguments: its name, each argument's name and
   value, and its result over the arguments placed by name. *)
type case = {
  by : string;
  label : string;
  args : (string * Nx.host Nx.float64_t) list;
  run : 'd. (string -> 'd Nx.Placement.t) -> float array;
}

let ints s lo hi =
  let open Gen in
  map
    (fun vs -> Nx.create Nx.float64 s (Array.map Float.of_int vs))
    (array ~size:(constant (Array.fold_left ( * ) 1 s)) (int_range lo hi))

let placed name args at =
  let x = List.assoc name args in
  Nx.place (at name) x

let to_int64 x = Nx.cast Nx.int64 x

let segments =
  let open Gen in
  let* c = of_list [ Nx.Add; Max; Min; Set ] in
  let* n = of_list [ 0; 1; 3; 4; 5 ] in
  let* cols = int_range 1 3 in
  let* k = int_range 0 3 in
  let+ ids = ints [| n |] (-1) k and+ x = ints [| n; cols |] (-4) 4 in
  let args = [ ("ids", ids); ("x", x) ] in
  let run at =
    Nx.to_array
      (Nx.reduce_segments c ~segments:k
         (to_int64 (placed "ids" args at))
         (placed "x" args at))
  in
  {
    by = "Nx.reduce_segments";
    label =
      Format.asprintf "reduce_segments %a, %d rows of %d" pp_combine c n cols;
    args;
    run;
  }

let ranges =
  let open Gen in
  let* c = of_list [ Nx.Add; Max; Min; Set ] in
  let* n = of_list [ 1; 3; 4; 5 ] in
  let* cols = int_range 1 3 in
  let* r = of_list [ 1; 2; 4 ] in
  let+ lo = ints [| r |] (-1) (n + 1)
  and+ hi = ints [| r |] (-1) (n + 1)
  and+ x = ints [| n; cols |] (-4) 4 in
  let args = [ ("lo", lo); ("hi", hi); ("x", x) ] in
  let run at =
    Nx.to_array
      (Nx.reduce_ranges c
         ~lo:(to_int64 (placed "lo" args at))
         ~hi:(to_int64 (placed "hi" args at))
         (placed "x" args at))
  in
  {
    by = "Nx.reduce_ranges";
    label =
      Format.asprintf "reduce_ranges %a, %d rows of %d" pp_combine c n cols;
    args;
    run;
  }

let placed_case =
  let open Gen in
  with_pp
    (fun ppf (c, at) ->
      Format.fprintf ppf "%s, %a" c.label
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
           (fun ppf (name, p) ->
             Format.fprintf ppf "%s at %a" name Nx.Placement.pp p))
        at)
    (let* c = one_of [ segments; ranges ] in
     let place (name, x) =
       map (fun p -> (name, p)) (of_list (candidates (Nx.ndim x)))
     in
     let+ at =
       List.fold_right
         (fun arg rest ->
           let+ a = place arg and+ rest = rest in
           a :: rest)
         c.args (constant [])
     in
     (c, at))

(* The positions and bounds reach the composite as an int64 cast of the
   argument, which lies where the argument does: the cast is the caller's,
   and the composite names the value it receives. *)
let law (c, at) =
  let calls = ref 0 in
  let run at =
    incr calls;
    c.run at
  in
  let host = c.run (fun _ -> Nx.Host.on) in
  let candidates name = candidates (Nx.ndim (List.assoc name c.args)) in
  let got = remedied ~by:c.by ~candidates at run in
  cover "a refusal followed" (!calls > 1);
  cover "computed where placed" (!calls = 1);
  cover "reduce_segments refused" (c.by = "Nx.reduce_segments" && !calls > 1);
  cover "reduce_ranges refused" (c.by = "Nx.reduce_ranges" && !calls > 1);
  equal_elements Nx_array.Dtype.Float64 host got

let () =
  exit
    (run "nx composites"
       [
         group "refusals"
           [
             prop
               "a composite's refusal names the caller's argument and the \
                place that resolves it"
               placed_case law;
           ];
       ])
