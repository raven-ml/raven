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
  cover "reduce_ranges refused" (c.by = "Nx.reduce_ranges" && !calls > 1);
  equal_elements Nx_array.Dtype.Float64 host got

(* Nx.Prim.composite's promises, over programs of calls that share the
   domain's state: each call's refusal is named as nx.mli states, whatever
   the calls before it did, a collection, an exception or a domain. *)

module S2 = (val Nx.devices [ m 0; m 1 ])

let message f =
  match f () with
  | _ -> "no refusal"
  | exception Invalid_argument m -> m
  | exception e -> Printexc.to_string e

(* The refusal by [by] of its argument [name], read whole where it lies
   split. *)
let read_whole by name =
  Printf.sprintf
    "%s: %s lies at split ~axis:0 [m0; m1] and is read at on [m0; m1]; \
     Nx.place (Placement.on s) %s"
    by name name

(* The operation [by]'s own refusal of its first operand so. *)
let unnamed by =
  Printf.sprintf
    "%s: operand 0 lies at split ~axis:0 [m0; m1] and is read at on [m0; \
     m1]; Nx.place (Placement.on s) x"
    by

(* A sort reads its axis whole, so it refuses an operand split along it. *)
let sorted x = fst (Nx.sort x)

let value = abstract "v"

let split () = Nx.place (S2.split ~axis:0) (Nx.arange Nx.int64 0 4 1)

let open_ ~by v f = Nx.Prim.composite ~by [ Arg ("x", v) ] f

(* [depth] composites, each over a value the outer one derives, with a
   collection before the refusal: the outermost's argument is named. *)
let nested (depth, collect) v =
  let rec go d x =
    if d > depth then begin
      if collect then Gc.full_major ();
      sorted (Nx.neg x)
    end
    else open_ ~by:(Printf.sprintf "c%d" d) x (fun () -> go (d + 1) (Nx.neg x))
  in
  message (fun () -> open_ ~by:"c0" v (fun () -> go 1 (Nx.neg v)))

(* A composite that returned a value derived from its argument, an outer
   one's: refused after it ends, the outer one's argument is named. *)
let after_inner v =
  message (fun () ->
      open_ ~by:"c0" v (fun () ->
          let a = Nx.neg v in
          let inner = open_ ~by:"c1" a (fun () -> Nx.neg a) in
          sorted (Nx.neg inner)))

(* An argument whole on every device beside the split one: the remedy
   leaves it where it lies, so the split one is named. *)
let passed_over v =
  let whole = Nx.place S2.on (Nx.arange Nx.int64 0 4 1) in
  message (fun () ->
      Nx.Prim.composite ~by:"two" [ Arg ("a", whole); Arg ("b", v) ] (fun () ->
          sorted (Nx.add whole v)))

(* A value the composite makes, derived from no argument: the operation's
   own message. *)
let underived v =
  message (fun () ->
      open_ ~by:"c0" v (fun () ->
          sorted (Nx.zeros_at (S2.split ~axis:0) Nx.int64 [| 4 |])))

(* A body that raises another exception leaves nothing open behind it. *)
let raising v =
  message (fun () ->
      open_ ~by:"c0" v (fun () ->
          ignore (Nx.neg v);
          raise Exit))

(* Outside any composite, the operation's own message. *)
let plain v = message (fun () -> sorted (Nx.neg v))

(* Another domain notes nothing: its refusal is the operation's own. *)
let other_domain v =
  open_ ~by:"c0" v (fun () ->
      let derived = Nx.neg v in
      Domain.join (Domain.spawn (fun () -> message (fun () -> sorted derived))))

(* map_segments over ids whole and rows split: a refusal inside [f], of a
   value derived from its rows, names [x]. *)
let in_f v =
  let ids = Nx.place S2.on (Nx.create Nx.int64 [| 4 |] [| 0L; 1L; 0L; 1L |]) in
  let x = Nx.reshape [| 4; 1 |] (Nx.cast Nx.float32 v) in
  let entered = ref false in
  let f _ rows =
    entered := true;
    Nx.add_s (fst (Nx.sort ~axis:0 rows)) 0.
  in
  let m = message (fun () -> Nx.map_segments ~segments:2 ids f x) in
  if !entered then m else "f not entered"

let commands =
  let sort = "Nx.sort" in
  let state =
    Gen.with_pp
      (fun ppf (d, c) -> Format.fprintf ppf "%d%s" d (if c then " gc" else ""))
      Gen.(pair (int_range 1 3) bool)
  in
  [
    command "split" (Gen.unit @-> makes value) (fun () -> ()) split;
    command "nested" (state @-> value ^-> returns string)
      (fun _ () -> read_whole "c0" "x") nested;
    command "after_inner" (value ^-> returns string)
      (fun () -> read_whole "c0" "x") after_inner;
    command "passed_over" (value ^-> returns string)
      (fun () -> read_whole "two" "b") passed_over;
    command "underived" (value ^-> returns string)
      (fun () -> unnamed sort) underived;
    command "raising" (value ^-> returns string)
      (fun () -> Printexc.to_string Exit) raising;
    command "plain" (value ^-> returns string) (fun () -> unnamed sort) plain;
    command "other_domain" (value ^-> returns string)
      (fun () -> unnamed sort) other_domain;
    command "in_f" (value ^-> returns string)
      (fun () -> read_whole "Nx.map_segments" "x") in_f;
  ]

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
             stateful ~count:50
               "Nx.Prim.composite names the outermost frame's argument, \
                passes over one the remedy leaves, and leaves no state"
               commands;
           ];
       ])
