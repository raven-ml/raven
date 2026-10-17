(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module M = Nx_array.Move
open Nx_array_gen

let ( let* ) = Option.bind
let ints = array int
let guard b = if b then Some () else None
let max_numel = Nx_array.Layout.max_numel

(* The shape a movement gives, by its preconditions *)

(* The product of [s]'s extents, [None] if the product of those other than 0
   passes [max_numel]. *)
let numel s =
  let* n =
    Array.fold_left
      (fun n d ->
        let* n = n in
        if d = 0 then Some n else if d > max_numel / n then None else Some (n * d))
      (Some 1) s
  in
  Some (if Array.mem 0 s then 0 else n)

let in_axis d i = 0 <= i && i < d

(* Whether [r]'s last element lies in [\[0, d)], without forming it. *)
let ends_in d (r : M.range) =
  let k = r.count - 1 in
  if k = 0 then true
  else if r.step > 0 then k <= (d - 1 - r.start) / r.step
  else r.step <> min_int && k <= r.start / -r.step

let valid_range d (r : M.range) =
  r.step <> 0 && r.count >= 0
  && (r.count = 0 || (in_axis d r.start && ends_in d r))

(* Whether [w]'s windows fit an axis of extent [d]: [dilation·(size - 1) <= d -
   1]. *)
let fits d (w : M.window) =
  w.size >= 1 && w.step >= 1 && w.dilation >= 1 && d >= 1
  && (w.size = 1 || w.dilation <= (d - 1) / (w.size - 1))

let windows d (w : M.window) =
  ((d - 1 - (w.dilation * (w.size - 1))) / w.step) + 1

let rec increasing = function
  | (a : M.window) :: (b :: _ as rest) -> a.axis < b.axis && increasing rest
  | _ -> true

(* The shape of [m]'s result on [s], or [None] where [shape] must raise. *)
let expected m s =
  let r = Array.length s in
  let* () = guard (Array.for_all (fun d -> d >= 0) s) in
  let* _ = numel s in
  let* s' =
    match m with
    | M.Reshape s' ->
        let* () = guard (Array.for_all (fun d -> d >= 0) s') in
        let* n = numel s in
        let* n' = numel s' in
        let* () = guard (n = n') in
        Some s'
    | M.Broadcast s' ->
        let off = Array.length s' - r in
        let* () = guard (Array.for_all (fun d -> d >= 0) s' && off >= 0) in
        let* () =
          guard
            (Array.for_all Fun.id
               (Array.mapi (fun i d -> d = 1 || d = s'.(off + i)) s))
        in
        Some s'
    | M.Permute p ->
        let sorted = List.sort compare (Array.to_list p) in
        let* () = guard (sorted = List.init r Fun.id) in
        Some (Array.map (fun a -> s.(a)) p)
    | M.Slice rs ->
        let* () = guard (Array.length rs = r) in
        let* () =
          guard
            (Array.for_all Fun.id
               (Array.mapi (fun i x -> valid_range s.(i) x) rs))
        in
        Some (Array.map (fun (x : M.range) -> x.count) rs)
    | M.Window ws ->
        let ws = Array.to_list ws in
        let* () =
          guard
            (increasing ws
            && List.for_all
                 (fun (w : M.window) -> in_axis r w.axis && fits s.(w.axis) w)
                 ws)
        in
        let s' = Array.copy s in
        List.iter (fun (w : M.window) -> s'.(w.axis) <- windows s.(w.axis) w) ws;
        Some
          (Array.append s'
             (Array.of_list (List.map (fun (w : M.window) -> w.size) ws)))
  in
  let* () = guard (Array.length s' <= Nx_array.Layout.max_rank) in
  let* _ = numel s' in
  Some s'

let constructor = function
  | M.Reshape _ -> "reshape"
  | M.Broadcast _ -> "broadcast"
  | M.Permute _ -> "permute"
  | M.Slice _ -> "slice"
  | M.Window _ -> "window"

let constructors = [ "reshape"; "broadcast"; "permute"; "slice"; "window" ]

let law_shape (s, m) =
  let e = expected m s in
  List.iter
    (fun c ->
      let here = c = constructor m in
      cover (c ^ " accepted") (here && e <> None);
      cover (c ^ " refused") (here && e = None))
    constructors;
  match e with
  | Some s' -> equal ints s' (M.shape m s)
  | None -> raises_match Exn.invalid_arg (fun () -> M.shape m s)

(* Values the interface states, at the bounds *)

type case = { name : string; m : M.t; s : int array; shape : int array option }

let r start count step = { M.start; count; step }
let w axis size step dilation = { M.axis; size; step; dilation }
let ok name m s shape = { name; m; s; shape = Some shape }
let refused name m s = { name; m; s; shape = None }

let bounds =
  [
    ok "a window counts its windows and appends its extent"
      (M.Window [| w 1 3 2 2 |])
      [| 2; 11; 4 |] [| 2; 4; 4; 3 |];
    ok "a window whose reach is its axis is one window"
      (M.Window [| w 0 3 1 2 |])
      [| 5 |] [| 1; 3 |];
    refused "a window whose reach passes its axis"
      (M.Window [| w 0 3 1 2 |])
      [| 4 |];
    ok "a window of the greatest step is one window"
      (M.Window [| w 0 1 max_int 1 |])
      [| 3 |] [| 1; 1 |];
    refused "a dilation whose reach overflows"
      (M.Window [| w 0 3 1 max_int |])
      [| 3 |];
    refused "a window on an empty axis" (M.Window [| w 0 1 1 1 |]) [| 0 |];
    ok "windows on the most axes"
      (M.Window [| w 0 1 1 1 |])
      (Array.make 31 1) (Array.make 32 1);
    refused "a window past the most axes"
      (M.Window [| w 0 1 1 1 |])
      (Array.make 32 1);
    ok "a slice of no element may start anywhere"
      (M.Slice [| r 100 0 (-7) |])
      [| 3 |] [| 0 |];
    ok "a slice of one element may take any step"
      (M.Slice [| r 2 1 min_int |])
      [| 4 |] [| 1 |];
    ok "a slice may end on the axis' last element"
      (M.Slice [| r 0 2 3 |])
      [| 4 |] [| 2 |];
    refused "a slice whose step overflows" (M.Slice [| r 1 2 max_int |]) [| 4 |];
    refused "a reversed slice of the least step"
      (M.Slice [| r 3 2 min_int |])
      [| 4 |];
    ok "a reshape to the most axes"
      (M.Reshape (Array.make 32 1))
      [| 1 |] (Array.make 32 1);
    refused "a reshape past the most axes" (M.Reshape (Array.make 33 1)) [| 1 |];
    refused "a reshape whose elements overflow"
      (M.Reshape [| (max_int / 2) + 1; 2 |])
      [| 4 |];
    ok "a reshape to max_numel elements"
      (M.Reshape [| max_numel / 2; 2 |])
      [| max_numel |] [| max_numel / 2; 2 |];
    refused "a reshape past max_numel elements"
      (M.Reshape [| (max_numel / 2) + 1; 2 |])
      [| max_numel + 2 |];
    refused "a reshape of no element to some" (M.Reshape [| 3 |]) [| 0 |];
    refused "a reshape to negative extents whose product matches"
      (M.Reshape [| -2; -3 |])
      [| 6 |];
    ok "a reshape to no element takes other extents up to max_numel"
      (M.Reshape [| max_numel / 2; 2; 0 |])
      [| 0 |] [| max_numel / 2; 2; 0 |];
    refused "a reshape to no element whose other extents pass max_numel"
      (M.Reshape [| (max_numel / 2) + 1; 2; 0 |])
      [| 0 |];
    refused "a reshape to no element whose other extents overflow"
      (M.Reshape [| max_int; 2; 0 |])
      [| 0 |];
    ok "a broadcast of an extent 1 to 0"
      (M.Broadcast [| 0; 3 |])
      [| 1; 3 |] [| 0; 3 |];
    refused "a broadcast past the most axes"
      (M.Broadcast (Array.make 33 1))
      [| 1 |];
    refused "a broadcast whose elements overflow"
      (M.Broadcast [| max_int; 2 |])
      [| 1; 2 |];
    ok "a broadcast to max_numel elements"
      (M.Broadcast [| max_numel / 2; 2 |])
      [| 1; 2 |] [| max_numel / 2; 2 |];
    refused "a broadcast past max_numel elements"
      (M.Broadcast [| (max_numel / 2) + 1; 2 |])
      [| 1; 2 |];
    refused "a broadcast to no element whose other extents pass max_numel"
      (M.Broadcast [| (max_numel / 2) + 1; 2; 0 |])
      [| 1 |];
    refused "a window of an empty argument whose other extents pass max_numel"
      (M.Window [| w 1 1 1 1 |])
      [| 0; 2; (max_numel / 2) + 1 |];
    refused "a permutation whose elements overflow"
      (M.Permute [| 1; 0 |])
      [| max_int; 2 |];
    refused "a permutation of an argument past max_numel elements"
      (M.Permute [| 1; 0 |])
      [| max_numel + 1; 1 |];
    refused "a slice of an empty argument whose other extents pass max_numel"
      (M.Slice [| r 0 0 1; r 0 1 1; r 0 1 1 |])
      [| 0; 1 lsl 27; 1 lsl 27 |];
    ok "a permutation of no axis" (M.Permute [||]) [||] [||];
    refused "a permutation of an argument past the most axes"
      (M.Permute (Array.init 33 Fun.id))
      (Array.make 33 1);
    refused "a permutation of 64 axes"
      (M.Permute (Array.init 64 Fun.id))
      (Array.make 64 1);
    refused "a slice of an argument past the most axes"
      (M.Slice (Array.make 40 (r 0 1 1)))
      (Array.make 40 1);
  ]

let test_bound c =
  match c.shape with
  | Some s' -> equal ints s' (M.shape c.m c.s)
  | None -> raises_match Exn.invalid_arg (fun () -> M.shape c.m c.s)

(* Ownership *)

let test_fresh () =
  let s' = [| 3; 2 |] and s = [| 6 |] in
  let out = M.shape (M.Reshape s') s in
  out.(0) <- 0;
  equal ints [| 3; 2 |] s';
  let s' = [| 4; 6 |] in
  let out = M.shape (M.Broadcast s') s in
  out.(0) <- 0;
  equal ints [| 4; 6 |] s';
  let out = M.shape (M.Permute [| 0 |]) s in
  out.(0) <- 0;
  equal ints [| 6 |] s

(* A refusal names the axis at fault, so that the caller sees which extent
   breaks the rule. *)
let messages =
  let w axis = { M.axis; size = 1; step = 1; dilation = 1 } in
  [
    ( "a broadcast names the axis that neither is 1 nor matches",
      (fun () -> M.shape (M.Broadcast [| 3; 5 |]) [| 3; 4 |]),
      "Move.Broadcast: [3; 4] does not broadcast to [3; 5]: axis 1 has 4, \
       neither 1 nor 5" );
    ( "a broadcast's axis counts from the argument's first",
      (fun () -> M.shape (M.Broadcast [| 2; 3; 5 |]) [| 4; 1 |]),
      "axis 0 has 4, neither 1 nor 3" );
    ( "a negative extent names its axis",
      (fun () -> M.shape (M.Reshape [| 2; -3 |]) [| 6 |]),
      "Move.Reshape: axis 1 of [2; -3] is -3, below 0" );
    ( "a repeated axis names its entry",
      (fun () -> M.shape (M.Permute [| 0; 1; 0 |]) [| 2; 3; 4 |]),
      "entry 2 repeats axis 0" );
    ( "an axis past the argument names its entry",
      (fun () -> M.shape (M.Permute [| 0; 3 |]) [| 2; 3 |]),
      "entry 1, 3, is not an axis of 2" );
    ( "windows out of order name both",
      (fun () -> M.shape (M.Window [| w 1; w 0 |]) [| 3; 3 |]),
      "window 1's axis 0 is not after window 0's axis 1" );
  ]

let test_message (_, f, text) =
  raises_match (Exn.invalid_arg ~substring:text) (fun () -> ignore (f ()))

let tests =
  [
    cases
      ~name:(fun (n, _, _) -> n)
      "refusals name the axis at fault" messages test_message;
    group "shape"
      [
        prop ~count:1000
          "is the result's shape, and raises iff a precondition fails"
          shape_and_move law_shape;
        cases ~name:(fun c -> c.name) "states the bounds" bounds test_bound;
        test "returns an array it does not hold" test_fresh;
      ];
  ]

let () = exit (run "nx_array.move" tests)
