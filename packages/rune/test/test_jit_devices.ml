(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jit over the memory devices of one set: a compiled call over values cut
   across two, three or four devices, a 2 x 2 mesh, or whole on each, gives
   eager's placements, exact functions' bits and the other functions' values
   within rune.mli's classes, for even, uneven and empty windows. It compiles
   (no call is evaluated), each device runs its own kernels, and its gradient is
   eager's. *)

open Windtrap
open Rune_support
open Witness
module T = Nx.Ptree

let f32 = Nx.float32
let m = Nx_support.memory

module Two = (val Nx.devices [ m 0; m 1 ])
module Three = (val Nx.devices [ m 0; m 1; m 2 ])
module Four = (val Nx.devices [ m 0; m 1; m 2; m 3 ])

let mesh = Nx.Mesh.v Four.v [ ("x", 2); ("y", 2) ]

(* Placements

   A placement over one of the sets: cut along an axis, whole on each device,
   whole on one, or over the mesh's two axes. *)

type cut = On | Split of int | Device of int | Mesh of int * int

let pp_cut ppf = function
  | On -> Format.pp_print_string ppf "on"
  | Split a -> Format.fprintf ppf "split ~axis:%d" a
  | Device k -> Format.fprintf ppf "device %d" k
  | Mesh (a, b) -> Format.fprintf ppf "mesh axis %d over x, %d over y" a b

(* A set, and the mesh over it where it has four devices. *)
type 'd set = { s : 'd Nx.devices; mesh : 'd Nx.Mesh.t option }

let at (type d) (set : d set) c : d Nx.Placement.t =
  match (c, set.mesh) with
  | On, _ -> Nx.Placement.on set.s
  | Split axis, _ -> Nx.Placement.split ~axis set.s
  | Device k, _ -> Nx.Placement.device set.s (m k)
  | Mesh (a, b), Some mesh ->
      Nx.Placement.mesh mesh [ (a, [ "x" ]); (b, [ "y" ]) ]
  | Mesh _, None -> invalid_arg "a mesh of four devices"

(* A law over the set of [n] devices. *)
type law = { law : 'd. 'd set -> unit }

let over n { law } =
  match n with
  | 2 -> law { s = Two.v; mesh = None }
  | 3 -> law { s = Three.v; mesh = None }
  | _ -> law { s = Four.v; mesh = Some mesh }

(* A value of [shape] on the host, of elements in [-1, 1] that depend on [seed]
   and their index. *)
let value ?(seed = 0) shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create f32 shape
    (Array.init n (fun i -> sin (float_of_int ((seed * 7919) + i + 1))))

(* [f] compiled over [x] and [y] gives [eager], computing no call evaluated. *)
let compiled f x y =
  let before = Rune.evaluations () in
  let r =
    outcome (fun () -> Rune.jit T.(tensor @-> tensor @-> returns tensor) f x y)
  in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  r

let omap f = function Value v -> Value (f v) | Raised m -> Raised m
let placement () = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

(* [jit] lies where [eager] does, within [bound] of it elementwise, NaN where it
   is NaN. *)
let within ~bound eager jit =
  equal ~msg:"placement" (placement ()) (Nx.placement eager) (Nx.placement jit);
  let e = Nx.to_array (Nx.cast Nx.float64 eager) in
  let j = Nx.to_array (Nx.cast Nx.float64 jit) in
  let b = Nx.to_array bound in
  Array.iteri
    (fun i ei ->
      if Float.is_nan ei then equal ~msg:"NaN" bool true (Float.is_nan j.(i))
      else
        less
          ~msg:(Printf.sprintf "distance at %d" i)
          float_exact
          ~than:(b.(i) +. Float.min_float)
          (Float.abs (j.(i) -. ei)))
    e

(* [γ(k)] at [float32]. *)
let gamma k =
  let u = ldexp 1. (-24) in
  let k = float_of_int k in
  k *. u /. (1. -. (k *. u))

(* Cases

   A set of [n] devices, a shape whose cut axis has [0] to [2n + 1] elements, so
   that windows come uneven and empty, and placements of the operands. *)

type case = { n : int; shape : int array; x : cut; y : cut }

let pp_case ppf c =
  Format.fprintf ppf "%d devices, shape %a, x %a, y %a" c.n Operand.pp_shape
    c.shape pp_cut c.x pp_cut c.y

let cut_of ~n ~rank =
  let open Gen in
  let mesh =
    if n = 4 && rank = 2 then
      [ (1, constant (Mesh (0, 1))); (1, constant (Mesh (1, 0))) ]
    else []
  in
  frequency
    ([
       (1, constant On);
       (4, map (fun a -> Split a) (int_range 0 (rank - 1)));
       (1, map (fun k -> Device k) (int_range 0 (n - 1)));
     ]
    @ mesh)

let cases =
  let open Gen in
  (let* n = int_range 2 4 in
   let* rank = int_range 1 2 in
   let* shape = array ~size:(constant rank) (int_range 0 ((2 * n) + 1)) in
   let* x = cut_of ~n ~rank and+ y = cut_of ~n ~rank in
   constant { n; shape; x; y })
  |> with_pp pp_case

let cover_windows c =
  let cut_axis = function
    | Split a -> Some a
    | Mesh (a, _) -> Some a
    | On | Device _ -> None
  in
  match cut_axis c.x with
  | None -> cover "x whole" true
  | Some a ->
      let e = c.shape.(a) in
      cover "an empty window" (e < c.n);
      cover "uneven windows" (e mod c.n <> 0 && e > c.n);
      cover "no element" (e = 0);
      cover "even windows" (e mod c.n = 0 && e > 0)

(* Elementwise functions are exact, so a compiled call gives eager's bits. *)
let elementwise set c =
  cover_windows c;
  cover "a mesh" (match c.x with Mesh _ -> true | _ -> false);
  cover "three devices" (c.n = 3);
  let x = Nx.place (at set c.x) (value c.shape) in
  let y = Nx.place (at set c.y) (value ~seed:1 c.shape) in
  let f x y = Nx.maximum (Nx.mul x y) (Nx.sub x (Nx.neg y)) in
  let eager = outcome (fun () -> f x y) in
  let jit = compiled f x y in
  match (eager, jit) with
  | Value e, Value j ->
      equal ~msg:"placement" (placement ()) (Nx.placement e) (Nx.placement j);
      equal bits_t (bits e) (bits j)
  | _ -> equal (outcome_t bits_t) (omap bits eager) (omap bits jit)

(* A sum over each axis, cut or not, and over all of them, within twice the
   bound of each. *)
let folds set c =
  cover_windows c;
  let x = Nx.place (at set c.x) (value c.shape) in
  let rank = Array.length c.shape in
  let check axes =
    let f x _ = Nx.sum ~axes x in
    let eager = f x x in
    match compiled f x x with
    | Value jit ->
        let n = List.fold_left (fun k a -> k * c.shape.(a)) 1 axes in
        let mass =
          Nx.sum ~axes (Nx.abs (Nx.cast Nx.float64 (Nx.place Nx.Host.on x)))
        in
        within ~bound:(Nx.mul_s mass (2. *. gamma n)) eager jit
    | r -> equal (outcome_t bits_t) (Value (bits eager)) (omap bits r)
  in
  List.iter (fun a -> check [ a ]) (List.init rank Fun.id);
  check (List.init rank Fun.id)

(* A matrix product with each operand cut along either axis or whole, within
   twice the bound of each. *)
let matmul set c =
  match c.shape with
  | [| rows; inner |] -> (
      let cols = 3 in
      let a = Nx.place (at set c.x) (value [| rows; inner |]) in
      let b = Nx.place (at set c.y) (value ~seed:1 [| inner; cols |]) in
      cover "a cut along k" (c.x = Split 1 || c.y = Split 0);
      cover "a cut along rows" (c.x = Split 0);
      let f a b = Nx.matmul a b in
      let eager = outcome (fun () -> f a b) in
      match (eager, compiled f a b) with
      | Value e, Value j ->
          let host x = Nx.abs (Nx.cast Nx.float64 (Nx.place Nx.Host.on x)) in
          let mass = Nx.matmul (host a) (host b) in
          within ~bound:(Nx.mul_s mass (2. *. gamma (inner + 1))) e j
      | e, j -> equal (outcome_t bits_t) (omap bits e) (omap bits j))
  | _ -> assume false

(* A broadcast of a vector against a batch cut across the devices, and its sum,
   as a bias and a loss read them. *)
let biased set c =
  let x = Nx.place (at set c.x) (value c.shape) in
  let last = c.shape.(Array.length c.shape - 1) in
  let b = Nx.place (at set On) (value ~seed:2 [| last |]) in
  let f x b = Nx.add x b in
  let eager = outcome (fun () -> f x b) in
  equal (outcome_t bits_t) (omap bits eager) (omap bits (compiled f x b))

(* Gradients

   The gradient of a loss over a batch cut across the devices, under jit, lies
   where eager's does, within its error. *)

let grads set rows =
  let x = Nx.place (at set (Split 0)) (value [| rows; 4 |]) in
  let w = Nx.place (at set On) (value ~seed:1 [| 4; 3 |]) in
  let loss x w = Nx.sum (Nx.tanh (Nx.matmul x w)) in
  let g x w = Rune.grad' (loss x) w in
  let eager = g x w in
  let jit = compiled g x w in
  match jit with
  | Value jit ->
      (* [g = xᵀ (1 - t²)] for [t = tanh (x w)]: [x w] is within [γ(5) |x| |w|],
         which moves [1 - t²] by twice that at most; [t]'s 18 ulps move it by 36
         [u], its square and difference by 2 [u] more; the product with [xᵀ]
         adds [γ(rows + 1)]. Eager and compiled each hold the bound. *)
      let host x = Nx.abs (Nx.cast Nx.float64 (Nx.place Nx.Host.on x)) in
      let ax = host x in
      let u = ldexp 1. (-24) in
      let term =
        Nx.add_s
          (Nx.mul_s (Nx.matmul ax (host w)) (2. *. gamma 5))
          ((38. *. u) +. gamma (rows + 1))
      in
      let bound = Nx.mul_s (Nx.matmul (Nx.transpose ax) term) 2. in
      within ~bound eager jit
  | r -> equal (outcome_t bits_t) (Value (bits eager)) (omap bits r)

(* Kernels

   Each device that holds elements of an operation's results runs a kernel of
   its own for its window, and a device whose window is empty runs none. A
   memory device's kernels run on its host, which profiles a submission's
   kernels by name, joined by commas. *)

let kernels set n rows =
  let x = Nx.place (at set (Split 0)) (value [| rows; 3 |]) in
  let f x _ = Nx.exp x in
  let step = Rune.jit T.(tensor @-> tensor @-> returns tensor) f in
  ignore (Nx.to_array (step x x));
  let _, events =
    Rig.Profile.take (fun () -> ignore (Nx.to_array (step x x)))
  in
  let launched =
    List.concat_map
      (function
        | Rig.Profile.Work { name; _ } when name <> "copies" ->
            String.split_on_char ',' name
        | _ -> [])
      events
  in
  let holding = min n rows in
  equal ~msg:"kernels" int holding (List.length launched)

(* Moves

   Positions that a placement leaves whole along an axis they repeat, as a
   broadcast does, move one element along it, as eager moves them: a take along
   axis 1 of a batch cut along axis 0, by a row of positions cut along axis 1
   and broadcast along axis 0, copies between the devices the row's windows
   alone. *)

let moves set =
  let x = Nx.place (at set (Split 0)) (value [| 4; 3 |]) in
  let row = Nx.create Nx.int64 [| 1; 3 |] [| 2L; 0L; 1L |] in
  let row = Nx.place (at set (Split 1)) row in
  let f x row = Nx.take_along_axis ~axis:1 (Nx.broadcast_to [| 4; 3 |] row) x in
  let step = Rune.jit T.(tensor @-> tensor @-> returns tensor) f in
  equal bits_t (bits (f x row)) (bits (step x row));
  let _, events =
    Rig.Profile.take (fun () -> ignore (Nx.to_array (step x row)))
  in
  let copied =
    List.fold_left
      (fun n -> function
        | Rig.Profile.Copy { src; dst; bytes; _ } when not (Rig.equal src dst)
          ->
            n + bytes
        | _ -> n)
      0 events
  in
  equal ~msg:"bytes copied" int (3 * 8) copied

(* Programs

   A program for one device serves every device of its target, so a
   data-parallel step over [n] memory devices makes as many programs whatever
   [n]: each device's kernels, its moves of windows included, are another's.
   The hidden width depends on [n], so no two counts share a program, and every
   cut is even, so each device's windows have one shape. *)

let sixteen =
  Array.init 16 (fun k ->
      Result.get_ok (Rig.memory_device (Printf.sprintf "programs-%d" k)))

let programs n =
  let module S = (val Nx.devices (Array.to_list (Array.sub sixteen 0 n))) in
  let hidden = 16 + n in
  let x = Nx.place (Nx.Placement.split ~axis:0 S.v) (value [| 4 * n; 32 |]) in
  let on = Nx.Placement.on S.v in
  let w1 = Nx.place on (value ~seed:1 [| 32; hidden |]) in
  let w2 = Nx.place on (value ~seed:2 [| hidden; 4 |]) in
  let step x (w1, w2) =
    let loss (w1, w2) =
      let y = Nx.matmul (Nx.tanh (Nx.matmul x w1)) w2 in
      Nx.mean (Nx.mul y y)
    in
    let g1, g2 = Rune.grad T.(pair tensor tensor) loss (w1, w2) in
    (Nx.sub w1 (Nx.mul_s g1 0.1), Nx.sub w2 (Nx.mul_s g2 0.1))
  in
  let f =
    Rune.jit
      T.(tensor @-> pair tensor tensor @-> returns (pair tensor tensor))
      step
  in
  let before = Tolk.Codegen.programs () and evaluated = Rune.evaluations () in
  ignore (f x (w1, w2));
  equal ~msg:"calls evaluated" int evaluated (Rune.evaluations ());
  Tolk.Codegen.programs () - before

let examples =
  [
    { n = 2; shape = [| 4; 3 |]; x = Split 0; y = On };
    { n = 3; shape = [| 5; 2 |]; x = Split 0; y = Split 0 };
    { n = 4; shape = [| 2; 3 |]; x = Split 0; y = Device 1 };
    { n = 4; shape = [| 4; 4 |]; x = Mesh (0, 1); y = On };
    { n = 2; shape = [| 0; 3 |]; x = Split 0; y = On };
  ]

let () =
  exit
    (run "Rune.jit over several devices"
       [
         group "a compiled call gives eager's placement and values"
           [
             prop ~count:30 ~examples "elementwise, bit for bit" cases (fun c ->
                 over c.n { law = (fun set -> elementwise set c) });
             prop ~count:20 ~examples "sums over cut and whole axes" cases
               (fun c -> over c.n { law = (fun set -> folds set c) });
             prop ~count:20 ~examples "a matrix product" cases (fun c ->
                 over c.n { law = (fun set -> matmul set c) });
             prop ~count:20 ~examples "a vector broadcast to a cut batch" cases
               (fun c -> over c.n { law = (fun set -> biased set c) });
           ];
         group "gradients"
           [
             test "over two devices" (fun () ->
                 over 2 { law = (fun s -> grads s 6) });
             test "over three, uneven" (fun () ->
                 over 3 { law = (fun s -> grads s 7) });
             test "over four, one window empty" (fun () ->
                 over 4 { law = (fun s -> grads s 3) });
           ];
         group "moves"
           [
             test "positions repeated along an axis move one element along it"
               (fun () -> over 2 { law = moves });
           ];
         group "programs"
           [
             test "a data-parallel step makes as many over 2, 4, 8 and 16"
               (fun () ->
                 let counts = List.map programs [ 2; 4; 8; 16 ] in
                 let first = List.hd counts in
                 equal (list int) (List.map (fun _ -> first) counts) counts);
           ];
         group "each device runs its own kernels"
           [
             test "over two devices" (fun () ->
                 over 2 { law = (fun s -> kernels s 2 4) });
             test "over four, one window empty" (fun () ->
                 over 4 { law = (fun s -> kernels s 4 3) });
           ];
       ])
