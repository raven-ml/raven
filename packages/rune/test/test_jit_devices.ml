(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jit over the software devices of one set: a compiled call over values
   cut across two, three or four devices, a 2 x 2 mesh, or whole on each, gives
   eager's placements, exact functions' bits and the other functions' values
   within rune.mli's classes, for even, uneven and empty windows. It compiles
   (no call is evaluated), each device runs its own kernels, and its gradient
   is eager's. A map over an axis cut one row per device compiles as
   per-device code. *)

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
type 'd set = { s : 'd Nx.Devices.t; mesh : 'd Nx.Mesh.t option }

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

(* [γ(k, 2u)] at [float32], a contraction's bound ({!Nx.contract}). *)
let product k =
  let v = ldexp 1. (-23) in
  let k = float_of_int k in
  k *. v /. (1. -. (k *. v))

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
          within ~bound:(Nx.mul_s mass (2. *. product (inner + 1))) e j
      | e, j -> equal (outcome_t bits_t) (omap bits e) (omap bits j))
  | _ -> assume false

(* An expert product of an MXFP4 weight cut along [k], its scales cut with
   its codes where the devices divide its blocks and whole on each device
   otherwise, beside an input whole or cut alike: within twice the bound of
   each, and the same bits on a second compiled call. *)

type experts = {
  devices : int;
  e : int;
  outs : int;
  blocks : int;
  ids : int array;
  x_cut : bool;
}

let pp_experts ppf c =
  Format.fprintf ppf "%d devices, %d experts of %d by %d blocks, ids [%s]%s"
    c.devices c.e c.outs c.blocks
    (String.concat "; " (Array.to_list (Array.map string_of_int c.ids)))
    (if c.x_cut then ", x cut alike" else "")

let experts_cases =
  let open Gen in
  (let* devices = int_range 2 4 in
   let* e = int_range 1 4 and+ outs = int_range 1 3 in
   (* Three devices divide the [16 b] bytes of codes only where they divide
      the blocks. *)
   let* blocks = if devices = 3 then constant 3 else int_range 1 4 in
   let* ids = array ~size:(int_range 0 5) (int_range (-1) e) in
   let+ x_cut = bool in
   { devices; e; outs; blocks; ids; x_cut })
  |> with_pp pp_experts

(* Three devices over three blocks, an input cut alike and an id below the
   range; scales whole over four devices with no position; scales cut over
   two. *)
let expert_examples =
  [
    {
      devices = 3;
      e = 2;
      outs = 2;
      blocks = 3;
      ids = [| 1; -1; 0 |];
      x_cut = true;
    };
    { devices = 4; e = 3; outs = 1; blocks = 1; ids = [||]; x_cut = false };
    {
      devices = 2;
      e = 2;
      outs = 3;
      blocks = 2;
      ids = [| 0; 2; 1 |];
      x_cut = false;
    };
  ]

let expert_product set c =
  let k = 32 * c.blocks and p = Array.length c.ids in
  cover "scales whole" (c.blocks mod c.devices <> 0);
  cover "scales cut with the codes" (c.blocks mod c.devices = 0);
  cover "x cut alike" c.x_cut;
  cover "three devices" (c.devices = 3);
  cover "no position" (p = 0);
  let shape = [| c.e; c.outs; k / 2 |] in
  let bytes = Array.fold_left ( * ) 1 shape in
  let codes =
    Nx.create Nx.uint8 shape (Array.init bytes (fun i -> i * 37 mod 256))
  in
  let scales =
    Nx.init Nx.uint8 [| c.e; c.outs; c.blocks |] (fun i ->
        118 + (Array.fold_left ( + ) 0 i mod 19))
  in
  let ids = Nx.create Nx.int64 [| p |] (Array.map Int64.of_int c.ids) in
  let x = value [| p; k |] in
  let cut = at set (Split 2) in
  let scales_at = if c.blocks mod c.devices = 0 then cut else at set On in
  let x_at = at set (if c.x_cut then Split 1 else On) in
  let args =
    ( (Nx.place cut codes, Nx.place scales_at scales),
      (Nx.place (at set On) ids, Nx.place x_at x) )
  in
  let f (codes, scales) (ids, x) =
    Nx_quant.apply_experts ~ids (Nx_quant.mxfp4 ~scales codes) x
  in
  let step =
    Rune.jit
      T.(pair tensor tensor @-> pair tensor tensor @-> returns tensor)
      f
  in
  let eager = f (fst args) (snd args) in
  let before = Rune.evaluations () in
  let jit = step (fst args) (snd args) in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  (* Each row's magnitudes: the codes without their sign bits by [|x|],
     summed at float64, whose values are exact. *)
  let magnitudes = Nx.bitwise_and codes (Nx.scalar_like codes 0x77) in
  let mass =
    Nx_quant.apply_experts ~ids
      (Nx_quant.mxfp4 ~scales magnitudes)
      (Nx.abs (Nx.cast Nx.float64 x))
  in
  within ~bound:(Nx.mul_s mass (2. *. product (k + 1))) eager jit;
  equal ~msg:"a second compiled call" bits_t (bits jit)
    (bits (step (fst args) (snd args)))

(* Quantised products

   [Nx_quant.apply] compiled, at each format and each input dtype, its weight
   on one device of the set, whole on each device or cut along its rows, under
   a vector or rows of [x]: eager and compiled are each within the distance
   {!Quants.reference} states of its float64 product of [x] with the weight's
   values: {!Nx.contract}'s bound [γ(k + 1, 2u) Σ|x||v|] in the accumulator,
   float64's for a float64 [x], the reference's own, and half an ulp of
   [x]'s dtype where it rounds into it. *)

type input = X : (float, 'b) Nx.dtype -> input

(* Where the weight lies: on one device, whole on each, or cut along its
   rows. *)
type lies = One | Whole | Rows

let inputs = [ X Nx.float32; X Nx.bfloat16; X Nx.float16; X Nx.float64 ]
let input_name (X dt) = Nx_array.Dtype.name dt

type quant_case = {
  qn : int;
  format : Quants.format;
  input : input;
  rows : int;
  blocks : int;
  m : int option;
  lies : lies;
}

let pp_quant_case ppf c =
  Format.fprintf ppf "%d devices, %s by %s, %d rows of %d blocks, %s%s" c.qn
    (Quants.name c.format) (input_name c.input) c.rows c.blocks
    (match c.m with None -> "a vector" | Some m -> Printf.sprintf "%d rows" m)
    (match c.lies with
    | One -> ", the weight on one device"
    | Whole -> ", the weight whole on each device"
    | Rows -> ", the weight cut along its rows")

let quant_cases =
  let open Gen in
  (let* qn = int_range 2 4 in
   let+ format = of_list ~pp:(fun ppf f -> Format.pp_print_string ppf
         (Quants.name f)) Quants.formats
   and+ input = of_list ~pp:(fun ppf i -> Format.pp_print_string ppf
         (input_name i)) inputs
   and+ rows = int_range 1 (qn + 1)
   and+ blocks = int_range 1 2
   and+ m = option (int_range 0 3)
   and+ lies = of_list [ One; Whole; Rows ] in
   { qn; format; input; rows; blocks; m; lies })
  |> with_pp pp_quant_case

(* Every format at every input dtype, each once, and each format's vector by
   its weight cut along its rows and whole on each of two devices. *)
let quant_examples =
  List.concat_map
    (fun format ->
      let vector lies =
        {
          qn = 2;
          format;
          input = X Nx.float32;
          rows = 3;
          blocks = 1;
          m = None;
          lies;
        }
      in
      vector Rows :: vector Whole
      :: List.mapi
           (fun i input ->
             {
               qn = 2 + (i mod 3);
               format;
               input;
               rows = 3;
               blocks = 2;
               m = (if i = 0 then None else Some 2);
               lies = List.nth [ One; Rows; Whole; Rows ] i;
             })
           inputs)
    Quants.formats

let quant_product set c =
  let (X dt) = c.input in
  cover
    (Printf.sprintf "%s by %s" (Quants.name c.format) (input_name c.input))
    true;
  let one_row = match c.m with None | Some 1 -> true | Some _ -> false in
  cover "a weight on one device" (c.lies = One);
  cover "a weight whole on each device" (c.lies = Whole);
  cover "a weight cut along its rows" (c.lies = Rows);
  cover "one row of x by a weight cut along its rows"
    (one_row && c.lies = Rows);
  cover "one row of x by a weight whole on each device"
    (one_row && c.lies = Whole);
  let k = c.blocks * Quants.block_values c.format in
  let w = Quants.weight c.format ~rows:c.rows ~blocks:c.blocks in
  let shape = match c.m with None -> [| k |] | Some m -> [| m; k |] in
  let x = Nx.cast dt (value ~seed:3 shape) in
  let w_at, x_at =
    match c.lies with
    | One -> (Device 1, Device 1)
    | Whole -> (On, On)
    | Rows -> (Split 0, On)
  in
  let wp = Nx_quant.place (at set w_at) w in
  let xp = Nx.place (at set x_at) x in
  let f x _ = Nx_quant.apply wp x in
  match compiled f xp xp with
  | Raised m -> fail m
  | Value jit ->
      let eager = f xp xp in
      let exact, side = Quants.reference x w in
      let host t = Nx.cast Nx.float64 (Nx.place Nx.Host.on t) in
      let near msg v =
        let e = Nx.to_array exact and v = Nx.to_array (host v) in
        let b = Nx.to_array side in
        Array.iteri
          (fun i e ->
            less
              ~msg:(Printf.sprintf "%s at %d" msg i)
              float_exact
              ~than:(b.(i) +. Float.min_float)
              (Float.abs (v.(i) -. e)))
          e
      in
      near "eager" eager;
      near "compiled" jit;
      within ~bound:(Nx.mul_s side 2.) eager jit

(* A broadcast of a vector against a batch cut across the devices, and its sum,
   as a bias and a loss read them. *)
let biased set c =
  let x = Nx.place (at set c.x) (value c.shape) in
  let last = c.shape.(Array.length c.shape - 1) in
  let b = Nx.place (at set On) (value ~seed:2 [| last |]) in
  let f x b = Nx.add x b in
  let eager = outcome (fun () -> f x b) in
  equal (outcome_t bits_t) (omap bits eager) (omap bits (compiled f x b))

(* Functions that read their coordinates over cut values, exact, so that a
   compiled call gives eager's bits: each device's loop reads its window's
   start, which its call passes. *)
let coordinates set c =
  cover_windows c;
  let x = Nx.place (at set c.x) (value c.shape) in
  let same name f =
    let eager = outcome (fun () -> f x x) in
    let jit = compiled f x x in
    match (eager, jit) with
    | Value e, Value j ->
        equal ~msg:(name ^ " placement") (placement ()) (Nx.placement e)
          (Nx.placement j);
        equal ~msg:name bits_t (bits e) (bits j)
    | _ -> equal ~msg:name (outcome_t bits_t) (omap bits eager) (omap bits jit)
  in
  let rank = Array.length c.shape in
  List.iter
    (fun a ->
      same
        (Printf.sprintf "argmax along %d" a)
        (fun x _ -> Nx.cast f32 (Nx.argmax ~axis:a x)))
    (List.init rank Fun.id);
  if rank = 2 then begin
    same "tril" (fun x _ -> Nx.tril x);
    same "eye" (fun x _ ->
        Nx.add x
          (Nx.eye_at ~m:c.shape.(1) (Nx.placement x) f32 c.shape.(0)))
  end

(* Values computed from constants

   A value the program computes from constants alone, stored before the kernels
   that read it, is read on each device: the shifts that unpack a sub-byte
   bitcast, and a table of [exp2] along an axis, as a rotary embedding reads its
   angles. Over a weight whole on each device or cut along its rows, a compiled
   call computes no call evaluated and gives eager's placement, bit for bit
   where its function is exact. *)

type packing = U4 | I4 | I8

let pp_packing ppf p =
  Format.pp_print_string ppf
    (match p with U4 -> "uint4" | I4 -> "int4" | I8 -> "int8")

(* The bytes [w] as elements of [p], each row's elements along one axis, at
   float32. *)
let decode p w =
  let rows = Nx.dim 0 w and cols = Nx.dim 1 w in
  let wide dt = Nx.reshape [| rows; 2 * cols |] (Nx.bitcast dt w) in
  match p with
  | U4 -> Nx.cast f32 (wide Nx.uint4)
  | I4 -> Nx.cast f32 (wide Nx.int4)
  | I8 -> Nx.cast f32 (Nx.bitcast Nx.int8 w)

(* A sum of each row of the decoded bytes and their product with a vector of
   small integers: exact, since every partial sum is an integer below
   [2^24]. *)
let decoded set ~packing ~whole =
  let rows = 3 and cols = 256 in
  let bytes =
    Nx.create Nx.uint8 [| rows; cols |]
      (Array.init (rows * cols) (fun i -> ((i * 37) + 11) mod 256))
  in
  let w = Nx.place (at set (if whole then On else Split 0)) bytes in
  let k = Nx.dim 1 (decode packing bytes) in
  let x =
    Nx.place (at set On)
      (Nx.create f32 [| k |] (Array.init k (fun i -> float_of_int ((i mod 7) - 3))))
  in
  let same name f =
    let name =
      Format.asprintf "%s of %a, %s" name pp_packing packing
        (if whole then "whole" else "cut")
    in
    let eager = f w x in
    match compiled f w x with
    | Value jit ->
        equal ~msg:(name ^ " placement") (placement ()) (Nx.placement eager)
          (Nx.placement jit);
        equal ~msg:name bits_t (bits eager) (bits jit)
    | Raised m -> fail (name ^ ": " ^ m)
  in
  same "a sum" (fun w _ -> Nx.sum ~axes:[ 1 ] (decode packing w));
  same "a product" (fun w x -> Nx.matmul (decode packing w) x)

let decodes n =
  List.iter
    (fun packing ->
      List.iter
        (fun whole -> over n { law = (fun set -> decoded set ~packing ~whole) })
        [ true; false ])
    [ U4; I4; I8 ]

(* A table of [f j] along each row of [x], summed against it, for [f] [exp2]
   or [sin]: each side is within [f]'s 4 ulps of the exact table and [γ(4)]
   of its sum, so the two lie within twice that bound of each other. *)
let table set ~whole =
  let x = Nx.place (at set (if whole then On else Split 0)) (value [| 5; 4 |]) in
  let check name f f64 =
    let g x _ =
      Nx.sum ~axes:[ 1 ] (Nx.mul x (f (Nx.arange_at (Nx.beside x) f32 0 4 1)))
    in
    let eager = g x x in
    match compiled g x x with
    | Value jit ->
        let host = Nx.abs (Nx.cast Nx.float64 (Nx.place Nx.Host.on x)) in
        let entries = Nx.abs (f64 (Nx.arange Nx.float64 0 4 1)) in
        let mass = Nx.sum ~axes:[ 1 ] (Nx.mul host entries) in
        let ulps = 4. *. ldexp 1. (-23) in
        within ~bound:(Nx.mul_s mass (2. *. (ulps +. gamma 4))) eager jit
    | Raised m -> fail (name ^ ": " ^ m)
  in
  check "exp2" Nx.exp2 Nx.exp2;
  check "sin" Nx.sin Nx.sin

(* [Nx_quant.apply] of a Q4_K or Q6_K weight to a vector, the weight whole on
   each device or cut along its rows: within twice the bound of each side, the
   decoded weight's products with [x] summed at float32. *)
let quant_vector set ~q6 ~whole =
  let rows = 3 and size = if q6 then 210 else 144 in
  let b = Array.init (rows * size) (fun i -> ((i * 37) + (i / 7)) mod 256) in
  (* Each block's float16 fields of either sign in [2^-9, 2^-4). *)
  List.iter
    (fun o ->
      for j = 0 to rows - 1 do
        b.((j * size) + o + 1) <- (0x18 + j) lor (j mod 2 * 0x80)
      done)
    (if q6 then [ 208 ] else [ 0; 2 ]);
  let bytes = Nx.create Nx.uint8 [| rows; size |] b in
  let w = if q6 then Nx_quant.q6_k bytes else Nx_quant.q4_k bytes in
  let wp = Nx_quant.place (at set (if whole then On else Split 0)) w in
  let x = value ~seed:3 [| 256 |] in
  let xp = Nx.place (at set On) x in
  let f x _ = Nx_quant.apply wp x in
  let eager = f xp xp in
  match compiled f xp xp with
  | Value jit ->
      let v = Nx.abs (Nx_quant.dequant Nx.float64 w) in
      let mass =
        Nx.matmul (Nx.abs (Nx.cast Nx.float64 x)) (Nx.matrix_transpose v)
      in
      within ~bound:(Nx.mul_s mass (2. *. gamma 257)) eager jit
  | Raised m -> fail m

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
      (* [g = xᵀ (1 - t²)] for [t = tanh (x w)]: [x w] is within
         [γ(5, 2u) |x| |w|], which moves [1 - t²] by twice that at most; [t]'s
         18 ulps move it by 36 [u], its square and difference by 2 [u] more;
         the product with [xᵀ] adds [γ(rows + 1, 2u)]. Eager and compiled each
         hold the bound. *)
      let host x = Nx.abs (Nx.cast Nx.float64 (Nx.place Nx.Host.on x)) in
      let ax = host x in
      let u = ldexp 1. (-24) in
      let term =
        Nx.add_s
          (Nx.mul_s (Nx.matmul ax (host w)) (2. *. product 5))
          ((38. *. u) +. product (rows + 1))
      in
      let bound = Nx.mul_s (Nx.matmul (Nx.transpose ax) term) 2. in
      within ~bound eager jit
  | r -> equal (outcome_t bits_t) (Value (bits eager)) (omap bits r)

(* Eigenvalues

   [Nx.eigvalsh] and [Nx.svdvals] of a batch of symmetric 6 x 6 matrices cut
   along the batch: a diagonal matrix, which takes no sweep, one with NaN, and
   uniform and graded ones, which take several, so that each device's sweeps
   stop apart. The values lie where eager's do, each within the bound rune.mli
   states, the eigenvalues [γ(16 (R + n)) ‖a‖_F] with [R] at most 8 sweeps of
   5 rounds, and eager's own [γ(16 n) ‖a‖_F]; the failing matrix's alone are
   NaN. *)
let eigh_batch set =
  let n = 6 in
  let entry k i j =
    let i, j = (max i j, min i j) in
    let s = sin (float_of_int ((k * 7919) + (i * n) + j + 1)) in
    match k with
    | 0 -> if i = j then s else 0.
    | 1 -> if i = 2 && j = 1 then Float.nan else s
    | 2 -> s
    | _ -> if i = j then 10. ** float_of_int (-i) else 1e-3 *. s
  in
  let data =
    Array.init (4 * n * n) (fun e ->
        entry (e / (n * n)) (e / n mod n) (e mod n))
  in
  let x = Nx.place (at set (Split 0)) (Nx.create f32 [| 4; n; n |] data) in
  let eager = Nx.eigvalsh f32 x in
  let before = Rune.evaluations () in
  let jit = Rune.jit T.(tensor @-> returns tensor) (Nx.eigvalsh f32) x in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  let norm k =
    let s = ref 0. in
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        let v = entry k i j in
        if not (Float.is_nan v) then s := !s +. (v *. v)
      done
    done;
    Float.sqrt !s
  in
  let tol k = (gamma (16 * (40 + n)) +. gamma (16 * n)) *. norm k in
  let bound =
    Nx.create Nx.float64 [| 4; n |] (Array.init (4 * n) (fun e -> tol (e / n)))
  in
  within ~bound eager jit;
  (* The singular values, of a symmetric matrix its eigenvalues' magnitudes,
     within three [g = γ(8 k (p (p + 6) + 8 R))] of the norm, [R] at most 10
     sweeps of 5 rounds, and eager's own [γ(16 p + 100)]. *)
  let eager = Nx.svdvals f32 x in
  let before = Rune.evaluations () in
  let jit = Rune.jit T.(tensor @-> returns tensor) (Nx.svdvals f32) x in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  let g = gamma (8 * n * ((n * (n + 6)) + (8 * 50))) in
  let tol k = ((3. *. g) +. gamma ((16 * n) + 100)) *. norm k in
  let bound =
    Nx.create Nx.float64 [| 4; n |] (Array.init (4 * n) (fun e -> tol (e / n)))
  in
  within ~bound eager jit

(* Loops

   A scan over a carry cut across the devices, reading rows of values cut along
   their rows' axes or along the rows themselves, and an iterate whose stop
   reads a fold over a cut axis compile, and give eager's placements and bits:
   their steps are exact. Rows cut along axis 0 raise as eagerly. A step that
   gives the carry another placement evaluates. *)

type loop_case = { ln : int; lshape : int array; carry : cut; rows : cut }

let pp_loop_case ppf c =
  Format.fprintf ppf "%d devices, shape %a, carry %a, rows %a" c.ln
    Operand.pp_shape c.lshape pp_cut c.carry pp_cut c.rows

let loop_cases =
  let open Gen in
  (let* ln = int_range 2 4 in
   let* rank = int_range 1 2 in
   let* lshape = array ~size:(constant rank) (int_range 0 ((2 * ln) + 1)) in
   let* carry = cut_of ~n:ln ~rank and+ rows = cut_of ~n:ln ~rank:(rank + 1) in
   constant { ln; lshape; carry; rows })
  |> with_pp pp_loop_case

(* [f] under jit over [x] and [y], compiled, computing no call evaluated,
   where [compiles]. *)
let jit_pair ~compiles f x y =
  let before = Rune.evaluations () in
  let s = T.(tensor @-> tensor @-> returns (pair tensor tensor)) in
  let r = outcome (fun () -> Rune.jit s f x y) in
  if compiles then
    equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  r

let same_pair eager jit =
  match (eager, jit) with
  | Value (a, b), Value (a', b') ->
      equal ~msg:"carry placement" (placement ()) (Nx.placement a)
        (Nx.placement a');
      equal ~msg:"rows placement" (placement ()) (Nx.placement b)
        (Nx.placement b');
      equal ~msg:"carry" bits_t (bits a) (bits a');
      equal ~msg:"rows" bits_t (bits b) (bits b')
  | Raised m, Raised m' -> equal string m m'
  | Value _, Raised m -> fail ("compiled raised: " ^ m)
  | Raised m, Value _ -> fail ("eager raised: " ^ m)

let carry_cut c = match c.carry with Split _ | Mesh _ -> true | _ -> false

let row_count = 3

let along_rows c =
  match c.rows with
  | Split 0 -> true
  | Mesh (a, b) -> a = 0 || b = 0
  | Split _ | On | Device _ -> false

let scanned set c =
  cover "carry cut" (carry_cut c);
  cover "rows cut along the rows" (along_rows c);
  cover "rows cut inside a row"
    (match c.rows with Split a -> a > 0 | _ -> false);
  let x = Nx.place (at set c.carry) (value c.lshape) in
  let xs =
    Nx.place (at set c.rows)
      (value ~seed:1 (Array.append [| row_count |] c.lshape))
  in
  let step carry row =
    let carry =
      Nx.maximum (Nx.mul carry row) (Nx.sub carry (Nx.mul_s row 0.5))
    in
    (carry, Nx.add carry row)
  in
  let f x xs = Rune.scan' step ~init:x xs in
  let eager = outcome (fun () -> f x xs) in
  let kept =
    match eager with
    | Value (y, _) -> Nx.Placement.equal (Nx.placement y) (Nx.placement x)
    | Raised _ -> false
  in
  cover "the step keeps the carry's placement" kept;
  let compiles = kept in
  cover "compiled over a cut carry" (compiles && carry_cut c);
  same_pair eager (jit_pair ~compiles f x xs)

(* A scan whose step reads the coordinates of its carry, cut along either
   axis over three devices, one window uneven, gives eager's bits: each
   device's step reads its window's start. *)
let coordinate_scan set =
  List.iter
    (fun a ->
      let x = Nx.place (at set (Split a)) (value [| 7; 4 |]) in
      let xs =
        Nx.place (at set (Split (a + 1))) (value ~seed:1 [| row_count; 7; 4 |])
      in
      let step carry row =
        let peak = Nx.argmax ~axis:a ~keepdims:true carry in
        let carry = Nx.add (Nx.tril (Nx.mul carry row)) (Nx.cast f32 peak) in
        (carry, Nx.add carry row)
      in
      let f x xs = Rune.scan' step ~init:x xs in
      same_pair (outcome (fun () -> f x xs)) (jit_pair ~compiles:true f x xs))
    [ 0; 1 ]

(* A batch of one

   A batch of one cut across the devices, read by the operations a
   data-parallel step makes, computes where a batch of two does, the host's
   values, compiled as eagerly, and lies where a batch of two does: a mean
   over it on every device. A scan over rows cut along their batch axis,
   which eager computation reads by [Nx.get], runs at a batch of one and
   compiles. *)

type batched = Rows_at of int | Squeezed | Unsqueezed | Added | Product | Mean

let pp_batched ppf = function
  | Rows_at t -> Format.fprintf ppf "get [%d] of [3; n; f] split 1" t
  | Squeezed -> Format.pp_print_string ppf "squeeze [1] of [n; 1; f] split 0"
  | Unsqueezed -> Format.pp_print_string ppf "unsqueeze [1] of [n; f] split 0"
  | Added -> Format.pp_print_string ppf "[n; f] split 0 + [f] on every device"
  | Product ->
      Format.pp_print_string ppf "[n; f] split 0 @ [f; 3] on every device"
  | Mean -> Format.pp_print_string ppf "mean over [n; f] split 0's batch"

let batched op x w =
  match op with
  | Rows_at t -> Nx.get [ t ] x
  | Squeezed -> Nx.squeeze ~axes:[ 1 ] x
  | Unsqueezed -> Nx.unsqueeze ~axes:[ 1 ] x
  | Added -> Nx.add x w
  | Product -> Nx.matmul x w
  | Mean -> Nx.mean ~axes:[ 0 ] x

(* The host value of [s] counting from [0] in C order: sums and products of
   its elements are exact. *)
let iota s =
  Nx.create f32 s (Array.init (Array.fold_left ( * ) 1 s) float_of_int)

let batch_of_one (type d) (set : d set) (op, f) =
  let lies = ref None in
  List.iter
    (fun n ->
      let xh, axis =
        match op with
        | Rows_at _ -> (iota [| 3; n; f |], 1)
        | Squeezed -> (iota [| n; 1; f |], 0)
        | Unsqueezed | Added | Product | Mean -> (iota [| n; f |], 0)
      in
      let wh = match op with Product -> iota [| f; 3 |] | _ -> iota [| f |] in
      let x = Nx.place (at set (Split axis)) xh
      and w = Nx.place (at set On) wh in
      let msg = Printf.sprintf "a batch of %d" n in
      let eager = batched op x w in
      equal ~msg bits_t (bits (batched op xh wh)) (bits eager);
      (match !lies with
      | None -> lies := Some (Nx.placement eager)
      | Some two ->
          equal ~msg:"where a batch of one lies" (placement ()) two
            (Nx.placement eager));
      (match op with
      | Mean ->
          equal ~msg:"the mean lies on every device" (placement ()) (at set On)
            (Nx.placement eager)
      | Rows_at _ | Squeezed | Unsqueezed | Added | Product -> ());
      match compiled (batched op) x w with
      | Value jit ->
          equal ~msg:"compiled placement" (placement ()) (Nx.placement eager)
            (Nx.placement jit);
          equal ~msg:"compiled" bits_t (bits eager) (bits jit)
      | Raised m -> fail ("compiled raised: " ^ m))
    [ 2; 1 ]

let batched_cases =
  let open Gen in
  let op =
    one_of
      [
        map (fun t -> Rows_at t) (int_range 0 2);
        of_list [ Squeezed; Unsqueezed; Added; Product; Mean ];
      ]
  in
  pair op (int_range 1 4)
  |> with_pp (fun ppf (op, f) ->
         Format.fprintf ppf "%a, f = %d" pp_batched op f)

(* A scan over rows [3; n; f], or [3; n; f; 1], cut along the batch, beside
   a carry [n; f] or [n; f; 1] cut along it: each row lies where the carry
   does, at a batch of one, beside features of one element too, as at two. *)
let scan_batch_of_one set =
  List.iter
    (fun (n, f, trailing) ->
      let tail = if trailing then [| 1 |] else [||] in
      let carry = Array.append [| n; f |] tail in
      let x = Nx.place (at set (Split 0)) (value carry) in
      let xs =
        Nx.place (at set (Split 1))
          (value ~seed:1 (Array.append [| row_count |] carry))
      in
      let step carry row = (Nx.add (Nx.mul carry row) row, Nx.neg carry) in
      let f x xs = Rune.scan' step ~init:x xs in
      let eager = outcome (fun () -> f x xs) in
      (match eager with
      | Value (y, _) ->
          equal ~msg:"the carry stays" (placement ()) (Nx.placement x)
            (Nx.placement y)
      | Raised m ->
          fail
            (Format.asprintf "rows %a raised: %s" Operand.pp_shape
               (Nx.shape xs) m));
      same_pair eager (jit_pair ~compiles:true f x xs))
    [
      (2, 4, false);
      (1, 4, false);
      (1, 1, false);
      (2, 1, false);
      (1, 4, true);
      (1, 1, true);
    ]

(* Bytes copied between devices by [run]. *)
let moved_bytes run =
  let _, events = Rig.Profile.take run in
  List.fold_left
    (fun n -> function
      | Rig.Profile.Copy { src; dst; bytes; _ } when not (Rig.equal src dst) ->
          n + bytes
      | _ -> n)
    0 events

(* A scan whose carry and rows are cut alike copies nothing between devices,
   as eager computation copies nothing. *)
let local_scan set =
  let x = Nx.place (at set (Split 0)) (value [| 6; 2 |]) in
  let xs = Nx.place (at set (Split 1)) (value ~seed:1 [| row_count; 6; 2 |]) in
  let step carry row = (Nx.add (Nx.mul carry row) row, Nx.neg carry) in
  let f x xs = Rune.scan' step ~init:x xs in
  same_pair (Value (f x xs)) (jit_pair ~compiles:true f x xs);
  let s = T.(tensor @-> tensor @-> returns (pair tensor tensor)) in
  let run () = ignore (Nx.to_array (fst (Rune.jit s f x xs))) in
  equal ~msg:"bytes copied" int 0 (moved_bytes run)

(* Values a loop makes itself

   A loop's first value made in the program, a constant, and a step's constant
   output have no device of their own: each piece lies where the values it is
   stored with lie, on any device, the program's or another's. *)

type first = Zeros_like | Placed

type made_case = {
  mn : int;
  row : int array;
  trips : int;
  rows : cut;
  arg : int;
  first : first;
}

let pp_made_case ppf c =
  Format.fprintf ppf "%d devices, row %a, %d trips, rows %a, argument on %d, %s"
    c.mn Operand.pp_shape c.row c.trips pp_cut c.rows c.arg
    (match c.first with Zeros_like -> "zeros_like" | Placed -> "placed")

(* Rows whole, on one device or cut inside a row. *)
let made_cases =
  let open Gen in
  (let* mn = int_range 2 4 in
   let* rank = int_range 1 2 in
   let* row = array ~size:(constant rank) (int_range 0 ((2 * mn) + 1)) in
   let mesh =
     if mn = 4 && rank = 2 then
       [ (1, constant (Mesh (1, 2))); (1, constant (Mesh (2, 1))) ]
     else []
   in
   let* rows =
     frequency
       ([
          (1, constant On);
          (2, map (fun k -> Device k) (int_range 0 (mn - 1)));
          (3, map (fun a -> Split a) (int_range 1 rank));
        ]
       @ mesh)
   in
   let* trips = frequency [ (1, constant 0); (3, int_range 1 3) ] in
   let* arg = int_range 0 (mn - 1) in
   let+ first = one_of [ constant Zeros_like; constant Placed ] in
   { mn; row; trips; rows; arg; first })
  |> with_pp pp_made_case

let made_examples =
  let c mn row trips rows arg first = { mn; row; trips; rows; arg; first } in
  [
    (* A cut carry, an empty one, no trips. *)
    c 2 [| 4 |] 3 (Split 1) 0 Placed;
    c 2 [| 4 |] 3 (Split 1) 1 Zeros_like;
    c 3 [| 0 |] 2 (Split 1) 0 Placed;
    c 2 [| 4 |] 0 (Split 1) 1 Placed;
    (* On a device that is not the program's. *)
    c 3 [| 4 |] 2 (Device 2) 0 Placed;
    c 3 [| 4 |] 0 (Device 1) 0 Zeros_like;
    c 2 [| 3 |] 2 On 1 Placed;
    (* The mesh's axes in both orders. *)
    c 4 [| 2; 3 |] 2 (Mesh (1, 2)) 3 Placed;
    c 4 [| 2; 3 |] 1 (Mesh (2, 1)) 0 Zeros_like;
  ]

(* A row's placement, where a value of rows lies at [c]. *)
let row_cut = function
  | Split a -> Split (a - 1)
  | Mesh (a, b) -> Mesh (a - 1, b - 1)
  | (On | Device _) as c -> c

let first_of set c like =
  match c.first with
  | Zeros_like when c.trips > 0 -> Nx.zeros_like (like ())
  | Zeros_like | Placed ->
      Nx.place (at set (row_cut c.rows)) (Nx.zeros f32 c.row)

let cover_made c =
  cover "no trips" (c.trips = 0);
  cover "a carry cut across devices"
    (match c.rows with Split _ | Mesh _ -> true | On | Device _ -> false);
  cover "a device that is not the program's"
    (match c.rows with Device k -> k <> c.arg | _ -> false)

(* A scan's first value and its stacked outputs, made in the program, over
   rows of [c.trips], compiled, gives eager's bits and placements. The
   program's device, that of its first argument, need not hold a piece. *)
let scan_made set c =
  cover_made c;
  let arg = Nx.place (at set (Device c.arg)) (value [||]) in
  let rows = value (Array.append [| c.trips |] c.row) in
  let xs = Nx.place (at set c.rows) rows in
  let f _ xs =
    let init = first_of set c (fun () -> Nx.get [ 0 ] xs) in
    Rune.scan' (fun carry x -> (Nx.add carry x, Nx.zeros_like x)) ~init xs
  in
  same_pair (outcome (fun () -> f arg xs)) (jit_pair ~compiles:true f arg xs)

(* An iterate's first value made in the program, its stop a fold of a carry
   that may hold no element, compiled, gives eager's bits and placements. *)
let iterate_made set c =
  cover_made c;
  let row_at = at set (row_cut c.rows) in
  let arg = Nx.place (at set (Device c.arg)) (value [||]) in
  let x = Nx.place row_at (Nx.ones f32 c.row) in
  let elements = Array.fold_left ( * ) 1 c.row in
  let limit = float_of_int (c.trips * elements) in
  let f a x =
    let init =
      match c.first with
      | Zeros_like -> Nx.zeros_like x
      | Placed -> Nx.place row_at (Nx.zeros f32 c.row)
    in
    let until carry = Nx.greater_equal_s (Nx.sum carry) limit in
    (Nx.neg a, Rune.iterate' ~max:(c.trips + 1) ~until (Nx.add x) init)
  in
  same_pair (outcome (fun () -> f arg x)) (jit_pair ~compiles:true f arg x)

(* A scan over rows the program makes, cut inside a row or on one device that
   is not the program's, compiled, gives eager's bits. *)
let rows_made set =
  let check rows =
    let arg = Nx.place (at set (Device 0)) (value [| 2 |]) in
    let row = at set (row_cut rows) in
    let f a _ =
      let xs = Nx.place (at set rows) (Nx.ones f32 [| 3; 6 |]) in
      let init = Nx.place row (Nx.zeros f32 [| 6 |]) in
      let step carry x = (Nx.add carry x, x) in
      (Nx.neg a, fst (Rune.scan' step ~init xs))
    in
    same_pair
      (outcome (fun () -> f arg arg))
      (jit_pair ~compiles:true f arg arg)
  in
  List.iter check [ Split 1; Device 1 ]

(* A step's constant output is stored on its piece's device, and a first value
   made in the program too: no byte moves between devices. *)
let made_local set =
  let xs = Nx.place (at set (Split 1)) (value [| 3; 6 |]) in
  let s = T.(tensor @-> returns (pair tensor tensor)) in
  let f xs =
    let init = Nx.zeros_like (Nx.get [ 0 ] xs) in
    Rune.scan' (fun carry x -> (Nx.add carry x, Nx.zeros_like x)) ~init xs
  in
  let run () = ignore (Nx.to_array (fst (Rune.jit s f xs))) in
  equal ~msg:"bytes copied" int 0 (moved_bytes run)

(* An iterate whose stop every device holds reads it on the program's device,
   that of its first argument: no trip copies the stop between devices. *)
let stop_home set =
  let s = Nx.place (at set (Device 1)) (value [| 1 |]) in
  let x = Nx.place (at set On) (value ~seed:1 [| 4 |]) in
  let f s x =
    let halve x = Nx.mul_s x 0.5 in
    let small x = Nx.less_s (Nx.max (Nx.abs x)) 0.01 in
    (Nx.neg s, Rune.iterate' ~max:20 ~until:small halve x)
  in
  same_pair (outcome (fun () -> f s x)) (jit_pair ~compiles:true f s x);
  let sig_ = T.(tensor @-> tensor @-> returns (pair tensor tensor)) in
  let run () = ignore (Nx.to_array (snd (Rune.jit sig_ f s x))) in
  equal ~msg:"bytes copied" int 0 (moved_bytes run)

let iterated set c =
  cover "carry cut" (carry_cut c);
  let x = Nx.place (at set c.carry) (value c.lshape) in
  let f x y =
    let halve x = Nx.mul_s x 0.5 in
    let small x = Nx.less_s (Nx.max (Nx.abs x)) 0.01 in
    let x = Rune.iterate' ~max:20 ~until:small halve x in
    (x, Nx.add x y)
  in
  same_pair (outcome (fun () -> f x x)) (jit_pair ~compiles:true f x x)

(* The gradient in its input of an expert product's sum, its MXFP4 weight
   cut along [n]: the transposed product contracts each row with its
   expert's matrix over [n], which each device sums over its window and the
   devices' sums close by partials. Eager and compiled lie where the input
   does, each within twice the bound of the sum over [n] of the expert's
   decoded columns. *)
let expert_grad set =
  let e = 3 and n = 6 and k = 64 and p = 5 in
  let codes =
    Nx.create Nx.uint8 [| e; n; k / 2 |]
      (Array.init (e * n * k / 2) (fun i -> i * 37 mod 256))
  in
  let scales =
    Nx.init Nx.uint8 [| e; n; k / 32 |] (fun i ->
        118 + (Array.fold_left ( + ) 0 i mod 19))
  in
  let ids = Nx.create Nx.int64 [| p |] [| 2L; 0L; -1L; 2L; 1L |] in
  let x = value [| p; k |] in
  (* Each position's expert's columns summed over [n], at float64: zeros for
     an id out of range. *)
  let column t =
    let summed = Nx.sum ~axes:[ 1 ] t in
    let named = Nx.maximum ids (Nx.scalar_like ids 0L) in
    let picked = Nx.take ~axis:0 named summed in
    let inside =
      Nx.logical_and (Nx.greater_equal_s ids 0L)
        (Nx.less_s ids (Int64.of_int e))
    in
    Nx.where (Nx.unsqueeze ~axes:[ 1 ] inside) picked (Nx.zeros_like picked)
  in
  let decoded = Nx_quant.dequant Nx.float64 (Nx_quant.mxfp4 ~scales codes) in
  let cut = at set (Split 1) and on = at set On in
  let w = Nx_quant.mxfp4 ~scales:(Nx.place cut scales) (Nx.place cut codes) in
  let g ids x =
    Rune.grad' (fun x -> Nx.sum (Nx_quant.apply_experts ~ids w x)) x
  in
  let ids' = Nx.place on ids and x' = Nx.place on x in
  let eager = g ids' x' in
  let exact = column decoded and mass = column (Nx.abs decoded) in
  let bound = Nx.mul_s mass (2. *. product (n + 1)) in
  within ~bound exact (Nx.cast Nx.float64 (Nx.place Nx.Host.on eager));
  equal ~msg:"placement" (placement ()) on (Nx.placement eager);
  match compiled g ids' x' with
  | Value jit -> within ~bound:(Nx.mul_s bound 2.) eager jit
  | Raised m -> fail m

(* Replicas

   A compiled sum over a cut axis into a result whole on each device holds one
   value on every device, bit for bit: a result shorter than the devices, which
   each computes whole, and one whose share each computes and copies to the
   others. The terms' magnitudes span twenty binades, so that two orders of
   addition round apart. *)

let replicas set cols =
  let rows = 9 in
  let x =
    Nx.create f32 [| rows; cols |]
      (Array.init (rows * cols) (fun i ->
           ldexp (sin (float_of_int (i + 1))) (i * 7 mod 20)))
  in
  let x = Nx.place (at set (Split 0)) x in
  let f x _ = Nx.sum ~axes:[ 0 ] x in
  match compiled f x x with
  | Value y ->
      equal ~msg:"placement" (placement ()) (at set On) (Nx.placement y);
      let shard a = Array.map Int32.bits_of_float (Nx_array.to_array a) in
      let bs = Array.map shard (Iarray.to_array (Nx.Repr.shards y)) in
      Array.iteri
        (fun k b ->
          equal ~msg:(Printf.sprintf "device %d" k) (array int32) bs.(0) b)
        bs
  | Raised m -> fail m

(* Assemblies

   An assembly of pieces on one device of a split program gives eager's bits
   ({!Nx.Prim.Assemble}: the last piece's element where pieces overlap, the
   fill where none holds the index), whether its pieces tile it, overlap,
   leave a gap or both, run backwards or by steps, are runs of the storage or
   not, of a real, a complex or a sub-byte dtype. *)

let range start count step = { Nx_array.Move.start; count; step }

let whole n = range 0 n 1

(* Each case: its shape, and each piece's region, a range per axis. *)
let assemblies =
  [
    ("pieces that tile it", [| 6 |], [ [ range 3 3 1 ]; [ range 0 3 1 ] ]);
    ( "overlapping pieces that leave a gap",
      [| 6 |],
      [ [ range 0 3 1 ]; [ range 2 3 1 ] ] );
    ("a gap alone", [| 6 |], [ [ range 0 2 1 ]; [ range 3 2 1 ] ]);
    ("one piece over another", [| 6 |], [ [ range 0 6 1 ]; [ range 1 2 1 ] ]);
    ("a piece backwards", [| 6 |], [ [ range 5 3 (-1) ]; [ range 0 3 1 ] ]);
    ("pieces by steps", [| 6 |], [ [ range 0 3 2 ]; [ range 1 2 2 ] ]);
    ("no piece", [| 6 |], []);
    ( "rows that tile it",
      [| 4; 6 |],
      [ [ range 2 2 1; whole 6 ]; [ range 0 2 1; whole 6 ] ] );
    ( "columns that tile it",
      [| 4; 6 |],
      [ [ whole 4; range 3 3 1 ]; [ whole 4; range 0 3 1 ] ] );
    ( "boxes apart along one axis or the other",
      [| 4; 6 |],
      [
        [ range 0 2 1; range 0 3 1 ];
        [ range 0 2 1; range 3 3 1 ];
        [ range 2 2 1; whole 6 ];
      ] );
    ( "boxes that meet on both axes",
      [| 4; 6 |],
      [ [ range 0 3 1; range 0 4 1 ]; [ range 2 2 1; range 3 3 1 ] ] );
  ]

(* The assembly of pieces at [regions] into a value of [shape], [dt] and
   [fill] on the first device of [set], each piece made from a value cut
   across the set. *)
let assembly (type v s) set (dt : (v, s) Nx.dtype) of_real (fill : v) shape
    regions =
  let x = Nx.place (at set (Split 0)) (value [| 24 |]) in
  let f x _ =
    let p = Nx.place (at set (Device 0)) x in
    let piece i rs =
      let counts =
        Array.of_list (List.map (fun (r : Nx_array.Move.range) -> r.count) rs)
      in
      let n = Array.fold_left ( * ) 1 counts in
      let y = Nx.reshape counts (Nx.slice [ R (i, i + n) ] p) in
      (Iarray.of_list rs, of_real (Nx.mul_s y (float_of_int (i + 2))))
    in
    Nx.Prim.eval ~by:"assembly"
      (Assemble
         {
           dtype = dt;
           shape = Iarray.of_array shape;
           fill;
           pieces = List.mapi piece regions;
         })
  in
  let eager = outcome (fun () -> f x x) in
  let before = Rune.evaluations () in
  let s = T.(tensor @-> tensor @-> returns tensor) in
  let jit = outcome (fun () -> Rune.jit s f x x) in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  equal (outcome_t codes_t) (omap codes eager) (omap codes jit)

(* The bytes [run] copies between devices. *)
let copied run =
  let _, events = Rig.Profile.take run in
  List.fold_left
    (fun n -> function
      | Rig.Profile.Copy { src; dst; bytes; _ } when not (Rig.equal src dst) ->
          n + bytes
      | _ -> n)
    0 events

(* A device's window of a value gathered whole on it reads the pieces within
   the window. The gradient of [sum (x · place on (w ⊙ w))], [w] of 4 rows cut
   across [n] devices and [x] of 8 rows by 4 cut along its rows, is
   [2 w ⊙ xᵀ 1]: each device sums its rows of [x] into a partial [xᵀ 1] of 4
   rows by 3, the closing gathers its sums whole on every device, and the
   cotangent placed back at [w] keeps each device's window. Each device then
   reads its rows of every device's partial, copied to it, [n - 1] windows of
   [4 / n] rows by 3 float32s from each device, [48 (n - 1)] bytes in all, and
   its own sums: the others' sums are copied nowhere. *)
let own_window set =
  let n = List.length (Nx.Devices.to_list set.s) in
  let x = Nx.place (at set (Split 0)) (value [| 8; 4 |]) in
  let w = Nx.place (at set (Split 0)) (value ~seed:1 [| 4; 3 |]) in
  let loss w = Nx.sum (Nx.matmul x (Nx.place (at set On) (Nx.mul w w))) in
  let g w _ = Rune.grad' loss w in
  let eager = g w w in
  let step = Rune.jit T.(tensor @-> tensor @-> returns tensor) g in
  (match compiled g w w with
  | Value jit ->
      equal ~msg:"placement" (placement ()) (Nx.placement eager)
        (Nx.placement jit);
      let host x = Nx.abs (Nx.cast Nx.float64 (Nx.place Nx.Host.on x)) in
      let sums = Nx.reshape [| 4; 1 |] (Nx.sum ~axes:[ 0 ] (host x)) in
      let mass = Nx.mul (Nx.mul_s (host w) 2.) sums in
      within ~bound:(Nx.mul_s mass (2. *. gamma 9)) eager jit
  | Raised m -> fail m);
  equal ~msg:"bytes copied" int
    (48 * (n - 1))
    (copied (fun () -> ignore (Nx.to_array (step w w))))

let assembled set shape regions =
  assembly set f32 Fun.id (-0.) shape regions;
  let c64 y = Nx.complex Nx.complex64 ~re:y ~im:(Nx.neg y) in
  assembly set Nx.complex64 c64 { Complex.re = 7.; im = -0. } shape regions;
  let int4 y = Nx.cast Nx.int4 (Nx.mul_s y 3.) in
  assembly set Nx.int4 int4 (-5) shape regions

(* Two assemblies of the same computed pieces, crossed, as [[a; b]] and
   [[b; a]], on each device of a cut argument, give eager's bits, each piece
   correctly rounded. *)
let crossed set =
  let x = Nx.place (at set (Split 0)) (value [| 8 |]) in
  let f x _ =
    let a = Nx.mul_s x 0.25 and b = Nx.add_s x 1. in
    Nx.sub (Nx.stack ~axis:1 [ a; b ]) (Nx.stack ~axis:1 [ b; a ])
  in
  let eager = outcome (fun () -> f x x) in
  equal (outcome_t codes_t) (omap codes eager) (omap codes (compiled f x x))

(* Gathers

   A program that places an argument so that a device holds more of it, a cut
   argument placed whole on each device or, on the mesh, whole along one of
   its axes, compiles, as one that gathers a value it computes or whose
   gradient places a cut cotangent at a whole weight does. Each gives eager's
   bits. *)

let gather_kinds set =
  let x = Nx.place (at set (Split 0)) (value [| 8; 3 |]) in
  let w = Nx.place (at set On) (value ~seed:1 [| 8; 3 |]) in
  let whole x _ = Nx.place (at set On) x in
  let same got f =
    equal (outcome_t bits_t)
      (omap bits (outcome (fun () -> f x w)))
      (omap bits got)
  in
  same (compiled whole x w) whole;
  let computed x _ = Nx.place (at set On) (Nx.mul x x) in
  same (compiled computed x w) computed;
  let grad x w =
    Rune.grad' (fun w -> Nx.sum (Nx.mul (Nx.place (at set (Split 0)) w) x)) w
  in
  same (compiled grad x w) grad;
  match set.mesh with
  | None -> ()
  | Some mesh ->
      let x = Nx.place (at set (Mesh (0, 1))) (value [| 8; 4 |]) in
      let rows x _ = Nx.place (Nx.Placement.mesh mesh [ (0, [ "x" ]) ]) x in
      let got = compiled rows x x in
      equal (outcome_t bits_t)
        (omap bits (outcome (fun () -> rows x x)))
        (omap bits got)

(* Small integers, so that every sum is exact in any order. *)
let integers ?(seed = 0) shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create f32 shape
    (Array.init n (fun i -> float_of_int ((((seed * 7) + (i * 5)) mod 9) - 4)))

(* Other sets

   A program that places a value onto another set compiles, an argument or a
   value it computes: it computes on the devices of every set it reaches, and
   gives eager's placements and bits. Between calls the other set's devices
   hold nothing of it, since a split program takes its storage per run. *)

let others =
  Array.init 2 (fun k ->
      Result.get_ok (Rig.software (Printf.sprintf "other-%d" k)))

module Other = (val Nx.devices (Array.to_list others))

(* A set no kernel library computes on: a software device and a device that does
   not run on the host. *)
module Bare = (val Nx.devices [ m 3; Nx_array_support.off_host () ])

(* An offload: weights on one device placed whole on each device of the other
   set; and a value cut across it. *)
let elsewhere_examples =
  [
    { n = 2; shape = [| 4; 3 |]; x = Device 0; y = On };
    { n = 4; shape = [| 3; 2 |]; x = Device 2; y = On };
    { n = 2; shape = [| 4; 3 |]; x = Split 0; y = Split 1 };
    { n = 3; shape = [| 6 |]; x = Split 0; y = Split 0 };
  ]

(* Eager's bits and placement of [f x y], from a compiled call. *)
let same_compiled f x y =
  let eager = outcome (fun () -> f x y) and got = compiled f x y in
  equal (outcome_t bits_t) (omap bits eager) (omap bits got);
  equal
    (outcome_t (placement ()))
    (omap Nx.placement eager) (omap Nx.placement got)

let elsewhere set c =
  cover "cut across the other set"
    (match c.y with Split _ -> true | _ -> false);
  cover "an offload"
    (match (c.x, c.y) with Device _, (On | Mesh _) -> true | _ -> false);
  let x = Nx.place (at set c.x) (value c.shape) in
  let there =
    match c.y with
    | On | Mesh _ -> Other.on
    | Split a -> Other.split ~axis:a
    | Device k -> Nx.Placement.device Other.v others.(k mod 2)
  in
  let back = at set (match c.y with Device _ -> On | y -> y) in
  let computed x y =
    let z = Nx.place there (Nx.mul x y) in
    Nx.place back (Nx.sub (Nx.mul_s z 2.) z)
  in
  let argument x _ = Nx.place back (Nx.place there x) in
  same_compiled computed x x;
  same_compiled argument x x

(* A partial sum placed onto another set closes there. *)
let closing_cases =
  let open Gen in
  (let* n = int_range 2 4 and+ k = int_range 0 9 in
   constant (n, k))
  |> with_pp (fun ppf (n, k) -> Format.fprintf ppf "%d devices, width %d" n k)

let closing_examples = [ (2, 3); (4, 2); (2, 4); (3, 0) ]

let closing set n k =
  cover "windows of unequal extents" (k mod n <> 0);
  cover "an empty window" (k < n);
  let x = Nx.place (at set (Split 1)) (integers [| 3; k |]) in
  let w = Nx.place (at set (Split 0)) (integers ~seed:1 [| k; 2 |]) in
  let f x w =
    Nx.place (at set On) (Nx.place Other.on (Nx.matmul ~sum:true x w))
  in
  same_compiled f x w

(* A detour through a set without kernels: eagerly the operation raises, under
   jit it compiles and gives the arithmetic computed on the host. *)
let bare_detour () =
  let there = Nx.Placement.device Bare.v (m 3) in
  let xs = value [| 4; 3 |] in
  let x = Nx.place Two.on xs in
  let f x y =
    let z = Nx.place there (Nx.mul x y) in
    Nx.place Two.on (Nx.sub (Nx.add z z) z)
  in
  raises_match (Exn.invalid_arg ~sub:"has no kernels") (fun () -> f x x);
  let expected = bits (Nx.mul xs xs) in
  match compiled f x x with
  | Value y -> equal ~msg:"the host's arithmetic" bits_t expected (bits y)
  | Raised why -> fail ("compiled raised: " ^ why)

(* The other set shares a device with the program's: of a value cut across
   it, the piece already on that device moves nowhere, so only the other
   device's piece is copied there and back. *)
module Both = (val Nx.devices [ m 1; others.(0) ])

let shared_device () =
  let on_one = Nx.Placement.device Two.v (m 1) in
  let rows = 4 and columns = 3 in
  let x = Nx.place on_one (value [| rows; columns |]) in
  let f x y =
    let z = Nx.place (Both.split ~axis:0) (Nx.mul x y) in
    Nx.place on_one (Nx.add z z)
  in
  same_compiled f x x;
  let step = Rune.jit T.(tensor @-> tensor @-> returns tensor) f in
  ignore (Nx.to_array (step x x));
  let _, events =
    Rig.Profile.take (fun () -> ignore (Nx.to_array (step x x)))
  in
  let between a b =
    List.fold_left
      (fun n -> function
        | Rig.Profile.Copy { src; dst; bytes; _ }
          when Rig.equal src a && Rig.equal dst b ->
            n + bytes
        | _ -> n)
      0 events
  in
  let piece = rows / 2 * columns * 4 in
  equal ~msg:"bytes to the other device" int piece (between (m 1) others.(0));
  equal ~msg:"bytes back" int piece (between others.(0) (m 1))

(* A detour inside a scan's step. *)
let scan_detour set =
  let here = at set (Device 0) in
  let x = Nx.place here (value [| 3; 2 |]) in
  let xs = Nx.place here (value ~seed:1 [| row_count; 3; 2 |]) in
  let step carry row =
    let z = Nx.place Other.on (Nx.mul carry row) in
    (Nx.place here (Nx.sub (Nx.mul_s z 2.) z), Nx.neg row)
  in
  let f x xs = Rune.scan' step ~init:x xs in
  same_pair (Value (f x xs)) (jit_pair ~compiles:true f x xs)

(* What a detour leaves on the devices of the set it visits, the argument's
   and the computed value's, once a call returns, the first included, while its
   program lives: nothing. A device's bytes are counted over devices of their
   own. *)
let held_rigs =
  List.init 2 (fun k ->
      Result.get_ok (Rig.software (Printf.sprintf "place-held-%d" k)))

module Held = (val Nx.devices held_rigs)

let settled_on rigs =
  for _ = 1 to 4 do
    Gc.full_major ();
    List.iter Rig.free_cache rigs
  done;
  List.map Rig.allocated rigs

let held set =
  let n = 65536 in
  let there = Held.on in
  let back = at set On in
  let x = Nx.place back (value [| n |]) in
  let computed x y =
    let z = Nx.place there (Nx.mul x y) in
    Nx.place back (Nx.sub (Nx.add z z) z)
  in
  let argument x _ = Nx.place back (Nx.place there x) in
  let stays name f =
    let g = Rune.jit T.(tensor @-> tensor @-> returns tensor) f in
    let run () = ignore (Nx.to_array (g x x)) in
    let none = settled_on held_rigs in
    let calls = Rune.evaluations () in
    let holds what =
      equal
        ~msg:(name ^ ": bytes on the other set " ^ what)
        (list int) none (settled_on held_rigs)
    in
    run ();
    holds "after the first call";
    run ();
    holds "after the second";
    equal ~msg:(name ^ ": no call evaluated") int 0
      (Rune.evaluations () - calls);
    (* A last call keeps the program reachable through every count. *)
    run ()
  in
  stays "computed" computed;
  stays "an argument" argument

(* Kernels

   Each device that holds elements of an operation's results runs a kernel of
   its own for its window, and a device whose window is empty runs none, nor
   does the host. A profile names a submission's kernels, joined by commas, on
   the device that ran them. The first windows of a cut hold its elements. *)

let kernels set n rows =
  let x = Nx.place (at set (Split 0)) (value [| rows; 3 |]) in
  let f x _ = Nx.exp x in
  let step = Rune.jit T.(tensor @-> tensor @-> returns tensor) f in
  ignore (Nx.to_array (step x x));
  let _, events =
    Rig.Profile.take (fun () -> ignore (Nx.to_array (step x x)))
  in
  let ran =
    List.concat_map
      (function
        | Rig.Profile.Work { device; name; _ }
          when not (List.mem name [ "copies"; "fills"; "words" ]) ->
            List.map (fun _ -> Rig.name device) (String.split_on_char ',' name)
        | _ -> [])
      events
  in
  let holding =
    List.filteri (fun k _ -> k < min n rows) (Nx.Devices.to_list set.s)
  in
  equal ~msg:"the devices of the kernels" (list string)
    (List.sort compare (List.map Rig.name holding))
    (List.sort compare ran)

(* Searches

   A function searched at a width compiles over software devices, its candidates
   timed on a device of the set: no call is evaluated, and it gives eager's
   bits. The search ignores its cache, so that it times candidates. *)

let searched set c =
  let x = Nx.place (at set c) (value [| 8; 3 |]) in
  let f x = Nx.add_s (Nx.mul x x) 0.5 in
  let before = Rune.evaluations () in
  let y =
    Tolk.Setting.context
      [ B (Tolk.Setting.ignore_beam_cache, true) ]
      (fun () -> Rune.jit' ~beam:1 f x)
  in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  equal bits_t (bits (f x)) (bits y)

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

(* A value whole on each device placed onto a mesh moves a window along each
   axis from one start, axes of other extents. *)
let onto_mesh set =
  match set.mesh with
  | None -> ()
  | Some mesh ->
      let x = Nx.place (at set On) (value [| 4; 2 |]) in
      let at = Nx.Placement.mesh mesh [ (0, [ "x" ]); (1, [ "y" ]) ] in
      let f x _ = Nx.place at (Nx.mul x x) in
      let eager = outcome (fun () -> f x x) in
      equal (outcome_t bits_t) (omap bits eager) (omap bits (compiled f x x))

(* Keys

   A key's placement inside a program is its argument's, and a draw at it,
   plus a constant at its uncut placement, gives eager's placement and bits:
   a key whole on each device, one on device 1, and a batch of [4] cut along
   axis 0. *)
let key_placement set =
  let draw k =
    let at = Nx.Rng.placement k in
    Nx.add
      (Nx.Rng.normal_at at ~key:k f32 [| 3 |])
      (Nx.full_at (Nx.Placement.uncut at) f32 [| 3 |] 0.5)
  in
  let root = Nx.Rng.key 11 in
  List.iter
    (fun (name, c, key) ->
      let k = Nx.Rng.place (at set c) key in
      let inside = ref None in
      let jit =
        Rune.jit
          T.(Nx.Rng.ptree @-> returns tensor)
          (fun k ->
            inside := Some (Nx.Rng.placement k);
            draw k)
          k
      in
      let eager = draw k in
      equal ~msg:(name ^ ": inside the program")
        (option (placement ()))
        (Some (Nx.Rng.placement k))
        !inside;
      equal ~msg:(name ^ ": placement") (placement ()) (Nx.placement eager)
        (Nx.placement jit);
      equal ~msg:name bits_t (bits eager) (bits jit))
    [
      ("a key on each device", On, root);
      ("a key on device 1", Device 1, root);
      ("a batch cut along axis 0", Split 0, Nx.Rng.split_batch ~n:4 root);
    ]

(* Programs

   A program for one device serves every device of its target, so the
   programs a data-parallel step over [n] software devices takes do not grow
   with [n]: each device's kernels, its stores of pieces into their runs of an
   assembly included, are another's. The count is the same over 4, 8 and 16
   devices, and over 2 at most that: a closing whose sum another device
   copies sums its slots in a kernel of its own from 4 devices on, where over
   2 tolk inlines a sum of two slots into each reader (a value reading at
   most 3 buffers is inlined). A step's programs are counted by their keys,
   made or kept, so that one this process made before counts. Over 2, 4, 8
   and 16 devices the hidden width grows with [n], and over 2 to 8, 3, 5 and 6
   among them, every width is 120, so that every cut, the gradients' closing
   sums' included, is even, and each device's windows have one shape. *)

let sixteen =
  Array.init 16 (fun k ->
      Result.get_ok (Rig.software (Printf.sprintf "programs-%d" k)))

(* The number of program keys among the kernels [f ()] compiles, and the
   name and a digest of each key, for a failure to name them. *)
let keys f =
  let (), ks = Tolk.Realize.kernels f in
  let ks = List.map (fun (k : Tolk.Realize.kernel) -> (k.name, k.key)) ks in
  let keys = List.sort_uniq String.compare (List.map snd ks) in
  let named k =
    let name, _ = List.find (fun (_, k') -> String.equal k k') ks in
    name ^ " " ^ String.sub (Digest.to_hex (Digest.string k)) 0 8
  in
  (List.length keys, String.concat ", " (List.map named keys))

(* Counts over the device counts [ns], among them 4, that do not grow with the
   devices: those over 4 devices or more are equal, and those over fewer are
   at most theirs. Each count's kernels name what differs. *)
let flat ?(ns = [ 2; 4; 8; 16 ]) what counts =
  let msg =
    String.concat "\n"
      (List.map2
         (fun n (c, names) ->
           Printf.sprintf "%s over %d: %d: %s" what n c names)
         ns counts)
  in
  let at = List.combine ns (List.map fst counts) in
  let four = List.assoc 4 at in
  List.iter
    (fun (n, c) ->
      if n < 4 then at_most ~msg int ~than:four c else equal ~msg int four c)
    at

(* [inner] is the step's input width and [hidden] its hidden one, by default
   [32] and [16 + n]. *)
let programs ?(inner = 32) ?hidden n =
  let module S = (val Nx.devices (Array.to_list (Array.sub sixteen 0 n))) in
  let hidden = Option.value hidden ~default:(16 + n) in
  let x =
    Nx.place (Nx.Placement.split ~axis:0 S.v) (value [| 4 * n; inner |])
  in
  let on = Nx.Placement.on S.v in
  let w1 = Nx.place on (value ~seed:1 [| inner; hidden |]) in
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
  let evaluated = Rune.evaluations () in
  let programs = keys (fun () -> ignore (f x (w1, w2))) in
  equal ~msg:"calls evaluated" int evaluated (Rune.evaluations ());
  programs

(* Maps over a cut axis

   A map over an axis cut one row per device is per-device code: [Rune.lanes]
   gathers every device's row, a sum over the lane axis closes as a fold over a
   cut axis does, and each device computes its lanes' [Rune.lane_index]. Its
   elements are small integers, so that every sum is exact in any order and a
   compiled call gives eager's bits. *)

let around = Rune.axis ()

(* Each lane's row scaled by the sum of every lane's rows, plus its index
   times [w]. *)
let per_device x w =
  Rune.vmap' ~axis:around
    (fun r ->
      let total = Nx.sum ~axes:[ 0 ] (Rune.lanes around r) in
      let index = Nx.cast f32 (Rune.lane_index around r) in
      Nx.add (Nx.mul r total) (Nx.mul index w))
    x

type mapped = { devices : int; rows : int; cols : int }

let pp_mapped ppf c =
  Format.fprintf ppf "%d devices, %d rows of %d" c.devices c.rows c.cols

let mapped =
  let open Gen in
  (let* devices = int_range 2 4 in
   let* rows =
     frequency [ (1, constant devices); (1, int_range 0 ((2 * devices) + 1)) ]
   and+ cols = int_range 1 5 in
   constant { devices; rows; cols })
  |> with_pp pp_mapped

let shard_map set c =
  cover "one row per device" (c.rows = c.devices);
  cover "uneven rows" (c.rows mod c.devices <> 0);
  cover "an empty window" (c.rows < c.devices);
  cover "three devices" (c.devices = 3);
  let x = Nx.place (at set (Split 0)) (integers [| c.rows; c.cols |]) in
  let w = Nx.place (at set On) (integers ~seed:1 [| c.cols |]) in
  let eager = outcome (fun () -> per_device x w) in
  let jit = compiled per_device x w in
  match (eager, jit) with
  | Value e, Value j ->
      equal ~msg:"placement" (placement ()) (Nx.placement e) (Nx.placement j);
      equal bits_t (bits e) (bits j)
  | _ -> equal (outcome_t bits_t) (omap bits eager) (omap bits jit)

(* Under a map over a cut axis, the lanes' indices lie where their rows lie,
   and so do those of a compiled call inside the map. *)
let indices set =
  let x = Nx.place (at set (Split 0)) (integers [| 4; 2 |]) in
  let index r = Rune.lane_index around r in
  let check name i =
    equal ~msg:(name ^ " placement") (placement ()) (at set (Split 0))
      (Nx.placement i);
    equal ~msg:name (array int32) [| 0l; 1l; 2l; 3l |]
      (Nx.to_array (Nx.place Nx.Host.on i))
  in
  check "map" (Rune.vmap' ~axis:around index x);
  check "map of a call" (Rune.vmap' ~axis:around (Rune.jit' index) x);
  (* Under forward mode the index reads its operand's primal. *)
  let forward r =
    let i, _ =
      Rune.jvp' (fun y -> Nx.cast f32 (Rune.lane_index around y)) r r
    in
    Nx.cast Nx.int32 i
  in
  check "map of a derivative" (Rune.vmap' ~axis:around forward x);
  (* An inner map's lane reads the outer map's index of its row. *)
  let inner = Rune.axis () in
  let nested =
    Rune.vmap' ~axis:around
      (fun r -> Rune.vmap' ~axis:inner (fun e -> index e) r)
      x
  in
  equal ~msg:"nested placement" (placement ()) (at set (Split 0))
    (Nx.placement nested);
  equal ~msg:"nested" (array int32)
    [| 0l; 0l; 1l; 1l; 2l; 2l; 3l; 3l |]
    (Nx.to_array (Nx.place Nx.Host.on nested))

(* Gradient accumulation by stacked partials: each device's gradient of its
   rows' loss, stacked along the lane axis, stays where its rows lie and moves
   nothing between devices; one sum over that axis closes them, to the
   gradient of the whole loss. The loss [Σ x w²] has the gradient [2 x w],
   exact on small integers. *)
let partials set =
  let x = Nx.place (at set (Split 0)) (integers [| 4; 3 |]) in
  let w = Nx.place (at set On) (integers ~seed:1 [| 3 |]) in
  let loss x w = Nx.sum (Nx.mul x (Nx.mul w w)) in
  let stacked x w =
    Rune.vmap' ~axis:around (fun x -> Rune.grad' (loss x) w) x
  in
  let step = Rune.jit T.(tensor @-> tensor @-> returns tensor) stacked in
  let before = Rune.evaluations () in
  let p = step x w in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  equal ~msg:"partials' placement" (placement ()) (at set (Split 0))
    (Nx.placement p);
  equal ~msg:"partials" bits_t (bits (stacked x w)) (bits p);
  let _, events =
    Rig.Profile.take (fun () -> ignore (Nx.to_array (step x w)))
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
  equal ~msg:"bytes between devices" int 0 copied;
  let accumulated x w = Nx.sum ~axes:[ 0 ] (stacked x w) in
  let g = compiled accumulated x w in
  let whole = Rune.grad' (loss x) w in
  equal (outcome_t bits_t) (Value (bits whole)) (omap bits g);
  match g with
  | Value g ->
      equal ~msg:"placement" (placement ()) (Nx.placement whole)
        (Nx.placement g)
  | Raised _ -> ()

(* A data-parallel step written as per-device code, each device's loss offset
   by its lane's index and gathered by a sum over the lanes, and one that
   accumulates stacked partial gradients, each take programs that do not grow
   over 2, 4, 8 and 16 software devices, as the data-parallel step above does.
   Their widths are that step's none, so that no program of one is the
   other's. *)
let mapped_programs n =
  let module S = (val Nx.devices (Array.to_list (Array.sub sixteen 0 n))) in
  let hidden = 48 + n in
  let x = Nx.place (Nx.Placement.split ~axis:0 S.v) (value [| n; 4; 32 |]) in
  let on = Nx.Placement.on S.v in
  let w1 = Nx.place on (value ~seed:1 [| 32; hidden |]) in
  let w2 = Nx.place on (value ~seed:2 [| hidden; 4 |]) in
  let loss x (w1, w2) =
    let y = Nx.matmul (Nx.tanh (Nx.matmul x w1)) w2 in
    Nx.add y (Nx.cast f32 (Rune.lane_index around x))
  in
  let descend (w1, w2) (g1, g2) =
    (Nx.sub w1 (Nx.mul_s g1 0.1), Nx.sub w2 (Nx.mul_s g2 0.1))
  in
  let per_device x w =
    let total w =
      let lane x =
        let y = loss x w in
        Nx.sum (Rune.lanes around (Nx.mean (Nx.mul y y)))
      in
      Nx.sum (Rune.vmap' ~axis:around lane x)
    in
    descend w (Rune.grad T.(pair tensor tensor) total w)
  in
  let accumulated x w =
    let lane x =
      Rune.grad T.(pair tensor tensor)
        (fun w ->
          let y = loss x w in
          Nx.mean (Nx.mul y y))
        w
    in
    let g1, g2 =
      Rune.vmap T.(tensor @-> returns (pair tensor tensor)) ~axis:around lane x
    in
    descend w (Nx.sum ~axes:[ 0 ] g1, Nx.sum ~axes:[ 0 ] g2)
  in
  let count f =
    let evaluated = Rune.evaluations () in
    let step =
      Rune.jit
        T.(tensor @-> pair tensor tensor @-> returns (pair tensor tensor))
        f
    in
    let programs = keys (fun () -> ignore (step x (w1, w2))) in
    equal ~msg:"calls evaluated" int evaluated (Rune.evaluations ());
    programs
  in
  let per_device = count per_device in
  (per_device, count accumulated)

(* Partials over a cut axis

   A fold, a scan, a gather or a scatter over an axis cut across devices
   computes by partials, each device over its tile of the axis, and gives
   eager's placement and bits. Each device's positions in the whole axis are
   its tile's start, and each device stores its part of a closing into the
   whole at its part's start, both of which each device's call binds: every
   such operation makes programs that do not grow over 2, 4 and 8 software
   devices ({!flat}). Each cut is even, so each device's tiles have one shape;
   the other axis's width depends on [n], so that no two counts share a
   program. A tile holds six rows: their bytes are a whole number of 16-byte
   runs, so that every device's piece of the operand lies at one alignment,
   and they are no count of devices, so that a tile's scan is never the
   program of the scan of the tiles' totals. The elements are small integers,
   so that every sum is exact in any order. *)

let positions ~seed ~bound shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create Nx.int64 shape
    (Array.init n (fun i -> Int64.of_int (((seed * 7) + (i * 5)) mod bound)))

(* Each operation's programs over [n] devices ({!keys}), in order. *)
let partial_programs n =
  let module S = (val Nx.devices (Array.to_list (Array.sub sixteen 0 n))) in
  let rows = 6 * n and width = 2 + n in
  let cut = Nx.Placement.split ~axis:0 S.v and on = Nx.Placement.on S.v in
  let x = Nx.place cut (integers [| rows; width |]) in
  let along = Nx.place on (positions ~seed:1 ~bound:rows [| 3; width |]) in
  let cut_positions =
    Nx.place cut (positions ~seed:2 ~bound:rows [| rows; width |])
  in
  let target = Nx.place on (integers ~seed:3 [| rows; width |]) in
  let updates = Nx.place on (integers ~seed:4 [| 3; width |]) in
  let ops =
    [
      ("argmax", fun x -> Nx.cast f32 (Nx.argmax ~axis:0 x));
      ("cumsum", fun x -> Nx.cumsum ~axis:0 x);
      ("cummax's tangent", fun x -> snd (Rune.jvp' (Nx.cummax ~axis:0) x x));
      ("cummax's gradient", Rune.grad' (fun x -> Nx.sum (Nx.cummax ~axis:0 x)));
      ("take_along_axis", fun x -> Nx.take_along_axis ~axis:0 along x);
      ( "scatter ~combine:Add",
        fun x -> Nx.scatter ~combine:Add ~axis:0 cut_positions x target );
      ("scatter", fun x -> Nx.scatter ~axis:0 along updates x);
      ("sum", fun x -> Nx.sum ~axes:[ 0 ] x);
    ]
  in
  List.map
    (fun (name, f) ->
      let evaluated = Rune.evaluations () in
      let y = ref None in
      let made = keys (fun () -> y := Some (Rune.jit' f x)) in
      equal
        ~msg:(name ^ ": calls evaluated")
        int evaluated (Rune.evaluations ());
      let y = Option.get !y and e = f x in
      equal ~msg:(name ^ ": placement") (placement ()) (Nx.placement e)
        (Nx.placement y);
      equal ~msg:name bits_t (bits e) (bits y);
      (name, made))
    ops

let examples =
  [
    { n = 2; shape = [| 4; 3 |]; x = Split 0; y = On };
    { n = 3; shape = [| 5; 2 |]; x = Split 0; y = Split 0 };
    { n = 4; shape = [| 2; 3 |]; x = Split 0; y = Device 1 };
    { n = 4; shape = [| 4; 4 |]; x = Mesh (0, 1); y = On };
    { n = 2; shape = [| 0; 3 |]; x = Split 0; y = On };
  ]

(* Values on an io device

   An io device computes nothing: a program with a value on it evaluates, and
   a value on it is copied to where an operation computes, as eagerly. *)

module Disk = (val Nx.devices [ Rig_disk.device ])
module Io = (val Nx.devices [ m 3; Rig_disk.device ])

(* Emissions

   An emission of a program over several devices is offered as eagerly, at
   its placement, once per site, and the program compiles. *)

let offers p f =
  match Rune.Probe.collect p f with
  | Ok _, rows -> rows
  | Error (e, bt), _ -> Printexc.raise_with_backtrace e bt

let same_offers eager jit =
  equal ~msg:"offers" int (List.length eager) (List.length jit);
  List.iter2
    (fun e j ->
      equal ~msg:"placement" (placement ()) (Nx.placement e) (Nx.placement j);
      equal ~msg:"offer" bits_t (bits e) (bits j))
    eager jit

let emitted set c =
  cover_windows c;
  let x = Nx.place (at set c.x) (value c.shape) in
  let p = Rune.Probe.make T.tensor in
  let f x =
    Rune.Probe.emit p (Nx.mul x x);
    Nx.add x x
  in
  let eager = offers p (fun () -> f x) in
  let before = Rune.evaluations () in
  let jit = offers p (fun () -> Rune.jit' f x) in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  same_offers eager jit

let emitted_scan set (c : loop_case) =
  cover "rows cut inside a row"
    (match c.rows with Split a -> a > 0 | _ -> false);
  assume (c.rows <> Split 0);
  assume (match c.rows with Mesh (a, b) -> a > 0 && b > 0 | _ -> true);
  let xs =
    Nx.place (at set c.rows)
      (value ~seed:1 (Array.append [| row_count |] c.lshape))
  in
  let p = Rune.Probe.make T.tensor in
  let step carry row =
    Rune.Probe.emit p (Nx.mul row row);
    (Nx.add carry row, row)
  in
  let f xs = Rune.scan' step ~init:(Nx.get [ 0 ] xs) xs in
  let eager = offers p (fun () -> f xs) in
  let before = Rune.evaluations () in
  let s = T.(tensor @-> returns (pair tensor tensor)) in
  let jit = offers p (fun () -> Rune.jit s f xs) in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  same_offers eager jit

(* What stays declined

   Over a set without kernels, nothing evaluates a declined program: the call
   raises, naming the decline. *)

(* An emission or a check in a branch stays declined: evaluated on software
   devices, named on a set without kernels. *)
let declined_branches () =
  let x = Nx.place (Two.split ~axis:0) (value [| 4 |]) in
  let i = Nx.place Two.on (Nx.scalar Nx.int32 0l) in
  let f i x =
    Rune.switch T.tensor i
      [
        (fun () ->
          Nx.check T.tensor (Nx.less_s x 2.) x (fun _ _ -> Exit);
          x);
        (fun () -> x);
      ]
  in
  let s = T.(tensor @-> tensor @-> returns tensor) in
  let before = Rune.evaluations () in
  ignore (Nx.to_array (Nx.place Nx.Host.on (Rune.jit s f i x)));
  greater ~msg:"evaluated" int ~than:0 (Rune.evaluations () - before);
  let p = Rune.Probe.make T.tensor in
  let g i x =
    Rune.switch T.tensor i
      [
        (fun () ->
          Rune.Probe.emit p x;
          x);
        (fun () -> x);
      ]
  in
  let before = Rune.evaluations () in
  ignore (Rune.Probe.collect p (fun () -> Rune.jit s g i x));
  greater ~msg:"an emission in a branch is evaluated" int ~than:0
    (Rune.evaluations () - before)

let declined () =
  let at = Bare.split ~axis:0 in
  let xs = Nx.place at (value [| 3; 2 |]) in
  let f xs =
    let init = Nx.sum ~axes:[ 0 ] xs in
    fst (Rune.scan' (fun c x -> (Nx.add c x, x)) ~init xs)
  in
  raises_match
    (Exn.invalid_arg ~sub:"the compiler declines this program")
    (fun () -> Rune.jit' f xs)

(* Checks

   A check over several devices fails as eager computation does, whichever
   device holds the data: the first false flag in C order of the whole value,
   the data's element there, the first failing check of the program. It
   compiles. *)

type check_case = {
  cn : int;
  cshape : int array;
  flags : cut;
  cdata : cut;
  falses : int list;
}

let pp_check_case ppf c =
  Format.fprintf ppf "%d devices, shape %a, flags %a, data %a, false at [%s]"
    c.cn Operand.pp_shape c.cshape pp_cut c.flags pp_cut c.cdata
    (String.concat "; " (List.map string_of_int c.falses))

let check_cases =
  let open Gen in
  (let* cn = int_range 2 4 in
   let* rank = int_range 0 2 in
   let* cshape = array ~size:(constant rank) (int_range 0 ((2 * cn) + 1)) in
   let cut =
     if rank = 0 then
       frequency
         [
           (1, constant On);
           (1, map (fun k -> Device k) (int_range 0 (cn - 1)));
         ]
     else cut_of ~n:cn ~rank
   in
   let* flags = cut and+ cdata = cut in
   let n = Array.fold_left ( * ) 1 cshape in
   let* falses =
     if n = 0 then constant []
     else list ~size:(int_range 0 3) (int_range 0 (n - 1))
   in
   constant { cn; cshape; flags; cdata; falses })
  |> with_pp pp_check_case

let check_examples =
  let mesh_xy = Mesh (0, 1) and mesh_yx = Mesh (1, 0) in
  let c cn cshape flags cdata falses = { cn; cshape; flags; cdata; falses } in
  [
    (* Passing, and an empty value, cut across devices. *)
    c 2 [| 4 |] (Split 0) (Split 0) [];
    c 2 [| 0; 3 |] (Split 1) (Split 1) [];
    (* The first false on either side of a cut, data cut alike. *)
    c 2 [| 8 |] (Split 0) (Split 0) [ 6 ];
    c 2 [| 8 |] (Split 0) (Split 0) [ 1; 6 ];
    (* A 2 x 2 mesh in both axis orders, the first false in C order not the
       first device's. *)
    c 4 [| 4; 4 |] mesh_xy mesh_xy [ 3; 4; 10 ];
    c 4 [| 4; 4 |] mesh_yx mesh_yx [ 3; 4; 10 ];
    (* Data whole on each device, on one device or cut otherwise. *)
    c 4 [| 4; 4 |] mesh_xy On [ 7 ];
    c 4 [| 4; 4 |] mesh_xy (Device 2) [ 7 ];
    c 4 [| 4; 4 |] mesh_yx mesh_xy [ 7 ];
    c 2 [| 6 |] (Split 0) (Device 1) [ 1 ];
    c 2 [| 6; 4 |] (Split 0) (Split 1) [ 5 ];
    (* One flag, whole on each device. *)
    c 3 [||] On (Device 1) [ 0 ];
    c 4 [| 3 |] On On [ 1; 2 ];
  ]

let index_text i =
  String.concat "," (Array.to_list (Array.map string_of_int i))

(* A check's failure as its index and datum, in a [Failure]. *)
let check_fail (type d) (idx : int array) (v : (float, _, d) Nx.t) =
  let d = Nx.item [] (Nx.place Nx.Host.on v) in
  Failure (Printf.sprintf "%s: %h" (index_text idx) d)

type 'a check_outcome = Passed of 'a | Failed of string

let pp_outcome pp ppf = function
  | Passed a -> pp ppf a
  | Failed s -> Format.fprintf ppf "raised %s" s

let check_outcome_t =
  Testable.make
    ~pp:(pp_outcome (Testable.pp bits_t))
    ~equal:(fun a b ->
      match (a, b) with
      | Passed a, Passed b -> Testable.equal bits_t a b
      | Failed a, Failed b -> String.equal a b
      | _ -> false)

(* The outcome of [f ()] once its check, here read at a wait, has been read:
   its value's bits, or its failure. *)
let waited f =
  match f () with
  | y -> (
      match Nx.place Nx.Host.on y with
      | h -> Passed (bits h)
      | exception Failure s -> Failed s)
  | exception Failure s -> Failed s

let cut_devices c =
  match c with
  | Mesh _ | Split _ | On -> true
  | Device _ -> false

let checked_case set c =
  cover "passes" (c.falses = []);
  cover "a cut across devices" (cut_devices c.flags);
  cover "a mesh" (match c.flags with Mesh _ -> true | _ -> false);
  cover "data apart from the flags" (c.flags <> c.cdata);
  cover "the first false on a mesh, in two tiles"
    (match c.flags with Mesh _ -> List.length c.falses >= 2 | _ -> false);
  let n = Array.fold_left ( * ) 1 c.cshape in
  let flags =
    Nx.create Nx.bool c.cshape
      (Array.init n (fun i -> not (List.mem i c.falses)))
  in
  let d = value c.cshape in
  let ok = Nx.place (at set c.flags) flags in
  let data = Nx.place (at set c.cdata) d in
  let f ok data =
    Nx.check T.tensor ok data check_fail;
    Nx.add data data
  in
  let eager = waited (fun () -> f ok data) in
  let before = Rune.evaluations () in
  let jit =
    waited (fun () ->
        Rune.jit T.(tensor @-> tensor @-> returns tensor) f ok data)
  in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  equal check_outcome_t eager jit

(* Two checks made in order fail as the first made, wherever each lies. *)
let two_checks set =
  let check_at a i msg d =
    let n = 4 in
    let flags = Nx.create Nx.bool [| n |] (Array.init n (fun k -> k <> i)) in
    Nx.check T.unit (Nx.place (at set a) flags) ()
      (fun _ () -> Failure msg)
    |> ignore;
    d
  in
  let run a b =
    let d = value [| 4 |] in
    let x = Nx.place (at set On) d in
    let f _ x =
      let x = check_at (Split 0) a "first" x in
      check_at (Split 0) b "second" x
    in
    let eager = waited (fun () -> f () x) in
    let jit =
      waited (fun () ->
          Rune.jit T.(unit @-> tensor @-> returns tensor) f () x)
    in
    equal check_outcome_t eager jit
  in
  run 3 0;
  run 0 3;
  run (-1) 1

(* Elementwise exact functions, gradients of rules and remats over several
   devices *)

let exact_compare eager jit =
  match (eager, jit) with
  | Value e, Value j ->
      equal ~msg:"placement" (placement ()) (Nx.placement e) (Nx.placement j);
      equal bits_t (bits e) (bits j)
  | _ -> equal (outcome_t bits_t) (omap bits eager) (omap bits jit)

(* A custom_vjp and a custom_jvp whose functions and rules are exact: compiled
   over several devices their values and derivatives are eager's bits. *)
let customs set c =
  cover_windows c;
  let x = Nx.place (at set c.x) (value c.shape) in
  let square =
    Rune.custom_vjp T.tensor T.tensor (fun x ->
        (Nx.mul x x, fun ct -> Nx.mul_s ct 3.))
  in
  let forward =
    Rune.custom_jvp T.tensor T.tensor (fun x ->
        (Nx.mul x x, fun t -> Nx.mul_s t 3.))
  in
  let f x _ = Nx.add (square x) (forward x) in
  exact_compare (outcome (fun () -> f x x)) (compiled f x x);
  let d x _ = Rune.grad' (fun x -> Nx.sum (square x)) x in
  exact_compare (outcome (fun () -> d x x)) (compiled d x x);
  let t = Nx.ones_like x in
  let jv x t = snd (Rune.jvp' forward x t) in
  exact_compare (outcome (fun () -> jv x t)) (compiled jv x t)

(* A remat's value and gradient compiled over several devices are eager's. *)
let remats set c =
  cover_windows c;
  let x = Nx.place (at set c.x) (value c.shape) in
  let g = Rune.remat T.(tensor @-> returns tensor) (fun x -> Nx.mul x x) in
  let f x _ = Nx.add (g x) x in
  exact_compare (outcome (fun () -> f x x)) (compiled f x x);
  let d x _ = Rune.grad' (fun x -> Nx.sum (g x)) x in
  exact_compare (outcome (fun () -> d x x)) (compiled d x x)

(* A call of a compiled function and a custom transpose's application inside
   a compiled function run as eagerly. *)
let calls set c =
  cover_windows c;
  let x = Nx.place (at set c.x) (value c.shape) in
  let g = Rune.jit' (fun x -> Nx.mul x x) in
  let linear =
    Rune.custom_transpose T.tensor T.tensor
      (fun v -> Nx.mul_s v 2.)
      ~transpose:(fun w -> Nx.mul_s w 2.)
  in
  let f x _ = Nx.add (g x) (linear x) in
  exact_compare (outcome (fun () -> f x x)) (compiled f x x)

(* A check inside a remat or a custom rule, compiled over several devices,
   raises as eagerly: first false in C order of the whole value and eager's
   datum. *)
let nested_check set =
  let ok =
    Nx.place (at set (Split 0))
      (Nx.create Nx.bool [| 4 |] [| true; true; false; true |])
  in
  let d =
    Nx.place (at set (Split 0)) (Nx.create f32 [| 4 |] [| 1.; 2.; 3.; 4. |])
  in
  let checked ok d =
    Nx.check T.tensor ok d check_fail;
    d
  in
  let r =
    Rune.remat T.(tensor @-> tensor @-> returns tensor) checked
  in
  let c = Rune.custom_jvp T.(pair tensor tensor) T.tensor
      (fun (ok, d) ->
        (checked ok d, fun (_, dd) -> dd))
  in
  let f1 ok d = Nx.add (r ok d) d in
  let f2 ok d = Nx.add (c (ok, d)) d in
  let jit f =
    waited (fun () -> Rune.jit T.(tensor @-> tensor @-> returns tensor) f ok d)
  in
  equal check_outcome_t (waited (fun () -> f1 ok d)) (jit f1);
  equal check_outcome_t (waited (fun () -> f2 ok d)) (jit f2)

(* A passing check moves only its flags: one per device other than the one that
   computes. *)
let check_moves set =
  let n = List.length (Nx.Devices.to_list set.s) in
  let ok = Nx.place (at set (Split 0)) (Nx.full Nx.bool [| 8 |] true) in
  let d = Nx.place (at set (Split 0)) (value [| 8 |]) in
  let f ok d =
    Nx.check T.tensor ok d check_fail;
    Nx.neg d
  in
  let s = T.(tensor @-> tensor @-> returns tensor) in
  (* The first call compiles and copies constants: the second is the one
     measured. *)
  ignore (Nx.to_array (Nx.place Nx.Host.on (Rune.jit s f ok d)));
  let run () = ignore (Nx.to_array (Nx.place Nx.Host.on (Rune.jit s f ok d))) in
  equal ~msg:"bytes copied" int (8 * (n - 1)) (moved_bytes run)

(* Additions to a total compiled over several devices give eager's bits and
   placement of the running value, wherever the running value starts. A
   running value placed where no sum with the additions exists raises as
   eagerly. *)
let totals set c =
  cover_windows c;
  cover "the running value elsewhere" (c.x <> c.y);
  let x = Nx.place (at set c.x) (value c.shape) in
  let t = Rune.Total.make () in
  let init = Nx.place (at set c.y) (Nx.zeros f32 c.shape) in
  let f x =
    Rune.Total.add t (Nx.mul x x);
    Rune.Total.add t x;
    x
  in
  let eager = outcome (fun () -> Rune.Total.collect t ~init (fun () -> f x)) in
  let before = Rune.evaluations () in
  let jit =
    outcome (fun () -> Rune.Total.collect t ~init (fun () -> Rune.jit' f x))
  in
  if Nx.Placement.equal (Nx.placement init) (Nx.placement x) then
    equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  match (eager, jit) with
  | Value (a, e), Value (b, j) ->
      equal ~msg:"result" bits_t (bits a) (bits b);
      equal ~msg:"placement" (placement ()) (Nx.placement e) (Nx.placement j);
      equal bits_t (bits e) (bits j)
  | Raised _, Raised _ -> ()
  | Value _, Raised m -> fail ("compiled raised: " ^ m)
  | Raised m, Value _ -> fail ("eager raised: " ^ m)

(* A check in a program of no arguments and no results, over several devices,
   has no set to issue a call's check on: it evaluates, as eagerly. *)
let no_value set =
  let ok =
    Nx.place (at set (Split 0))
      (Nx.create Nx.bool [| 4 |] [| true; false; true; true |])
  in
  let run f =
    match
      f ();
      Nx.sync ok
    with
    | () -> "passed"
    | exception Failure s -> s
  in
  let g () = Nx.check T.unit ok () (fun _ () -> Failure "failed") in
  equal string (run g) (run (Rune.jit T.(unit @-> returns unit) g))

(* A check in a loop's step keeps declining: evaluated on software devices. *)
let in_loops set =
  let x = Nx.place (at set (Split 0)) (Nx.ones f32 [| 4 |]) in
  let xs = Nx.place (at set (Split 1)) (Nx.ones f32 [| 2; 4 |]) in
  let check_step c row =
    Nx.check T.tensor (Nx.greater_s row 0.) row check_fail;
    (Nx.add c row, row)
  in
  let f xs _ = fst (Rune.scan' check_step ~init:x xs) in
  let before = Rune.evaluations () in
  let jit () = Rune.jit T.(tensor @-> tensor @-> returns tensor) f xs x in
  exact_compare (outcome (fun () -> f xs x)) (outcome jit);
  greater ~msg:"a loop's check is evaluated" int ~than:0
    (Rune.evaluations () - before)

(* Switches

   A switch over several devices compiles: its index lies whole on the
   program's device or whole on another, and one branch computes, its pieces
   each device's. It gives eager's bits and placement, and computes nothing
   for a branch not taken. *)

type switch_case = {
  sn : int;
  sshape : int array;
  scut : cut;
  index_at : cut;
  taken : int;
}

let pp_switch_case ppf c =
  Format.fprintf ppf "%d devices, shape %a, x %a, index %a, branch %d" c.sn
    Operand.pp_shape c.sshape pp_cut c.scut pp_cut c.index_at c.taken

let switch_cases =
  let open Gen in
  (let* sn = int_range 2 4 in
   let* rank = int_range 1 2 in
   let* sshape = array ~size:(constant rank) (int_range 0 ((2 * sn) + 1)) in
   let* scut = cut_of ~n:sn ~rank in
   let* index_at =
     frequency
       [
         (1, constant On); (2, map (fun k -> Device k) (int_range 0 (sn - 1)));
       ]
   in
   let* taken =
     frequency [ (4, int_range 0 2); (1, constant 7); (1, constant (-1)) ]
   in
   constant { sn; sshape; scut; index_at; taken })
  |> with_pp pp_switch_case

let switch_examples =
  let c sn sshape scut index_at taken =
    { sn; sshape; scut; index_at; taken }
  in
  [
    (* Each branch, over two devices and a mesh, the index on every kind of
       placement, with even, uneven and empty windows. *)
    c 2 [| 4 |] (Split 0) On 0;
    c 2 [| 4 |] (Split 0) (Device 1) 1;
    c 3 [| 7 |] (Split 0) (Device 0) 2;
    c 4 [| 2; 6 |] (Mesh (0, 1)) (Device 3) 1;
    c 2 [| 0 |] (Split 0) On 0;
    (* On a device apart, out of range. *)
    c 3 [| 4 |] (Device 2) (Device 0) 1;
    c 2 [| 4 |] (Split 0) On 7;
    c 2 [| 4 |] (Split 0) On (-1);
  ]

let switch_branches (x : (float, Nx.float32_elt, 'd) Nx.t) =
  [
    (fun () -> Nx.neg x);
    (fun () -> Nx.mul_s x 2.);
    (fun () -> Nx.add x x);
  ]

let same_switch eager jit =
  match (eager, jit) with
  | Value e, Value j ->
      equal ~msg:"placement" (placement ()) (Nx.placement e) (Nx.placement j);
      equal bits_t (bits e) (bits j)
  | _ -> equal (outcome_t bits_t) (omap bits eager) (omap bits jit)

(* An evaluation that must compile, read at a wait so that a deferred failure
   surfaces; its failure's message, the index out of range, is [Raised]. *)
let compile_read f =
  let before = Rune.evaluations () in
  let r = outcome f in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  r

let switched set c =
  cover "taken inside the branches" (c.taken >= 0 && c.taken < 3);
  cover "an index out of range" (c.taken < 0 || c.taken >= 3);
  cover "x cut across devices"
    (match c.scut with Split _ | Mesh _ -> true | _ -> false);
  cover "an index on another device"
    (match c.index_at with Device k -> k > 0 | _ -> false);
  let x = Nx.place (at set c.scut) (value c.sshape) in
  let i =
    Nx.place (at set c.index_at) (Nx.scalar Nx.int32 (Int32.of_int c.taken))
  in
  let f i x = Rune.switch T.tensor i (switch_branches x) in
  let eager = outcome (fun () -> f i x) in
  let jit =
    compile_read (fun () ->
        let y = Rune.jit T.(tensor @-> tensor @-> returns tensor) f i x in
        ignore (Nx.to_array (Nx.place Nx.Host.on y));
        y)
  in
  same_switch eager jit

(* A branch not taken computes nothing: the move in the other branch's work
   happens only when it runs. *)
let untaken (type d) (set : d set) =
  let x = Nx.place (at set (Split 0)) (value [| 8 |]) in
  let f i x =
    Rune.switch T.tensor i
      [
        (fun () -> Nx.neg x);
        (* A sum over the cut axis closes across devices. *)
        (fun () -> Nx.add x (Nx.sum x));
      ]
  in
  let s = T.(tensor @-> tensor @-> returns tensor) in
  let step = Rune.jit s f in
  let index k = Nx.place (at set On) (Nx.scalar Nx.int32 k) in
  ignore (Nx.to_array (Nx.place Nx.Host.on (step (index 0l) x)));
  let moved k =
    moved_bytes (fun () ->
        ignore (Nx.to_array (Nx.place Nx.Host.on (step (index k) x))))
  in
  equal ~msg:"the branch not taken moves nothing" int 0 (moved 0l);
  greater ~msg:"the taken branch moves its window" int ~than:0 (moved 1l)

(* A map whose lanes take a switch's branches apart, its lanes cut across
   devices: the branches' results lie apart there, so the call is evaluated,
   with eager's values. *)
let lane_switch set =
  let x = Nx.place (at set (Split 0)) (value [| 4 |]) in
  let f x _ =
    Rune.vmap'
      (fun xi ->
        Rune.cond T.tensor (Nx.greater_s xi 0.)
          (fun () -> Nx.neg xi)
          (fun () -> xi))
      x
  in
  let eager = outcome (fun () -> f x x) in
  let jit =
    outcome (fun () ->
        let s = T.(tensor @-> tensor @-> returns tensor) in
        Rune.jit s f x x)
  in
  same_switch eager jit

(* A switch inside a scan's step, and one nested in a branch, as eagerly. *)
let nested_switch set =
  let x = Nx.place (at set (Split 0)) (value [| 6 |]) in
  let xs = Nx.place (at set (Split 1)) (value ~seed:1 [| 3; 6 |]) in
  let step (i, c) r =
    let c' =
      Rune.switch T.tensor i
        [ (fun () -> Nx.add c r); (fun () -> Nx.mul c r) ]
    in
    ((Nx.sub (Nx.scalar_like i 1l) i, c'), c')
  in
  let s = T.(tensor @-> tensor @-> returns (pair tensor tensor)) in
  let f x xs =
    let i0 = Nx.place (at set On) (Nx.scalar Nx.int32 0l) in
    let (_, c), ys =
      Rune.scan T.(pair tensor tensor) T.tensor T.tensor step ~init:(i0, x) xs
    in
    let inner =
      Rune.switch T.tensor i0
        [
          (fun () -> Rune.switch T.tensor i0 [ (fun () -> c); (fun () -> x) ]);
          (fun () -> c);
        ]
    in
    (inner, ys)
  in
  let eager = f x xs in
  let before = Rune.evaluations () in
  let jit = Rune.jit s f x xs in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  equal ~msg:"carry" bits_t (bits (fst eager)) (bits (fst jit));
  equal ~msg:"rows" bits_t (bits (snd eager)) (bits (snd jit))

(* [xs] in a file of the disk, on [set]. *)
let on_disk (type d) (set : d Nx.Devices.t) xs =
  let n = Array.length xs in
  let path = Filename.temp_file "rune-io-" ".bin" in
  Sys.remove path;
  let b =
    match Rig_disk.create_file path (n * 4) with
    | Ok f -> Rig_disk.buffer f
    | Error why -> failf "create_file: %s" why
  in
  Sys.remove path;
  let x = Nx.create f32 [| n |] xs in
  if n > 0 then Rig.Buffer.copy ~src:(Nx.Repr.read x Nx_array.buffer) ~dst:b;
  Nx.Repr.of_array set
    (Nx_array.v f32 (Nx_array.Layout.contiguous (Iarray.of_list [ n ])) b)

let io_xs = Array.init 8 (fun i -> Float.of_int (i * i))
let io_ys = Array.init 8 (fun i -> Float.of_int i -. 2.5)

(* A set with a device that computes nothing has no kernels, so the call
   raises as eager computation does. *)
let io_argument () =
  let at = Nx.Placement.device Io.v (m 3) in
  let f x y = Nx.add x (Nx.place at y) in
  let x = Nx.place at (Nx.create f32 [| 8 |] io_xs) in
  let y = on_disk Io.v io_ys in
  raises_match (Exn.invalid_arg ~sub:"has no kernels") (fun () -> f x y);
  raises_match (Exn.invalid_arg ~sub:"computes nothing") (fun () ->
      Rune.jit T.(tensor @-> tensor @-> returns tensor) f x y)

let io_capture () =
  let y = on_disk Disk.v io_ys in
  let at = Two.split ~axis:0 in
  let x = Nx.place at (Nx.create f32 [| 8 |] io_xs) in
  let r = Rune.jit' (fun x -> Nx.add x (Nx.place at y)) x in
  let expected =
    Nx.add (Nx.create f32 [| 8 |] io_xs) (Nx.create f32 [| 8 |] io_ys)
  in
  equal (array float_exact) (Nx.to_array expected)
    (Nx.to_array (Nx.place Nx.Host.on r))

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
             prop ~count:10 ~examples:expert_examples
               "an expert product cut along its inputs" experts_cases (fun c ->
                 over c.devices { law = (fun set -> expert_product set c) });
             prop ~count:30 ~examples "coordinates over cut values" cases
               (fun c -> over c.n { law = (fun set -> coordinates set c) });
             prop ~count:20 ~examples "a vector broadcast to a cut batch" cases
               (fun c -> over c.n { law = (fun set -> biased set c) });
             prop ~count:10 ~examples:quant_examples
               "a quantised product at every format and input dtype"
               quant_cases (fun c ->
                 over c.qn { law = (fun set -> quant_product set c) });
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
         group "gathers"
           [
             test "over two devices" (fun () -> over 2 { law = gather_kinds });
             test "over four and their mesh" (fun () ->
                 over 4 { law = gather_kinds });
             test "a window of a gathered value reads its own pieces"
               (fun () ->
                 over 2 { law = own_window };
                 over 4 { law = own_window });
           ];
         group "assemblies give eager's bits"
           (List.map
              (fun (name, shape, regions) ->
                test name (fun () ->
                    over 2 { law = (fun s -> assembled s shape regions) }))
              assemblies
           @ [
               test "crossed assemblies of computed pieces" (fun () ->
                   over 2 { law = crossed });
             ]);
         group "values on an io device"
           [
             test "an argument on the disk raises, named, as eagerly"
               io_argument;
             test "a capture on the disk of a cut program gives eager's values"
               io_capture;
           ];
         group "other sets"
           [
             prop ~count:20 ~examples:elsewhere_examples
               "a value placed onto another set and back compiles" cases
               (fun c -> over c.n { law = (fun set -> elsewhere set c) });
             prop ~count:20 ~examples:closing_examples
               "a partial placed onto another set closes there" closing_cases
               (fun (n, k) -> over n { law = (fun set -> closing set n k) });
             test "a detour through a set without kernels" bare_detour;
             test "a device in both sets holds its piece" shared_device;
             test "a detour inside a scan's step" (fun () ->
                 over 3 { law = scan_detour });
             test "the other set holds nothing between calls" (fun () ->
                 over 2 { law = held });
           ];
         group "loops"
           [
             prop ~count:30 "a scan over a cut carry, bit for bit" loop_cases
               (fun c -> over c.ln { law = (fun set -> scanned set c) });
             prop ~count:20 "an iterate stopped by a fold over a cut axis"
               loop_cases (fun c ->
                 over c.ln { law = (fun set -> iterated set c) });
             prop ~count:15 ~examples:made_examples
               "a scan's first value and outputs made in the program"
               made_cases (fun c ->
                 over c.mn { law = (fun set -> scan_made set c) });
             prop ~count:15 ~examples:made_examples
               "an iterate's first value made in the program" made_cases
               (fun c -> over c.mn { law = (fun set -> iterate_made set c) });
             test "a scan over rows made in the program" (fun () ->
                 over 2 { law = rows_made };
                 over 3 { law = rows_made });
             test "a scan's constant outputs stay on their devices" (fun () ->
                 over 2 { law = made_local };
                 over 3 { law = made_local });
             test "a scan cut alike copies nothing between devices" (fun () ->
                 over 3 { law = local_scan });
             test "a step that reads its coordinates gives eager's bits"
               (fun () -> over 3 { law = coordinate_scan });
             test "an iterate reads its stop on the program's device"
               (fun () -> over 2 { law = stop_home });
             test "a scan over a batch of one cut across devices" (fun () ->
                 over 2 { law = scan_batch_of_one };
                 over 4 { law = scan_batch_of_one });
             test "eigenvalues and singular values of matrices that stop apart"
               (fun () ->
                 over 2 { law = eigh_batch };
                 over 3 { law = eigh_batch });
           ];
         group "switches"
           [
             prop ~count:20 ~examples:switch_examples
               "a branch's value and placement, its index anywhere"
               switch_cases (fun c ->
                 over c.sn { law = (fun set -> switched set c) });
             test "a branch not taken computes nothing" (fun () ->
                 over 2 { law = untaken });
             test "lanes cut across devices that take branches apart" (fun () ->
                 over 2 { law = lane_switch };
                 over 4 { law = lane_switch });
             test "inside a scan's step and nested" (fun () ->
                 over 2 { law = nested_switch };
                 over 4 { law = nested_switch });
           ];
         group "checks"
           [
             prop ~count:25 ~examples:check_examples
               "a check fails as eagerly, wherever flags and data lie"
               check_cases (fun c ->
                 over c.cn { law = (fun set -> checked_case set c) });
             test "two checks in order, the first made fails" (fun () ->
                 over 2 { law = two_checks });
             test "a check inside a remat or a custom rule" (fun () ->
                 over 2 { law = nested_check });
             test "a passing check moves one flag per device" (fun () ->
                 over 2 { law = check_moves };
                 over 3 { law = check_moves });
             test "a check of no value evaluates" (fun () ->
                 over 2 { law = no_value });
             test "a check in a loop is evaluated" (fun () ->
                 over 2 { law = in_loops });
           ];
         group "custom rules, remats and totals"
           [
             prop ~count:10 ~examples "custom rules, values and derivatives"
               cases (fun c ->
                 over c.n { law = (fun set -> customs set c) });
             prop ~count:10 ~examples "a remat, value and gradient" cases
               (fun c -> over c.n { law = (fun set -> remats set c) });
             prop ~count:10 ~examples "a call and a custom transpose" cases
               (fun c -> over c.n { law = (fun set -> calls set c) });
             prop ~count:10 ~examples "additions to a total" cases (fun c ->
                 over c.n { law = (fun set -> totals set c) });
           ];
         group "emissions"
           [
             prop ~count:10 ~examples
               "made at the top level, as eagerly, compiled" cases (fun c ->
                 over c.n { law = (fun set -> emitted set c) });
             prop ~count:10
               ~examples:
                 [
                   {
                     ln = 2;
                     lshape = [| 1 |];
                     carry = On;
                     rows = Split 1;
                   };
                 ]
               "made in a scan's step, stacked, compiled" loop_cases (fun c ->
                 over c.ln { law = (fun set -> emitted_scan set c) });
           ];
         group "what stays declined"
           [
             test "raises at the call on a set without kernels, named" declined;
             test "a check or an emission in a branch is evaluated"
               declined_branches;
           ];
         group "a batch of one"
           [
             prop "computes where a batch of two does, compiled as eagerly"
               batched_cases (fun c ->
                 over 2 { law = (fun set -> batch_of_one set c) };
                 over 4 { law = (fun set -> batch_of_one set c) });
           ];
         group "an expert product's gradient in its input"
           [
             test "its weight cut along n over two devices" (fun () ->
                 over 2 { law = expert_grad });
             test "over three, uneven" (fun () -> over 3 { law = expert_grad });
           ];
         group "values computed from constants"
           [
             test "decoded bytes over two devices" (fun () -> decodes 2);
             test "decoded bytes over four" (fun () -> decodes 4);
             test "a table of exp2 or sin over two and four devices" (fun () ->
                 List.iter
                   (fun n ->
                     List.iter
                       (fun whole -> over n { law = (fun s -> table s ~whole) })
                       [ true; false ])
                   [ 2; 4 ]);
             test "Q4_K and Q6_K by a vector over two and four devices"
               (fun () ->
                 List.iter
                   (fun n ->
                     List.iter
                       (fun (q6, whole) ->
                         over n
                           { law = (fun s -> quant_vector s ~q6 ~whole) })
                       [ (false, true); (false, false); (true, true); (true, false) ])
                   [ 2; 4 ]);
           ];
         group "replicas"
           [
             test "a result shorter than the devices" (fun () ->
                 over 4 { law = (fun s -> replicas s 3) });
             test "a share on each device, uneven" (fun () ->
                 over 4 { law = (fun s -> replicas s 9) });
             test "a share on each of three devices" (fun () ->
                 over 3 { law = (fun s -> replicas s 6) });
           ];
         group "moves"
           [
             test "positions repeated along an axis move one element along it"
               (fun () -> over 2 { law = moves });
             test "a whole value placed onto a mesh" (fun () ->
                 over 4 { law = onto_mesh });
           ];
         group "keys"
           [
             test "a key's placement in a program, and draws there" (fun () ->
                 over 2 { law = key_placement });
           ];
         group "programs"
           [
             test "a data-parallel step's programs do not grow over 2 to 16"
               (fun () ->
                 flat "programs" (List.map programs [ 2; 4; 8; 16 ]));
             test "nor over 3, 5 and 6, every cut even" (fun () ->
                 let ns = [ 2; 3; 4; 5; 6; 8 ] in
                 flat ~ns "programs"
                   (List.map (programs ~inner:120 ~hidden:120) ns));
           ];
         group "maps over a cut axis"
           [
             prop ~count:60 "lanes, lane_index and a sum over the lanes"
               mapped (fun c ->
                 over c.devices { law = (fun set -> shard_map set c) });
             test "the lanes' indices lie where their rows lie" (fun () ->
                 over 4 { law = indices });
             test "stacked partials stay on their devices, then close"
               (fun () -> over 4 { law = partials });
             test "per-device code's programs do not grow over 2 to 16"
               (fun () ->
                 let counts = List.map mapped_programs [ 2; 4; 8; 16 ] in
                 flat "per-device" (List.map fst counts);
                 flat "accumulated" (List.map snd counts));
             test "partials give eager's bits, their programs do not grow"
               (fun () ->
                 let counts = List.map partial_programs [ 2; 4; 8 ] in
                 List.iter
                   (fun (name, _) ->
                     flat ~ns:[ 2; 4; 8 ] name
                       (List.map (List.assoc name) counts))
                   (List.hd counts));
           ];
         group "each device runs its own kernels"
           [
             test "over two devices" (fun () ->
                 over 2 { law = (fun s -> kernels s 2 4) });
             test "over four, one window empty" (fun () ->
                 over 4 { law = (fun s -> kernels s 4 3) });
           ];
         group "searches"
           [
             test "a value whole on one software device" (fun () ->
                 over 2 { law = (fun s -> searched s (Device 0)) });
             test "a value cut across two" (fun () ->
                 over 2 { law = (fun s -> searched s (Split 0)) });
           ];
       ])
