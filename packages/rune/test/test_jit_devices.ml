(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jit over the memory devices of one set: a compiled call over values cut
   across two, three or four devices, a 2 x 2 mesh, or whole on each, gives
   eager's placements, exact functions' bits and the other functions' values
   within rune.mli's classes, for even, uneven and empty windows. It compiles
   (no call is evaluated), each device runs its own kernels, and its gradient is
   eager's. A map over an axis cut one row per device compiles as per-device
   code. *)

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
   on one device of the set or cut along its rows: eager and compiled are each
   within {!Nx.contract}'s bound of the exact product of [x] with the weight's
   values, [γ(k + 1, 2u) Σ|x||v|] in its accumulator, float64's for a float64
   [x], and round once to [x]'s dtype, half its ulp. A sum over a decode that
   reads a sub-byte bitcast, as Q4_K's, Q6_K's and MXFP4 blocks' do, is
   evaluated today where the weight lies whole on each of several devices, or
   is cut along its rows under one row of [x]: the bitcast's table of shifts
   lies on the first device, beside a kernel on another. So the weight lies on
   one device, or is cut along its rows under two rows of [x] or more. *)

type format = Q8_0 | Q6_K | Q4_K | Mxfp4 | Mxfp4_blocks

let format_name = function
  | Q8_0 -> "q8_0"
  | Q6_K -> "q6_k"
  | Q4_K -> "q4_k"
  | Mxfp4 -> "mxfp4"
  | Mxfp4_blocks -> "mxfp4_blocks"

let formats = [ Q8_0; Q6_K; Q4_K; Mxfp4; Mxfp4_blocks ]

type input = X : (float, 'b) Nx.dtype -> input

let inputs = [ X Nx.float32; X Nx.bfloat16; X Nx.float16; X Nx.float64 ]
let input_name (X dt) = Nx_array.Dtype.name dt

type quant_case = {
  qn : int;
  format : format;
  input : input;
  rows : int;
  blocks : int;
  m : int option;
  by_rows : bool;
}

let pp_quant_case ppf c =
  Format.fprintf ppf "%d devices, %s by %s, %d rows of %d blocks, %s%s" c.qn
    (format_name c.format) (input_name c.input) c.rows c.blocks
    (match c.m with None -> "a vector" | Some m -> Printf.sprintf "%d rows" m)
    (if c.by_rows then ", the weight cut along its rows" else "")

(* Values and bytes of a block, and the offsets of its float16 fields. *)
let block_values = function Q8_0 | Mxfp4 | Mxfp4_blocks -> 32 | _ -> 256

let block_bytes = function
  | Q8_0 -> 34
  | Q6_K -> 210
  | Q4_K -> 144
  | Mxfp4 -> 16
  | Mxfp4_blocks -> 17

let halves = function
  | Q8_0 -> [ 0 ]
  | Q6_K -> [ 208 ]
  | Q4_K -> [ 0; 2 ]
  | Mxfp4 | Mxfp4_blocks -> []

(* A weight of [rows] rows of [blocks] blocks, its bytes a fixed pattern, its
   float16 fields of either sign in [2^-9, 2^-4), and its MXFP4 scale
   bytes in [118, 136], so that every value is finite. *)
let quant_weight format ~rows ~blocks =
  let size = block_bytes format in
  let count = rows * blocks in
  let b = Array.init (count * size) (fun i -> ((i * 37) + (i / 7)) mod 256) in
  for j = 0 to count - 1 do
    let at o = (j * size) + o in
    List.iter
      (fun o -> b.(at (o + 1)) <- (0x18 + (j mod 20)) lor (j mod 2 * 0x80))
      (halves format);
    if format = Mxfp4_blocks then b.(at 0) <- 118 + (j mod 19)
  done;
  let u8 last a = Nx.create Nx.uint8 [| rows; last |] a in
  match format with
  | Q8_0 -> Nx_quant.q8_0 (u8 (blocks * size) b)
  | Q6_K -> Nx_quant.q6_k (u8 (blocks * size) b)
  | Q4_K -> Nx_quant.q4_k (u8 (blocks * size) b)
  | Mxfp4_blocks -> Nx_quant.mxfp4_blocks (u8 (blocks * size) b)
  | Mxfp4 ->
      let scales = Array.init count (fun j -> 118 + (j mod 19)) in
      Nx_quant.mxfp4 ~scales:(u8 blocks scales) (u8 (blocks * size) b)

let quant_cases =
  let open Gen in
  (let* qn = int_range 2 4 in
   let+ format = of_list ~pp:(fun ppf f -> Format.pp_print_string ppf
         (format_name f)) formats
   and+ input = of_list ~pp:(fun ppf i -> Format.pp_print_string ppf
         (input_name i)) inputs
   and+ rows = int_range 1 (qn + 1)
   and+ blocks = int_range 1 2
   and+ m = option (int_range 0 3)
   and+ by_rows = bool in
   let rows_of_x = match m with Some m -> m >= 2 | None -> false in
   { qn; format; input; rows; blocks; m; by_rows = by_rows && rows_of_x })
  |> with_pp pp_quant_case

(* Every format at every input dtype, each once. *)
let quant_examples =
  List.concat_map
    (fun format ->
      List.mapi
        (fun i input ->
          {
            qn = 2 + (i mod 3);
            format;
            input;
            rows = 3;
            blocks = 2;
            m = (if i = 0 then None else Some 2);
            by_rows = i mod 2 = 1;
          })
        inputs)
    formats

let quant_product set c =
  let (X dt) = c.input in
  cover
    (Printf.sprintf "%s by %s" (format_name c.format) (input_name c.input))
    true;
  cover "a weight cut along its rows" c.by_rows;
  let k = c.blocks * block_values c.format in
  let w = quant_weight c.format ~rows:c.rows ~blocks:c.blocks in
  let shape = match c.m with None -> [| k |] | Some m -> [| m; k |] in
  let x = Nx.cast dt (value ~seed:3 shape) in
  let w_at = if c.by_rows then Split 0 else Device 1 in
  let wp = Nx_quant.place (at set w_at) w in
  let xp = Nx.place (at set (if c.by_rows then On else Device 1)) x in
  let f x _ = Nx_quant.apply wp x in
  match compiled f xp xp with
  | Raised m -> fail m
  | Value jit ->
      let eager = f xp xp in
      let x64 = Nx.cast Nx.float64 x
      and v64 = Nx.matrix_transpose (Nx_quant.dequant Nx.float64 w) in
      let exact = Nx.matmul x64 v64 in
      let mass = Nx.matmul (Nx.abs x64) (Nx.abs v64) in
      let g =
        match dt with
        | Nx_array.Dtype.Float64 ->
            let v = ldexp 1. (-52) and n = float_of_int (k + 1) in
            n *. v /. (1. -. (n *. v))
        | _ -> product (k + 1)
      in
      (* One side's distance from [exact]: its sum's bound, and half an ulp
         of [x]'s dtype at the sum and at least a subnormal step. *)
      let sum = Nx.mul_s mass g in
      let side =
        match dt with
        | Nx_array.Dtype.Float32 | Nx_array.Dtype.Float64 -> sum
        | _ ->
            let format = Nx_array.Dtype.float_format dt in
            Nx.add sum
              (Nx.add_s
                 (Nx.mul_s (Nx.add (Nx.abs exact) sum) (format.epsilon /. 2.))
                 (format.min_normal *. format.epsilon))
      in
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
  let _, events = Rig.Profile.take run in
  let copied =
    List.fold_left
      (fun n -> function
        | Rig.Profile.Copy { src; dst; bytes; _ } when not (Rig.equal src dst)
          ->
            n + bytes
        | _ -> n)
      0 events
  in
  equal ~msg:"bytes copied" int 0 copied

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
  let _, events = Rig.Profile.take run in
  let copied =
    List.fold_left
      (fun n -> function
        | Rig.Profile.Copy { src; dst; bytes; _ } when not (Rig.equal src dst)
          ->
            n + bytes
        | _ -> n)
      0 events
  in
  equal ~msg:"bytes copied" int 0 copied

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
  let n = List.length (Nx.rigs set.s) in
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

(* Other sets

   A program that places a value onto another set evaluates, an argument or a
   value it computes, and gives eager's placements and bits: a split program
   keeps what it derives from an argument between calls, which would hold a
   copy on the other set's devices. *)

(* [f] under jit over [x] and [y], evaluating a call. *)
let evaluated f x y =
  let before = Rune.evaluations () in
  let r =
    outcome (fun () -> Rune.jit T.(tensor @-> tensor @-> returns tensor) f x y)
  in
  equal ~msg:"evaluated" bool true (Rune.evaluations () > before);
  r

let others =
  Array.init 2 (fun k ->
      Result.get_ok (Rig.memory_device (Printf.sprintf "other-%d" k)))

module Other = (val Nx.devices (Array.to_list others))

(* An offload: weights on one device placed whole on each device of the
   other set. *)
let elsewhere_examples =
  [
    { n = 2; shape = [| 4; 3 |]; x = Device 0; y = On };
    { n = 4; shape = [| 3; 2 |]; x = Device 2; y = On };
  ]

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
  let same f =
    equal (outcome_t bits_t)
      (omap bits (outcome (fun () -> f x x)))
      (omap bits (evaluated f x x))
  in
  same computed;
  same argument

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

(* Programs

   A program for one device serves every device of its target, so the
   programs a data-parallel step over [n] memory devices takes do not grow
   with [n]: each device's kernels, its stores of pieces into their runs of an
   assembly included, are another's. The count is the same over 4, 8 and 16
   devices, and over 2 at most that: a closing whose sum another device
   copies sums its slots in a kernel of its own from 4 devices on, where over
   2 tolk inlines a sum of two slots into each reader (a value reading at
   most 3 buffers is inlined). A step's programs are counted by their keys,
   made or kept, so that one this process made before counts. The hidden
   width grows with [n] so that every cut, the gradients' closing sums'
   included, is even, and each device's windows have one shape. *)

let sixteen =
  Array.init 16 (fun k ->
      Result.get_ok (Rig.memory_device (Printf.sprintf "programs-%d" k)))

(* The number of program keys among the kernels [f ()] compiles, and the
   name and a digest of each key, for a failure to name them. *)
let keys f =
  let (), ks = Tolk.Realize.kernels f in
  let keys = List.sort_uniq String.compare (List.map snd ks) in
  let named k =
    let name, _ = List.find (fun (_, k') -> String.equal k k') ks in
    name ^ " " ^ String.sub (Digest.to_hex (Digest.string k)) 0 8
  in
  (List.length keys, String.concat ", " (List.map named keys))

(* Counts over 2, 4, 8 and 16 devices that do not grow with the devices:
   those over 4, 8 and 16 are equal, and that over 2 is at most theirs. Each
   count's kernels name what differs. *)
let flat what counts =
  let msg =
    String.concat "\n"
      (List.map2
         (fun n (c, names) ->
           Printf.sprintf "%s over %d: %d: %s" what n c names)
         [ 2; 4; 8; 16 ] counts)
  in
  match List.map fst counts with
  | [ c2; c4; c8; c16 ] ->
      at_most ~msg int ~than:c4 c2;
      equal ~msg (list int) [ c4; c4 ] [ c8; c16 ]
  | _ -> invalid_arg "flat: four counts"

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

let integers ?(seed = 0) shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create f32 shape
    (Array.init n (fun i -> float_of_int ((((seed * 7) + (i * 5)) mod 9) - 4)))

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
   over 2, 4, 8 and 16 memory devices, as the data-parallel step above does.
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
              assemblies);
         group "other sets"
           [
             prop ~count:20 ~examples:elsewhere_examples
               "a value placed onto another set and back evaluates" cases
               (fun c -> over c.n { law = (fun set -> elsewhere set c) });
           ];
         group "loops"
           [
             prop ~count:30 "a scan over a cut carry, bit for bit" loop_cases
               (fun c -> over c.ln { law = (fun set -> scanned set c) });
             prop ~count:20 "an iterate stopped by a fold over a cut axis"
               loop_cases (fun c ->
                 over c.ln { law = (fun set -> iterated set c) });
             test "a scan cut alike copies nothing between devices" (fun () ->
                 over 3 { law = local_scan });
             test "a step that reads its coordinates gives eager's bits"
               (fun () -> over 3 { law = coordinate_scan });
             test "an iterate reads its stop on the program's device"
               (fun () -> over 2 { law = stop_home });
             test "a scan over a batch of one cut across devices" (fun () ->
                 over 2 { law = scan_batch_of_one };
                 over 4 { law = scan_batch_of_one });
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
         group "programs"
           [
             test "a data-parallel step's programs do not grow over 2 to 16"
               (fun () ->
                 flat "programs" (List.map programs [ 2; 4; 8; 16 ]));
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
           ];
         group "each device runs its own kernels"
           [
             test "over two devices" (fun () ->
                 over 2 { law = (fun s -> kernels s 2 4) });
             test "over four, one window empty" (fun () ->
                 over 4 { law = (fun s -> kernels s 4 3) });
           ];
       ])
