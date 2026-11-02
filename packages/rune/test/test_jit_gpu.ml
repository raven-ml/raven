(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jit on this machine's first GPU: a scatter past its work bound gives
   the host's eager values, a cast between float32 and a 16-bit float gives
   eager's bits at every code and tie, a product of narrow floats or of MXFP4
   weights holds its bound, a call's check raises at the next wait, so the
   call returns while its work runs, a key split there has the host's words,
   and a program over the GPU and a memory device, devices of two targets,
   compiles to eager's values with replicas equal bit for bit. It skips
   without a GPU, or without the compiler the GPU's kernels need. *)

open Windtrap
open Rune_support
open Witness

let gpu =
  lazy
    (match Rig_gpus.first () with
    | Error _ -> None
    | Ok d -> Some (d, Nx.devices [ d ]))

(* A compile error that names a header the machine lacks is the machine's: the
   test skips. Half floats' kernels read headers no other kernel does. *)
let missing m =
  let has s =
    let n = String.length s in
    let rec at i =
      i + n <= String.length m && (String.sub m i n = s || at (i + 1))
    in
    at 0
  in
  if has "cannot open source file" then
    Some ("a header is missing, set CUDA_PATH: " ^ m)
  else None

let compiles (module S : Nx.Devices) =
  let neg (Nx.Dtype.Any dt) =
    let x = Nx.cast dt (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]) in
    ignore (Rune.jit' Nx.neg (Nx.place S.on x))
  in
  List.iter
    (fun d ->
      match neg d with
      | () -> ()
      | exception (Tolk.Renderer.Compiler.Compile_error m as e) -> (
          match missing m with
          | Some why -> skip ~reason:why ()
          | None -> raise e))
    Nx.Dtype.[ Any Nx.float32; Any Nx.float16; Any Nx.bfloat16 ]

(* The first GPU, skipping where its renderer cannot be made: without its
   compiler's library, jit evaluates each call uncompiled. *)
let on_gpu () =
  match Lazy.force gpu with
  | None -> skip ~reason:"no GPU" ()
  | Some (d, s) -> (
      match Tolk_engine.renderer d with
      | exception Failure why -> skip ~reason:why ()
      | _ ->
          compiles s;
          (d, s))

let law c = Scatters.law (snd (on_gpu ())) c
let casts16 dt () = Casts16.law (snd (on_gpu ())) dt

exception Big of int array * float

(* A step whose work is in flight when the call returns. An iterate's stop is
   read on the host each trip, so the work that stays in flight is the work
   after it: [products] products of [n; n] matrices, after a check of the first
   one, which fails at [0; 0]. They are 4.4 TFLOP: at least 68 ms at the RTX
   5000 Ada's float32 peak of 65 TFLOPS, tens of milliseconds on any current
   GPU, while the host reads the timeline microseconds after the call's last
   submit, so the GPU cannot finish them before the read. Each product by
   [w], whose elements are [1/n], keeps the values [n], exactly. *)
let n = 4096
let products = 32

let test_deferred () =
  let d, (module S) = on_gpu () in
  let step x =
    let x =
      Rune.iterate' ~max:4
        ~until:(fun x -> Nx.greater (Nx.sum x) (Nx.scalar_like (Nx.sum x) 0.))
        (fun x -> Nx.add x (Nx.scalar_like x 1.))
        x
    in
    let y = Nx.matmul x x in
    Nx.check Nx.Ptree.tensor
      (Nx.less y (Nx.scalar_like y (float_of_int n /. 2.)))
      y
      (fun i v -> Big (i, Nx.item [] v));
    let w = Nx.div x (Nx.scalar_like x (float_of_int n)) in
    let h = ref y in
    for _ = 1 to products do
      h := Nx.matmul !h w
    done;
    !h
  in
  let x = Nx.place S.on (Nx.zeros Nx.float32 [| n; n |]) in
  let g = Rune.jit' step in
  let before = Rune.evaluations () in
  (* The first call compiles. *)
  (try Nx.sync (g x) with Big _ -> ());
  let y = g x in
  let signaled = Rig.signaled d and submitted = Rig.submitted d in
  equal ~msg:"compiled" Testable.int 0 (Rune.evaluations () - before);
  less ~msg:"the call returned before its work completed" Testable.int
    ~than:submitted signaled;
  raises ~msg:"the next wait raises the check"
    (Big ([| 0; 0 |], float_of_int n))
    (fun () -> Nx.sync y)

(* Several targets

   A program over the GPU beside a memory device compiles, each device running
   kernels of its own target, and gives the placements the same function gives
   eagerly over two memory devices, an exact function's bits and another's
   values within rune.mli's classes. The devices of the first device's
   architecture compute a value whole on several devices and the others
   receive copies, so its replicas are equal bit for bit, whichever device
   comes first. *)

let f32 = Nx.float32

(* A placement: whole on each device, cut along an axis, or, over a 2 x 2
   mesh, cut along axis [0] over its first axis and whole along its second. *)
type cut = On | Split of int | Rows

(* A set, and the mesh over it where it has four devices. *)
type 'd set = { s : 'd Nx.devices; mesh : 'd Nx.Mesh.t option }

let set_of (type d) (s : d Nx.devices) =
  let mesh =
    if List.length (Nx.rigs s) = 4 then
      Some (Nx.Mesh.v s [ ("x", 2); ("y", 2) ])
    else None
  in
  { s; mesh }

let at set = function
  | On -> Nx.Placement.on set.s
  | Split axis -> Nx.Placement.split ~axis set.s
  | Rows -> Nx.Placement.mesh (Option.get set.mesh) [ (0, [ "x" ]) ]

(* A value of [shape] on the host, of elements in [-1, 1] that depend on [seed]
   and their index. *)
let value ?(seed = 0) shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create f32 shape
    (Array.init n (fun i -> sin (float_of_int ((seed * 7919) + i + 1))))

let host x = Nx.place Nx.Host.on x
let placement () = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

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

(* [jit] within [bound] of [eager] elementwise, NaN where it is NaN. *)
let within ~bound eager jit =
  let e = Nx.to_array (Nx.cast Nx.float64 (host eager)) in
  let j = Nx.to_array (Nx.cast Nx.float64 (host jit)) in
  let b = Nx.to_array (Nx.cast Nx.float64 (host bound)) in
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

(* A function of two values, its arguments on the host with their placements,
   its result's placement, and how its compiled result compares with eager's:
   [close x y eager jit]. *)
type case = {
  name : string;
  f : 'd. (float, Nx.float32_elt, 'd) Nx.t -> (float, Nx.float32_elt, 'd) Nx.t ->
      (float, Nx.float32_elt, 'd) Nx.t;
  x : (float, Nx.float32_elt, Nx.host) Nx.t * cut;
  y : (float, Nx.float32_elt, Nx.host) Nx.t * cut;
  result : cut;
  close :
    (float, Nx.float32_elt, Nx.host) Nx.t ->
    (float, Nx.float32_elt, Nx.host) Nx.t ->
    (float, Nx.float32_elt, Nx.host) Nx.t ->
    (float, Nx.float32_elt, Nx.host) Nx.t ->
    unit;
}

let exact _ _ e j = equal bits_t (bits e) (bits j)

(* A value whose magnitudes span twenty binades, so that two orders of
   addition round apart. *)
let spread shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create f32 shape
    (Array.init n (fun i -> ldexp (sin (float_of_int (i + 1))) (i * 7 mod 20)))

let mass x = Nx.abs (Nx.cast Nx.float64 x)

(* [tanh (3 x)]: on the GPU and eagerly each within 18 ulps of the correctly
   rounded result, an ulp at most [2 u] of it, so within [72 u] of each other,
   relatively, and [73 u] of eager's result. *)
let tanh x = Nx.tanh (Nx.mul_s x 3.)
let tanh_err = 73. *. ldexp 1. (-24)

let cases =
  [
    {
      name = "elementwise over a cut and a whole value";
      f = (fun x y -> Nx.maximum (Nx.mul x y) (Nx.sub x (Nx.neg y)));
      x = (value [| 8; 3 |], Split 0);
      y = (value ~seed:1 [| 8; 3 |], On);
      result = Split 0;
      close = exact;
    };
    {
      name = "elementwise over whole values";
      f = (fun x y -> Nx.maximum (Nx.mul x y) (Nx.sub x (Nx.neg y)));
      x = (value [| 8; 3 |], On);
      y = (value ~seed:1 [| 8; 3 |], On);
      result = On;
      close = exact;
    };
    {
      name = "a tanh of a whole value";
      f = (fun x _ -> tanh x);
      x = (value [| 1024 |], On);
      y = (value ~seed:1 [| 1 |], On);
      result = On;
      close = (fun _ _ e j -> within ~bound:(Nx.mul_s (mass e) tanh_err) e j);
    };
    {
      name = "a sum over the cut axis";
      f = (fun x _ -> Nx.sum ~axes:[ 0 ] x);
      x = (spread [| 9; 7 |], Split 0);
      y = (value ~seed:1 [| 1 |], On);
      result = On;
      close =
        (fun x _ e j ->
          within ~bound:(Nx.mul_s (Nx.sum ~axes:[ 0 ] (mass x)) (2. *. gamma 9))
            e j);
    };
    {
      name = "a matrix product cut along its contracted axis";
      f = Nx.matmul;
      x = (value [| 5; 6 |], Split 1);
      y = (value ~seed:1 [| 6; 3 |], Split 0);
      result = On;
      close =
        (fun x y e j ->
          within
            ~bound:(Nx.mul_s (Nx.matmul (mass x) (mass y)) (2. *. product 7))
            e j);
    };
    {
      (* As test_jit_devices.ml's gradient: [g = xᵀ (1 - t²)] for
         [t = tanh (x w)]. *)
      name = "the gradient of a loss over a cut batch";
      f = (fun x w -> Rune.grad' (fun w -> Nx.sum (Nx.tanh (Nx.matmul x w))) w);
      x = (value [| 6; 4 |], Split 0);
      y = (value ~seed:1 [| 4; 3 |], On);
      result = On;
      close =
        (fun x w e j ->
          let ax = mass x and u = ldexp 1. (-24) in
          let term =
            Nx.add_s
              (Nx.mul_s (Nx.matmul ax (mass w)) (2. *. product 5))
              ((38. *. u) +. product 7)
          in
          let bound = Nx.mul_s (Nx.matmul (Nx.transpose ax) term) 2. in
          within ~bound e j);
    };
  ]

(* Each device's piece of [y], on the host, and its window. *)
let pieces y =
  let s = Nx.Placement.devices (Nx.placement y) in
  let windows = Nx.Prim.pieces (Nx.placement y) (Nx.shape y) in
  Array.mapi
    (fun j a -> (bits (host (Nx.Repr.of_array s a)), snd windows.(j)))
    (Iarray.to_array (Nx.Repr.shards y))

(* Over a 2 x 2 mesh: a value cut along its first axis over the mesh's first
   axis, which its second repeats, so that each window lies on two devices and
   each fold over that axis closes in two groups. *)
let mesh_cases =
  (* The terms [t] of a sum or a cumulative sum of [n] of [tanh (3 x)]: within
     [(73 u + 2 γ(n)) Σ|t|] of eager's. *)
  let folded fold n x _ e j =
    let terms = Nx.abs (Nx.cast Nx.float64 (tanh x)) in
    within ~bound:(Nx.mul_s (fold terms) (tanh_err +. (2. *. gamma n))) e j
  in
  [
    {
      name = "a tanh of rows";
      f = (fun x _ -> tanh x);
      x = (value [| 256; 4 |], Rows);
      y = (value ~seed:1 [| 1 |], On);
      result = Rows;
      close = (fun _ _ e j -> within ~bound:(Nx.mul_s (mass e) tanh_err) e j);
    };
    {
      name = "a sum over the rows";
      f = (fun x _ -> Nx.sum ~axes:[ 0 ] (tanh x));
      x = (value [| 64; 7 |], Rows);
      y = (value ~seed:1 [| 1 |], On);
      result = On;
      close = folded (Nx.sum ~axes:[ 0 ]) 64;
    };
    {
      (* The positions of the column extremes, distinct values, as floats. *)
      name = "the positions of the extremes over the rows";
      f = (fun x _ -> Nx.cast f32 (Nx.argmax ~axis:0 (Nx.mul_s x 3.)));
      x = (value [| 64; 5 |], Rows);
      y = (value ~seed:1 [| 1 |], On);
      result = On;
      close = exact;
    };
    {
      name = "a cumulative sum along the rows";
      f = (fun x _ -> Nx.cumsum ~axis:0 (tanh x));
      x = (value [| 64; 3 |], Rows);
      y = (value ~seed:1 [| 1 |], On);
      result = Rows;
      close = folded (Nx.cumsum ~axis:0) 64;
    };
  ]

(* [c] compiled over [mixed] against [c] applied eagerly over [memory]. Devices
   that hold one window of the result hold one piece, bit for bit. *)
let check (type d e) (mixed : d set) (memory : e set) c =
  let (x, cx), (y, cy) = (c.x, c.y) in
  let eager = c.f (Nx.place (at memory cx) x) (Nx.place (at memory cy) y) in
  equal ~msg:"eager's placement" (placement ()) (at memory c.result)
    (Nx.placement eager);
  let f = Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) c.f in
  let before = Rune.evaluations () in
  let jit = f (Nx.place (at mixed cx) x) (Nx.place (at mixed cy) y) in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  equal ~msg:"placement" (placement ()) (at mixed c.result) (Nx.placement jit);
  c.close x y (host eager) (host jit);
  let ps = pieces jit and rigs = Array.of_list (Nx.rigs mixed.s) in
  Array.iteri
    (fun j (b, w) ->
      match Array.find_index (fun (_, w') -> w' = w) ps with
      | Some i when i < j ->
          equal
            ~msg:("the replica on " ^ Rig.name rigs.(j))
            bits_t (fst ps.(i)) b
      | _ -> ())
    ps

let memory k = Result.get_ok (Rig.memory_device ("rune-jit-gpu-" ^ k))

(* The sets the laws run over: the GPU and memory devices, in an order that
   puts the GPU at [first], and as many memory devices. *)
let sets =
  Hashtbl.create 4

let over ~devices ~first c () =
  let gpu, _ = on_gpu () in
  let (mixed, eager) =
    match Hashtbl.find_opt sets (devices, first) with
    | Some s -> s
    | None ->
        let ms = List.init devices (fun k -> memory (string_of_int k)) in
        let others = List.tl ms in
        let mixed =
          List.filteri (fun k _ -> k < first) others
          @ [ gpu ]
          @ List.filteri (fun k _ -> k >= first) others
        in
        let s = (Nx.devices mixed, Nx.devices ms) in
        Hashtbl.replace sets (devices, first) s;
        s
  in
  let (module S : Nx.Devices) = mixed and (module M : Nx.Devices) = eager in
  check (set_of S.v) (set_of M.v) c

(* [split k]'s second key under jit at the GPU's set. A GPU set has no
   kernels, so a key there derives only in a program. *)
type second =
  | Second : 'd Nx.Placement.t * ('d Nx.Rng.key -> 'd Nx.Rng.key) -> second

let second =
  lazy
    (let _, (module S) = on_gpu () in
     Second
       ( S.on,
         Rune.jit
           Nx.Ptree.(Nx.Rng.ptree @-> returns Nx.Rng.ptree)
           (fun k -> (Nx.Rng.split k).(1)) ))

let words k = Nx.to_array (Nx.Rng.to_tensor (Nx.Rng.place Nx.Host.on k))

let law_split seed =
  let (Second (p, f)) = Lazy.force second in
  let k = Nx.Rng.key seed in
  equal
    (Testable.array Testable.int32)
    (words (Nx.Rng.split k).(1))
    (words (f (Nx.Rng.place p k)))

(* Matrix units

   A compiled contraction of bfloat16 or float16 operands into float32 holds
   {!Nx.contract}'s bound, [γ(K + 1, 2u) (|init| + Σ|a||b|)], on the matrix
   units the compiler takes for it too. Each row is [c] then [K - 1] equal
   terms [t] against ones, with [t] just under [2^j 2u c] for [j] from [1] to
   [13], up to float16's unit in the last place at [c], and just under one and
   two of float32's. A unit whose sum keeps float32's width errs by less than
   one of its units per addition, so it meets the bound even where it
   truncates, as the rows with [t] near [u c] press it to. A narrower sum, or
   one that drops a term, loses about [t] per addition from [j = 1], twice the
   bound or more: a float16 accumulator errs by 2 to 1700 times it on these
   rows. *)

let units_k = 2880

(* The terms of each row, as multiples of [u c]: fifteen, and the first again,
   so that the rows fill a tensor core's tile of sixteen, which takes no
   product of fifteen rows. *)
let unit_terms =
  let terms =
    (255. /. 256.) :: (255. /. 128.)
    :: List.init 13 (fun j -> 255. /. 256. *. ldexp 1. (j + 2))
  in
  terms @ [ List.hd terms ]

let unit_case (type b) (dt : (float, b) Nx.dtype) c () =
  let _, (module S) = on_gpu () in
  let u = ldexp 1. (-24) in
  let ts = Array.of_list (List.map (fun m -> m *. u *. c) unit_terms) in
  let rows = Array.length ts and cols = 64 in
  let a =
    Nx.init Nx.float32 [| rows; units_k |] (fun i ->
        if i.(1) = 0 then c else ts.(i.(0)))
  in
  equal ~msg:"the terms are exact in the operands' dtype"
    (Testable.array float_exact) (Nx.to_array a)
    (Nx.to_array (Nx.cast Nx.float32 (Nx.cast dt a)));
  let a = Nx.place S.on (Nx.cast dt a) in
  let b = Nx.place S.on (Nx.ones dt [| units_k; cols |]) in
  let p = Nx.Pattern.v "i j, j k -> i k | j" in
  let f a b = Nx.contract Nx.float32 p a b in
  let jit = Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) f a b in
  let n = float_of_int (units_k + 1) and v = 2. *. u in
  let g = n *. v /. (1. -. (n *. v)) in
  Array.iteri
    (fun i y ->
      let t = ts.(i / cols) in
      let exact = c +. (float_of_int (units_k - 1) *. t) in
      less
        ~msg:(Printf.sprintf "row %d, terms %h, column %d" (i / cols) t
                (i mod cols))
        float_exact ~than:(g *. exact)
        (Float.abs (y -. exact)))
    (Nx.to_array (host jit))

let unit_cases =
  [
    test "bfloat16 into float32" (unit_case Nx.bfloat16 1.);
    test "float16 into float32" (unit_case Nx.float16 2048.);
  ]

(* A product of narrow floats: a [m; k] by [k; n] Nx.matmul compiled on the
   GPU, against the product of the same values computed on the host in
   float64, whose own error is 2^-29 of the bound's. {!Nx.contract}'s bound
   for [k] products summed at float32 is [γ(k + 1, 2u) Σ|a b|]; the result
   then rounds once into its dtype, by a relative [u] of that dtype or, below
   its least normal, half its least subnormal. A GPU's tensor cores take
   extents that 16 divides and refuse the others. *)

type narrow = Narrow : (float, 's) Nx.dtype * float * float -> narrow

let narrows =
  [
    Narrow (Nx.float16, 0x1p-11, 0x1p-25);
    Narrow (Nx.bfloat16, 0x1p-8, 0x1p-134);
  ]

let pp_product ppf (Narrow (dt, _, _), (m, k, n), seed) =
  Format.fprintf ppf "%a [%d; %d] by [%d; %d], seed %d" Nx.Dtype.pp dt m k k n
    seed

let products_gen =
  let extent = Gen.of_list [ 0; 1; 7; 16; 17; 32; 48 ] in
  let tiled = Gen.of_list [ 16; 32; 48 ] in
  let shape =
    Gen.frequency
      [
        (1, Gen.triple extent extent extent); (1, Gen.triple tiled tiled tiled);
      ]
  in
  Gen.with_pp pp_product
    (Gen.triple (Gen.of_list narrows) shape (Gen.int_range 0 1_000_000))

let product_law (Narrow (dt, u, tiny), (m, k, n), seed) =
  let _, (module S) = on_gpu () in
  cover "extents that 16 divides"
    (m > 0 && k > 0 && n > 0 && m mod 16 = 0 && k mod 16 = 0 && n mod 16 = 0);
  let draw seed s =
    let x = Nx.Rng.uniform ~key:(Nx.Rng.key seed) Nx.float32 s in
    Nx.cast dt (Nx.sub_s (Nx.mul_s x 4.) 2.)
  in
  let a = draw seed [| m; k |] and b = draw (seed + 1) [| k; n |] in
  let before = Rune.evaluations () in
  let y =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> returns tensor)
      Nx.matmul (Nx.place S.on a) (Nx.place S.on b)
  in
  equal ~msg:"compiled" int 0 (Rune.evaluations () - before);
  let y = Nx.to_array (Nx.cast Nx.float64 (host y)) in
  let a = Nx.cast Nx.float64 a and b = Nx.cast Nx.float64 b in
  let exact = Nx.to_array (Nx.matmul a b) in
  let mass = Nx.to_array (Nx.matmul (Nx.abs a) (Nx.abs b)) in
  Array.iteri
    (fun i z ->
      let e = product (k + 1) *. mass.(i) in
      let bound = e +. (u *. (Float.abs z +. e)) +. tiny in
      less
        ~msg:(Printf.sprintf "[%d; %d], exactly %h" (i / n) (i mod n) z)
        float_exact ~than:bound
        (Float.abs (y.(i) -. z)))
    exact

(* Sixteen products of [1] by [1; t; ...; t], [t = 255 2^-31] just under
   float32's [u], every value a bfloat16: rows of ones by columns of these
   terms, [32; 16] by [16; 32], extents a core takes. A sum that keeps
   float32's width rounds each addition of [t] to [1] at most to its unit in
   the last place, within [Nx.contract]'s bound. *)
let test_sixteen_terms () =
  let _, (module S) = on_gpu () in
  let k = 16 and n = 32 and t = 255. *. ldexp 1. (-31) in
  let a = Nx.ones Nx.bfloat16 [| n; k |] in
  let b =
    Nx.cast Nx.bfloat16
      (Nx.init Nx.float32 [| k; n |] (fun i -> if i.(0) = 0 then 1. else t))
  in
  equal ~msg:"the terms are bfloat16s" float_exact t
    (Nx.item [ 1; 0 ] (Nx.cast Nx.float32 b));
  let y =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> returns tensor)
      (Nx.contract Nx.float32 (Nx.Pattern.v "i j, j k -> i k | j"))
      (Nx.place S.on a) (Nx.place S.on b)
  in
  let exact = 1. +. (float_of_int (k - 1) *. t) in
  Array.iteri
    (fun i y ->
      less
        ~msg:(Printf.sprintf "[%d; %d]" (i / n) (i mod n))
        float_exact ~than:(product (k + 1) *. exact)
        (Float.abs (y -. exact)))
    (Nx.to_array (host y))

let () =
  exit
    (run "Rune.jit on a GPU"
       [
         group "scatters"
           [
             prop ~count:4 ~examples:Scatters.examples
               "a scatter past 16 times eager's work compiles to eager's values"
               (Gen.with_pp Scatters.pp Scatters.gen)
               law;
           ];
         group "16-bit casts"
           [
             test "float16 gives eager's bits" (casts16 Nx.float16);
             test "bfloat16 gives eager's bits" (casts16 Nx.bfloat16);
           ];
         group "products"
           [
             prop ~count:24
               "a float16 or bfloat16 matmul is within its bound of the exact \
                product"
               products_gen product_law;
             test "sixteen terms under float32's unit hold the bound"
               test_sixteen_terms;
             Windtrap.cases
               "an MXFP4 product is within its bound of the exact one"
               ~id:(Format.asprintf "%a" Mxfp4_products.pp)
               Mxfp4_products.cases
               (fun c -> Mxfp4_products.law (snd (on_gpu ())) c);
           ];
         group "checks"
           [
             test "a call's check raises at the next wait, not at the call"
               test_deferred;
           ];
         group "a GPU beside a memory device"
           (List.map
              (fun c -> test c.name (over ~devices:2 ~first:0 c))
              cases);
         group "a memory device beside a GPU"
           (List.map
              (fun c -> test c.name (over ~devices:2 ~first:1 c))
              cases);
         group "a mesh of a GPU first and three memory devices"
           (List.map
              (fun c -> test c.name (over ~devices:4 ~first:0 c))
              mesh_cases);
         group "a mesh of a memory device, a GPU and two memory devices"
           (List.map
              (fun c -> test c.name (over ~devices:4 ~first:1 c))
              mesh_cases);
         group "a contraction on matrix units holds its bound" unit_cases;
         group "keys"
           [
             prop ~count:20
               ~examples:[ 0; -1; max_int; min_int ]
               "a key split under jit has the eager split's words" Gen.int
               law_split;
           ];
       ])
