(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Shared fixtures and checks for the rune test suite. *)

open Windtrap

let f32 = Nx.float32
let f64 = Nx.float64
let vec32 xs = Nx.create f32 [| Array.length xs |] xs
let vec64 xs = Nx.create f64 [| Array.length xs |] xs
let mat64 r c xs = Nx.create f64 [| r; c |] xs
let to_arr t = Nx.to_array t

(* Collections that release every value no longer reachable. With backtraces
   recorded, the runtime keeps the last exception raised alive, and an
   operation's fallback catches [Effect.Unhandled] carrying its operands:
   raising one more exception first lets that value go. Each finaliser on the
   way to a storage (a compiled function's, then its bound values') takes a
   collection, so collect until the resident bytes stay put twice. *)
(* The cell a placed value's storage belongs to. *)
let cell_of (type a b) (x : (a, b) Nx.t) =
  match x with
  | Nx_effect.Placed r -> r.r_cell
  | Host _ | Traced _ -> fail "expected a placed value"

(* Whether the storage behind a placed value is bound by [n] programs and
   live. *)
let bound_by n x =
  let c = cell_of x in
  c.bound = n && match c.state with Live _ -> true | Consumed _ -> false

let[@inline never] raise_exit () = raise Exit

let full_major () =
  (try raise_exit () with Exit -> ());
  let resident () = (Rune.jit_stats ()).resident_bytes in
  let rec settle before unchanged =
    Gc.full_major ();
    let now = resident () in
    let unchanged = if now = before then unchanged + 1 else 0 in
    if unchanged < 2 then settle now unchanged
  in
  settle (resident ()) 0

let scalar t = (to_arr t).(0)

(* The transformation rules for the sliding-window movement are written against
   its effect; driving them through the public [Nx.sliding_window] pins that the
   exposed function reaches those rules end to end. *)
let sliding_window ~axis ~window ~step x =
  Nx.sliding_window ~axis ~window ~step x

let check_arr ?(eps = 1e-5) ~msg expected actual =
  let t = if eps = 0. then float_exact else float eps in
  let actual = to_arr actual in
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i e -> equal ~msg:(Printf.sprintf "%s[%d]" msg i) t e actual.(i))
    expected

let scalar_like (type a b) (t : (a, b) Nx.t) (v : float) : (a, b) Nx.t =
  let dt = Nx.dtype t in
  Nx.full dt [||] (Nx_core.Dtype.of_float dt v)

let as_f32 (type a b) (x : (a, b) Nx.t) : Nx.float32_t =
  match Nx_core.Dtype.equal_witness (Nx.dtype x) f32 with
  | Some Type.Equal -> x
  | None -> failwith "expected a float32 leaf"

(* A statically-typed parameter record with mixed dtypes: the canonical
   structure used across suites. *)

type params = { w : Nx.float32_t; b : Nx.float32_t; scale : Nx.float64_t }

module Params = struct
  type _ t = params

  let walk c { w; b; scale } =
    let open Nx.Ptree.Walk in
    let w = field c "w" tensor w in
    let b = field c "b" tensor b in
    let scale = field c "scale" tensor scale in
    { w; b; scale }
end

let params_ptree : params Nx.Ptree.t = Nx.Ptree.instantiate (module Params)

let params () =
  {
    w = vec32 [| 1.0; -2.0; 3.0 |];
    b = vec32 [| 0.5 |];
    scale = vec64 [| 2.0 |];
  }

(* A pair of float64 tensors, for differentiating binary operations. *)

type pair = { fst : Nx.float64_t; snd : Nx.float64_t }

module Pair = struct
  type _ t = pair

  let walk c { fst; snd } =
    let open Nx.Ptree.Walk in
    let fst = field c "fst" tensor fst in
    let snd = field c "snd" tensor snd in
    { fst; snd }
end

let pair_ptree : pair Nx.Ptree.t = Nx.Ptree.instantiate (module Pair)

(* Finite-difference oracle.

   Gradient rules are validated against central differences of a scalar float64
   loss. The loss weights the operation's output with a fixed non-uniform
   tensor: a uniform cotangent would let transposition and permutation mistakes
   cancel out in the comparison. *)

(* [weighted y] is [sum (w * y)] with deterministic non-uniform weights. *)
let weighted y =
  let n = Nx.numel y in
  let w =
    Nx.create f64 (Nx.shape y)
      (Array.init n (fun i -> float_of_int ((i mod 5) + 1) /. 2.0))
  in
  Nx.sum (Nx.mul y w)

let central_diff ~h (eval : float array -> float) (xs : float array) :
    float array =
  Array.init (Array.length xs) (fun i ->
      let at d =
        let ys = Array.copy xs in
        ys.(i) <- ys.(i) +. d;
        eval ys
      in
      (at h -. at (-.h)) /. (2.0 *. h))

let check_close ~tol ~msg expected actual =
  Array.iteri
    (fun i e ->
      equal ~msg:(Printf.sprintf "%s[%d]" msg i) (float tol) e actual.(i))
    expected

(* [check_grad ~msg f x] compares [grad' (weighted . f)] at [x] against central
   differences. [f] maps a float64 tensor to a float64 tensor. *)
let check_grad ?(h = 1e-5) ?(tol = 1e-3) ~msg (f : Nx.float64_t -> Nx.float64_t)
    (x : Nx.float64_t) =
  let shape = Nx.shape x in
  let loss x = weighted (f x) in
  let analytic = to_arr (Rune.grad' loss x) in
  let numeric =
    central_diff ~h
      (fun ys -> scalar (loss (Nx.create f64 shape ys)))
      (to_arr x)
  in
  check_close ~tol ~msg numeric analytic

(* [check_grad2 ~msg f a b] is {!check_grad} for a binary operation,
   differentiating with respect to both arguments through a {!Pair}. *)
let check_grad2 ?(h = 1e-5) ?(tol = 1e-3) ~msg
    (f : Nx.float64_t -> Nx.float64_t -> Nx.float64_t) (a : Nx.float64_t)
    (b : Nx.float64_t) =
  let loss p = weighted (f p.fst p.snd) in
  let g = Rune.grad pair_ptree loss { fst = a; snd = b } in
  let shape_a = Nx.shape a and shape_b = Nx.shape b in
  let arr_a = to_arr a and arr_b = to_arr b in
  let num_a =
    central_diff ~h
      (fun ys -> scalar (loss { fst = Nx.create f64 shape_a ys; snd = b }))
      arr_a
  in
  let num_b =
    central_diff ~h
      (fun ys -> scalar (loss { fst = a; snd = Nx.create f64 shape_b ys }))
      arr_b
  in
  check_close ~tol ~msg:(msg ^ ".fst") num_a (to_arr g.fst);
  check_close ~tol ~msg:(msg ^ ".snd") num_b (to_arr g.snd)

(* [tangent_like t] is a deterministic, non-uniform tangent for [t]: zero or
   uniform tangents would mask permutation and scaling mistakes. *)
let tangent_like t =
  let n = Nx.numel t in
  Nx.create f64 (Nx.shape t)
    (Array.init n (fun i -> float_of_int ((i * 7 mod 11) - 5) /. 4.0))

(* [check_jvp ~msg f x] compares the forward-mode tangent of [f] at [x] along
   [tangent_like x] against the central difference of [f] along the same
   direction, elementwise on the output. *)
let check_jvp ?(h = 1e-5) ?(tol = 1e-3) ~msg (f : Nx.float64_t -> Nx.float64_t)
    (x : Nx.float64_t) =
  let v = tangent_like x in
  let _, dy = Rune.jvp' f x v in
  let shape = Nx.shape x in
  let xs = to_arr x and vs = to_arr v in
  let eval d =
    to_arr
      (f
         (Nx.create f64 shape (Array.mapi (fun i xi -> xi +. (d *. vs.(i))) xs)))
  in
  let fp = eval h and fm = eval (-.h) in
  let numeric =
    Array.init (Array.length fp) (fun i -> (fp.(i) -. fm.(i)) /. (2.0 *. h))
  in
  check_close ~tol ~msg numeric (to_arr dy)

(* [check_jvp2 ~msg f a b] is {!check_jvp} for a binary operation, feeding
   tangents to both arguments through a {!Pair}. *)
let check_jvp2 ?(h = 1e-5) ?(tol = 1e-3) ~msg
    (f : Nx.float64_t -> Nx.float64_t -> Nx.float64_t) (a : Nx.float64_t)
    (b : Nx.float64_t) =
  let va = tangent_like a and vb = tangent_like b in
  let _, dy =
    Rune.jvp pair_ptree Nx.Ptree.tensor
      (fun p -> f p.fst p.snd)
      { fst = a; snd = b } { fst = va; snd = vb }
  in
  let shape_a = Nx.shape a and shape_b = Nx.shape b in
  let arr_a = to_arr a and arr_b = to_arr b in
  let arr_va = to_arr va and arr_vb = to_arr vb in
  let eval d =
    let bump xs vs = Array.mapi (fun i xi -> xi +. (d *. vs.(i))) xs in
    to_arr
      (f
         (Nx.create f64 shape_a (bump arr_a arr_va))
         (Nx.create f64 shape_b (bump arr_b arr_vb)))
  in
  let fp = eval h and fm = eval (-.h) in
  let numeric =
    Array.init (Array.length fp) (fun i -> (fp.(i) -. fm.(i)) /. (2.0 *. h))
  in
  check_close ~tol ~msg numeric (to_arr dy)

(* Complex finite-difference oracle.

   A complex tensor is a pair of real components, and every rule is a real
   linear map on them. These checks differentiate both components separately,
   assemble the real Jacobian, and compare against it under rune's packing: a
   tangent carries [dre + i*dim], a cotangent carries [dL/dre - i*dL/dim]. That
   makes them independent of the convention being right — they measure the
   operation, not another rule. *)

let c128 = Nx.complex128
let cx re im = Complex.{ re; im }

let cvec xs =
  Nx.create c128 [| Array.length xs |] (Array.map (fun (re, im) -> cx re im) xs)

let cmat r c xs =
  Nx.create c128 [| r; c |] (Array.map (fun (re, im) -> cx re im) xs)

let to_carr t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

(* Deterministic complex weights. Both components vary and neither is ever zero:
   a cotangent that is real everywhere is exactly what lets a missing
   conjugation agree with the oracle by coincidence. The two generators differ
   so that a binary rule reading the wrong argument cannot cancel out. *)

let cotangent_like t =
  Nx.create c128 (Nx.shape t)
    (Array.init (Nx.numel t) (fun i ->
         cx
           (float_of_int ((i mod 5) + 1) /. 2.0)
           (float_of_int (((i + 1) mod 4) - 5) /. 4.0)))

let ctangent_like t =
  Nx.create c128 (Nx.shape t)
    (Array.init (Nx.numel t) (fun i ->
         cx
           (float_of_int ((i * 3 mod 5) - 6) /. 4.0)
           (float_of_int ((i mod 4) + 2) /. 3.0)))

let check_cclose ~tol ~msg expected actual =
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i (e : Complex.t) ->
      let a : Complex.t = actual.(i) in
      equal
        ~msg:(Printf.sprintf "%s[%d].re" msg i)
        (float tol) e.Complex.re a.Complex.re;
      equal
        ~msg:(Printf.sprintf "%s[%d].im" msg i)
        (float tol) e.Complex.im a.Complex.im)
    expected

(* [check_carr ~msg expected t] is {!check_cclose} on the flattened complex
   tensor [t]. *)
let check_carr ?(eps = 1e-10) ~msg expected actual =
  check_cclose ~tol:eps ~msg expected (to_carr actual)

(* Loop oracle for vmap: [vmap' f x] must equal stacking [f] applied to each
   slice of [x] along the mapped axis. *)
let loop_map f x =
  let b = (Nx.shape x).(0) in
  Nx.stack ~axis:0 (List.init b (fun i -> f (Nx.slice [ Nx.I i ] x)))

let check_vmap ~msg f x =
  check_arr ~msg (to_arr (loop_map f x)) (Rune.vmap' f x)

let check_cvmap ~msg f x =
  check_carr ~msg (to_carr (loop_map f x)) (Rune.vmap' f x)

let raises_jit_error f =
  raises_match
    (fun exn -> match exn with Rune.Jit_error _ -> true | _ -> false)
    f

(* [cvjp_numeric ~h f z w] is the cotangent [w] pulled back through the real
   Jacobian of [f] at [z], measured by central differences: perturb each
   component of each input, read how each component of each output responds, and
   contract with [w] unpacked into [(dL/dre, dL/dim)]. *)
let cvjp_numeric ~h f (z : Nx.complex128_t) (w : Complex.t array) =
  let shape = Nx.shape z in
  let zs = to_carr z in
  let eval a = to_carr (f (Nx.create c128 shape a)) in
  Array.init (Array.length zs) (fun j ->
      let at dre dim =
        let ys = Array.copy zs in
        ys.(j) <- cx (ys.(j).Complex.re +. dre) (ys.(j).Complex.im +. dim);
        eval ys
      in
      let fpre = at h 0.0 and fmre = at (-.h) 0.0 in
      let fpim = at 0.0 h and fmim = at 0.0 (-.h) in
      let dre = ref 0.0 and dim = ref 0.0 in
      Array.iteri
        (fun i (wi : Complex.t) ->
          let slope p m component =
            (component p.(i) -. component m.(i)) /. (2.0 *. h)
          in
          let re (c : Complex.t) = c.Complex.re
          and im (c : Complex.t) = c.Complex.im in
          let contract p m =
            (slope p m re *. wi.Complex.re) -. (slope p m im *. wi.Complex.im)
          in
          dre := !dre +. contract fpre fmre;
          dim := !dim +. contract fpim fmim)
        w;
      cx !dre (-. !dim))

(* [cjvp_numeric ~h f z v] is the central difference of [f] at [z] along [v],
   taken on both components at once — the directional derivative that a
   pushforward must reproduce. *)
let cjvp_numeric ~h f (z : Nx.complex128_t) (v : Nx.complex128_t) =
  let shape = Nx.shape z in
  let zs = to_carr z and vs = to_carr v in
  let eval d =
    to_carr
      (f
         (Nx.create c128 shape
            (Array.mapi
               (fun i (zi : Complex.t) ->
                 cx
                   (zi.Complex.re +. (d *. vs.(i).Complex.re))
                   (zi.Complex.im +. (d *. vs.(i).Complex.im)))
               zs)))
  in
  let fp = eval h and fm = eval (-.h) in
  Array.init (Array.length fp) (fun i ->
      cx
        ((fp.(i).Complex.re -. fm.(i).Complex.re) /. (2.0 *. h))
        ((fp.(i).Complex.im -. fm.(i).Complex.im) /. (2.0 *. h)))

(* [check_cgrad ~msg f z] compares the reverse-mode pullback of a deterministic
   complex cotangent against the transposed finite-difference Jacobian. *)
let check_cgrad ?(h = 1e-5) ?(tol = 1e-5) ~msg
    (f : Nx.complex128_t -> Nx.complex128_t) (z : Nx.complex128_t) =
  let w = cotangent_like (f z) in
  let _, g = Rune.vjp' f z w in
  check_cclose ~tol ~msg (cvjp_numeric ~h f z (to_carr w)) (to_carr g)

(* [check_cjvp ~msg f z] compares the forward-mode pushforward of a
   deterministic complex tangent against the same central difference. *)
let check_cjvp ?(h = 1e-5) ?(tol = 1e-5) ~msg
    (f : Nx.complex128_t -> Nx.complex128_t) (z : Nx.complex128_t) =
  let v = ctangent_like z in
  let _, dy = Rune.jvp' f z v in
  check_cclose ~tol ~msg (cjvp_numeric ~h f z v) (to_carr dy)

(* A pair of complex128 tensors, for differentiating binary operations. *)

type cpair = { cfst : Nx.complex128_t; csnd : Nx.complex128_t }

module Cpair = struct
  type _ t = cpair

  let walk c { cfst; csnd } =
    let open Nx.Ptree.Walk in
    let cfst = field c "cfst" tensor cfst in
    let csnd = field c "csnd" tensor csnd in
    { cfst; csnd }
end

let cpair_ptree : cpair Nx.Ptree.t = Nx.Ptree.instantiate (module Cpair)

(* [check_cgrad2 ~msg f a b] is {!check_cgrad} for a binary operation. Each
   argument is checked against the Jacobian taken with the other one held fixed,
   so a contribution routed to the wrong argument shows up. *)
let check_cgrad2 ?(h = 1e-5) ?(tol = 1e-5) ~msg
    (f : Nx.complex128_t -> Nx.complex128_t -> Nx.complex128_t)
    (a : Nx.complex128_t) (b : Nx.complex128_t) =
  let w = cotangent_like (f a b) in
  let _, g =
    Rune.vjp cpair_ptree Nx.Ptree.tensor
      (fun p -> f p.cfst p.csnd)
      { cfst = a; csnd = b } w
  in
  let warr = to_carr w in
  check_cclose ~tol ~msg:(msg ^ ".fst")
    (cvjp_numeric ~h (fun x -> f x b) a warr)
    (to_carr g.cfst);
  check_cclose ~tol ~msg:(msg ^ ".snd")
    (cvjp_numeric ~h (fun y -> f a y) b warr)
    (to_carr g.csnd)

(* [check_cjvp2 ~msg f a b] is {!check_cjvp} for a binary operation. The
   pushforward is linear in the tangents, so the oracle is the sum of the two
   partial directional derivatives. *)
let check_cjvp2 ?(h = 1e-5) ?(tol = 1e-5) ~msg
    (f : Nx.complex128_t -> Nx.complex128_t -> Nx.complex128_t)
    (a : Nx.complex128_t) (b : Nx.complex128_t) =
  let va = ctangent_like a and vb = cotangent_like b in
  let _, dy =
    Rune.jvp cpair_ptree Nx.Ptree.tensor
      (fun p -> f p.cfst p.csnd)
      { cfst = a; csnd = b } { cfst = va; csnd = vb }
  in
  let from_a = cjvp_numeric ~h (fun x -> f x b) a va in
  let from_b = cjvp_numeric ~h (fun y -> f a y) b vb in
  check_cclose ~tol ~msg
    (Array.mapi (fun i d -> Complex.add d from_b.(i)) from_a)
    (to_carr dy)

(* Bitcast compiles to the bits eager reads, both ways, from a transposed view:
   both zeros, infinities, quiet and signalling NaN with payloads, subnormals.
   The bits leave and enter the compiled function as stored, and an eager
   bitcast, which is a view, reads them. *)
let check_bitcast_matches_eager ?devices () =
  let check (type a b c d) name (bits : (a, b) Nx.t) (float : (c, d) Nx.dtype) =
    let int = Nx.dtype bits in
    let to_float x = Nx.bitcast float (Nx.transpose x) in
    let to_bits x = Nx.bitcast int (Nx.transpose x) in
    let expected = Nx.to_array (Nx.transpose bits) in
    equal ~msg:(name ^ " from bits") bool true
      (Nx.to_array (Nx.bitcast int (Rune.jit' ?devices to_float bits))
      = expected);
    equal ~msg:(name ^ " to bits") bool true
      (Nx.to_array (Rune.jit' ?devices to_bits (Nx.bitcast float bits))
      = expected)
  in
  check "float32"
    (Nx.create Nx.int32 [| 2; 5 |]
       [|
         0l;
         0x80000000l;
         0x7F800000l;
         0xFF800000l;
         0x7FC00000l;
         0x7F800001l;
         0xFFC00123l;
         1l;
         0x807FFFFFl;
         0x3F800000l;
       |])
    Nx.float32;
  let halves =
    [| 0; 0x8000; 0x7C00; 0xFC00; 0x7E00; 0x7C01; 0xFE23; 1; 0x83FF; 0x3C00 |]
  in
  check "float16" (Nx.create Nx.uint16 [| 2; 5 |] halves) Nx.float16;
  check "bfloat16"
    (Nx.create Nx.uint16 [| 2; 5 |]
       [|
         0; 0x8000; 0x7F80; 0xFF80; 0x7FC0; 0x7F81; 0xFFC3; 1; 0x807F; 0x3F80;
       |])
    Nx.bfloat16

(* A compiled gather returns the elements it selects, so a -0 stays -0 through
   strided and listed slices, [take] and [take_along_axis], as eagerly. Each
   result is compared as the bits of its float width, widened to int32. *)
let check_gathers_keep_negative_zero ?devices () =
  let check (type b c d) name (dtype : (float, b) Nx.dtype)
      (bits : (c, d) Nx.dtype) =
    let row = [| -0.; 0.; -1.; 0.; -0.; -2. |] in
    let x = Nx.cast dtype (Nx.create f32 [| 1; 6 |] row) in
    let table =
      Nx.cast dtype
        (Nx.init f32 [| 100; 2 |] (fun i ->
             if (i.(0) + i.(1)) mod 3 = 0 then -0. else float_of_int i.(1)))
    in
    let rows = Nx.create Nx.int32 [| 4 |] [| 0l; 3l; 6l; 99l |] in
    let columns = Nx.create Nx.int32 [| 1; 3 |] [| 0l; 4l; 2l |] in
    let bits_of t = Nx.to_array (Nx.cast Nx.int32 (Nx.bitcast bits t)) in
    List.iter
      (fun (msg, f, x) ->
        equal
          ~msg:(Printf.sprintf "%s %s" name msg)
          (array int32)
          (bits_of (f x))
          (bits_of (Rune.jit' ?devices f x)))
      [
        ("strided slice", Nx.slice [ Nx.A; Nx.Rs (0, 6, 2) ], x);
        ("reversed strided slice", Nx.slice [ Nx.A; Nx.Rs (5, -7, -2) ], x);
        ("listed slice", Nx.slice [ Nx.A; Nx.L [ 4; 0; 1 ] ], x);
        ("strided rows", Nx.slice [ Nx.Rs (0, 100, 3) ], table);
        ("take", Nx.take ~axis:0 ~indices:rows, table);
        ("take_along_axis", Nx.take_along_axis ~axis:1 ~indices:columns, x);
      ]
  in
  check "float32" Nx.float32 Nx.int32;
  check "float16" Nx.float16 Nx.int16;
  check "bfloat16" Nx.bfloat16 Nx.int16

(* A row long enough for 2-bit rounds, and k * (n + 1) past int32, so the
   running count is int64: all but about a thousand entries tie at the
   threshold, more than 2^20 of them, so their count times k passes 2^31 along
   the row. Compiled, as eagerly, the first k of a stable descending sort. *)
let check_top_k_long_row ?devices () =
  let n = (1 lsl 20) + 8192 and k = 2048 in
  let st = Random.State.make [| 4 |] in
  let x =
    Nx.init f32 [| 1; n |] (fun _ ->
        if Random.State.int st 1024 = 0 then Random.State.float st 1. +. 1.
        else 0.)
  in
  let expected =
    Nx.to_array
      (Nx.shrink [| (0, 1); (0, k) |] (Nx.argsort ~descending:true ~axis:1 x))
  in
  let f x = snd (Nx.top_k ~k x) in
  equal ~msg:"eager" (array int32) expected (Nx.to_array (f x));
  equal ~msg:"compiled" (array int32) expected
    (Nx.to_array (Rune.jit' ?devices f x))

(* Sorting *)

(* [n] entries of [dtype] in runs of equal values, with NaN, both zeros and,
   when [infinities], both infinities for a float dtype, and the extremes for an
   integer one. *)
let sort_input (type a b) ?(infinities = true) (dtype : (a, b) Nx.dtype) n :
    (a, b) Nx.t =
  let is_float = Nx_core.Dtype.is_float dtype in
  let x =
    Nx.cast dtype
      (Nx.create f64 [| n |]
         (Array.init n (fun i ->
              match i mod 17 with
              | 3 when is_float -> Float.nan
              | 5 when is_float && infinities -> Float.infinity
              | 8 when is_float && infinities -> Float.neg_infinity
              | 11 when is_float -> -0.
              | 13 when is_float -> 0.
              | _ when is_float -> float_of_int ((i * 7 mod 13) - 6) /. 4.
              | _ -> float_of_int (i * 7 mod 13))))
  in
  if is_float then x
  else
    let at r =
      Nx.create Nx.bool [| n |] (Array.init n (fun i -> i mod 17 = r))
    in
    let full v = Nx.full dtype [| n |] v in
    Nx.where (at 3)
      (full (Nx_core.Dtype.max_value dtype))
      (Nx.where (at 8) (full (Nx_core.Dtype.min_value dtype)) x)

(* One vector cut into [pieces], each a shape sorted along an axis in both
   directions, so that one compiled program covers every case of a dtype. The
   result concatenates, for each piece and direction, the sorted values and then
   the indices, as [out]. *)
let sort_pieces out pieces x =
  let offset = ref 0 in
  Nx.concatenate ~axis:0
    (List.concat_map
       (fun (shape, axis) ->
         let size = Array.fold_left ( * ) 1 shape in
         let piece =
           Nx.reshape shape (Nx.shrink [| (!offset, !offset + size) |] x)
         in
         offset := !offset + size;
         List.concat_map
           (fun descending ->
             let values, indices = Nx.sort ~descending ~axis piece in
             [
               Nx.flatten (Nx.cast out values); Nx.flatten (Nx.cast out indices);
             ])
           [ false; true ])
       pieces)

(* A compiled sort returns the input's elements at the positions eager's argsort
   gives, bit for bit: both zeros, and NaNs of either sign with payloads, along
   a short axis and one of 600. The values are compared as bits, widened to
   int32, outside the compiled function, where a float8 bitcast is allowed. *)
let check_sort_values_are_elements ?devices () =
  let nan_of bits = Int64.float_of_bits bits in
  let short =
    [|
      1.;
      -0.;
      Float.nan;
      0.;
      -2.;
      nan_of 0xFFF8000000000123L;
      -0.;
      0.;
      3.;
      nan_of 0x7FF4000000000001L;
    |]
  in
  let long =
    Array.init 1200 (fun i ->
        match i mod 11 with
        | 1 -> -0.
        | 2 -> 0.
        | 3 -> nan_of 0xFFF8000000000123L
        | 5 -> Float.nan
        | _ -> float_of_int (i mod 7))
  in
  let check (type b c d) name (dtype : (float, b) Nx.dtype)
      (bits : (c, d) Nx.dtype) =
    let bits_of t = Nx.to_array (Nx.cast Nx.int32 (Nx.bitcast bits t)) in
    List.iter
      (fun (shape, row) ->
        let x = Nx.cast dtype (Nx.create f64 shape row) in
        let axis = Array.length shape - 1 in
        List.iter
          (fun descending ->
            let msg =
              Printf.sprintf "%s, %d along the axis, %s" name shape.(axis)
                (if descending then "descending" else "ascending")
            in
            let indices = snd (Nx.sort ~descending ~axis x) in
            let values x = fst (Nx.sort ~descending ~axis x) in
            equal ~msg (array int32)
              (bits_of (Nx.take_along_axis ~axis ~indices x))
              (bits_of (Rune.jit' ?devices values x)))
          [ false; true ])
      [ ([| 10 |], short); ([| 2; 600 |], long) ]
  in
  check "float32" Nx.float32 Nx.int32;
  check "float16" Nx.float16 Nx.int16;
  check "bfloat16" Nx.bfloat16 Nx.int16;
  check "float8_e4m3" Nx.float8_e4m3 Nx.uint8;
  check "float8_e5m2" Nx.float8_e5m2 Nx.uint8

(* Compiled [sort_pieces] of a [sort_input] against eager, segment by segment. A
   zero compares without its sign: eager's value sort leaves the order of -0 and
   +0 unspecified ([check_sort_values_are_elements] checks the compiled
   bits). *)
let check_sort_pieces (type a b) ?infinities out pieces
    (dtype : (a, b) Nx.dtype) =
  let size shape = Array.fold_left ( * ) 1 shape in
  let total = List.fold_left (fun acc (s, _) -> acc + size s) 0 pieces in
  let x = sort_input ?infinities dtype total in
  let unsigned_zero v = if v = 0. then 0. else v in
  let expected = Array.map unsigned_zero (to_arr (sort_pieces out pieces x)) in
  let actual =
    Array.map unsigned_zero (to_arr (Rune.jit' (sort_pieces out pieces) x))
  in
  let at = ref 0 in
  List.iter
    (fun (shape, axis) ->
      List.iter
        (fun segment ->
          let msg =
            Format.asprintf "%a, [%s] along %d, %s" Nx.pp_dtype dtype
              (String.concat "; "
                 (Array.to_list (Array.map string_of_int shape)))
              axis segment
          in
          check_arr ~eps:0. ~msg
            (Array.sub expected !at (size shape))
            (vec64 (Array.sub actual !at (size shape)));
          at := !at + size shape)
        [
          "ascending values";
          "ascending indices";
          "descending values";
          "descending indices";
        ])
    pieces

(* An index outside the axis drops a scatter's update, reads zero at a gather
   and passes no gradient back through one, where no range of the compiled
   destination's address meets the updates: one update beside a unit axis, or a
   scatter axis of extent 1, whose index may be broadcast inside the function.
   With and without the promise of unique indices. *)
let check_out_of_range_beside_unit_axes ?devices () =
  let check msg expected f t =
    check_arr ~msg:(msg ^ ", eager") expected (f t);
    check_arr ~msg:(msg ^ ", compiled") expected (Rune.jit' ?devices f t)
  in
  let column = Nx.create f32 [| 3; 1 |] [| 1.; 2.; 3. |] in
  let one = Nx.create f32 [| 1; 1 |] [| 1. |] in
  let row = Nx.create f32 [| 1; 3 |] [| 1.; 2.; 3. |] in
  List.iter
    (fun (t, ids, broadcast, set, add) ->
      let n = Array.length ids in
      let indices =
        Nx.create Nx.int32 [| n; 1 |] (Array.map Int32.of_int ids)
      in
      let shape = Option.value broadcast ~default:[| n; 1 |] in
      let values =
        Nx.create f32 shape
          (Array.init (Array.fold_left ( * ) 1 shape) (fun j ->
               9. -. float_of_int j))
      in
      let at =
        String.concat "; " (Array.to_list (Array.map string_of_int ids))
      in
      let dims =
        String.concat "x" (Array.to_list (Array.map string_of_int (Nx.shape t)))
      in
      List.iter
        (fun unique_indices ->
          let scatter mode t =
            let indices =
              match broadcast with
              | None -> indices
              | Some shape -> Nx.broadcast_to shape indices
            in
            Nx.scatter ~mode ~unique_indices ~axis:0 ~indices ~values t
          in
          let msg =
            Printf.sprintf "%s at [%s], unique %b" dims at unique_indices
          in
          check (msg ^ ", set") set (scatter `Set) t;
          check (msg ^ ", add") add (scatter `Add) t)
        [ false; true ])
    [
      (column, [| -1 |], None, [| 1.; 2.; 3. |], [| 1.; 2.; 3. |]);
      (column, [| 3 |], None, [| 1.; 2.; 3. |], [| 1.; 2.; 3. |]);
      (column, [| -1; 1 |], None, [| 1.; 8.; 3. |], [| 1.; 10.; 3. |]);
      (column, [| 1; 3 |], None, [| 1.; 9.; 3. |], [| 1.; 11.; 3. |]);
      (one, [| -1; -1 |], None, [| 1. |], [| 1. |]);
      (row, [| -1; -1 |], Some [| 2; 3 |], [| 1.; 2.; 3. |], [| 1.; 2.; 3. |]);
    ];
  List.iter
    (fun (ids, taken) ->
      let indices =
        Nx.create Nx.int32 [| Array.length ids |] (Array.map Int32.of_int ids)
      in
      check "take" taken (Nx.take ~axis:0 ~indices) column)
    [ ([| -1 |], [| 0. |]); ([| 3 |], [| 0. |]); ([| -1; 1 |], [| 0.; 2. |]) ];
  let far = Nx.create Nx.int32 [| 2 |] [| -1l; -1l |] in
  check "the gradient of take over an axis of size 1" [| 0.; 0.; 0. |]
    (Rune.grad' (fun t -> Nx.sum (Nx.take ~axis:0 ~indices:far t)))
    row
