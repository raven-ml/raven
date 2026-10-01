(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every row's tangent and pullback under the compiled call: compiled, each is
   eager's, on the host and on Metal, and so are the maps of every row and of
   its tangent and pullback; a row the compiled call cannot compute raises
   Jit_error at the trace. Named cases keep the old compiled assertions
   of the reductions' rules, and place a captured coefficient beside a traced
   argument, whose recorded tangent takes the argument's context. *)

open Windtrap
module Rune = Rune_next.Rune

let operands () = Nx.Ptree.(list tensor)

let pair_of_lists () =
  Nx.Ptree.(list tensor @-> list tensor @-> returns (list tensor))

let direction seed x =
  Reference.direction (Random.State.make [| seed |]) (operands ()) x

let at gen =
  Gen.with_pp
    (fun ppf (i, seed) ->
      Format.fprintf ppf "%a@ seed %d" Case.pp_instance i seed)
    (Gen.pair gen (Gen.int_range 0 1_000_000))

let leaves x = Reference.leaves (operands ()) x
let is_jit_error = function Rune.Jit_error _ -> true | _ -> false

(* A target of the compiled call: where it places the operands, and the dtypes
   it cannot compute. *)
type target = {
  place : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t;
  refuses : Nx_dtype.Scalar.t -> bool;
}

let host =
  {
    place = Fun.id;
    refuses =
      (fun d ->
        Nx_dtype.Scalar.equal d (Nx_dtype.Scalar.of_dtype Nx.complex128)
        || Nx_dtype.Scalar.equal d (Nx_dtype.Scalar.of_dtype Nx.complex64));
  }

(* [refused target c dtypes] is [true] if the compiled call cannot compute the
   row whose operands and results have [dtypes]: a row it does not lower (the
   transforms and the general and Hermitian eigendecompositions), a dtype the
   target refuses, or the float64 values svd always gives. *)
let scalars x = List.map (fun t -> Nx_dtype.Scalar.of_dtype (Nx.dtype t)) x

let refused target (c : Case.t) dtypes =
  let lowered =
    match c.row with Fft | Rfft | Irfft | Eig | Eigh -> false | _ -> true
  in
  let values =
    match c.row with Svd -> [ Nx_dtype.Scalar.of_dtype Nx.float64 ] | _ -> []
  in
  (not lowered) || List.exists target.refuses (values @ dtypes)

(* [compiled_is_eager ~rel ~place c] compares [jit] of the row's tangent and of
   its pullback with eager's, the operands moved by [place] first. *)
let compiled_is_eager ~rel ~target (c : Case.t) (Case.Instance i, seed) =
  let place = target.place in
  let x = List.map place i.x in
  let along x v = snd (Rune.jvp (operands ()) (operands ()) i.f x v) in
  let pull x w = snd (Rune.vjp (operands ()) (operands ()) i.f x) w in
  let v = List.map place (direction seed i.x) in
  let w = List.map place (direction (seed + 1) (i.f i.x)) in
  if refused target c (scalars i.x @ scalars (i.f i.x)) then begin
    raises_match ~msg:"the tangent" is_jit_error (fun () ->
        Rune.jit (pair_of_lists ()) along x v);
    raises_match ~msg:"the pullback" is_jit_error (fun () ->
        Rune.jit (pair_of_lists ()) pull x w)
  end
  else begin
    let close = Reference.close ~rel ~floor:rel () in
    equal ~msg:"the tangent" close
      (leaves (along i.x (direction seed i.x)))
      (leaves (Rune.jit (pair_of_lists ()) along x v));
    equal ~msg:"the pullback" close
      (leaves (pull i.x (direction (seed + 1) (i.f i.x))))
      (leaves (Rune.jit (pair_of_lists ()) pull x w))
  end

(* [rows ~points] runs the law on each row with a tangent at [points c] when it
   has some. *)
let rows ~count ~rel ~points ~target =
  List.filter_map
    (fun (c : Case.t) ->
      match (c.kind, c.row, points c) with
      | Case.Tangent, _, None -> None
      | Case.Tangent, _, Some points ->
          Some
            (prop ~count (Row.name c.row) (at points)
               (compiled_is_eager ~rel ~target c))
      | (Case.Plain | Case.Integer), _, _ -> None)
    Case.all

(* Under a map *)

(* How a batch is drawn: which operands carry it, how many rows, and the seed of
   the rows and of the directions. *)
type batch = { which : bool list; length : int; seed : int }

let mapped gen =
  let open Gen in
  with_pp
    (fun ppf (i, b) ->
      Format.fprintf ppf "%a@ batched %s, %d rows, seed %d" Case.pp_instance i
        (String.concat ""
           (List.map (fun t -> if t then "x" else "-") b.which))
        b.length b.seed)
    (let* (Case.Instance i as inst) = gen in
     let* which =
       map
         (fun bits ->
           if List.exists Fun.id bits then bits else true :: List.tl bits)
         (list ~size:(constant (List.length i.x)) bool)
     in
     let+ length = int_range 1 3 and+ seed = int_range 0 1_000_000 in
     (inst, { which; length; seed }))

let rec merge which all xs =
  match (which, all, xs) with
  | true :: which, _ :: all, x :: xs -> x :: merge which all xs
  | false :: which, a :: all, xs -> a :: merge which all xs
  | [], [], [] -> []
  | _ -> invalid_arg "merge"

(* [stack ~seed ~length x] is [length] rows around [x] stacked on a new axis 0,
   each moved a little along a direction, so that it stays in the row's domain.
*)
let stack ~seed ~length x =
  let v = Reference.direction (Random.State.make [| seed |]) Nx.Ptree.tensor x in
  Nx.stack
    (List.init length (fun k ->
         Nx.add x
           (Nx.mul v
              (Nx.full (Nx.dtype v) [||]
                 (Nx_dtype.of_float (Nx.dtype v) (1e-3 *. float_of_int k))))))

let map_of_lists () = Nx.Ptree.(list tensor @-> returns (list tensor))

(* [mapped_is_eager ~rel ~target c] compares [jit] of the map of the row with
   eager's and, for a row with a tangent, [jit] of the maps of its tangent and
   of its pullback; the batched operands are moved by [target.place] first. *)
let mapped_is_eager ~rel ~target (c : Case.t) (Case.Instance i, b) =
  let place = target.place in
  let f xs = i.f (merge b.which i.x xs) in
  let batched seed xs =
    List.mapi
      (fun j x -> stack ~seed:(seed + j) ~length:b.length x)
      (List.filteri (fun j _ -> List.nth b.which j) xs)
  in
  let x = batched b.seed i.x in
  let y = Rune.vmap (map_of_lists ()) f x in
  let refuses = refused target c (scalars i.x @ scalars (i.f i.x)) in
  let compare msg eager compiled =
    if refuses then raises_match ~msg is_jit_error compiled
    else
      match c.kind with
      | Case.Tangent ->
          equal ~msg (Reference.close ~rel ~floor:rel ())
            (leaves eager) (leaves (compiled ()))
      | Case.Plain | Case.Integer ->
          equal ~msg (list (Reference.exact ())) eager (compiled ())
  in
  compare "the map" y (fun () ->
      Rune.jit (map_of_lists ()) (Rune.vmap (map_of_lists ()) f)
        (List.map place x));
  match c.kind with
  | Case.Plain | Case.Integer -> ()
  | Case.Tangent ->
      let along x v = snd (Rune.jvp (operands ()) (operands ()) f x v) in
      let pull x w = snd (Rune.vjp (operands ()) (operands ()) f x) w in
      let v = direction (b.seed + 100) x in
      let w = direction (b.seed + 200) y in
      let both g p q =
        ( Rune.vmap (pair_of_lists ()) g p q,
          fun () ->
            Rune.jit (pair_of_lists ())
              (Rune.vmap (pair_of_lists ()) g)
              (List.map place p) (List.map place q) )
      in
      let eager, jitted = both along x v in
      compare "the tangent" eager jitted;
      let eager, jitted = both pull x w in
      compare "the pullback" eager jitted

let mapped_rows ~count ~rel ~points ~target =
  List.filter_map
    (fun (c : Case.t) ->
      Option.map
        (fun points ->
          prop ~count (Row.name c.row) (mapped points)
            (mapped_is_eager ~rel ~target c))
        (points c))
    Case.all

(* Named cases *)

let vec a = Nx.create Nx.float64 [| Array.length a |] a
let floats = array float_exact
let reduce k axes x = Nx.Op.eval (Reduce (k, axes, x))
let compiled_grad f x = Rune.jit' (Rune.grad' f) x

let reductions =
  [
    test "a compiled product keeps the multiplicity of its zeros" (fun () ->
        let x =
          Nx.create Nx.float64 [| 3; 3 |]
            [| 2.; 3.; 4.; 2.; 0.; 4.; 0.; 0.; 4. |]
        in
        equal floats
          [| 12.; 8.; 6.; 0.; 8.; 0.; 0.; 0.; 0. |]
          (Nx.to_array
             (compiled_grad (fun x -> Nx.sum (reduce Prod [| 1 |] x)) x)));
    test "compiled tied extrema share the derivative" (fun () ->
        let x =
          Nx.create Nx.float64 [| 2; 3 |] [| 2.; 2.; 0.; -1.; -1.; 3. |]
        in
        equal ~msg:"max" floats
          [| 0.5; 0.5; 0.; 0.; 0.; 1. |]
          (Nx.to_array
             (compiled_grad (fun x -> Nx.sum (reduce Max [| 1 |] x)) x));
        equal ~msg:"min" floats
          [| 0.; 0.; 1.; 0.5; 0.5; 0. |]
          (Nx.to_array
             (compiled_grad (fun x -> Nx.sum (reduce Min [| 1 |] x)) x)));
    test "a compiled float16 tie count above 65,504 does not overflow"
      (fun () ->
        let n = 65536 in
        let g =
          compiled_grad (reduce Max [| 0 |]) (Nx.ones Nx.float16 [| n |])
        in
        equal
          (array (float 1e-9))
          (Array.make n (1. /. float_of_int n))
          (Nx.to_array g));
  ]

(* A device over the host's memory whose programs are the host's, so that a
   traced argument lies off the host. *)
let device =
  Nx.Device.of_runtime
    (Nx_device.Driver.device ~name:"COMPILED-RULES" ~arch:"test" ~budget:max_int
       (Host_visible
          { memory = Nx_device.Driver.host_memory; mapping = Some Identity }))

let on_device x = Nx.place (Nx.Placement.device device) x

let coefficients =
  let c = vec [| 1.5; -2.; 0.25 |] in
  let m = Nx.create Nx.float64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  [
    test
      "a captured coefficient times a traced argument has the argument's \
       gradient" (fun () ->
        let g =
          compiled_grad
            (fun x -> Nx.sum (Nx.mul x c))
            (on_device (vec [| 1.; 2.; 3. |]))
        in
        is_true ~msg:"where the argument lies"
          (Nx.Placement.equal (Nx.Placement.device device) (Nx.placement g));
        equal floats [| 1.5; -2.; 0.25 |] (Nx.to_array g));
    test "a product with a captured matrix has the argument's gradient"
      (fun () ->
        let g =
          compiled_grad
            (fun x -> Nx.sum (Nx.matmul x m))
            (on_device (Nx.create Nx.float64 [| 1; 3 |] [| 1.; 2.; 3. |]))
        in
        is_true ~msg:"where the argument lies"
          (Nx.Placement.equal (Nx.Placement.device device) (Nx.placement g));
        equal floats [| 3.; 7.; 11. |] (Nx.to_array g));
  ]

(* nx's functions whose gradients reach several rows, compiled; those that
   compile a factorization are slow. *)

let mat r c a = Nx.create Nx.float64 [| r; c |] a

let iota shape =
  Nx.create Nx.float64 shape
    (Array.init (Array.fold_left ( * ) 1 shape) (fun i -> float_of_int (i + 1)))

let ints shape a = Nx.create Nx.int64 shape (Array.map Int64.of_int a)
let near = array (float_rel ~rel:1e-10 ~abs:1e-12)

let compiled_like_eager ?msg f x =
  equal ?msg near (Nx.to_array (f x)) (Nx.to_array (Rune.jit' f x))

let compositions =
  let spd = mat 3 3 [| 4.; 1.; 2.; 1.; 5.; 3.; 2.; 3.; 6. |] in
  [
    slow "the gradient of a Cholesky-using loss compiles" (fun () ->
        compiled_like_eager
          (Rune.grad' (fun m -> Nx.sum (Nx.mul (Nx.cholesky m) (Nx.cholesky m))))
          spd);
    slow "the gradient of det compiles" (fun () ->
        compiled_like_eager (Rune.grad' Nx.det)
          (mat 3 3 [| 0.3; 1.2; -0.4; 2.1; 0.5; 0.9; -0.7; 1.6; 3.2 |]));
    slow "the gradient of a QR-using loss compiles" (fun () ->
        let loss m =
          let q, r = Nx.qr ~mode:`Reduced m in
          let lq = Nx.tril q and ur = Nx.triu r in
          Nx.add (Nx.sum (Nx.mul lq lq)) (Nx.sum (Nx.mul ur ur))
        in
        compiled_like_eager (Rune.grad' loss)
          (mat 4 4
             [|
               12.;
               1.;
               3.;
               0.5;
               1.;
               13.;
               2.;
               1.;
               3.;
               2.;
               14.;
               0.25;
               0.5;
               1.;
               0.5;
               15.;
             |]));
    test "gradients through indices outside the axis" (fun () ->
        let indices = ints [| 4 |] [| -1; 2; 4; 0 |] in
        let weights = vec [| 1.; 2.; 3.; 4. |] in
        let through ~values t =
          Nx.sum (Nx.mul weights (Nx.scatter ~axis:0 ~indices ~values t))
        in
        let t = Nx.zeros Nx.float64 [| 4 |]
        and values = vec [| 10.; 20.; 30.; 40. |] in
        let both msg expected f x =
          equal ~msg:(msg ^ ", eager") floats expected (Nx.to_array (f x));
          equal ~msg:(msg ^ ", compiled") floats expected
            (Nx.to_array (Rune.jit' f x))
        in
        both "the values" [| 0.; 3.; 0.; 1. |]
          (Rune.grad' (fun values -> through ~values t))
          values;
        both "the target" [| 0.; 2.; 0.; 4. |]
          (Rune.grad' (fun t -> through ~values t))
          t;
        both "the table" [| 4.; 0.; 2.; 0. |]
          (Rune.grad' (fun table ->
               Nx.sum (Nx.mul weights (Nx.take ~indices table))))
          (vec [| 1.; 1.; 1.; 1. |]));
    test "the gradient of take with repeated tokens" (fun () ->
        let indices = ints [| 6 |] [| 3; 1; 3; 3; 0; 1 |] in
        let weights = iota [| 6; 2 |] in
        compiled_like_eager
          (Rune.grad' (fun table ->
               Nx.sum (Nx.mul weights (Nx.take ~axis:0 ~indices table))))
          (iota [| 5; 2 |]));
    test "the gradient of top_k lands on the chosen entries" (fun () ->
        let scores = mat 2 5 [| 3.; 9.; 1.; 7.; 5.; 4.; 2.; 8.; 6.; 0. |] in
        let weights = mat 1 2 [| 1.; 2. |] in
        let loss x = Nx.sum (Nx.mul weights (fst (Nx.top_k ~k:2 x))) in
        equal floats
          [| 0.; 1.; 0.; 2.; 0.; 0.; 0.; 1.; 2.; 0. |]
          (Nx.to_array (Rune.grad' loss scores));
        compiled_like_eager (Rune.grad' loss) scores);
    test "a compiled map of scatter is its eager map" (fun () ->
        let indices = ints [| 3; 2 |] [| 1; 0; 1; 2; 0; 0 |] in
        let values = iota [| 3; 2 |] and t = iota [| 3; 2 |] in
        let batch x = Nx.stack [ x; Nx.add x x ] in
        List.iter
          (fun (name, mode) ->
            compiled_like_eager
              ~msg:("over the target, " ^ name)
              (Rune.vmap' (fun t -> Nx.scatter ~mode ~axis:0 ~indices ~values t))
              (batch t);
            compiled_like_eager
              ~msg:("over the values, " ^ name)
              (Rune.vmap' (fun values ->
                   Nx.scatter ~mode ~axis:0 ~indices ~values t))
              (batch values))
          [ ("set", `Set); ("add", `Add) ];
        compiled_like_eager ~msg:"over the indices"
          (fun rows ->
            Nx.cast Nx.float64
              (Rune.vmap'
                 (fun indices ->
                   Nx.scatter ~mode:`Add ~axis:0 ~indices ~values t)
                 rows))
          (ints [| 2; 3; 2 |] [| 1; 0; 1; 2; 0; 0; 2; 2; 2; 1; 0; 1 |]));
    test
      "a compiled reduce_segments by extremes is its eager value, bit for bit"
      (fun () ->
        let x =
          Nx.bitcast Nx.float64
            (Nx.create Nx.uint64 [| 7 |]
               [|
                 Int64.bits_of_float 1.;
                 0x7ff8000000000005L;
                 Int64.bits_of_float (-0.);
                 0xfff8000000000006L;
                 0L;
                 Int64.bits_of_float 3.;
                 Int64.bits_of_float (-2.);
               |])
        in
        let ids = ints [| 7 |] [| 0; 0; 1; 0; 1; 3; 2 |] in
        List.iter
          (fun mode ->
            let f x =
              Nx.bitcast Nx.uint64 (Nx.reduce_segments mode ~segments:3 ids x)
            in
            equal (array int64)
              (Nx.to_array (f x))
              (Nx.to_array (Rune.jit' f x)))
          [ `Max; `Min ]);
  ]

let metal =
  match Metal.device with
  | None -> []
  | Some d ->
      let d = Nx.Device.of_runtime d in
      let float32 = Case.D Nx.float32 in
      let points (c : Case.t) =
        if List.mem float32 c.dtypes then Some (c.smooth float32) else None
      in
      let target =
        {
          place = (fun x -> Nx.place (Nx.Placement.device d) x);
          refuses =
            (fun s ->
              host.refuses s
              || Nx_dtype.Scalar.equal s (Nx_dtype.Scalar.of_dtype Nx.float64));
        }
      in
      [
        group ~tags:[ "slow" ] "metal" (rows ~count:2 ~rel:1e-3 ~points ~target);
        group ~tags:[ "slow" ] "metal under a map"
          (mapped_rows ~count:1 ~rel:1e-3 ~points ~target);
      ]

let () =
  exit
    (run "rune.next compiled rules"
       ([
          group "reductions" reductions;
          group "coefficients" coefficients;
          group "compositions" compositions;
          group ~tags:[ "slow" ] "host"
            (rows ~count:4 ~rel:1e-9
               ~points:(fun (c : Case.t) -> Some c.finite)
               ~target:host);
          group ~tags:[ "slow" ] "host under a map"
            (mapped_rows ~count:2 ~rel:1e-9
               ~points:(fun (c : Case.t) -> Some c.finite)
               ~target:host);
        ]
       @ metal))
