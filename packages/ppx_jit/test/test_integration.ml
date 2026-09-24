(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let f32 shape values = Nx.create Nx.float32 shape values

module Params = struct
  type t = { w : Nx.float32_t; b : Nx.float32_t } [@@deriving ptree]
end

module Vec = struct
  type t = { x : Nx.float32_t; y : Nx.float32_t } [@@deriving ptree]
end

(* The function under test, in eager form. *)
let affine (p : Params.t) (v : Vec.t) : Vec.t * Nx.float32_t =
  let y = Nx.add (Nx.mul p.w v.x) p.b in
  let z = Nx.add (Nx.mul p.w v.y) p.b in
  ({ Vec.x = y; y = z }, Nx.add y z)

(* The same function compiled with the manual boilerplate: the reference the
   [@jit] expansion is checked against. *)
module Affine_in = struct
  type t = { p : Params.t; v : Vec.t } [@@deriving ptree]
end

module Affine_out = struct
  type t = { result : Vec.t * Nx.float32_t } [@@deriving ptree]
end

let affine_manual =
  let impl =
    Rune.jit2
      (module Affine_in)
      (module Affine_out)
      (fun { Affine_in.p; v } -> { Affine_out.result = affine p v })
  in
  fun p v -> (impl { Affine_in.p; v }).Affine_out.result

let[@jit] affine_jit (p : Params.t) (v : Vec.t) : Vec.t * Nx.float32_t =
  affine p v

let test_matches_eager_and_manual () =
  let p =
    Params.{ w = f32 [| 2 |] [| 1.5; -0.5 |]; b = f32 [| 2 |] [| 0.25; 2. |] }
  in
  let v =
    Vec.{ x = f32 [| 2 |] [| 3.; -1. |]; y = f32 [| 2 |] [| 0.5; 4. |] }
  in
  let eager_out, eager_scalar = affine p v in
  let manual_out, manual_scalar = affine_manual p v in
  let jit_out, jit_scalar = affine_jit p v in
  let check_out expected actual =
    equal
      (array (float 1e-6))
      (Nx.to_array expected.Vec.x)
      (Nx.to_array actual.Vec.x);
    equal
      (array (float 1e-6))
      (Nx.to_array expected.Vec.y)
      (Nx.to_array actual.Vec.y)
  in
  check_out eager_out manual_out;
  check_out eager_out jit_out;
  equal
    (array (float 1e-6))
    (Nx.to_array eager_scalar)
    (Nx.to_array manual_scalar);
  equal (array (float 1e-6)) (Nx.to_array eager_scalar) (Nx.to_array jit_scalar)

let test_reuses_compiled_function () =
  let p = Params.{ w = f32 [| 1 |] [| 2. |]; b = f32 [| 1 |] [| 1. |] } in
  let v1 = Vec.{ x = f32 [| 1 |] [| 1. |]; y = f32 [| 1 |] [| 2. |] } in
  let v2 = Vec.{ x = f32 [| 1 |] [| 3. |]; y = f32 [| 1 |] [| 4. |] } in
  let out1, scalar1 = affine_jit p v1 in
  let out2, scalar2 = affine_jit p v2 in
  let exp1, exps1 = affine p v1 in
  let exp2, exps2 = affine p v2 in
  equal (array (float 1e-6)) (Nx.to_array exp1.x) (Nx.to_array out1.x);
  equal (array (float 1e-6)) (Nx.to_array exps1) (Nx.to_array scalar1);
  equal (array (float 1e-6)) (Nx.to_array exp2.y) (Nx.to_array out2.y);
  equal (array (float 1e-6)) (Nx.to_array exps2) (Nx.to_array scalar2)

(* Labelled and optional arguments keep their calling convention. *)
let[@jit] shifted ~(p : Params.t)
    ?(shift : Nx.float32_t = f32 [| 2 |] [| 0.; 0. |]) (v : Vec.t) : Vec.t =
  let out, _ = affine p v in
  Vec.{ x = Nx.add out.x shift; y = Nx.add out.y shift }

let test_labelled_and_optional () =
  let p =
    Params.{ w = f32 [| 2 |] [| 1.; 1. |]; b = f32 [| 2 |] [| 0.5; -0.5 |] }
  in
  let v =
    Vec.{ x = f32 [| 2 |] [| 2.; 3. |]; y = f32 [| 2 |] [| -1.; 1.5 |] }
  in
  let base, _ = affine p v in
  let defaulted = shifted ~p v in
  equal (array (float 1e-6)) (Nx.to_array base.x) (Nx.to_array defaulted.x);
  equal (array (float 1e-6)) (Nx.to_array base.y) (Nx.to_array defaulted.y);
  let shift = f32 [| 2 |] [| 10.; 20. |] in
  let explicit = shifted ~p ~shift v in
  equal
    (array (float 1e-6))
    (Nx.to_array (Nx.add base.x shift))
    (Nx.to_array explicit.x);
  equal
    (array (float 1e-6))
    (Nx.to_array (Nx.add base.y shift))
    (Nx.to_array explicit.y)

(* The payload is spliced into the [Rune.jit2] call, so it may mention values in
   scope at module level. *)
let cpu_device = "CPU"

let[@jit { device = cpu_device }] square (a : Nx.float32_t) : Nx.float32_t =
  Nx.mul a a

let test_runtime_device_payload () =
  let a = f32 [| 3 |] [| 1.; -2.; 3. |] in
  equal (array (float 1e-6)) (Nx.to_array (Nx.mul a a)) (Nx.to_array (square a))

(* [@jit] functions can live inside modules. *)
module Nested = struct
  let[@jit] negate (a : Nx.float32_t) : Nx.float32_t = Nx.neg a
end

let test_inside_module () =
  let a = f32 [| 2 |] [| 4.; -5. |] in
  equal
    (array (float 1e-6))
    (Nx.to_array (Nx.neg a))
    (Nx.to_array (Nested.negate a))

let tests =
  [
    test "matches eager and manual jit2" test_matches_eager_and_manual;
    test "reuses the compiled function across calls"
      test_reuses_compiled_function;
    test "labelled and optional arguments" test_labelled_and_optional;
    test "payload splices runtime values" test_runtime_device_payload;
    test "works inside modules" test_inside_module;
  ]

let () = run "ppx_jit integration" tests
