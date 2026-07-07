(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Vega.Loss_scale: constructors, scale/unscale round-trips,
   finiteness checks and the dynamic adjustment schedule. *)

open Windtrap
module Ls = Vega.Loss_scale

let f32 = Nx.float32
let vec xs = Nx.create f32 [| Array.length xs |] xs
let scale_of ls = Nx.item [] ls.Ls.scale
let steps_of ls = Nx.item [] ls.Ls.good_steps

(* Two float32 leaves, to exercise the structural operations. *)
module Pair = struct
  type t = { a : Nx.float32_t; b : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { a; b } =
    { a = f a; b = f b }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { a = f p.a q.a; b = f p.b q.b }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { a; b } =
    f a;
    f b
end

let raises_invalid_arg f =
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    f

let finite = Nx.scalar Nx.bool true
let nonfinite = Nx.scalar Nx.bool false

(* Constructors *)

let test_constructors () =
  let s = Ls.static 128.0 in
  equal ~msg:"static scale" (float 0.0) 128.0 (scale_of s);
  equal ~msg:"static marker" int32 (-1l) (steps_of s);
  let d = Ls.dynamic () in
  equal ~msg:"dynamic default init is 2^15" (float 0.0) 32768.0 (scale_of d);
  equal ~msg:"dynamic counter starts at 0" int32 0l (steps_of d);
  equal ~msg:"dynamic ~init" (float 0.0) 4.0
    (scale_of (Ls.dynamic ~init:4.0 ()));
  raises_invalid_arg (fun () -> Ls.static 0.0);
  raises_invalid_arg (fun () -> Ls.dynamic ~init:(-1.0) ())

(* Scaling and unscaling *)

let test_scale_unscale_round_trip () =
  let ls = Ls.dynamic ~init:1024.0 () in
  let loss = Nx.scalar f32 1.5 in
  equal ~msg:"scale multiplies" (float 0.0) 1536.0
    (Nx.item [] (Ls.scale ls loss));
  let grads = { Pair.a = vec [| 1.0; -0.5 |]; b = vec [| 0.25 |] } in
  let scaled = { Pair.a = Ls.scale ls grads.Pair.a; b = Ls.scale ls grads.b } in
  let back = Ls.unscale (module Pair) ls scaled in
  (* Powers of two scale exactly. *)
  equal ~msg:"round-trip leaf a"
    (array (float 0.0))
    [| 1.0; -0.5 |] (Nx.to_array back.Pair.a);
  equal ~msg:"round-trip leaf b"
    (array (float 0.0))
    [| 0.25 |] (Nx.to_array back.Pair.b)

let test_scale_half_dtype () =
  let ls = Ls.static 8.0 in
  let loss = Nx.scalar Nx.float16 2.0 in
  let scaled = Ls.scale ls loss in
  is_true ~msg:"scale keeps the input dtype"
    (Nx_core.Dtype.equal (Nx.dtype scaled) Nx.float16);
  equal ~msg:"scaled value" (float 0.0) 16.0 (Nx.item [] scaled)

(* Finiteness *)

let test_grads_finite () =
  let ok = { Pair.a = vec [| 1.0; 2.0 |]; b = vec [| 3.0 |] } in
  is_true ~msg:"finite gradients"
    (Nx.item [] (Ls.grads_finite (module Pair) ok));
  let inf = { ok with Pair.b = vec [| Float.infinity |] } in
  is_false ~msg:"an infinity in any leaf"
    (Nx.item [] (Ls.grads_finite (module Pair) inf));
  let nan = { ok with Pair.a = vec [| 1.0; Float.nan |] } in
  is_false ~msg:"a nan in any leaf"
    (Nx.item [] (Ls.grads_finite (module Pair) nan))

(* Adjustment *)

let test_adjust_backoff () =
  let ls = Ls.dynamic ~init:1024.0 () in
  let ls = Ls.adjust ls ~finite:nonfinite in
  equal ~msg:"overflow halves the scale" (float 0.0) 512.0 (scale_of ls);
  equal ~msg:"overflow resets the counter" int32 0l (steps_of ls);
  let ls = Ls.adjust ~backoff_factor:0.25 ls ~finite:nonfinite in
  equal ~msg:"backoff_factor" (float 0.0) 128.0 (scale_of ls)

let test_adjust_growth () =
  let ls = ref (Ls.dynamic ~init:1024.0 ()) in
  for i = 1 to 2 do
    ls := Ls.adjust ~growth_interval:3 !ls ~finite;
    equal
      ~msg:(Printf.sprintf "scale unchanged after %d finite steps" i)
      (float 0.0) 1024.0 (scale_of !ls);
    equal ~msg:"counter advances" int32 (Int32.of_int i) (steps_of !ls)
  done;
  ls := Ls.adjust ~growth_interval:3 !ls ~finite;
  equal ~msg:"scale doubles at the growth interval" (float 0.0) 2048.0
    (scale_of !ls);
  equal ~msg:"growth resets the counter" int32 0l (steps_of !ls);
  let grown = Ls.adjust ~growth_interval:1 ~growth_factor:4.0 !ls ~finite in
  equal ~msg:"growth_factor" (float 0.0) 8192.0 (scale_of grown)

let test_adjust_backoff_resets_progress () =
  let ls = Ls.dynamic ~init:1024.0 () in
  let ls = Ls.adjust ~growth_interval:3 ls ~finite in
  let ls = Ls.adjust ~growth_interval:3 ls ~finite in
  (* Two finite steps, then an overflow: the counter restarts from zero. *)
  let ls = Ls.adjust ~growth_interval:3 ls ~finite:nonfinite in
  equal ~msg:"overflow halves" (float 0.0) 512.0 (scale_of ls);
  let ls = Ls.adjust ~growth_interval:3 ls ~finite in
  equal ~msg:"no growth right after backoff" (float 0.0) 512.0 (scale_of ls);
  equal ~msg:"counter restarted" int32 1l (steps_of ls)

let test_adjust_static_identity () =
  let ls = Ls.static 64.0 in
  let after_ok = Ls.adjust ~growth_interval:1 ls ~finite in
  equal ~msg:"static scale ignores finite steps" (float 0.0) 64.0
    (scale_of after_ok);
  equal ~msg:"static marker preserved" int32 (-1l) (steps_of after_ok);
  let after_bad = Ls.adjust ls ~finite:nonfinite in
  equal ~msg:"static scale ignores overflows" (float 0.0) 64.0
    (scale_of after_bad);
  equal ~msg:"static marker preserved on overflow" int32 (-1l)
    (steps_of after_bad)

let test_adjust_validation () =
  let ls = Ls.dynamic () in
  raises_invalid_arg (fun () -> Ls.adjust ~growth_interval:0 ls ~finite);
  raises_invalid_arg (fun () -> Ls.adjust ~growth_factor:0.0 ls ~finite);
  raises_invalid_arg (fun () -> Ls.adjust ~backoff_factor:(-0.5) ls ~finite)

let tests =
  [
    group "constructors" [ test "static and dynamic" test_constructors ];
    group "scaling"
      [
        test "scale/unscale round-trip" test_scale_unscale_round_trip;
        test "scale at half dtype" test_scale_half_dtype;
      ];
    group "finiteness" [ test "grads_finite" test_grads_finite ];
    group "adjust"
      [
        test "overflow backs off" test_adjust_backoff;
        test "growth after the interval" test_adjust_growth;
        test "overflow resets growth progress"
          test_adjust_backoff_resets_progress;
        test "static is the identity" test_adjust_static_identity;
        test "validation" test_adjust_validation;
      ];
  ]

let () = run "vega loss scale" tests
