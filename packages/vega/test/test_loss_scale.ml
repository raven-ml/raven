(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Vega.Loss_scale: constructors, scaling, the loss-scaled step
   against the unscaled one, skipped steps and the dynamic schedule. *)

open Windtrap
module Ls = Vega.Loss_scale

let f32 = Nx.float32
let vec xs = Nx.create f32 [| Array.length xs |] xs
let scale_of ls = Nx.item [] ls.Ls.scale
let steps_of ls = Nx.item [] ls.Ls.good_steps

(* Two float32 leaves, to exercise the structural operations. *)
module Pair = struct
  type t = { a : Nx.float32_t; b : Nx.float32_t }

  module Walked = struct
    type nonrec _ t = t

    let walk c { a; b } =
      let open Nx.Ptree.Walk in
      let a = field c "a" tensor a in
      let b = field c "b" tensor b in
      { a; b }
  end

  let ptree : t Nx.Ptree.t = Nx.Ptree.instantiate (module Walked)
end

(* [after ls ~finite] is the scale after one step whose gradients are finite or
   not, over a scalar whose update is the identity. *)
let after ?growth_interval ?growth_factor ?backoff_factor ls ~finite =
  let grads = Nx.scalar f32 (if finite then 1.0 else Float.infinity) in
  let p = Nx.Ptree.tensor in
  snd
    (Ls.step ?growth_interval ?growth_factor ?backoff_factor p p ls ~grads
       Fun.id grads)

(* Constructors *)

let test_constructors () =
  let s = Ls.static 128.0 in
  equal ~msg:"static scale" float_exact 128.0 (scale_of s);
  equal ~msg:"static marker" int32 (-1l) (steps_of s);
  let d = Ls.dynamic () in
  equal ~msg:"dynamic default init is 2^15" float_exact 32768.0 (scale_of d);
  equal ~msg:"dynamic counter starts at 0" int32 0l (steps_of d);
  equal ~msg:"dynamic ~init" float_exact 4.0
    (scale_of (Ls.dynamic ~init:4.0 ()));
  raises_match Exn.invalid_arg (fun () -> Ls.static 0.0);
  raises_match Exn.invalid_arg (fun () -> Ls.dynamic ~init:(-1.0) ())

(* Scaling *)

let test_scale () =
  let ls = Ls.dynamic ~init:1024.0 () in
  equal ~msg:"scale multiplies" float_exact 1536.0
    (Nx.item [] (Ls.scale ls (Nx.scalar f32 1.5)))

let test_scale_half_dtype () =
  let ls = Ls.static 8.0 in
  let loss = Nx.scalar Nx.float16 2.0 in
  let scaled = Ls.scale ls loss in
  is_true ~msg:"scale keeps the input dtype"
    (Nx_dtype.equal (Nx.dtype scaled) Nx.float16);
  equal ~msg:"scaled value" float_exact 16.0 (Nx.item [] scaled)

(* The loss-scaled step *)

(* Adam on [1/2 |x - t|^2], whose gradient is [x - t]: [scaled] runs with the
   gradient of the scaled loss through [Ls.step], [plain] with the gradient
   itself. *)
let both = Nx.Ptree.pair Pair.ptree (Vega.adam_ptree Pair.ptree)
let lr = Vega.lr 0.1

let grads (x : Pair.t) (t : Pair.t) =
  { Pair.a = Nx.sub x.a t.a; b = Nx.sub x.b t.b }

let plain t (x, st) =
  Vega.adam_step Pair.ptree ~lr st ~params:x ~grads:(grads x t)

let scaled ?growth_interval t ((x, st), ls) =
  let g = grads x t in
  let grads = { Pair.a = Ls.scale ls g.a; b = Ls.scale ls g.b } in
  Ls.step ?growth_interval Pair.ptree both ls ~grads
    (fun grads -> Vega.adam_step Pair.ptree ~lr st ~params:x ~grads)
    (x, st)

let rec iterate n f x = if n = 0 then x else iterate (n - 1) f (f x)

(* A leaf of the parameters and the same leaf of the target. *)
let leaf_gen =
  let v = Gen.float_range (-10.0) 10.0 in
  Gen.(
    map
      (fun l ->
        ( vec (Array.of_list (List.map fst l)),
          vec (Array.of_list (List.map snd l)) ))
      (list ~size:(int_range 0 4) (pair v v)))

let run_gen =
  Gen.(
    map
      (fun (((xa, ta), (xb, tb)), (init, n)) ->
        ({ Pair.a = xa; b = xb }, { Pair.a = ta; b = tb }, init, n))
      (pair (pair leaf_gen leaf_gen)
         (pair (float_range 1.0 65536.0) (int_range 1 6))))

let close_trees x y =
  ignore
    (Nx.Ptree.map2 both
       (fun path a b ->
         let msg = Format.asprintf "%a" Nx.Ptree.Path.pp path in
         equal ~msg
           (array (float_rel ~rel:1e-5 ~abs:1e-6))
           (Nx.to_array (Nx.cast Nx.float64 a))
           (Nx.to_array (Nx.cast Nx.float64 b));
         a)
       x y)

let test_scaled_follows_plain =
  prop "without overflow, a scaled run follows the unscaled one" run_gen
    (fun (x, t, init, n) ->
      cover "zero-size leaf" (Nx.numel x.Pair.a = 0 || Nx.numel x.Pair.b = 0);
      cover "the scale grows during the run" (n >= 2);
      let start = (x, Vega.adam_init Pair.ptree x) in
      let expected = iterate n (plain t) start in
      let got, ls =
        iterate n (scaled ~growth_interval:2 t) (start, Ls.dynamic ~init ())
      in
      close_trees expected got;
      equal ~msg:"the scale doubles every two finite steps" float_exact
        (scale_of (Ls.dynamic ~init ()) *. (2.0 ** float_of_int (n / 2)))
        (scale_of ls))

(* A step whose gradients hold a NaN or an infinity in one element of one leaf
   leaves the parameters and the whole optimizer state as they were, counter
   included, and halves the scale. *)
let test_overflow_skips () =
  let x = { Pair.a = vec [| 1.0; 2.0 |]; b = vec [| 3.0 |] } in
  let t = { Pair.a = vec [| 0.0; 0.0 |]; b = vec [| 0.0 |] } in
  let start = (x, Vega.adam_init Pair.ptree x) in
  let warm, ls = iterate 2 (scaled t) (start, Ls.dynamic ~init:1024.0 ()) in
  let poisoned bad =
    let st = snd warm in
    let g = grads (fst warm) t in
    let grads = { g with Pair.a = Nx.add g.Pair.a (vec [| 0.0; bad |]) } in
    Ls.step Pair.ptree both ls ~grads
      (fun grads -> Vega.adam_step Pair.ptree ~lr st ~params:(fst warm) ~grads)
      warm
  in
  List.iter
    (fun bad ->
      let got, ls' = poisoned bad in
      close_trees warm got;
      equal ~msg:"the counter is not advanced" int32
        (Nx.item [] (snd warm).Vega.step)
        (Nx.item [] (snd got).Vega.step);
      equal ~msg:"overflow halves the scale" float_exact 512.0 (scale_of ls');
      equal ~msg:"overflow restarts the count" int32 0l (steps_of ls'))
    [ Float.nan; Float.infinity; Float.neg_infinity ]

(* Structure *)

let test_visits () =
  equal ~msg:"leaf paths" (list string)
    [ "scale: a leaf"; "good_steps: a leaf" ]
    (List.map
       (Format.asprintf "%a" Nx.Ptree.pp_visit)
       (Nx.Ptree.visits Ls.ptree (Ls.dynamic ())))

(* The dynamic schedule *)

let test_backoff () =
  let ls = Ls.dynamic ~init:1024.0 () in
  let ls = after ls ~finite:false in
  equal ~msg:"overflow halves the scale" float_exact 512.0 (scale_of ls);
  equal ~msg:"overflow resets the counter" int32 0l (steps_of ls);
  let ls = after ~backoff_factor:0.25 ls ~finite:false in
  equal ~msg:"backoff_factor" float_exact 128.0 (scale_of ls)

let test_growth () =
  let ls = ref (Ls.dynamic ~init:1024.0 ()) in
  for i = 1 to 2 do
    ls := after ~growth_interval:3 !ls ~finite:true;
    equal
      ~msg:(Printf.sprintf "scale unchanged after %d finite steps" i)
      float_exact 1024.0 (scale_of !ls);
    equal ~msg:"counter advances" int32 (Int32.of_int i) (steps_of !ls)
  done;
  ls := after ~growth_interval:3 !ls ~finite:true;
  equal ~msg:"scale doubles at the growth interval" float_exact 2048.0
    (scale_of !ls);
  equal ~msg:"growth resets the counter" int32 0l (steps_of !ls);
  let grown = after ~growth_interval:1 ~growth_factor:4.0 !ls ~finite:true in
  equal ~msg:"growth_factor" float_exact 8192.0 (scale_of grown)

let test_backoff_resets_progress () =
  let ls = Ls.dynamic ~init:1024.0 () in
  let ls = after ~growth_interval:3 ls ~finite:true in
  let ls = after ~growth_interval:3 ls ~finite:true in
  (* Two finite steps, then an overflow: the counter restarts from zero. *)
  let ls = after ~growth_interval:3 ls ~finite:false in
  equal ~msg:"overflow halves" float_exact 512.0 (scale_of ls);
  let ls = after ~growth_interval:3 ls ~finite:true in
  equal ~msg:"no growth right after backoff" float_exact 512.0 (scale_of ls);
  equal ~msg:"counter restarted" int32 1l (steps_of ls)

let test_static_unchanged () =
  let ls = Ls.static 64.0 in
  let after_ok = after ~growth_interval:1 ls ~finite:true in
  equal ~msg:"static scale ignores finite steps" float_exact 64.0
    (scale_of after_ok);
  equal ~msg:"static marker preserved" int32 (-1l) (steps_of after_ok);
  let after_bad = after ls ~finite:false in
  equal ~msg:"static scale ignores overflows" float_exact 64.0
    (scale_of after_bad);
  equal ~msg:"static marker preserved on overflow" int32 (-1l)
    (steps_of after_bad)

let test_validation () =
  let ls = Ls.dynamic () in
  raises_match Exn.invalid_arg (fun () ->
      after ~growth_interval:0 ls ~finite:true);
  raises_match Exn.invalid_arg (fun () ->
      after ~growth_factor:0.0 ls ~finite:true);
  raises_match Exn.invalid_arg (fun () ->
      after ~backoff_factor:(-0.5) ls ~finite:true)

let tests =
  [
    group "constructors" [ test "static and dynamic" test_constructors ];
    group "scaling"
      [
        test "scale" test_scale;
        test "scale at half dtype" test_scale_half_dtype;
      ];
    group "step"
      [
        test_scaled_follows_plain;
        test "overflow skips the step" test_overflow_skips;
      ];
    group "structure" [ test "visits" test_visits ];
    group "schedule"
      [
        test "overflow backs off" test_backoff;
        test "growth after the interval" test_growth;
        test "overflow resets growth progress" test_backoff_resets_progress;
        test "static is unchanged" test_static_unchanged;
        test "validation" test_validation;
      ];
  ]

let () = exit (run "vega loss scale" tests)
