(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Vega.Loss_scale. *)

open Windtrap
module Ls = Vega.Loss_scale

let f32 = Nx.float32
let vec xs = Nx.create f32 [| Array.length xs |] xs
let scale_of ls = Nx.item [] ls.Ls.scale
let good_steps_of ls = Nx.item [] ls.Ls.good_steps
let tensor = Nx.Ptree.tensor
let pp_hex ppf x = Format.fprintf ppf "%h" x

(* [to_f32 x] is [x] rounded to float32. *)
let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let not_positive = [ 0.0; -0.0; -1.0; Float.neg_infinity; Float.nan ]

(* Two float32 leaves, to exercise the structural operations. *)
module Pair = struct
  type t = { a : Nx.host Nx.float32_t; b : Nx.host Nx.float32_t }

  let walk c { a; b } =
    let open Nx.Ptree.Walk in
    let a = field c "a" tensor a in
    let b = field c "b" tensor b in
    { a; b }

  let ptree : (t, Nx.host) Nx.Ptree.t = Nx.Ptree.v walk
end

(* The schedule's model *)

(* A loss scale as the .mli states it: [good] is the count of consecutive finite
   steps, [-1] for a static scale. *)
type model = { scale : float; good : int }

let pp_model ppf r =
  Format.fprintf ppf "{ scale = %h; good = %d }" r.scale r.good

let model_dynamic init =
  cover "the default init" (Option.is_none init);
  { scale = Option.value init ~default:32768.0; good = 0 }

let model_static s = { scale = s; good = -1 }

(* A step divides each gradient by the scale at float32 and is finite iff every
   quotient is. A dynamic scale counts a finite step, and is multiplied by
   [growth_factor] when the count reaches [growth_interval]; a non-finite step
   multiplies it by [backoff_factor]. Either change restarts the count. *)
let model_step interval growth backoff grads r =
  let growth_interval = Option.value interval ~default:2000 in
  let growth_factor = Option.value growth ~default:2.0 in
  let backoff_factor = Option.value backoff ~default:0.5 in
  let quotient_finite g = Float.is_finite (to_f32 (g /. r.scale)) in
  let finite = List.for_all quotient_finite grads in
  let dynamic = r.good >= 0 in
  let grows = dynamic && finite && r.good + 1 >= growth_interval in
  let backs_off = dynamic && not finite in
  cover "a step on a static scale" (not dynamic);
  cover "finite gradients turn non-finite over the scale"
    (dynamic && List.for_all Float.is_finite grads && not finite);
  cover "a step on a zero scale" (dynamic && r.scale = 0.0);
  cover "a step on an infinite scale" (dynamic && r.scale = Float.infinity);
  cover "the scale grows" grows;
  cover "the scale grows by the default factor" (grows && Option.is_none growth);
  cover "the scale backs off" backs_off;
  cover "the scale backs off by the default factor"
    (backs_off && Option.is_none backoff);
  if not dynamic then r
  else if backs_off then
    { scale = to_f32 (r.scale *. backoff_factor); good = 0 }
  else if grows then { scale = to_f32 (r.scale *. growth_factor); good = 0 }
  else { r with good = r.good + 1 }

let system_step growth_interval growth_factor backoff_factor grads ls =
  let grads = vec (Array.of_list grads) in
  snd
    (Ls.step ?growth_interval ?growth_factor ?backoff_factor tensor tensor ls
       ~grads Fun.id grads)

(* Scales from the smallest float32 subnormal, which a backoff takes to zero, to
   [2^127], which growth takes to infinity. Factors are powers of two, so the
   model's products rounded to float32 are the tensors'. *)
let scale_gen = Gen.of_list ~pp:pp_hex [ 0x1p-149; 0.5; 0x1p127 ]
let interval_gen = Gen.option (Gen.int_range 1 2)
let growth_gen = Gen.option (Gen.of_list ~pp:pp_hex [ 0.5; 2.0; 4.0 ])
let backoff_gen = Gen.option (Gen.of_list ~pp:pp_hex [ 0.25; 0.5; 2.0 ])

(* [2^127] overflows float32 over a scale below one. *)
let grads_gen =
  Gen.list ~size:(Gen.int_range 0 3)
    (Gen.of_list ~pp:pp_hex
       [ 1.0; -0.0; 0x1p127; -0x1p127; Float.infinity; Float.nan ])

let test_schedule =
  let ls = abstract ~pp:pp_model "ls" in
  let step =
    command "step"
      (interval_gen @-> growth_gen @-> backoff_gen @-> grads_gen @-> ls
     ^-> makes ls)
      model_step system_step
  in
  (* [step] is listed twice to run chains of steps that reach a zero and an
     infinite scale. *)
  stateful ~count:300 ~steps:30 "gives the scale for the next step"
    [
      command "dynamic"
        (Gen.option scale_gen @-> makes ls)
        model_dynamic
        (fun init -> Ls.dynamic ?init Nx.Host.on);
      command "static"
        (scale_gen @-> makes ls)
        model_static (Ls.static Nx.Host.on);
      step;
      step;
      command "scale" (ls ^-> returns float_exact) (fun r -> r.scale) scale_of;
      command "good_steps"
        (ls ^-> returns int32)
        (fun r -> Int32.of_int r.good)
        good_steps_of;
    ]

(* A scaled run *)

(* Adam on [1/2 |x - t|^2], whose gradient is [x - t]: [scaled] runs with the
   gradient of the scaled loss through [Ls.step], [plain] with the gradient
   itself. *)
let both = Nx.Ptree.pair Pair.ptree (Vega.adam_ptree Pair.ptree)
let lr = Vega.rate 0.1

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

(* An initial scale [2^k]. *)
let run_gen =
  Gen.(
    map
      (fun (((xa, ta), (xb, tb)), (k, n)) ->
        ({ Pair.a = xa; b = xb }, { Pair.a = ta; b = tb }, k, n))
      (pair (pair leaf_gen leaf_gen) (pair (int_range 0 16) (int_range 1 6))))

let leaves x =
  List.map
    (fun (Nx.P t) -> Nx.to_array (Nx.cast Nx.float64 t))
    (fst (Nx.Ptree.flatten both x))

(* A float32 gradient times a power of two, divided by it again, is the gradient
   exactly unless the product overflows. Gradients here stay near 20 and scales
   at most [2^19], [2^16] grown three times. *)
let test_scaled_follows_plain =
  prop
    "with a power-of-two scale and no overflow, a scaled run is the unscaled \
     one"
    run_gen (fun (x, t, k, n) ->
      cover "zero-size leaf" (Nx.numel x.Pair.a = 0 || Nx.numel x.Pair.b = 0);
      cover "the scale grows during the run" (n >= 2);
      let start = (x, Vega.adam_init Pair.ptree x) in
      let expected = iterate n (plain t) start in
      let got, _ =
        iterate n
          (scaled ~growth_interval:2 t)
          (start, Ls.dynamic ~init:(Float.ldexp 1.0 k) Nx.Host.on)
      in
      equal (list (array float_exact)) (leaves expected) (leaves got))

(* Two Adam steps from [x], then one whose gradients hold [bad] in one element
   of one leaf: the skipped step is the state after two steps, the counter at
   2. *)
let test_skipped bad =
  let x = { Pair.a = vec [| 1.0; 2.0 |]; b = vec [| 3.0 |] } in
  let t = { Pair.a = vec [| 0.0; 0.0 |]; b = vec [| 0.0 |] } in
  let start = (x, Vega.adam_init Pair.ptree x) in
  let warm, ls =
    iterate 2 (scaled t) (start, Ls.dynamic ~init:1024.0 Nx.Host.on)
  in
  let params, st = warm in
  let g = grads params t in
  let grads = { g with Pair.a = Nx.add g.Pair.a (vec [| 0.0; bad |]) } in
  let got, _ =
    Ls.step Pair.ptree both ls ~grads
      (fun grads -> Vega.adam_step Pair.ptree ~lr st ~params ~grads)
      warm
  in
  equal (list (array float_exact)) (leaves warm) (leaves got)

let test_default_interval () =
  let at good_steps =
    {
      Ls.scale = Nx.scalar f32 1024.0;
      good_steps = Nx.scalar Nx.int32 good_steps;
    }
  in
  let one = Nx.scalar f32 1.0 in
  let finite_step ls = snd (Ls.step tensor tensor ls ~grads:one Fun.id one) in
  equal float_exact 1024.0 (scale_of (finite_step (at 1998l)));
  equal float_exact 2048.0 (scale_of (finite_step (at 1999l)))

let test_refuses_options (growth_interval, growth_factor, backoff_factor) =
  let ls = Ls.dynamic Nx.Host.on in
  let one = Nx.scalar f32 1.0 in
  raises_match Exn.invalid_arg (fun () ->
      Ls.step ?growth_interval ?growth_factor ?backoff_factor tensor tensor ls
        ~grads:one Fun.id one)

(* [update] returns one tensor of the two in [x]. *)
let test_structure_mismatch () =
  let one = Nx.scalar f32 1.0 in
  let x = [ one; one ] in
  raises
    (Invalid_argument
       "Nx.Ptree.map2: the root: length 1 in the first value, length 2 in the \
        second") (fun () ->
      Ls.step tensor (Nx.Ptree.list tensor) (Ls.dynamic Nx.Host.on) ~grads:one
        (fun _ -> [ one ])
        x)

let pp_option pp ppf = function
  | None -> Format.pp_print_string ppf "None"
  | Some v -> Format.fprintf ppf "Some %a" pp v

let id_options (gi, gf, bf) =
  Format.asprintf "growth_interval %a, growth_factor %a, backoff_factor %a"
    (pp_option Format.pp_print_int)
    gi (pp_option pp_hex) gf (pp_option pp_hex) bf

let tests =
  [
    group "static"
      [
        cases "raises Invalid_argument if s is not positive"
          ~id:(Format.asprintf "%h") not_positive (fun s ->
            raises_match Exn.invalid_arg (fun () -> Ls.static Nx.Host.on s));
      ];
    group "dynamic"
      [
        cases "raises Invalid_argument if init is not positive"
          ~id:(Format.asprintf "%h") not_positive (fun init ->
            raises_match Exn.invalid_arg (fun () -> Ls.dynamic ~init Nx.Host.on));
      ];
    group "scale"
      [
        test "is x times the current scale, at x's dtype" (fun () ->
            let ls = Ls.dynamic Nx.Host.on in
            equal float_exact 49152.0
              (Nx.item [] (Ls.scale ls (Nx.scalar f32 1.5)));
            (* 2 * 2^15 exceeds float16's largest finite value, 65504. *)
            equal float_exact Float.infinity
              (Nx.item [] (Ls.scale ls (Nx.scalar Nx.float16 2.0))));
      ];
    group "step"
      [
        test_scaled_follows_plain;
        cases
          "a skipped step is x, every tensor, the optimizer's counter included"
          ~id:(Format.asprintf "%h")
          [ Float.nan; Float.infinity; Float.neg_infinity ]
          test_skipped;
        test_schedule;
        test "a dynamic scale grows at 2000 consecutive finite steps by default"
          test_default_interval;
        cases
          "raises Invalid_argument if growth_interval, growth_factor or \
           backoff_factor is not positive"
          ~id:id_options
          [
            (Some 0, None, None);
            (Some (-1), None, None);
            (None, Some 0.0, None);
            (None, Some (-2.0), None);
            (None, Some Float.nan, None);
            (None, None, Some 0.0);
            (None, None, Some (-0.5));
            (None, None, Some Float.nan);
          ]
          test_refuses_options;
        test
          "raises as Nx.Ptree.map2 does if update grads' and x differ in \
           structure"
          test_structure_mismatch;
      ];
    group "walk"
      [
        test "visits scale then good_steps" (fun () ->
            equal (list string)
              [ "scale: a leaf"; "good_steps: a leaf" ]
              (List.map
                 (Format.asprintf "%a" Nx.Ptree.pp_visit)
                 (Nx.Ptree.visits (Nx.Ptree.v Ls.walk) (Ls.dynamic Nx.Host.on))));
      ];
  ]

let () = exit (run "vega loss scale" tests)
