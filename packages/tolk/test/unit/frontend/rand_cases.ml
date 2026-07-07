(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Shared random-number test cases, run by [test_rand] on the process-wide
   default device and by [test_rand_cuda] pinned to CUDA. Golden values were
   generated from the reference implementation (tinygrad @ a330bfffc) with:

     PYTHONPATH=. DEV=CPU .venv/bin/python -c "
       from tinygrad import Tensor, Context
       Tensor.manual_seed(42); print(Tensor.rand(4).numpy().tolist())"

   and the analogous snippets for each case below (each case reseeds with
   manual_seed(42) first). The values are identical on DEV=CPU and DEV=CUDA;
   only randn differs in the last ulp across devices (transcendentals), so it
   is compared with a small tolerance. *)

open Windtrap
module T = Tolk_frontend.Tensor
module El = Tolk_frontend.Elementwise
module Cr = Tolk_frontend.Creation
module Op = Tolk_frontend.Op
module Rand = Tolk_frontend.Rand
module Run = Tolk_frontend.Run
module Jit = Tolk_frontend.Jit
module D = Tolk_uop.Dtype

let check_floats_exact expected t =
  let got = Run.to_float_array t in
  equal int (Array.length expected) (Array.length got);
  Array.iteri
    (fun i e ->
      if not (Float.equal e got.(i)) then
        failf "element %d: expected %.17g, got %.17g" i e got.(i))
    expected

let check_floats_close ?(tol = 1e-5) expected t =
  let got = Run.to_float_array t in
  equal int (Array.length expected) (Array.length got);
  Array.iteri
    (fun i e ->
      if Float.abs (e -. got.(i)) > tol then
        failf "element %d: expected %.17g, got %.17g" i e got.(i))
    expected

let check_ints expected t = equal (array int) expected (Run.to_int_array t)

let hex bytes =
  String.concat ""
    (List.init (Bytes.length bytes) (fun i ->
         Printf.sprintf "%02x" (Char.code (Bytes.get bytes i))))

(* sha256 goldens: hashlib.sha256((0).to_bytes(4, "big")) *)

let sha256_tests =
  group "sha256"
    [
      test "digest of four zero bytes" (fun () ->
          equal string
            "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
            (hex (Tolk.Helpers.sha256 (Bytes.make 4 '\000'))));
      test "digest of empty message" (fun () ->
          equal string
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            (hex (Tolk.Helpers.sha256 Bytes.empty)));
      test "derived key word for device index 0" (fun () ->
          let digest = Tolk.Helpers.sha256 (Bytes.make 4 '\000') in
          equal int 347607321
            (Int32.to_int (Bytes.get_int32_be digest 28) land 0xFFFFFFFF));
    ]

(* Exact-value goldens from the reference implementation, seed 42. *)

let golden_tests =
  group "goldens"
    [
      test "rand 4" (fun () ->
          Rand.manual_seed 42;
          check_floats_exact
            [|
              0.5334206819534302; 0.6701551675796509; 0.7630789279937744;
              0.4320552349090576;
            |]
            (Rand.rand [ 4 ]));
      test "second rand 4 differs and matches reference" (fun () ->
          Rand.manual_seed 42;
          ignore (Run.to_float_array (Rand.rand [ 4 ]));
          check_floats_exact
            [|
              0.18275272846221924; 0.051476120948791504; 0.31145310401916504;
              0.5078401565551758;
            |]
            (Rand.rand [ 4 ]));
      test "rand 2x3" (fun () ->
          Rand.manual_seed 42;
          check_floats_exact
            [|
              0.38104248046875; 0.009775400161743164; 0.1128007173538208;
              0.1176995038986206; 0.505361795425415; 0.3721102476119995;
            |]
            (Rand.rand [ 2; 3 ]));
      test "randn 2x3" (fun () ->
          Rand.manual_seed 42;
          check_floats_close
            [|
              1.9576417207717896; -0.1859397292137146; 1.6404170989990234;
              -0.7646574378013611; -0.8694872856140137; -0.43787896633148193;
            |]
            (Rand.randn [ 2; 3 ]));
      test "randint 2x3 low 5 high 10" (fun () ->
          Rand.manual_seed 42;
          let t = Rand.randint ~low:5 ~high:10 [ 2; 3 ] in
          is_true (D.equal (T.dtype t) D.int32);
          check_ints [| 6; 5; 5; 5; 7; 6 |] t);
      test "uniform 2x3 low 2 high 10" (fun () ->
          Rand.manual_seed 42;
          check_floats_exact
            [|
              5.04833984375; 2.0782032012939453; 2.9024057388305664;
              2.941596031188965; 6.04289436340332; 4.976881980895996;
            |]
            (Rand.uniform ~low:2. ~high:10. [ 2; 3 ]));
      test "multinomial 20 samples with replacement" (fun () ->
          Rand.manual_seed 42;
          let w = Run.of_float_array ~shape:[ 4 ] [| 1.; 2.; 3.; 4. |] in
          let t = Rand.multinomial ~num_samples:20 ~replacement:true w in
          is_true (D.equal (T.dtype t) D.int32);
          check_ints
            [| 3; 2; 3; 1; 2; 3; 2; 3; 0; 1; 2; 3; 1; 2; 2; 1; 2; 2; 3; 3 |]
            t);
      test "randperm 6" (fun () ->
          Rand.manual_seed 42;
          check_ints [| 1; 2; 3; 5; 0; 4 |] (Rand.randperm 6));
      test "dropout mask on ones under training" (fun () ->
          Rand.manual_seed 42;
          Tolk.Helpers.Context_var.with_context
            [ Tolk.Helpers.Context_var.B (Tolk.Helpers.training, 1) ]
            (fun () ->
              check_floats_exact
                [| 0.; 2.; 0.; 0.; 2.; 0.; 2.; 0. |]
                (Rand.dropout (Cr.ones [ 8 ]))));
    ]

(* Counter-state parity with the reference generator. *)

let counter_tests =
  group "counter"
    [
      test "counter reads 4,0 after rand 4" (fun () ->
          Rand.manual_seed 42;
          ignore (Run.to_float_array (Rand.rand [ 4 ]));
          match Rand.device_rng_counter (Run.device_name ()) with
          | Some counter -> check_ints [| 4; 0 |] counter
          | None -> failf "no rng counter for the default device");
      test "counter accumulates across draws" (fun () ->
          Rand.manual_seed 42;
          ignore (Run.to_float_array (Rand.rand [ 4 ]));
          ignore (Run.to_float_array (Rand.rand [ 2; 3 ]));
          match Rand.device_rng_counter (Run.device_name ()) with
          | Some counter -> check_ints [| 10; 0 |] counter
          | None -> failf "no rng counter for the default device");
      test "manual_seed resets the state" (fun () ->
          Rand.manual_seed 42;
          ignore (Run.to_float_array (Rand.rand [ 4 ]));
          Rand.manual_seed 42;
          is_true (Rand.device_rng_counter (Run.device_name ()) = None);
          check_floats_exact
            [|
              0.5334206819534302; 0.6701551675796509; 0.7630789279937744;
              0.4320552349090576;
            |]
            (Rand.rand [ 4 ]));
      test "consecutive draws are distinct" (fun () ->
          Rand.manual_seed 7;
          let a = Run.to_float_array (Rand.rand [ 4 ]) in
          let b = Run.to_float_array (Rand.rand [ 4 ]) in
          is_true (a <> b));
    ]

(* Random draws inside a JIT capture: replays must advance the stream, and
   the stream must stay reproducible under reseeding. *)

let jit_tests =
  let f inputs ~vars:_ =
    Run.realize
      (El.mul (El.add inputs.(0) inputs.(1)) (Rand.randn [ 10; 10 ]))
  in
  let five_draws a b =
    let jf = Jit.create f in
    List.init 5 (fun _ -> (Run.to_float_array (Jit.call jf [| a; b |])).(0))
  in
  group "jit"
    [
      test "rand regenerates and reproduces under capture" (fun () ->
          let a = Run.realize (Rand.randn [ 10; 10 ]) in
          let b = Run.realize (Rand.randn [ 10; 10 ]) in
          Rand.manual_seed 1234;
          let r1 = five_draws a b in
          equal int 5 (List.length (List.sort_uniq Float.compare r1));
          Rand.manual_seed 1234;
          let r2 = five_draws a b in
          equal (list (float 0.)) r1 r2;
          Rand.manual_seed 3421;
          let r3 = five_draws a b in
          equal int 5 (List.length (List.sort_uniq Float.compare r3));
          is_true (r3 <> r2));
    ]

(* Statistical smoke: seeded, so the bounds are deterministic. *)

let stat_tests =
  group "statistics"
    [
      test "rand mean and range" (fun () ->
          Rand.manual_seed 0;
          let xs = Run.to_float_array (Rand.rand [ 10000 ]) in
          Array.iter (fun x -> is_true (x >= 0. && x < 1.)) xs;
          let mean = Array.fold_left ( +. ) 0. xs /. 10000. in
          is_true (Float.abs (mean -. 0.5) < 0.02));
      test "randn mean and variance" (fun () ->
          Rand.manual_seed 0;
          let xs = Run.to_float_array (Rand.randn [ 10000 ]) in
          let mean = Array.fold_left ( +. ) 0. xs /. 10000. in
          let var =
            Array.fold_left (fun acc x -> acc +. ((x -. mean) *. (x -. mean)))
              0. xs
            /. 10000.
          in
          is_true (Float.abs mean < 0.05);
          is_true (Float.abs (var -. 1.) < 0.05));
    ]

(* Dropout behaviour beyond the golden mask. *)

let dropout_tests =
  let with_training f =
    Tolk.Helpers.Context_var.with_context
      [ Tolk.Helpers.Context_var.B (Tolk.Helpers.training, 1) ]
      f
  in
  group "dropout"
    [
      test "identity outside training" (fun () ->
          let t = Cr.ones [ 4 ] in
          is_true (Rand.dropout t == t));
      test "identity at rate 0" (fun () ->
          with_training (fun () ->
              let t = Cr.ones [ 4 ] in
              is_true (Rand.dropout ~p:0. t == t)));
      test "all zero at rate 1" (fun () ->
          with_training (fun () ->
              (* The result is a pure zero constant, which has no storage of
                 its own to read back; write it into a buffer instead. *)
              let dst = Run.realize (Cr.ones [ 4 ]) in
              ignore
                (Run.realize
                   (Op.assign dst (Rand.dropout ~p:1. (Cr.ones [ 4 ]))));
              check_floats_exact [| 0.; 0.; 0.; 0. |] dst));
      test "mask is reproducible under a fixed seed" (fun () ->
          with_training (fun () ->
              Rand.manual_seed 123;
              let a = Run.to_float_array (Rand.dropout (Cr.ones [ 32 ])) in
              Rand.manual_seed 123;
              let b = Run.to_float_array (Rand.dropout (Cr.ones [ 32 ])) in
              equal (array (float 0.)) a b));
      test "survivors scale by 1/(1-p)" (fun () ->
          with_training (fun () ->
              Rand.manual_seed 5;
              let xs =
                Run.to_float_array (Rand.dropout ~p:0.25 (Cr.ones [ 64 ]))
              in
              Array.iter
                (fun x ->
                  if not (Float.equal x 0. || Float.abs (x -. (4. /. 3.)) < 1e-6)
                  then failf "unexpected dropout value %.17g" x)
                xs));
      test "rate outside range raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Rand.dropout ~p:1.5 (Cr.ones [ 4 ])));
    ]

(* Groups whose values must be identical on every device. *)
let exact_groups = [ sha256_tests; golden_tests; counter_tests ]

let all_groups =
  exact_groups @ [ jit_tests; stat_tests; dropout_tests ]
