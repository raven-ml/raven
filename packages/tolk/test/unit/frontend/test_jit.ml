(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tensor-level JIT capture and replay. Each JIT warms up on the first call,
   captures on the second, and replays on all later calls; the wrapped
   function body therefore runs exactly twice however many times the JIT is
   called. Results are asserted against eagerly computed references. Runs on
   the process-wide device (DEV selects the backend). *)

open Windtrap
module T = Tolk_frontend.Tensor
module Mv = Tolk_frontend.Movement
module El = Tolk_frontend.Elementwise
module Rd = Tolk_frontend.Reduce
module Op = Tolk_frontend.Op
module Run = Tolk_frontend.Run
module Jit = Tolk_frontend.Jit
module U = Tolk_uop.Uop

let vec data = Run.of_float_array ~shape:[ Array.length data ] data
let close a b = Float.abs (a -. b) < 1e-4

let check_floats ?msg expected t =
  let got = Run.to_float_array t in
  equal ?msg int (Array.length expected) (Array.length got);
  Array.iteri
    (fun i e ->
      if not (close e got.(i)) then
        failf "%selement %d: expected %g, got %g"
          (match msg with Some m -> m ^ ": " | None -> "")
          i e got.(i))
    expected

let is_jit_error = function Jit.Jit_error _ -> true | _ -> false

let elementwise_tests =
  group "elementwise"
    [
      test "chain replays without re-running the function" (fun () ->
          let traces = ref 0 in
          let jit =
            Jit.create (fun inputs ~vars:_ ->
                incr traces;
                let x = inputs.(0) in
                Run.realize (El.mul (El.add x x) (T.f 3.)))
          in
          List.iteri
            (fun i data ->
              let out = Jit.call jit [| vec data |] in
              check_floats
                ~msg:(Printf.sprintf "call %d" i)
                (Array.map (fun x -> x *. 6.) data)
                out;
              equal bool
                ~msg:(Printf.sprintf "captured after call %d" i)
                (i >= 1) (Jit.captured jit))
            [
              [| 1.; 2.; 3.; 4. |];
              [| 5.; 6.; 7.; 8. |];
              [| -1.; 0.5; 2.; -3. |];
              [| 100.; 0.; -0.25; 7. |];
            ];
          equal int ~msg:"function ran only for warmup and capture" 2 !traces);
      test "unrealized inputs are realized by call" (fun () ->
          let jit =
            Jit.create (fun inputs ~vars:_ ->
                Run.realize (El.add inputs.(0) (T.f 1.)))
          in
          let lazy_input () = El.add (vec [| 1.; 2. |]) (vec [| 10.; 20. |]) in
          check_floats [| 12.; 23. |] (Jit.call jit [| lazy_input () |]);
          check_floats [| 12.; 23. |] (Jit.call jit [| lazy_input () |]);
          check_floats [| 12.; 23. |] (Jit.call jit [| lazy_input () |]));
    ]

(* The decode-loop pattern: a persistent cache updated in place at a symbolic
   position, a prefix read back through the same symbolic bound, one capture
   serving every position. *)
let symbolic_tests =
  let plus1 u = U.O.(u + U.const_int 1) in
  let bound_start_pos ~max_context pos =
    U.bind
      ~var:
        (U.variable ~name:"start_pos" ~min_val:0 ~max_val:(max_context - 1) ())
      ~value:(U.const_int pos)
  in
  group "symbolic decode loop"
    [
      test "kv-cache assign at start_pos, prefix reduce, one capture"
        (fun () ->
          let max_context = 8 and width = 2 in
          let cache =
            Run.of_float_array ~shape:[ max_context; width ]
              (Array.make (max_context * width) 0.)
          in
          ignore (Run.realize cache);
          let traces = ref 0 in
          let jit =
            Jit.create (fun inputs ~vars ->
                incr traces;
                let row = inputs.(0) in
                let pos = vars.(0) in
                let view =
                  Mv.symbolic_shrink cache [ Some (pos, plus1 pos); None ]
                in
                ignore (Op.assign view row);
                let prefix =
                  Mv.symbolic_shrink cache
                    [ Some (U.const_int 0, plus1 pos); None ]
                in
                Run.realize (Rd.sum prefix))
          in
          let reference = Array.make (max_context * width) 0. in
          for pos = 1 to 5 do
            let v = float_of_int pos *. 10. in
            let out =
              Jit.call jit
                ~vars:[| bound_start_pos ~max_context pos |]
                [| Run.of_float_array ~shape:[ 1; width ] [| v; v +. 0.5 |] |]
            in
            reference.(pos * width) <- v;
            reference.((pos * width) + 1) <- v +. 0.5;
            let expected =
              Array.fold_left ( +. ) 0.
                (Array.sub reference 0 ((pos + 1) * width))
            in
            check_floats
              ~msg:(Printf.sprintf "prefix sum at pos %d" pos)
              [| expected |] out
          done;
          equal int ~msg:"one capture serves every step" 2 !traces;
          (* The cache buffer survived replay: every written row is present,
             position 0 and the tail are untouched. *)
          check_floats ~msg:"cache contents after the loop" reference cache);
    ]

let error_tests =
  group "errors"
    [
      test "input size mismatch on replay raises Jit_error" (fun () ->
          let jit =
            Jit.create (fun inputs ~vars:_ ->
                Run.realize (El.add inputs.(0) (T.f 1.)))
          in
          ignore (Jit.call jit [| vec [| 1.; 2. |] |]);
          ignore (Jit.call jit [| vec [| 3.; 4. |] |]);
          raises_match is_jit_error (fun () ->
              ignore (Jit.call jit [| vec [| 1.; 2.; 3. |] |])));
      test "duplicate inputs raise Jit_error" (fun () ->
          let jit =
            Jit.create (fun inputs ~vars:_ ->
                Run.realize (El.add inputs.(0) inputs.(1)))
          in
          let x = vec [| 1.; 2. |] in
          raises_match is_jit_error (fun () -> ignore (Jit.call jit [| x; x |])));
      test "malformed vars raise Jit_error" (fun () ->
          let jit =
            Jit.create (fun inputs ~vars:_ ->
                Run.realize (El.add inputs.(0) (T.f 1.)))
          in
          raises_match is_jit_error (fun () ->
              ignore
                (Jit.call jit ~vars:[| U.const_int 3 |] [| vec [| 1.; 2. |] |])));
    ]

let () =
  run "Tolk_frontend_jit" [ elementwise_tests; symbolic_tests; error_tests ]
