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

let storage_view ~src ~offset ~size ~dtype =
  let module U = Tolk_uop.Uop in
  let module D = Tolk_uop.Dtype in
  let offset = U.alu_binary ~op:Tolk_uop.Ops.Mul ~lhs:offset
      ~rhs:(U.const_int (D.itemsize (U.dtype src))) in
  let bytes = U.bitcast ~src ~dtype:D.int8 in
  U.bitcast ~dtype ~src:(U.shrink ~src:bytes ~offset
      ~size:(U.const_int (size * D.itemsize dtype)))

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
      test "signed int64 endpoints survive binding, capture and replay" (fun () ->
          let module D = Tolk_uop.Dtype in
          let scalar = U.param ~slot:(-1) ~name:"wide" ~dtype:D.int64
              ~addrspace:D.Alu ~vmin_vmax:(D.min D.int64, D.max D.int64) () in
          let scalar = U.replace scalar ~op:Tolk_uop.Ops.Buffer () in
          let traces = ref 0 in
          let jit = Jit.create ~outputs:(fun tensor -> [tensor]) (fun inputs ~vars ->
              incr traces;
              Run.realize (El.add inputs.(0) (T.of_uop vars.(0)))) in
          List.iter (fun value ->
              let bound = U.bind ~var:scalar
                  ~value:(U.const (Tolk_uop.Const.int64 D.int64 value)) in
              let input = Run.of_bytes ~dtype:D.int64 ~shape:[1] (Bytes.make 8 '\000') in
              let output = Jit.call jit ~vars:[|bound|] [|input|] in
              equal int64 value (Bytes.get_int64_le (Run.data output) 0))
            [Int64.min_int; Int64.max_int; Int64.min_int; Int64.max_int];
          equal int 2 !traces);
      test "storage slices remain views through capture and replay" (fun () ->
          List.iter (fun offset ->
              let traces = ref 0 in
              let jit = Jit.create ~outputs:(fun tensor -> [tensor]) (fun inputs ~vars:_ ->
                  incr traces;
                  let view = storage_view ~src:(U.base (T.uop inputs.(0)))
                      ~offset:(U.const_int offset) ~size:4 ~dtype:Tolk_uop.Dtype.float32 in
                  Run.realize (El.add (T.of_uop view) (T.f 1.))) in
              for call = 0 to 3 do
                let data = Array.init 8 (fun i -> Float.of_int (call * 10 + i)) in
                let out = Jit.call jit [| vec data |] in
                check_floats (Array.init 4 (fun i -> data.(i + offset) +. 1.)) out
              done;
              equal int 2 !traces) [ 1; 4 ]);
      test "chain replays without re-running the function" (fun () ->
          let traces = ref 0 in
          let jit =
            Jit.create ~outputs:(fun tensor -> [tensor]) (fun inputs ~vars:_ ->
                incr traces;
                let x = inputs.(0) in
                Run.realize (El.mul (El.add x x) (T.f 3.)))
          in
          List.iteri
            (fun i data ->
              Gc.full_major ();
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
      test "empty host inputs participate in capture and replay" (fun () ->
          let traces = ref 0 in
          let jit = Jit.create ~outputs:(fun tensor -> [tensor]) (fun inputs ~vars:_ ->
              incr traces;
              Run.realize (El.add (Rd.sum inputs.(0)) (Rd.sum inputs.(1)))) in
          for call = 0 to 3 do
            let empty = Run.of_float_array ~shape:[ 0; 3 ] [||] in
            let input = vec [| Float.of_int call; 2. |] in
            check_floats [| Float.of_int call +. 2. |]
              (Jit.call jit [| empty; input |]);
            equal bool (call >= 1) (Jit.captured jit)
          done;
          equal int 2 !traces);
      test "unrealized inputs are realized by call" (fun () ->
          let jit =
            Jit.create ~outputs:(fun tensor -> [tensor]) (fun inputs ~vars:_ ->
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
      test "returned symbolic views follow replay bindings" (fun () ->
          let traces = ref 0 in
          let fixed = U.bind
              ~var:(U.variable ~name:"inner_size" ~min_val:1 ~max_val:8 ())
              ~value:(U.const_int 3) in
          let jit = Jit.create ~outputs:snd (fun inputs ~vars ->
              incr traces;
              let computed = Run.realize (El.add inputs.(0) (T.f 1.)) in
              let prefix bound =
                Mv.symbolic_shrink computed [Some (U.const_int 0, bound)] in
              ("views", [prefix vars.(0); prefix fixed])) in
          let concrete tensor =
            let node = T.uop tensor in
            let bindings = List.filter_map (fun u ->
                match U.as_bind u with
                | Some {value; _} -> Some (u, value)
                | None -> None) (U.toposort node) in
            T.of_uop (U.substitute ~walk:true bindings node) in
          let step n =
            let bound = U.bind
                ~var:(U.variable ~name:"view_size" ~min_val:1 ~max_val:8 ())
                ~value:(U.const_int n) in
            let label, views = Jit.call jit ~vars:[|bound|]
                [|vec (Array.init 8 (fun i -> Float.of_int (10 * n + i)))|] in
            equal string "views" label;
            List.iter2 (fun size view ->
                let view = concrete view in
                equal (list int) [size] (T.shape view);
                check_floats (Array.init size (fun i -> Float.of_int (10 * n + i + 1))) view)
              [n; 3] views in
          List.iter step [1; 2; 4; 3; 8];
          equal int 2 !traces;
          Jit.reset jit;
          List.iter step [4; 1; 6];
          equal int 4 !traces);
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
            Jit.create ~outputs:(fun tensor -> [tensor]) (fun inputs ~vars ->
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
            Jit.create ~outputs:(fun tensor -> [tensor]) (fun inputs ~vars:_ ->
                Run.realize (El.add inputs.(0) (T.f 1.)))
          in
          ignore (Jit.call jit [| vec [| 1.; 2. |] |]);
          ignore (Jit.call jit [| vec [| 3.; 4. |] |]);
          raises_match is_jit_error (fun () ->
              ignore (Jit.call jit [| vec [| 1.; 2.; 3. |] |])));
      test "duplicate inputs raise Jit_error" (fun () ->
          let jit =
            Jit.create ~outputs:(fun tensor -> [tensor]) (fun inputs ~vars:_ ->
                Run.realize (El.add inputs.(0) inputs.(1)))
          in
          let x = vec [| 1.; 2. |] in
          raises_match is_jit_error (fun () -> ignore (Jit.call jit [| x; x |])));
      test "malformed vars raise Jit_error" (fun () ->
          let jit =
            Jit.create ~outputs:(fun tensor -> [tensor]) (fun inputs ~vars:_ ->
                Run.realize (El.add inputs.(0) (T.f 1.)))
          in
          raises_match is_jit_error (fun () ->
              ignore
                (Jit.call jit ~vars:[| U.const_int 3 |] [| vec [| 1.; 2. |] |])));
    ]

let () =
  run "Tolk_frontend_jit" [ elementwise_tests; symbolic_tests; error_tests ]
