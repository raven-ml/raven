(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module Run = Sub.Run
module Support = Rig_support
module P = Support.Polled

let fill ?(params = 16) ?(refs = [| Sub.write 0 ~at:0 |]) fn =
  Sub.fill ~refs fn ~params

let submit run buffers = Rig.submit run ~buffers
let word b at = Support.load (Support.Reader.host b + at)

(* Takes [run] for the stores that follow, as no other call holds it. *)
let take run = if not (Run.take run) then fail "the run is taken"

let store run block ~offset value =
  Run.int64 run block 0 (Int64.of_int offset);
  Run.int64 run block 8 (Int64.of_int value)

(* The same run can serve successive hand-overs while the first still waits in a
   queue. Both its parameter bytes and its resolved addresses belong to each
   handed-over call. *)
let test_reuse () =
  let d, p = P.open_ "args:reuse" in
  let s = Sub.make d [| fill Support.poke |] in
  let run = Run.make s and block = Sub.block s 0 in
  let a = B.create d 24 and b = B.create d 24 in
  store run block ~offset:8 31;
  ignore (submit run [| a |]);
  take run;
  store run block ~offset:16 47;
  ignore (submit run [| b |]);
  take run;
  store run block ~offset:0 99;
  Run.release run;
  equal int 2 (P.queued p);
  equal int 2 (P.run p);
  equal int 31 (word a 8);
  equal int 47 (word b 16);
  (* Repeating a submit adds the address to the saved offset exactly once. *)
  take run;
  store run block ~offset:8 73;
  ignore (submit run [| a |]);
  ignore (P.run p);
  ignore (submit run [| b |]);
  ignore (P.run p);
  equal int 73 (word a 8);
  equal int 73 (word b 8)

let test_synchronous () =
  let d = Result.get_ok (Rig.memory_device "args:sync") in
  let s = Sub.make d [| fill Support.poke |] in
  let run = Run.make s and block = Sub.block s 0 in
  let out = B.create d 16 in
  store run block ~offset:8 41;
  ignore (submit run [| out |]);
  equal int 41 (word out 8);
  ignore (submit run [| out |]);
  equal int 41 (word out 8)

(* Mixed blocks distinguish their offsets even when the preceding Fill has the
   largest parameter block and the following Fill has none. *)
let test_mixed () =
  let d, p = P.open_ "args:mixed" in
  let image =
    require_ok ~pp:Format.pp_print_string (Rig.Image.load d "functions")
  in
  let launch =
    let fill = require_some (Rig.Image.resolve image "fill") in
    Sub.launch ~refs:[| Sub.write 0 ~at:0 |] fill ~params:16
  in
  let s =
    Sub.make d
      [|
        fill ~params:4096 Support.poke;
        launch;
        fill ~params:0 ~refs:[||] Support.noop;
        fill Support.poke;
      |]
  in
  let run = Run.make s in
  store run (Sub.block s 0) ~offset:0 17;
  Run.int64 run (Sub.block s 0) 4088 127L;
  let block = Sub.block s 1 in
  Run.groups run block 1 1 1;
  Run.threads run block 1 1 1;
  store run block ~offset:8 29;
  store run (Sub.block s 3) ~offset:16 43;
  let out = B.create d 24 in
  ignore (submit run [| out |]);
  equal int 1 (P.run p);
  equal (list int) [ 17; 29; 43 ] (List.init 3 (fun i -> word out (i * 8)))

(* A fill's argument holds 8,192 bytes, past the 4,096 a part had: [far]
   stores its word at byte 4,096 through the ref in its last 8, on a device
   that runs it at its submit and on one that runs it later. *)
let test_wide_fill () =
  let wide d =
    let s =
      Sub.make d
        [| fill ~params:8192 ~refs:[| Sub.write 0 ~at:8184 |] Support.far |]
    in
    let run = Run.make s and block = Sub.block s 0 in
    Run.int64 run block 4096 59L;
    Run.int64 run block 8184 8L;
    let out = B.create d 16 in
    (out, submit run [| out |])
  in
  let out, _ = wide (Result.get_ok (Rig.memory_device "args:wide-memory")) in
  equal ~msg:"the memory device" int 59 (word out 8);
  let d, p = P.open_ "args:wide-polled" in
  let out, _ = wide d in
  ignore (P.run p);
  equal ~msg:"Polled" int 59 (word out 8)

let test_refs () =
  let d, p = P.open_ "args:refs" in
  let refs = [| Sub.write 0 ~at:0; Sub.read 1 ~at:8 |] in
  let s = Sub.make d [| fill ~params:24 ~refs Support.carry |] in
  refs.(0) <- Sub.read 1 ~at:16;
  let run = Run.make s and block = Sub.block s 0 in
  let src = B.create d 24 and dst = B.create d 24 in
  Support.store (Support.Reader.host src + 8) 101;
  Run.int64 run block 0 16L;
  Run.int64 run block 8 8L;
  Run.int64 run block 16 8L;
  ignore (submit run [| dst; src |]);
  ignore (P.run p);
  equal int 101 (word dst 16)

let test_refusals () =
  let d, _ = P.open_ "args:refusals" in
  let base = Rig.submitted d in
  let make params refs =
    Sub.make d
      [| fill ~params ~refs Support.noop |]
  in
  let at at slot = Sub.write slot ~at in
  List.iter
    (fun (msg, f) -> raises_match ~msg Exn.invalid_arg f)
    [
      ("negative parameters", fun () -> make (-1) [||]);
      ("unaligned ref", fun () -> make 16 [| at 4 0 |]);
      ("negative ref", fun () -> make 16 [| at (-8) 0 |]);
      ("overflowing ref", fun () -> make 16 [| at (max_int - 7) 0 |]);
      ("ref beyond parameters", fun () -> make 16 [| at 16 0 |]);
      ("ref in empty parameters", fun () -> make 0 [| at 0 0 |]);
      ("a slot no ref names", fun () -> make 16 [| at 0 1 |]);
      ( "a slot unordered and ordered",
        fun () -> make 16 [| at 0 0; Sub.unordered 0 ~at:8 |] );
      ("negative slot", fun () -> make 16 [| at 0 (-1) |]);
      ("overlapping refs", fun () -> make 16 [| at 0 0; at 0 0 |]);
    ];
  ignore (make 4096 [| at 0 0; at 4088 0 |]);
  let s = make 16 [| at 0 0 |] in
  let other = make 16 [| at 0 0 |] in
  let out = B.create d 8 in
  raises_match ~msg:"too few buffers" Exn.invalid_arg (fun () ->
      submit (Run.make s) [||]);
  raises_match ~msg:"too many buffers" Exn.invalid_arg (fun () ->
      submit (Run.make s) [| out; out |]);
  equal ~msg:"a fresh run holds the blocks" int (base + 1)
    (Rig.Point.value (submit (Run.make s) [| out |]));
  let run = Run.make s in
  raises_match ~msg:"another submission's block" Exn.invalid_arg (fun () ->
      store run (Sub.block other 0) ~offset:0 1)

let test_handle_only () =
  let d, _ = P.open_ ~addresses:false "args:handle-only" in
  let base = Rig.submitted d in
  let out = B.create d 8 in
  let s =
    Sub.make
      ~fixed:[ (out, Rig.Read_write) ]
      d
      [| fill Support.poke |]
  in
  raises_match Exn.invalid_arg (fun () -> Sub.address s out);
  let run = Run.make s in
  store run (Sub.block s 0) ~offset:0 1;
  raises_match Exn.invalid_arg (fun () -> submit run [| out |]);
  equal int base (Rig.submitted d)

let test_address () =
  let d, _ = P.open_ "args:address" in
  let host = B.create Rig.host (1 lsl 16) in
  let mapped = require_some (B.borrow d host) in
  let fixed = B.view mapped ~first:16 ~length:32 in
  let s = Sub.make ~fixed:[ (fixed, Rig.Read) ] d [||] in
  let expected = Support.at mapped + 16 in
  equal int expected (Sub.address s fixed);
  let nested = B.view (B.view host ~first:8 ~length:64) ~first:16 ~length:8 in
  equal int (expected + 8) (Sub.address s nested);
  let again = require_some (B.borrow d host) in
  equal int (expected + 8) (Sub.address s (B.view again ~first:24 ~length:8));
  equal int (expected + 32) (Sub.address s (B.view host ~first:48 ~length:0));
  raises_match ~msg:"whole allocation is not fixed" Exn.invalid_arg (fun () ->
      Sub.address s host);
  raises_match ~msg:"partial overlap" Exn.invalid_arg (fun () ->
      Sub.address s (B.view host ~first:8 ~length:16));
  raises_match ~msg:"unrelated allocation" Exn.invalid_arg (fun () ->
      Sub.address s (B.create d 8));
  (* Independently wrapped aliases have independent claims. Sharing a host
     address does not make the submission keep or order that claim. *)
  let alias = B.of_bigarray (B.export host Bigarray.char) in
  raises_match ~msg:"independently wrapped alias" Exn.invalid_arg (fun () ->
      Sub.address s (B.view alias ~first:16 ~length:8));
  let copy =
    Sub.make d
      [|
        Sub.(
          copy ~src:(Buffer mapped)
            ~dst:(Buffer (B.create d (B.length mapped))));
      |]
  in
  raises_match ~msg:"part memory is not explicit fixed memory" Exn.invalid_arg
    (fun () -> Sub.address copy mapped)

(* Fixed memory never dies, since the submission holds it; the address of a
   dead buffer raises. *)
let test_dead_address () =
  let d, _ = P.open_ "args:dead-address" in
  let b = B.create d 8 and dead = B.create d 8 in
  let s = Sub.make ~fixed:[ (b, Rig.Read) ] d [||] in
  Rig.Claim.with_ ~read:[] ~donate:[ dead ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"reused" dead));
  raises_match (Exn.invalid_arg ~substring:"reused") (fun () ->
      Sub.address s dead)

let () =
  exit
  @@ run "args"
       [
         test "Fill snapshots each run before a deferred hand-over" test_reuse;
         test "synchronous Fill preserves offsets across submits"
           test_synchronous;
         test "mixed Launch and Fill blocks stay independent" test_mixed;
         test "Fill resolves distinct refs copied at make" test_refs;
         test "a fill's argument holds 8,192 bytes" test_wide_fill;
         test "Fill refuses invalid blocks and refs" test_refusals;
         test "address refs refuse handle-only memory before submit"
           test_handle_only;
         test "fixed addresses follow memory identity and view bounds"
           test_address;
         test "an address of a dead buffer raises" test_dead_address;
       ]
