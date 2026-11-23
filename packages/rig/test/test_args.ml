(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module Support = Rig_support
module P = Support.Polled

let fill ?(params = 16) ?(refs = [| Sub.write 0 ~at:0 |]) fn =
  Sub.fill ~refs fn ~params

let submit s buffers args = Rig.submit s ~buffers ~args
let word b at = Support.load (Support.Reader.host b + at)

(* Writes the values of a [Support.poke] fill into [params]: its ref adds slot
   0's address to [offset], where the fill stores [value]. *)
let store params ~offset value =
  Bytes.set_int64_le params 0 (Int64.of_int offset);
  Bytes.set_int64_le params 8 (Int64.of_int value)

(* The values of a submission's one fill, with [params]. *)
let values params = [| { Support.zero with params } |]

(* A submission serves successive submits while the first still waits in a
   queue: each handed-over call holds the parameter bytes and the resolved
   addresses of its own submit, whatever the caller writes into its values
   after. *)
let test_reuse () =
  let d, p = P.open_ "args:reuse" in
  let s = Sub.make d [| fill Support.poke |] in
  let params = Bytes.create 16 in
  let args = values params in
  let a = B.create d 24 and b = B.create d 24 in
  store params ~offset:8 31;
  ignore (submit s [| a |] args);
  store params ~offset:16 47;
  ignore (submit s [| b |] args);
  store params ~offset:0 99;
  equal int 2 (P.queued p);
  equal int 2 (P.run p);
  equal int 31 (word a 8);
  equal int 47 (word b 16);
  (* Repeating a submit adds the address to the passed offset exactly once. *)
  store params ~offset:8 73;
  ignore (submit s [| a |] args);
  ignore (P.run p);
  ignore (submit s [| b |] args);
  ignore (P.run p);
  equal int 73 (word a 8);
  equal int 73 (word b 8)

let test_synchronous () =
  let d = Result.get_ok (Rig.software "args:sync") in
  let s = Sub.make d [| fill Support.poke |] in
  let params = Bytes.create 16 in
  let out = B.create d 16 in
  store params ~offset:8 41;
  ignore (submit s [| out |] (values params));
  equal int 41 (word out 8);
  ignore (submit s [| out |] (values params));
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
  let wide = Bytes.make 4096 '\000' in
  store wide ~offset:0 17;
  Bytes.set_int64_le wide 4088 127L;
  let launched = Bytes.create 16 and last = Bytes.create 16 in
  store launched ~offset:8 29;
  store last ~offset:16 43;
  let one = (1, 1, 1) in
  let args =
    [|
      { Support.zero with params = wide };
      { Sub.groups = one; threads = one; shared = 0; params = launched };
      Support.zero;
      { Support.zero with params = last };
    |]
  in
  let out = B.create d 24 in
  ignore (submit s [| out |] args);
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
    let params = Bytes.make 8192 '\000' in
    Bytes.set_int64_le params 4096 59L;
    Bytes.set_int64_le params 8184 8L;
    let out = B.create d 16 in
    (out, submit s [| out |] (values params))
  in
  let out, _ = wide (Result.get_ok (Rig.software "args:wide-memory")) in
  equal ~msg:"the software device" int 59 (word out 8);
  let d, p = P.open_ "args:wide-polled" in
  let out, _ = wide d in
  ignore (P.run p);
  equal ~msg:"Polled" int 59 (word out 8)

(* The parameters past a submit's values read 0, also where an earlier
   submit of the submission wrote them: [far] stores the word at its byte
   4,096 where its last 8 bytes point, and a submit of 8 bytes stores 0 at
   the slot's start, on a device that runs it at its submit and on one that
   runs it later. *)
let test_short_values () =
  let short d run =
    let s =
      Sub.make d
        [| fill ~params:8192 ~refs:[| Sub.write 0 ~at:8184 |] Support.far |]
    in
    let full = Bytes.make 8192 '\000' in
    Bytes.set_int64_le full 4096 59L;
    Bytes.set_int64_le full 8184 8L;
    let out = B.create d 16 in
    B.blit_from_string (String.make 16 '\xff') 0 out 0 16;
    ignore (submit s [| out |] (values full));
    run ();
    equal ~msg:"the longer submit's word" int 59 (word out 8);
    ignore (submit s [| out |] (values (Bytes.make 8 '\001')));
    run ();
    equal ~msg:"the word at the slot's start" int 0 (word out 0);
    equal ~msg:"the longer submit's word, kept" int 59 (word out 8)
  in
  short (Result.get_ok (Rig.software "args:short-memory")) ignore;
  let d, p = P.open_ "args:short-polled" in
  short d (fun () -> ignore (P.run p))

let test_refs () =
  let d, p = P.open_ "args:refs" in
  let refs = [| Sub.write 0 ~at:0; Sub.read 1 ~at:8 |] in
  let s = Sub.make d [| fill ~params:24 ~refs Support.carry |] in
  refs.(0) <- Sub.read 1 ~at:16;
  let src = B.create d 24 and dst = B.create d 24 in
  Support.store (Support.Reader.host src + 8) 101;
  let params = Bytes.create 24 in
  Bytes.set_int64_le params 0 16L;
  Bytes.set_int64_le params 8 8L;
  Bytes.set_int64_le params 16 8L;
  ignore (submit s [| dst; src |] (values params));
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
  let out = B.create d 8 in
  let one = Support.zeros 1 in
  raises_match ~msg:"too few buffers" Exn.invalid_arg (fun () ->
      submit s [||] one);
  raises_match ~msg:"too many buffers" Exn.invalid_arg (fun () ->
      submit s [| out; out |] one);
  raises
    ~msg:"values longer than the fill's parameters"
    (Invalid_argument
       "Rig.submit: args.(0) has 17 bytes of params, past the 16 of the fill \
        of part 0")
    (fun () -> submit s [| out |] (values (Bytes.create 17)));
  raises ~msg:"no values"
    (Invalid_argument
       "Rig.submit: args has 0 elements, not 1, one per launch or fill")
    (fun () -> submit s [| out |] [||]);
  equal ~msg:"no value assigned" int base (Rig.submitted d);
  equal ~msg:"a submit of the right buffers and values" int (base + 1)
    (Rig.Point.value (submit s [| out |] one))

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
  let params = Bytes.create 16 in
  store params ~offset:0 1;
  raises_match Exn.invalid_arg (fun () -> submit s [| out |] (values params));
  equal int base (Rig.submitted d)

let test_address () =
  let d, _ = P.open_ "args:address" in
  let host = B.create Rig.host B.paged_from in
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
  ignore (require_some (Rig.Buffer.take ~why:"reused" [ dead ]));
  raises_match (Exn.invalid_arg ~sub:"reused") (fun () -> Sub.address s dead)

let () =
  exit
  @@ run "args"
       [
         test "a fill reads its own submit's values in a deferred hand-over"
           test_reuse;
         test "synchronous Fill preserves offsets across submits"
           test_synchronous;
         test "mixed Launch and Fill blocks stay independent" test_mixed;
         test "Fill resolves distinct refs copied at make" test_refs;
         test "a fill's argument holds 8,192 bytes" test_wide_fill;
         test "a fill's parameters past its values read 0" test_short_values;
         test "Fill refuses invalid blocks, refs and values" test_refusals;
         test "address refs refuse handle-only memory before submit"
           test_handle_only;
         test "fixed addresses follow memory identity and view bounds"
           test_address;
         test "an address of a dead buffer raises" test_dead_address;
       ]
