(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Mac's GPU as a device: opening it, its shared memory and borrows, its
   programs, work submitted to it and its timeline, nx's runtime laws over it,
   and last, a command buffer Metal fails. *)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar

external compile : string -> string = "test_metal_compile"
external signal : nativeint -> nativeint -> int -> unit = "test_metal_signal"
external contains : nativeint -> nativeint -> bool = "test_metal_contains"
external allocation_count : nativeint -> int = "test_metal_allocation_count"
external weak : nativeint -> nativeint = "test_metal_weak"
external weak_live : nativeint -> bool = "test_metal_weak_live"

external dispatch :
  nativeint ->
  nativeint ->
  nativeint ->
  nativeint array ->
  nativeint ->
  nativeint ->
  int ->
  int ->
  nativeint ->
  unit = "test_metal_dispatch_byte" "test_metal_dispatch"

external execute :
  nativeint ->
  nativeint ->
  nativeint ->
  nativeint array ->
  nativeint ->
  int ->
  int ->
  unit = "test_metal_execute_byte" "test_metal_execute"

let metal = Result.get_ok (Nx_metal_device.get 0)
let m = Option.get (Nx_metal_device.of_device metal)

let borrow_host b =
  match B.borrow Nx_device.host b with Ok b -> b | Error why -> failwith why

(* Launches [p] over [threads] threads writing [out] to Metal, and is its
   value. *)
let launch ?(stamps = 0n) ?(record = ignore) p out threads =
  Nx_device.submit [ metal ] ~touches:[ out ] (fun s ->
      let v = Nx_device.Submission.value s metal in
      record s;
      dispatch (Nx_metal_device.queue m)
        (Nx_metal_device.signaler m)
        (Nx_metal_device.fence m)
        (Nx_metal_device.resources m)
        (Nx_device.Program.handle p)
        (B.address out) threads v stamps;
      v)

(* [metal]'s borrow of [b], which it maps. *)
let borrow b =
  match B.borrow metal b with Ok b -> b | Error why -> failwith why

(* The file at [path], opened, or created with [n] bytes. *)
let of_file path =
  match B.of_file path with Ok b -> b | Error why -> failwith why

let create_file path n =
  match B.create_file path n with Ok b -> b | Error why -> failwith why

(* The function [name] of [binary], which [metal] loads. *)
let program ~binary ~name =
  match Nx_device.Program.load metal ~binary ~name with
  | Ok p -> p
  | Error why -> failwith why

let library =
  lazy
    (compile
       {|#include <metal_stdlib>
using namespace metal;
struct args { device uint *out; };
kernel void fill(constant args &a [[buffer(0)]],
                 uint i [[threadgroup_position_in_grid]]) {
  a.out[i] = i * 3u + 1u;
}|})

type chars =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

let chars n : chars = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

let read b =
  let ba = chars (B.nbytes b) in
  B.copy ~src:b ~dst:(B.of_bigarray ba);
  String.init (Bigarray.Array1.dim ba) (Bigarray.Array1.get ba)

let write b s =
  B.copy
    ~src:
      (B.of_bigarray
         (Bigarray.Array1.init Bigarray.char Bigarray.c_layout (String.length s)
            (String.get s)))
    ~dst:b

let pattern seed n =
  String.init n (fun i -> Char.chr ((seed + (i * 7)) land 0xff))

(* Host buffers of this many bytes start on a page. *)
let page = 1 lsl 16

let opening =
  group "opening"
    [
      test "count is one, and get opens that device once" (fun () ->
          equal int 1 (Nx_metal_device.count ());
          is_true (Result.get_ok (Nx_metal_device.get 0) == metal));
      test
        "it is METAL, of an Apple GPU family, with a budget of its working set"
        (fun () ->
          equal string "METAL" (Nx_device.name metal);
          starts_with ~affix:"Apple" (Nx_device.arch metal);
          greater int ~than:0 (Nx_device.budget metal));
      test "its low-level accessors are Metal objects" (fun () ->
          List.iter (not_equal nativeint 0n)
            Nx_metal_device.[ queue m; signaler m; fence m ]);
      cases ~name:fst "refuse"
        [
          ("a device past the count", fun () -> is_error (Nx_metal_device.get 1));
          ( "a device past the count, naming the device first",
            fun () ->
              let why = Result.get_error (Nx_metal_device.get 1) in
              is_true ~msg:"the device's name first"
                (String.starts_with ~prefix:"METAL:1: " why) );
          ( "a device of index -1",
            fun () ->
              raises_match Exn.invalid_arg (fun () -> Nx_metal_device.get (-1))
          );
          ( "the Metal objects of the host",
            fun () ->
              is_true
                (Option.is_none (Nx_metal_device.of_device Nx_device.host)) );
        ]
        (fun (_, check) -> check ());
    ]

let formats =
  Gen.of_list
    ~pp:(fun ppf s -> Format.pp_print_string ppf (S.to_string s))
    S.[ UInt8; Int4; BFloat16; Float32; Float64; Complex128 ]

let memory =
  group "memory"
    [
      prop
        "a copy there and back is the identity, through memory the host \
         addresses, and counts its bytes"
        (Gen.quad formats
           (Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 3; 17; 1000 ])
           (Gen.int_range 0 3) (Gen.int_range 0 255))
        (fun (s, n, k, seed) ->
          let o = k * Int.max 1 (S.bitsize s / 8) in
          let bytes = ((n * S.bitsize s) + 7) / 8 in
          let whole = B.create metal S.UInt8 (o + bytes + 5) in
          let b = B.view whole ~offset:o s n in
          let bytes = pattern seed (B.nbytes b)
          and s0 = Nx_device.stats metal in
          write b bytes;
          equal string bytes (read b);
          if B.nbytes b > 0 then not_equal nativeint 0n (B.address b);
          let d = Nx_device.Stats.diff s0 (Nx_device.stats metal) in
          equal (pair int int)
            (B.nbytes b, B.nbytes b)
            Nx_device.Stats.(bytes_in d, bytes_out d));
      test
        "a file's bytes are read straight into its memory and written from it, \
         with no work on its timeline" (fun () ->
          let n = (3 lsl 20) + 12345 in
          let bytes = pattern 11 n in
          let path = temp_file () in
          Out_channel.with_open_bin path (fun oc -> output_string oc bytes);
          let b =
            B.view (B.create metal S.UInt8 (n + 16)) ~offset:16 S.UInt8 n
          in
          let v = Nx_device.submitted metal in
          B.copy ~src:(of_file path) ~dst:b;
          is_true ~msg:"read" (read b = bytes);
          let out = temp_file () in
          B.copy ~src:b ~dst:(create_file out n);
          is_true ~msg:"written"
            (In_channel.with_open_bin out In_channel.input_all = bytes);
          equal ~msg:"work submitted" int v (Nx_device.submitted metal));
      test "a borrow of a file's bytes is the file's pages, read where they lie"
        (fun () ->
          let bytes = pattern 13 ((1 lsl 20) + 4099) in
          let path = temp_file () in
          Out_channel.with_open_bin path (fun oc -> output_string oc bytes);
          let before = Nx_device.stats Nx_device.disk in
          let b =
            borrow
              (B.view (of_file path) ~offset:4 S.UInt8
                 (String.length bytes - 4))
          in
          equal (pair bool string) (true, "METAL")
            (B.is_borrowed b, Nx_device.name (B.device b));
          is_true ~msg:"its bytes"
            (read b = String.sub bytes 4 (String.length bytes - 4));
          equal ~msg:"bytes read from the file" int 0
            Nx_device.Stats.(
              bytes_out (diff before (Nx_device.stats Nx_device.disk))));
      test "a borrow of host memory shares its bytes" (fun () ->
          let whole = B.create Nx_device.host S.UInt8 page in
          let host = B.view whole ~offset:37 S.UInt8 5 in
          write host "abcde";
          let b = borrow host in
          equal (pair bool string) (true, "METAL")
            (B.is_borrowed b, Nx_device.name (B.device b));
          equal string "abcde" (read b);
          write host "vwxyz";
          equal string "vwxyz" (read b);
          let ba = B.bigarray Bigarray.char whole in
          let off_page = B.of_bigarray (Bigarray.Array1.sub ba 1 3) in
          is_error ~msg:"memory off a page" (B.borrow metal off_page));
      test "an allocation over its budget raises Out_of_memory with it"
        (fun () ->
          let budget = Nx_device.budget metal in
          Fun.protect ~finally:(fun () -> Nx_device.set_budget metal budget)
          @@ fun () ->
          Gc.full_major ();
          let held = Nx_device.Stats.allocated (Nx_device.stats metal) in
          Nx_device.set_budget metal (held + 1024);
          let b = B.create metal S.UInt8 1024 in
          raises_match
            (function
              | Nx_device.Out_of_memory (d, 1) -> d == metal | _ -> false)
            (fun () -> B.create metal S.UInt8 1);
          ignore (Sys.opaque_identity b));
      test
        "every allocation and borrow is resident while it lives, in the \
         residency set where Metal has one" (fun () ->
          let set = Nx_metal_device.residency_set m in
          let resident b =
            let buffer = Nx_device.Driver.Region.(handle (of_buffer b)) in
            match set with
            | Some set -> contains set buffer
            | None -> Array.mem buffer (Nx_metal_device.resources m)
          in
          let held () =
            Gc.full_major ();
            ignore (Nx_device.stats metal);
            match set with
            | Some set -> allocation_count set
            | None -> Array.length (Nx_metal_device.resources m)
          in
          if set <> None then
            equal ~msg:"no resources beside the set" int 0
              (Array.length (Nx_metal_device.resources m));
          let b = B.create metal S.UInt8 16 in
          is_true ~msg:"an allocation" (resident b);
          let before = held () in
          (fun () ->
            let bm = borrow (B.create Nx_device.host S.UInt8 page) in
            is_true ~msg:"a borrow" (resident bm);
            let now = held () in
            ignore (Sys.opaque_identity bm);
            equal ~msg:"one more" int (before + 1) now)
            ();
          equal ~msg:"a collected borrow leaves" int before (held ());
          ignore (Sys.opaque_identity b));
    ]

(* A metallib whose [fill] writes [i * 3 + c] at each index [i]. *)
let filling c =
  compile
    (Printf.sprintf
       {|#include <metal_stdlib>
using namespace metal;
struct args { device uint *out; };
kernel void fill(constant args &a [[buffer(0)]],
                 uint i [[threadgroup_position_in_grid]]) {
  a.out[i] = i * 3u + %du;
}|}
       c)

(* Loads and drops 600 distinct binaries, copies of one metallib padded with [k]
   zeros, which Metal reads as the same library, while another stays held: the
   pipelines of the dropped ones are released once collected, and the held one
   still runs. *)
let test_churn () =
  let held = program ~binary:(filling 1) ~name:"fill" in
  let base = filling 2 in
  let padded k = program ~binary:(base ^ String.make k '\000') ~name:"fill" in
  let one = padded 0 and other = padded 1 in
  not_equal ~msg:"a pipeline per binary" nativeint
    (Nx_device.Program.handle one)
    (Nx_device.Program.handle other);
  ignore (Sys.opaque_identity (one, other));
  let dropped =
    Array.init 600 (fun k -> weak (Nx_device.Program.handle (padded (k + 2))))
  in
  for _ = 1 to 4 do
    Gc.full_major ();
    Nx_device.synchronize metal
  done;
  let live =
    Array.fold_left (fun n w -> if weak_live w then n + 1 else n) 0 dropped
  in
  equal ~msg:"pipelines of dropped programs alive" int 0 live;
  let out = B.create metal S.UInt32 8 in
  ignore (launch held out 8);
  Nx_device.synchronize metal;
  let words = Array.init 8 (fun i -> (i * 3) + 1) in
  equal ~msg:"the held program runs" string
    (String.init 32 (fun k ->
         Char.chr ((words.(k / 4) lsr (8 * (k mod 4))) land 0xff)))
    (read out)

let work =
  group "work"
    [
      test
        "600 programs loaded and dropped release their pipelines, and a held \
         one runs"
        test_churn;
      test "a metallib's function loads once" (fun () ->
          let binary = Lazy.force library in
          let p = program ~binary ~name:"fill" in
          not_equal nativeint 0n (Nx_device.Program.handle p);
          equal nativeint
            (Nx_device.Program.handle p)
            (Nx_device.Program.handle (program ~binary ~name:"fill")));
      cases
        ~name:(fun (name, _, _) -> name)
        "a program load is the driver's Error, after the device's name, for"
        [
          ("a missing function", (fun () -> Lazy.force library), "missing");
          ("a binary that is no metallib", (fun () -> "not a metallib"), "fill");
        ]
        (fun (_, binary, name) ->
          match Nx_device.Program.load metal ~binary:(binary ()) ~name with
          | Ok _ -> fail "loaded"
          | Error why ->
              is_true ~msg:"the device's name first"
                (String.starts_with ~prefix:"METAL: " why));
      test "a kernel submitted as timeline work writes its buffer and signals"
        (fun () ->
          let p = program ~binary:(Lazy.force library) ~name:"fill" in
          let out = B.create metal S.UInt32 8 in
          let v = launch p out 8 in
          equal int v (Nx_device.submitted metal);
          let words = Array.init 8 (fun i -> (i * 3) + 1) in
          let bytes =
            String.init 32 (fun k ->
                Char.chr ((words.(k / 4) lsr (8 * (k mod 4))) land 0xff))
          in
          equal string bytes (read out);
          at_least int ~than:v (Nx_device.signaled metal));
      test
        "the host borrows a Metal buffer where it lies, and reads what a \
         kernel wrote there once Metal is synchronized" (fun () ->
          let p = program ~binary:(Lazy.force library) ~name:"fill" in
          let out = B.create metal S.UInt32 8 in
          ignore (launch p out 8);
          Nx_device.synchronize metal;
          let on_host = borrow_host out in
          is_true ~msg:"on the host, over the Metal buffer"
            (Nx_device.equal (B.device on_host) Nx_device.host
            && B.overlaps on_host out);
          let v = B.bigarray Bigarray.int32 on_host in
          equal ~msg:"what the kernel wrote" (list int32)
            (List.init 8 (fun i -> Int32.of_int ((i * 3) + 1)))
            (List.init 8 (Bigarray.Array1.get v)));
      test
        "the host waits for work that touched it, which signals through the \
         signaler and leaves its signal word at 0" (fun () ->
          let signaled = Atomic.make false in
          let hb = B.create Nx_device.host S.UInt8 8 in
          let domain =
            Nx_device.submit [ metal ] ~touches:[ hb ] (fun s ->
                let v = Nx_device.Submission.value s metal in
                Domain.spawn (fun () ->
                    Unix.sleepf 0.05;
                    Atomic.set signaled true;
                    signal (Nx_metal_device.queue m)
                      (Nx_metal_device.signaler m)
                      v))
          in
          Nx_device.synchronize Nx_device.host;
          is_true (Atomic.get signaled);
          Domain.join domain;
          let word = B.bigarray Bigarray.int64 (Nx_device.signal_word metal) in
          equal int64 0L word.{0});
      test
        "its work is waited for by no queue of another device, and a \
         submission sees its previous kernel complete before it rewrites its \
         buffer" (fun () ->
          let p = program ~binary:(Lazy.force library) ~name:"fill" in
          let out = B.create metal S.UInt32 8 in
          let words = B.bigarray Bigarray.int32 (borrow_host out) in
          let first = launch p out 8 in
          Nx_device.submit [ metal ] ~touches:[ out ] (fun s ->
              equal ~msg:"no waits" int 0
                (List.length (Nx_device.Submission.waits s));
              Nx_device.Submission.wait s metal first;
              at_least ~msg:"the previous kernel signaled" int ~than:first
                (Nx_device.signaled metal);
              for i = 0 to 7 do
                words.{i} <- 0l
              done;
              signal (Nx_metal_device.queue m)
                (Nx_metal_device.signaler m)
                (Nx_device.Submission.value s metal));
          equal ~msg:"rewritten after the kernel" string (String.make 32 '\000')
            (read out));
    ]

module P = Nx_device.Profile

let profiled f =
  let p = P.start () in
  match f () with
  | () -> P.stop p
  | exception e ->
      ignore (P.stop p);
      raise e

let twice =
  lazy
    (compile
       {|#include <metal_stdlib>
using namespace metal;
struct args { device uint *out; };
kernel void twice(constant args &a [[buffer(0)]],
                  uint i [[threadgroup_position_in_grid]]) {
  a.out[i] = 2u * i;
}|})

let dispatch_profile =
  test
    "a dispatch is a span of its command buffer's GPU times on the host clock, \
     and the first load of its program an event" (fun () ->
      let out = B.create metal S.UInt32 64 in
      let stamps = B.create Nx_device.host S.UInt64 4 in
      let binary = Lazy.force twice in
      let before = ref 0 and after = ref 0 in
      let events =
        profiled (fun () ->
            let p = program ~binary ~name:"twice" in
            before := P.now ();
            ignore
              (launch ~stamps:(B.address stamps)
                 ~record:(fun s ->
                   Nx_device.Submission.record s metal ~lane:"compute"
                     ~name:"twice" stamps)
                 p out 64);
            Nx_device.synchronize metal;
            after := P.now ())
      in
      let ours =
        List.filter
          (function
            | P.Allocation _ | P.Counters _ | P.Trace _ | P.Overwritten _ ->
                false
            | P.Span _ | P.Load _ -> true)
          events
      in
      match ours with
      | [ P.Load p; P.Span s ] ->
          equal (pair string bool) ("twice", true)
            (Nx_device.Program.name p.program, p.binary = binary);
          equal (pair string string) ("METAL", "compute")
            (Nx_device.name s.device, s.lane);
          is_true
            ~msg:
              (Printf.sprintf "GPU %d to %d within host %d to %d" s.start s.stop
                 !before !after)
            (!before <= s.start && s.start <= s.stop && s.stop <= !after)
      | l -> fail (Printf.sprintf "%d events" (List.length l)))

(* A command of [fill] over [threads] threads, its arguments at [offset]. *)
let fill_command ?(local = (1, 1, 1)) ~offset threads =
  {
    Nx_metal_device.program = program ~binary:(Lazy.force library) ~name:"fill";
    offset;
    global = (threads, 1, 1);
    local;
  }

let submitters =
  group "submitters"
    [
      test "msg_send is the address of a function" (fun () ->
          not_equal nativeint 0n Nx_metal_device.msg_send);
      test "a selector is registered once per name" (fun () ->
          let s = Nx_metal_device.selector "commandBuffer" in
          not_equal nativeint 0n s;
          equal nativeint s (Nx_metal_device.selector "commandBuffer");
          not_equal nativeint s (Nx_metal_device.selector "commit"));
      test "an indirect command buffer runs each dispatch on its arguments"
        (fun () ->
          let outs = List.init 2 (fun _ -> B.create metal S.UInt32 4) in
          let args = B.create metal S.UInt8 512 in
          let words = B.bigarray Bigarray.int64 (borrow_host args) in
          List.iteri
            (fun i out -> words.{32 * i} <- Int64.of_nativeint (B.address out))
            outs;
          let cmds =
            List.init 2 (fun i -> fill_command ~offset:(256 * i) (4 - (2 * i)))
          in
          match Nx_metal_device.indirect_commands m args cmds with
          | Error why -> fail why
          | Ok (icb, commands) ->
              equal int 2 (List.length commands);
              List.iter (not_equal nativeint 0n) (icb :: commands);
              Nx_device.submit [ metal ] ~touches:(args :: outs) (fun s ->
                  execute (Nx_metal_device.queue m)
                    (Nx_metal_device.signaler m)
                    (Nx_metal_device.fence m)
                    (Nx_metal_device.resources m)
                    icb 2
                    (Nx_device.Submission.value s metal));
              Nx_device.synchronize metal;
              let values out =
                let a = B.bigarray Bigarray.int32 (borrow_host out) in
                List.init 2 (fun i -> Int32.to_int a.{i})
              in
              equal
                (list (list int))
                [ [ 1; 4 ]; [ 1; 4 ] ]
                (List.map values outs));
      test
        "a dispatch of more threads per threadgroup than its pipeline's is \
         refused" (fun () ->
          let args = B.create metal S.UInt8 256 in
          match
            Nx_metal_device.indirect_commands m args
              [ fill_command ~local:(1 lsl 16, 1, 1) ~offset:0 1 ]
          with
          | Ok _ -> fail "made"
          | Error why -> Windtrap.contains ~sub:"bigger than" why);
      test "arguments of another device are refused" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Nx_metal_device.indirect_commands m
                (B.create Nx_device.host S.UInt8 256)
                [ fill_command ~offset:0 1 ]));
    ]

let spin =
  lazy
    (compile
       {|#include <metal_stdlib>
using namespace metal;
struct args { device uint *out; };
kernel void spin(constant args &a [[buffer(0)]],
                 uint i [[threadgroup_position_in_grid]]) {
  uint acc = i;
  for (uint k = 0; k < 1000u; k++)
    for (uint j = 0; j < 1000000u; j++) acc = acc * 1664525u + 1013904223u + j;
  a.out[i] = acc;
}|})

let spin_groups = 1 lsl 15

(* Last: a command buffer that Metal fails, as macOS ends one that keeps the GPU
   from the display for about half a second on a Mac that drives one, loses the
   device with Metal's reason, though a failed command buffer still runs the
   signals encoded in it. *)
let failed =
  test "a command buffer Metal fails loses the device with Metal's reason"
    (fun () ->
      let p = program ~binary:(Lazy.force spin) ~name:"spin" in
      let out = B.create metal S.UInt32 spin_groups in
      ignore (launch p out spin_groups);
      match Nx_device.synchronize metal with
      | () -> skip ~reason:"macOS ended no command buffer" ()
      | exception Nx_device.Lost (d, why) ->
          is_true ~msg:"Metal's device" (d == metal);
          Windtrap.contains ~msg:"Metal's reason"
            ~sub:"kIOGPUCommandBufferCallbackError" why;
          raises_match
            (function Nx_device.Lost (d, _) -> d == metal | _ -> false)
            (fun () -> B.create metal S.UInt8 1))

(* After a failed command buffer, the device signals nothing more: the value of
   a later command buffer that completes does not read as signaled, and the
   failure loses the device when its value is read. *)
let failed_watched =
  test
    "a command buffer that fails before the one that signals keeps the work \
     unsignaled, and a read of the value loses the device" (fun () ->
      let d = Result.get_ok (Nx_metal_device.get 0) in
      let m = Option.get (Nx_metal_device.of_device d) in
      let load name binary =
        match Nx_device.Program.load d ~binary ~name with
        | Ok p -> p
        | Error why -> failwith why
      in
      let spin = load "spin" (Lazy.force spin)
      and fill = load "fill" (Lazy.force library) in
      let out = B.create d S.UInt32 spin_groups in
      let dispatch p threads v =
        dispatch (Nx_metal_device.queue m)
          (Nx_metal_device.signaler m)
          (Nx_metal_device.fence m)
          (Nx_metal_device.resources m)
          (Nx_device.Program.handle p)
          (B.address out) threads v 0n
      in
      let v =
        Nx_device.submit [ d ] ~touches:[ out ] (fun s ->
            let v = Nx_device.Submission.value s d in
            dispatch spin spin_groups 0;
            dispatch fill 1 v;
            v)
      in
      let rec settled n =
        if n > 0 && Nx_device.lost d = None && Nx_device.signaled d < v then begin
          Unix.sleepf 0.05;
          settled (n - 1)
        end
      in
      settled 200;
      match Nx_device.lost d with
      | None when Nx_device.signaled d >= v ->
          skip ~reason:"macOS ended no command buffer" ()
      | None -> fail "the work neither signaled nor failed in 10 s"
      | Some why ->
          Windtrap.contains ~msg:"Metal's reason"
            ~sub:"kIOGPUCommandBufferCallbackError" why;
          less ~msg:"the value" int ~than:v (Nx_device.signaled d);
          raises_match
            (function Nx_device.Lost (d', _) -> d' == d | _ -> false)
            (fun () -> Nx_device.synchronize d))

let () =
  exit
    (run "nx.metal.device"
       [
         opening;
         memory;
         work;
         submitters;
         group "profiles" (dispatch_profile :: Nx_test.Profiles.copies [ metal ]);
         group "nx" (Nx_test.Runtimes.laws [ metal ]);
         group "failures" [ failed; failed_watched ];
       ])
