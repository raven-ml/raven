(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Mac's GPU as a device: opening it, its shared memory and borrows, its
   programs, work submitted to it and its timeline, and nx's runtime laws over
   it. *)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar

external compile : string -> string = "test_metal_compile"
external set_signaled : nativeint -> int -> unit = "test_metal_set_signaled"
external contains : nativeint -> nativeint -> bool = "test_metal_contains"
external allocation_count : nativeint -> int = "test_metal_allocation_count"

external dispatch :
  nativeint ->
  nativeint ->
  nativeint ->
  nativeint array ->
  nativeint ->
  nativeint ->
  int ->
  int ->
  unit = "test_metal_dispatch_byte" "test_metal_dispatch"

let metal = Nx_metal_device.v 0

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
      test "count is one, and get and v open that device once" (fun () ->
          equal int 1 (Nx_metal_device.count ());
          is_true (Nx_metal_device.v 0 == metal);
          is_true (Result.get_ok (Nx_metal_device.get 0) == metal));
      test
        "it is METAL, of an Apple GPU family, with a budget of its working set"
        (fun () ->
          equal string "METAL" (Nx_device.name metal);
          starts_with ~affix:"Apple" (Nx_device.arch metal);
          greater int ~than:0 (Nx_device.budget metal));
      test "its handles are Metal objects" (fun () ->
          let h = Nx_metal_device.handles metal in
          List.iter (not_equal nativeint 0n)
            [ h.device; h.queue; h.event; h.fence ]);
      cases ~name:fst "refuse"
        [
          ("a device past the count", fun () -> is_error (Nx_metal_device.get 1));
          ( "v of a device past the count, with get's message",
            fun () ->
              let why = Result.get_error (Nx_metal_device.get 1) in
              raises (Invalid_argument why) (fun () -> Nx_metal_device.v 1) );
          ( "a device of index -1",
            fun () ->
              raises_match Exn.invalid_arg (fun () -> Nx_metal_device.get (-1))
          );
          ( "the handles of the host",
            fun () ->
              raises_match Exn.invalid_arg (fun () ->
                  Nx_metal_device.handles Nx_device.host) );
          ( "the resources of the host",
            fun () ->
              raises_match Exn.invalid_arg (fun () ->
                  Nx_metal_device.resources Nx_device.host) );
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
          if B.nbytes b > 0 then not_equal nativeint 0n (B.host_address b);
          let d = Nx_device.Stats.diff s0 (Nx_device.stats metal) in
          equal (pair int int)
            (B.nbytes b, B.nbytes b)
            Nx_device.Stats.(bytes_in d, bytes_out d));
      test "a borrow of host memory shares its bytes" (fun () ->
          let whole = B.create Nx_device.host S.UInt8 page in
          let host = B.view whole ~offset:37 S.UInt8 5 in
          write host "abcde";
          let b = B.borrow metal host in
          equal (pair bool string) (true, "METAL")
            (B.is_borrowed b, Nx_device.name (B.device b));
          equal string "abcde" (read b);
          write host "vwxyz";
          equal string "vwxyz" (read b);
          let ba = B.bigarray Bigarray.char whole in
          let off_page = B.of_bigarray (Bigarray.Array1.sub ba 1 3) in
          raises_match Exn.invalid_arg (fun () -> B.borrow metal off_page));
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
          let set = (Nx_metal_device.handles metal).residency_set in
          let resident b =
            match set with
            | Some set -> contains set (B.handle b)
            | None -> Array.mem (B.handle b) (Nx_metal_device.resources metal)
          in
          let held () =
            Gc.full_major ();
            ignore (Nx_device.stats metal);
            match set with
            | Some set -> allocation_count set
            | None -> Array.length (Nx_metal_device.resources metal)
          in
          if set <> None then
            equal ~msg:"no resources beside the set" int 0
              (Array.length (Nx_metal_device.resources metal));
          let b = B.create metal S.UInt8 16 in
          is_true ~msg:"an allocation" (resident b);
          let before = held () in
          (fun () ->
            let bm = B.borrow metal (B.create Nx_device.host S.UInt8 page) in
            is_true ~msg:"a borrow" (resident bm);
            let now = held () in
            ignore (Sys.opaque_identity bm);
            equal ~msg:"one more" int (before + 1) now)
            ();
          equal ~msg:"a collected borrow leaves" int before (held ());
          ignore (Sys.opaque_identity b));
    ]

let work =
  group "work"
    [
      test "a metallib's function loads once" (fun () ->
          let binary = Lazy.force library in
          let p = Nx_device.Program.load metal ~binary ~name:"fill" in
          not_equal nativeint 0n (Nx_device.Program.handle p);
          is_true (Nx_device.Program.load metal ~binary ~name:"fill" == p));
      cases
        ~name:(fun (name, _, _) -> name)
        "a program load raises the driver's Failure for"
        [
          ("a missing function", (fun () -> Lazy.force library), "missing");
          ("a binary that is no metallib", (fun () -> "not a metallib"), "fill");
        ]
        (fun (_, binary, name) ->
          raises_match (Exn.failure ?substring:None) (fun () ->
              Nx_device.Program.load metal ~binary:(binary ()) ~name));
      test "a kernel submitted as timeline work writes its buffer and signals"
        (fun () ->
          let p =
            Nx_device.Program.load metal ~binary:(Lazy.force library)
              ~name:"fill"
          in
          let out = B.create metal S.UInt32 8 in
          let h = Nx_metal_device.handles metal in
          let v =
            Nx_device.submit metal ~touches:[] (fun v ->
                dispatch h.queue h.event h.fence
                  (Nx_metal_device.resources metal)
                  (Nx_device.Program.handle p)
                  (B.address out) 8 v;
                v)
          in
          equal int v (Nx_device.submitted metal);
          let words = Array.init 8 (fun i -> (i * 3) + 1) in
          let bytes =
            String.init 32 (fun k ->
                Char.chr ((words.(k / 4) lsr (8 * (k mod 4))) land 0xff))
          in
          equal string bytes (read out);
          at_least int ~than:v (Nx_device.signaled metal));
      test
        "the host waits for work that touched it, which signals on the shared \
         event and leaves the timeline's signal word at 0" (fun () ->
          let h = Nx_metal_device.handles metal in
          let signaled = Atomic.make false in
          let domain =
            Nx_device.submit metal ~touches:[ Nx_device.host ] (fun v ->
                Domain.spawn (fun () ->
                    Unix.sleepf 0.05;
                    Atomic.set signaled true;
                    set_signaled h.event v))
          in
          Nx_device.synchronize Nx_device.host;
          is_true (Atomic.get signaled);
          Domain.join domain;
          let words = B.bigarray Bigarray.int64 (Nx_device.timeline metal) in
          equal (pair int64 int)
            (0L, Nx_device.submitted metal)
            (words.{0}, Int64.to_int words.{1}));
    ]

let () =
  exit
    (run "nx.metal.device"
       [ opening; memory; work; group "nx" (Nx_test.Runtimes.laws [ metal ]) ])
