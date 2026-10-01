(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* CUDA devices on a real GPU: opening, memory and its budget, every copy route,
   borrowing, memory of another library, programs, a kernel launched as timeline
   work, nx's runtime laws over the devices, and a fault. Every test skips on a
   machine without a CUDA device, and those of two GPUs on one without two. *)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar

external launch :
  nativeint -> nativeint -> nativeint -> nativeint -> nativeint -> int -> unit
  = "test_launch_byte" "test_launch"

external foreign_alloc : nativeint -> int -> nativeint = "test_alloc"

external foreign_host_alloc : nativeint -> int -> nativeint * nativeint
  = "test_host_alloc"

external other_context_alloc : int -> int -> nativeint
  = "test_other_context_alloc"

let gpus = List.init (Nx_cuda_device.count ()) Nx_cuda_device.v

(* The host address of [b]'s first byte, if the host addresses [b]'s memory. *)
let hosted b =
  let r = Nx_device.Driver.Region.of_buffer b in
  Option.map
    (fun a -> Nativeint.add a (Nativeint.of_int (B.offset b)))
    (Nx_device.Driver.Region.host_address r)

(* [d]'s borrow of [b], which it maps. *)
let borrow d b = match B.borrow d b with Ok b -> b | Error why -> failwith why

(* The function [name] of [binary], which [d] loads. *)
let program d ~binary ~name =
  match Nx_device.Program.load d ~binary ~name with
  | Ok p -> p
  | Error why -> failwith why

(* [f ()] is [Error why], [why] containing [sub]. *)
let refused ~sub f =
  match f () with
  | Ok _ -> fail "not refused"
  | Error why -> contains ~msg:"the reason" ~sub why

let cuda () =
  match gpus with [] -> skip ~reason:"no CUDA device" () | d :: _ -> d

let two () =
  match gpus with
  | d :: d1 :: _ -> (d, d1)
  | _ -> skip ~reason:"fewer than two CUDA devices" ()

let mib = 1 lsl 20

(* More bytes than a page on every platform. *)
let pages = 1 lsl 16

type bytes_ba =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

let bytes_of b : bytes_ba = B.bigarray Bigarray.int8_unsigned b

(* A host buffer of [n] bytes holding [f i] at [i]. *)
let host_of n f =
  let b = B.create Nx_device.host S.UInt8 n in
  let ba = bytes_of b in
  for i = 0 to n - 1 do
    ba.{i} <- f i land 0xff
  done;
  b

(* The first byte of [b] that does not hold [f i] at [i], if any. *)
let differs b f =
  let host = B.create Nx_device.host S.UInt8 (B.nbytes b) in
  B.copy ~src:b ~dst:host;
  let ba = bytes_of host in
  let rec go i =
    if i = B.nbytes b then None
    else if ba.{i} <> f i land 0xff then Some i
    else go (i + 1)
  in
  go 0

let holds ?msg b f = is_none ?msg ~pp:Format.pp_print_int (differs b f)

(* The timeline values [f] used on [d]. *)
let values d f =
  let v = Nx_device.submitted d in
  f ();
  Nx_device.submitted d - v

let collect d =
  Gc.full_major ();
  ignore (Nx_device.stats d)

let out_of_memory d = function
  | Nx_device.Out_of_memory (d', _) -> d' == d
  | _ -> false

let opening =
  test
    "a GPU opens once as CUDA, of its compute capability, with a budget of its \
     memory, and none past the count" (fun () ->
      let d = cuda () in
      equal string "CUDA" (Nx_device.name d);
      starts_with ~affix:"sm_" (Nx_device.arch d);
      greater int ~than:0 (Nx_device.budget d);
      is_true (Nx_cuda_device.v 0 == d);
      let n = Nx_cuda_device.count () in
      starts_with
        ~affix:(Printf.sprintf "CUDA:%d: no such device" n)
        (require_error (Nx_cuda_device.get n));
      is_true ~msg:"no driver objects for the host"
        (Option.is_none (Nx_cuda_device.of_device Nx_device.host)))

let context d = Nx_cuda_device.context (Option.get (Nx_cuda_device.of_device d))

let memory =
  group "memory"
    [
      test
        "device memory is not the host's, host memory is page-locked and \
         addressed, and both count" (fun () ->
          let d = cuda () in
          let s0 = Nx_device.stats d in
          let b = B.create d S.Float32 250 in
          equal ~msg:"GPU memory the host does not address" (option nativeint)
            None (hosted b);
          let p = B.create ~memory:Pinned d S.UInt8 100 in
          is_true ~msg:"pinned memory the host addresses" (hosted p <> None);
          equal int 1100
            Nx_device.Stats.(allocated (diff s0 (Nx_device.stats d)));
          ignore (Sys.opaque_identity (b, p)));
      test
        "an allocation past the budget, or past the GPU's memory, raises \
         Out_of_memory" (fun () ->
          let d = cuda () in
          let budget = Nx_device.budget d in
          Fun.protect ~finally:(fun () -> Nx_device.set_budget d budget)
          @@ fun () ->
          raises_match (out_of_memory d) (fun () ->
              B.create d S.UInt8 (budget + 1));
          Nx_device.set_budget d max_int;
          raises_match (out_of_memory d) (fun () ->
              B.create d S.UInt8 (budget + 1)));
    ]

let formats =
  Gen.of_list
    ~pp:(fun ppf s -> Format.pp_print_string ppf (S.to_string s))
    S.[ UInt8; Int4; BFloat16; Float32; Float64; Complex128 ]

let copies =
  group "copies"
    [
      prop
        "a copy there and back is the identity, through a view at any aligned \
         offset, and counts its bytes"
        (Gen.triple formats
           (Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 3; 17; 1000 ])
           (Gen.int_range 0 3))
        (fun (s, n, k) ->
          let d = cuda () in
          let o = k * Int.max 1 (S.bitsize s / 8) in
          let bytes = ((n * S.bitsize s) + 7) / 8 in
          let b = B.view (B.create d S.UInt8 (o + bytes + 5)) ~offset:o s n in
          let s0 = Nx_device.stats d in
          B.copy ~src:(host_of bytes (fun i -> i + k)) ~dst:b;
          holds b (fun i -> i + k);
          equal (pair int int) (bytes, bytes)
            Nx_device.Stats.(
              let d = diff s0 (Nx_device.stats d) in
              (bytes_in d, bytes_out d)));
      test
        "a copy of three staging slots takes three timeline values each way, \
         into a view at an offset" (fun () ->
          let d = cuda () in
          let n = (130 * mib) + 7 in
          let pattern i = (i * 7) + (i lsr 20) in
          let b =
            B.view (B.create d S.UInt8 (n + 1000)) ~offset:1000 S.UInt8 n
          in
          equal ~msg:"in" int 3
            (values d (fun () -> B.copy ~src:(host_of n pattern) ~dst:b));
          let back = B.create Nx_device.host S.UInt8 n in
          equal ~msg:"out" int 3 (values d (fun () -> B.copy ~src:b ~dst:back));
          holds back pattern);
      test
        "a copy from page-locked memory, between device memory and into mapped \
         host memory takes one timeline value" (fun () ->
          let d = cuda () in
          let n = 100 * mib in
          let pinned = B.create ~memory:Pinned d S.UInt8 n in
          B.copy ~src:(host_of n (fun i -> i * 3)) ~dst:pinned;
          let b = B.create d S.UInt8 n and b' = B.create d S.UInt8 n in
          let mapped = B.create Nx_device.host S.UInt8 n in
          let bm = borrow d mapped in
          equal (list int) [ 1; 1; 1 ]
            [
              values d (fun () -> B.copy ~src:pinned ~dst:b);
              values d (fun () -> B.copy ~src:b ~dst:b');
              values d (fun () -> B.copy ~src:b' ~dst:mapped);
            ];
          holds mapped (fun i -> i * 3);
          ignore (Sys.opaque_identity bm));
      test
        "a copy to another GPU is one transfer by the source, and one from its \
         page-locked memory is direct" (fun () ->
          let d, d1 = two () in
          equal string "CUDA:1" (Nx_device.name d1);
          let src = B.create d S.UInt8 1000
          and dst = B.create d1 S.UInt8 1000 in
          B.copy ~src:(host_of 1000 Fun.id) ~dst:src;
          equal ~msg:"transfer" int 1 (values d (fun () -> B.copy ~src ~dst));
          holds dst Fun.id;
          let pinned = B.create ~memory:Pinned d1 S.UInt8 1000 in
          B.copy ~src:dst ~dst:pinned;
          let on_d = B.create d S.UInt8 1000 in
          equal ~msg:"from page-locked memory" int 1
            (values d (fun () -> B.copy ~src:pinned ~dst:on_d));
          holds on_d Fun.id);
      test "a copy runs from a domain whose thread has no context current"
        (fun () ->
          let d = cuda () in
          let b = B.create d S.UInt8 64 in
          Domain.join
            (Domain.spawn (fun () ->
                 B.copy ~src:(host_of 64 (fun i -> 64 - i)) ~dst:b));
          holds b (fun i -> 64 - i));
      test "two domains copy to two GPUs at once" (fun () ->
          let d, d1 = two () in
          let copy_on dev k =
            Domain.spawn (fun () ->
                let b = B.create dev S.UInt8 (8 * mib) in
                B.copy ~src:(host_of (8 * mib) (fun i -> i + k)) ~dst:b;
                differs b (fun i -> i + k))
          in
          let a = copy_on d 1 and b = copy_on d1 2 in
          equal
            (pair (option int) (option int))
            (None, None)
            (Domain.join a, Domain.join b));
    ]

(* Host memory whose pages the first page of a borrowed buffer shares: the
   driver refuses to register it again while that registration lives. *)
let overlapping hb =
  B.of_bigarray (Bigarray.Array1.sub (bytes_of hb) pages pages)

let borrowing =
  group "borrowing"
    [
      test
        "a borrow maps the whole host memory once, until its last borrow is \
         collected" (fun () ->
          let d = cuda () in
          let hb = host_of (2 * pages) Fun.id in
          (fun () ->
            let whole = borrow d hb in
            let tail = borrow d (B.view hb ~offset:pages S.UInt8 16) in
            equal ~msg:"the host's bytes" (option nativeint)
              (Some (B.address hb))
              (hosted whole);
            let b = B.create d S.UInt8 16 in
            B.copy ~src:tail ~dst:b;
            holds ~msg:"read through the mapping" b (fun i -> pages + i);
            refused ~sub:"does not start on a page" (fun () ->
                B.borrow d
                  (B.of_bigarray (Bigarray.Array1.sub (bytes_of hb) 1 16)));
            refused ~sub:"cannot map" (fun () -> B.borrow d (overlapping hb)))
            ();
          collect d;
          is_true ~msg:"unmapped with its last borrow"
            (B.is_borrowed (borrow d (overlapping hb))));
      test
        "a host buffer borrowed on two GPUs shares one registration, released \
         when both devices' borrows are collected" (fun () ->
          let d, d1 = two () in
          let hb = host_of (2 * pages) Fun.id in
          (fun () ->
            let on_d = borrow d hb and on_d1 = borrow d1 hb in
            let a = B.create d S.UInt8 16 and b = B.create d1 S.UInt8 16 in
            B.copy ~src:(B.view on_d ~offset:0 S.UInt8 16) ~dst:a;
            B.copy ~src:(B.view on_d1 ~offset:pages S.UInt8 16) ~dst:b;
            holds ~msg:"on the first" a Fun.id;
            holds ~msg:"on the second" b (fun i -> pages + i))
            ();
          collect d;
          refused ~sub:"cannot map" (fun () -> B.borrow d (overlapping hb));
          collect d1;
          is_true ~msg:"unregistered with the last device's borrows"
            (B.is_borrowed (borrow d (overlapping hb))));
      test
        "of_address is a borrowed view of another library's allocation, which \
         the host addresses when it is page-locked" (fun () ->
          let d = cuda () in
          let ctx = context d in
          let a = foreign_alloc ctx 64 in
          let b = Nx_cuda_device.of_address d a S.Int32 16 in
          is_true ~msg:"borrowed" (B.is_borrowed b);
          B.copy ~src:(host_of 64 Fun.id) ~dst:b;
          holds b Fun.id;
          let inner =
            Nx_cuda_device.of_address d (Nativeint.add a 16n) S.Int32 4
          in
          equal ~msg:"the allocation and the offset" (pair nativeint int) (a, 16)
            (Nx_device.Driver.Region.(handle (of_buffer inner)), B.offset inner);
          holds ~msg:"inner bytes" inner (fun i -> 16 + i);
          let host, device = foreign_host_alloc ctx 64 in
          equal ~msg:"page-locked memory" (option nativeint) (Some host)
            (hosted (Nx_cuda_device.of_address d device S.UInt8 64)));
      cases
        ~name:(fun (name, _, _) -> name)
        "of_address refuses, with the reason"
        [
          ( "elements past the allocation",
            "do not fit",
            fun d h ->
              Nx_cuda_device.of_address d (foreign_alloc h 64) S.Int32 17 );
          ( "an address the driver knows no memory at",
            "knows no memory",
            fun d _ -> Nx_cuda_device.of_address d 0x1000n S.UInt8 1 );
          ( "memory of another context",
            "not memory of",
            fun d _ ->
              Nx_cuda_device.of_address d (other_context_alloc 0 64) S.UInt8 64
          );
        ]
        (fun (_, why, f) ->
          let d = cuda () in
          let h = context d in
          raises_match (Exn.invalid_arg ~substring:why) (fun () -> f d h));
    ]

(* Writes [2 * i] into element [i] of its argument, for 256 threads. *)
let ptx =
  {|.version 7.0
.target sm_50
.address_size 64
.visible .entry double_index(.param .u64 out) {
  .reg .u32 %r<3>;
  .reg .u64 %rd<4>;
  ld.param.u64 %rd1, [out];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r1, %tid.x;
  shl.b32 %r2, %r1, 1;
  mul.wide.u32 %rd3, %r1, 4;
  add.s64 %rd2, %rd2, %rd3;
  st.global.u32 [%rd2], %r2;
  ret;
}
|}

(* Launches [double_index] over the address [out] as timeline work that touches
   [touches]. *)
let double_index d ~touches out =
  let c = Option.get (Nx_cuda_device.of_device d) in
  let f = program d ~binary:ptx ~name:"double_index" in
  Nx_device.submit [ d ] ~touches (fun s ->
      launch (Nx_cuda_device.context c) (Nx_cuda_device.compute c)
        (B.address (Nx_device.signal_word d))
        (Nx_device.Program.handle f)
        out
        (Nx_device.Submission.value s d))

let programs =
  group "programs"
    [
      test
        "a module's function loads once, and the driver's error names a bad \
         module or a missing function" (fun () ->
          let d = cuda () in
          let f = program d ~binary:ptx ~name:"double_index" in
          is_true (program d ~binary:ptx ~name:"double_index" == f);
          refused ~sub:"CUDA: CUDA_ERROR_" (fun () ->
              Nx_device.Program.load d ~binary:"not a module" ~name:"f");
          refused ~sub:"CUDA_ERROR_NOT_FOUND" (fun () ->
              Nx_device.Program.load d ~binary:ptx ~name:"missing"));
      test "a kernel launched through submit computes, and signals its value"
        (fun () ->
          let d = cuda () in
          let out = B.create d S.Int32 256 in
          double_index d ~touches:[ out ] (B.address out);
          Nx_device.synchronize d;
          equal int (Nx_device.submitted d) (Nx_device.signaled d);
          let host = B.create Nx_device.host S.Int32 256 in
          B.copy ~src:out ~dst:host;
          let ba = B.bigarray Bigarray.int32 host in
          equal (array int32)
            (Array.init 256 (fun i -> Int32.of_int (2 * i)))
            (Array.init 256 (Bigarray.Array1.get ba)));
      test "the signal word is page-locked memory of the device" (fun () ->
          let d = cuda () in
          let t = Nx_device.signal_word d in
          is_true (Nx_device.equal (B.device t) d);
          Nx_device.synchronize d;
          let words = B.create Nx_device.host S.UInt64 1 in
          B.copy ~src:t ~dst:words;
          equal int (Nx_device.signaled d)
            (Int64.to_int (B.bigarray Bigarray.int64 words).{0}));
    ]

(* A kernel that writes to address 0 faults the device, which is lost for good
   with the driver's error. It runs last: the fault ends the context. *)
let fault =
  test "a fault loses the device with the driver's error" (fun () ->
      let d = cuda () in
      let b = B.create d S.UInt8 8 in
      double_index d ~touches:[] 0n;
      let illegal = function
        | Nx_device.Lost (d', why) ->
            d' == d && String.starts_with ~prefix:"CUDA_ERROR_" why
        | _ -> false
      in
      raises_match illegal (fun () -> Nx_device.synchronize d);
      raises_match illegal (fun () -> B.create d S.UInt8 1);
      raises_match illegal (fun () -> differs b (fun _ -> 0)))

(* The entry points a batch's host program calls. *)
let entry_points =
  [
    "cuCtxSetCurrent";
    "cuLaunchKernel";
    "cuMemcpyAsync";
    "cuStreamWaitValue64_v2";
    "cuStreamWriteValue64_v2";
    "cuLaunchHostFunc";
  ]

let submitters =
  group "submitters"
    [
      test
        "driver_function finds every entry point of a loaded driver, and none \
         without one" (fun () ->
          let found = List.map Nx_cuda_device.driver_function entry_points in
          if List.for_all Option.is_none found then
            equal int ~msg:"no driver, no device" 0 (List.length gpus)
          else List.iter (is_some ~msg:"an entry point") found;
          is_none ~msg:"no such entry point"
            (Nx_cuda_device.driver_function "cuNoSuchEntryPoint"));
      test "stamp is a host function's address" (fun () ->
          not_equal nativeint 0n Nx_cuda_device.stamp);
    ]

let () =
  exit
    (run "nx.cuda.device"
       [
         opening;
         submitters;
         memory;
         copies;
         borrowing;
         programs;
         group "profiles" (Nx_test.Profiles.copies gpus);
         group "nx" (Nx_test.Runtimes.laws gpus);
         fault;
       ])
