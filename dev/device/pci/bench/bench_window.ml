(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Windows over 4 KiB of process memory, as a driver reaches a GPU's registers
   and rings.

   Every row makes 1024 accesses at successive offsets aligned to their width,
   so a row's time over 1024 is one access's cost. [mapped] reads and writes a
   mapped window from OCaml, [c] stores from C through device_pci.h as a
   submission does, with [store32-bare] the same stores through a bare pointer,
   and [through] reaches a buffer through an in-process transport: the cost a
   remote or USB machine adds above its wire.

   [today/] makes the same accesses through nx.device's Mmio, whose remote
   ranges call OCaml functions where a transport calls C ones. *)

module Window = Device_pci.Window
module Mmio = Nx_device_support.Mmio

external buffer : int -> int = "test_memory"
external far : int -> int -> int = "test_far"
external store32 : Window.t -> int = "bench_store32" [@@noalloc]
external store64 : Window.t -> int = "bench_store64" [@@noalloc]
external store32_bare : int -> int = "bench_store32_bare" [@@noalloc]
external write_c : Window.t -> int = "bench_write" [@@noalloc]

let span = 4096
let accesses = 1024
let x64 = Thumper.black_box 0x0102_0304_0506_0708L
let page = String.make span 'x'

(* Window *)

let get8 w () =
  let s = ref 0 in
  for i = 0 to accesses - 1 do
    s := !s + Window.get8 w i
  done;
  !s

let set8 w () =
  for i = 0 to accesses - 1 do
    Window.set8 w i i
  done

let get32 w () =
  let s = ref 0 in
  for i = 0 to accesses - 1 do
    s := !s + Window.get32 w (i * 4 mod span)
  done;
  !s

let set32 w () =
  for i = 0 to accesses - 1 do
    Window.set32 w (i * 4 mod span) i
  done

let get64 w () =
  let s = ref 0L in
  for i = 0 to accesses - 1 do
    s := Int64.add !s (Window.get64 w (i * 8 mod span))
  done;
  Int64.to_int !s

let set64 w () =
  for i = 0 to accesses - 1 do
    Window.set64 w (i * 8 mod span) x64
  done

(* Mmio *)

let mmio_get8 m () =
  let s = ref 0 in
  for i = 0 to accesses - 1 do
    s := !s + Mmio.get8 m i
  done;
  !s

let mmio_set8 m () =
  for i = 0 to accesses - 1 do
    Mmio.set8 m i i
  done

let mmio_get32 m () =
  let s = ref 0 in
  for i = 0 to accesses - 1 do
    s := !s + Mmio.get32 m (i * 4 mod span)
  done;
  !s

let mmio_set32 m () =
  for i = 0 to accesses - 1 do
    Mmio.set32 m (i * 4 mod span) i
  done

let mmio_get64 m () =
  let s = ref 0L in
  for i = 0 to accesses - 1 do
    s := Int64.add !s (Mmio.get64 m (i * 8 mod span))
  done;
  Int64.to_int !s

let mmio_set64 m () =
  for i = 0 to accesses - 1 do
    Mmio.set64 m (i * 8 mod span) x64
  done

(* Today's remote range over a buffer of its own. *)
let mmio_far =
  let b = Bytes.make span '\000' in
  let read a n = Bytes.sub_string b (Nativeint.to_int a) n in
  let write a s =
    Bytes.blit_string s 0 b (Nativeint.to_int a) (String.length s)
  in
  Mmio.remote { read; write } 0n span

let mapped = Window.v (buffer span) span
let through = Window.through (Window.transport (far 0 span)) 0 span
let mmio = Mmio.v (Nativeint.of_int (buffer span)) span
let bench = Thumper.bench

let () =
  exit
    (Thumper.run "device_pci_window"
       [
         Thumper.group "mapped"
           [
             bench "get8" (get8 mapped);
             bench "set8" (set8 mapped);
             bench "get32" (get32 mapped);
             bench "set32" (set32 mapped);
             bench "get64" (get64 mapped);
             bench "set64" (set64 mapped);
             bench "write-4KiB" (fun () -> Window.write mapped 0 page);
           ];
         Thumper.group "c"
           [
             bench "store32" (fun () -> store32 mapped);
             bench "store64" (fun () -> store64 mapped);
             bench "store32-bare" (fun () ->
                 store32_bare (Window.address mapped));
             bench "write-4KiB" (fun () -> write_c mapped);
             bench "through-store32" (fun () -> store32 through);
           ];
         Thumper.group "through"
           [
             bench "get32" (get32 through);
             bench "set32" (set32 through);
             bench "get64" (get64 through);
             bench "set64" (set64 through);
           ];
         Thumper.group "today"
           [
             Thumper.group "mapped"
               [
                 bench "get8" (mmio_get8 mmio);
                 bench "set8" (mmio_set8 mmio);
                 bench "get32" (mmio_get32 mmio);
                 bench "set32" (mmio_set32 mmio);
                 bench "get64" (mmio_get64 mmio);
                 bench "set64" (mmio_set64 mmio);
                 bench "write-4KiB" (fun () -> Mmio.write mmio 0 page);
               ];
             Thumper.group "through"
               [
                 bench "get32" (mmio_get32 mmio_far);
                 bench "set32" (mmio_set32 mmio_far);
                 bench "get64" (mmio_get64 mmio_far);
                 bench "set64" (mmio_set64 mmio_far);
               ];
           ];
       ])
