(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jit on this machine's first GPU: a scatter past its work bound gives
   the host's eager values. It skips without a GPU, or without the compiler
   the GPU's kernels need. *)

open Windtrap
open Rune2_support

let gpu =
  lazy
    (match Rig_gpus.first () with
    | Error _ -> None
    | Ok d -> Some (Nx.devices [ d ]))

(* A compile error that names a compiler library or a header the machine
   lacks is the machine's: the test skips. Half floats' kernels read headers
   no other kernel does. *)
let missing m =
  let has s =
    let n = String.length s in
    let rec at i =
      i + n <= String.length m && (String.sub m i n = s || at (i + 1))
    in
    at 0
  in
  if has "cannot open source file" then
    Some ("a header is missing, set CUDA_PATH: " ^ m)
  else if has "NVRTC_PATH" || has "COMGR_PATH" then Some m
  else None

let compiles (module S : Nx.Devices) =
  let neg (Nx.Dtype.Any dt) =
    let x = Nx.cast dt (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]) in
    ignore (Rune.jit' Nx.neg (Nx.place S.on x))
  in
  List.iter
    (fun d ->
      match neg d with
      | () -> ()
      | exception (Tolk.Renderer.Compiler.Compile_error m as e) -> (
          match missing m with
          | Some why -> skip ~reason:why ()
          | None -> raise e))
    Nx.Dtype.[ Any Nx.float32; Any Nx.float16; Any Nx.bfloat16 ]

let on_gpu () =
  match Lazy.force gpu with
  | None -> skip ~reason:"no GPU" ()
  | Some s ->
      compiles s;
      s

let law c = Scatters.law (on_gpu ()) c

let () =
  exit
    (run "Rune.jit on a GPU"
       [
         group "scatters"
           [
             prop ~count:4 ~examples:Scatters.examples
               "a scatter past 16 times eager's work compiles to eager's values"
               (Gen.with_pp Scatters.pp Scatters.gen)
               law;
           ];
       ])
