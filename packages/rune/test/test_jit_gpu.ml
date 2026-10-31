(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jit on this machine's first GPU: a scatter past its work bound gives
   the host's eager values, a call's check raises at the next wait, so the
   call returns while its work runs, and a program over the GPU and a memory
   device, devices of two targets, is evaluated. It skips without a GPU, or
   without the compiler the GPU's kernels need. *)

open Windtrap
open Rune_support

let gpu =
  lazy
    (match Rig_gpus.first () with
    | Error _ -> None
    | Ok d -> Some (d, Nx.devices [ d ]))

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
  | Some (d, s) ->
      compiles s;
      (d, s)

let law c = Scatters.law (snd (on_gpu ())) c

exception Big of int array * float

(* A step whose work is in flight when the call returns. An iterate's stop is
   read on the host each trip, so the work that stays in flight is the work
   after it: [products] products of [n; n] matrices, after a check of the first
   one, which fails at [0; 0]. They are 4.4 TFLOP: at least 68 ms at the RTX
   5000 Ada's float32 peak of 65 TFLOPS, tens of milliseconds on any current
   GPU, while the host reads the timeline microseconds after the call's last
   submit, so the GPU cannot finish them before the read. Each product by
   [w], whose elements are [1/n], keeps the values [n], exactly. *)
let n = 4096
let products = 32

let test_deferred () =
  let d, (module S) = on_gpu () in
  let step x =
    let x =
      Rune.iterate' ~max:4
        ~until:(fun x -> Nx.greater (Nx.sum x) (Nx.scalar_like (Nx.sum x) 0.))
        (fun x -> Nx.add x (Nx.scalar_like x 1.))
        x
    in
    let y = Nx.matmul x x in
    Nx.check Nx.Ptree.tensor
      (Nx.less y (Nx.scalar_like y (float_of_int n /. 2.)))
      y
      (fun i v -> Big (i, Nx.item [] v));
    let w = Nx.div x (Nx.scalar_like x (float_of_int n)) in
    let h = ref y in
    for _ = 1 to products do
      h := Nx.matmul !h w
    done;
    !h
  in
  let x = Nx.place S.on (Nx.zeros Nx.float32 [| n; n |]) in
  let g = Rune.jit' step in
  let before = Rune.evaluations () in
  (* The first call compiles. *)
  (try Nx.sync (g x) with Big _ -> ());
  let y = g x in
  let signaled = Rig.signaled d and submitted = Rig.submitted d in
  equal ~msg:"compiled" Testable.int 0 (Rune.evaluations () - before);
  less ~msg:"the call returned before its work completed" Testable.int
    ~than:submitted signaled;
  raises ~msg:"the next wait raises the check"
    (Big ([| 0; 0 |], float_of_int n))
    (fun () -> Nx.sync y)

(* A program over the GPU and a memory device lies on devices of two targets,
   so it is evaluated, as nx evaluates it: the set has no kernels, so the copy
   of its result raises, as rune.mli's evaluated programs on such a set do. *)
let mixed () =
  match Rig_gpus.first () with
  | Error _ -> skip ~reason:"no GPU" ()
  | Ok gpu ->
      let memory = Result.get_ok (Rig.memory_device "rune-jit-gpu-mixed") in
      let module S = (val Nx.devices [ gpu; memory ]) in
      let x =
        Nx.place S.on
          (Nx.create Nx.float32 [| 4; 2 |] (Array.init 8 Float.of_int))
      in
      let f x = Nx.place (S.split ~axis:0) x in
      let before = Rune.evaluations () in
      raises_match
        (Exn.invalid_arg ~sub:"has no kernels")
        (fun () -> ignore (Rune.jit' f x));
      equal ~msg:"evaluated" int 1 (Rune.evaluations () - before)

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
         group "checks"
           [
             test "a call's check raises at the next wait, not at the call"
               test_deferred;
           ];
         group "several targets"
           [
             test "a program over a GPU and a memory device is evaluated"
               mixed;
           ];
       ])
