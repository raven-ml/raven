(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What a compiled call does where it cannot compile. An operation the compiler
   does not lower is evaluated, to eager's values. A compiler that is missing or
   broken raises its own failure at the call, on the host and on a GPU: the
   tests tagged no-clang run where CC names no program, no-nvrtc where no NVRTC
   is found, and broken-nvrtc where NVRTC_PATH names a file that is no
   library. *)

open Windtrap
open Rune_support
open Witness

let f64 = Nx.float64

let has_infix ~affix s =
  let n = String.length affix in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = affix || at (i + 1))
  in
  at 0

(* Evaluated *)

type unlowered =
  | Unlowered :
      string * (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) * ('v, 's, 'd) Nx.t
      -> unlowered

(* Operations the compiler does not lower, each a function and its argument on
   the host. *)
let unlowered =
  let a = Nx.create f64 [| 2; 2 |] [| 2.; 1.; -1.; 3. |] in
  let z =
    Nx.create Nx.complex64 [| 3 |]
      Complex.[| { re = 1.; im = 2. }; { re = -0.5; im = 0. }; one |]
  in
  [
    Unlowered ("Eig_values", Nx.eigvals Nx.complex128, a);
    Unlowered ("a complex Prod", (fun z -> Nx.prod z), z);
  ]

let evaluated (Unlowered (_, f, x)) =
  let eager = codes (f x) in
  let before = Rune.evaluations () in
  let got = codes (Rune.jit' f x) in
  equal ~msg:"evaluated" int 1 (Rune.evaluations () - before);
  equal codes_t eager got

(* Compiled *)

(* [f ()], whose compiled calls must compile: none is evaluated. *)
let compiled f =
  let before = Rune.evaluations () in
  ignore (f ());
  equal ~msg:"evaluated" int 0 (Rune.evaluations () - before)

(* A compiled call that adds to a total, under reverse mode: its known part
   threads the total's running value. *)
let total_under_grad () =
  let t = Rune.Total.make () in
  let x = Nx.create f64 [| 3 |] [| 1.; 2.; 3. |] in
  let g =
    Rune.jit' (fun x ->
        Rune.Total.add t (Nx.sum x);
        Nx.mul x x)
  in
  let loss x =
    let y, total =
      Rune.Total.collect t ~init:(Nx.zeros f64 [||]) (fun () -> g x)
    in
    Nx.add (Nx.sum y) total
  in
  compiled (fun () -> Nx.to_array (Rune.grad' loss x))

(* The derivative of a scan whose step gives back its carry: the carry and its
   tangent each store a value into themselves. *)
let scan_carried_back () =
  let p = Nx.Ptree.(pair tensor tensor) in
  let step c _ = (Nx.neg (Nx.neg c), c) in
  let scanned (c, xs) = Rune.scan' step ~init:c xs in
  let f =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> tensor @-> tensor @-> returns (pair p p))
      (fun c xs tc txs -> Rune.jvp p p scanned (c, xs) (tc, txs))
  in
  let ones s = Nx.ones Nx.float32 s in
  compiled (fun () ->
      let (c, _), (tc, _) =
        f (ones [| 2 |]) (ones [| 1; 2 |]) (ones [| 2 |]) (ones [| 1; 2 |])
      in
      (Nx.to_array c, Nx.to_array tc))

(* Missing compilers *)

let double x = Nx.add x x
let host_x () = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |]

(* The program CC names in the no-clang run, which the dune file sets. *)
let missing_clang = "/nonexistent/clang"

let without_clang () =
  raises_match
    (function
      | Tolk.Renderer.Compiler.Compile_error m ->
          has_infix ~affix:missing_clang m
      | _ -> false)
    (fun () -> Rune.jit' double (host_x ()))

(* This machine's first GPU and a set of it, where its kernels compile with
   NVRTC. *)
let nvidia =
  lazy
    (match Rig_gpus.first () with
    | Error _ -> None
    | Ok d -> (
        match (Tolk_engine.target d).Tolk.Helpers.Target.device with
        | "CUDA" | "NV" -> Some (d, Nx.devices [ d ])
        | _ -> None))

let on_nvidia () =
  match Lazy.force nvidia with
  | None -> skip ~reason:"no GPU whose kernels compile with NVRTC" ()
  | Some g -> g

let gpu_call (module S : Nx.Devices) () =
  ignore (Rune.jit' double (Nx.place S.on (host_x ())))

(* Where no NVRTC is found, making the GPU's renderer fails; a compiled call
   raises that failure, which names nvrtc. *)
let without_nvrtc () =
  let d, s = on_nvidia () in
  match Tolk_engine.renderer d with
  | _ -> skip ~reason:"NVRTC is on the machine" ()
  | exception Failure why ->
      contains ~sub:"nvrtc" why;
      raises (Failure why) (gpu_call s)

(* Where NVRTC_PATH names a file that is no library, a compile fails, naming
   nvrtc. *)
let broken_nvrtc () =
  let _, s = on_nvidia () in
  raises_match
    (function
      | Tolk.Renderer.Compiler.Compile_error m -> has_infix ~affix:"nvrtc" m
      | _ -> false)
    (gpu_call s)

let () =
  exit
    (run "Rune.jit where it cannot compile"
       [
         cases "an operation the compiler does not lower is evaluated"
           ~id:(fun (Unlowered (name, _, _)) -> name)
           unlowered evaluated;
         test "a call adding to a total under grad compiles" total_under_grad;
         test "a jit of jvp of a scan giving back its carry compiles"
           scan_carried_back;
         tag "no-clang"
         @@ test "a call on the host raises Compile_error naming CC's program"
              without_clang;
         tag "no-nvrtc"
         @@ test "a call on a GPU raises the Failure naming nvrtc" without_nvrtc;
         tag "broken-nvrtc"
         @@ test "a call on a GPU raises Compile_error naming nvrtc"
              broken_nvrtc;
       ])
