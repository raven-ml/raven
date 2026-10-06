(* Kernels of emulated data types compiled on several domains at once, the first
   compilations of the process. *)

open Windtrap
open Tolk

let target =
  {
    Helpers.Target.device = "CPU";
    renderer = "CLANG";
    arch = "x86_64,x86-64";
    interface = "";
    indices = "";
  }

(* The binary is the source's bytes: only the code generation runs. *)
let clang =
  Renderer.with_compiler (Renderer.Compiler.v Fun.id) (Cstyle.clang target)

(* [add dt n] stores into slot 0 the sum of slots 1 and 2, of [n] elements of
   [dt]. *)
let add dt n =
  let r = Ops.range (Int n) [ 0 ] in
  let at slot = Ops.index (Call.param ~shape:[ Int n ] slot dt) [ r ] in
  let sum = Ops.add (Ops.load (at 1) []) (Ops.load (at 2) []) in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (at 0) sum) [ r ] ]

let kernels =
  List.concat_map
    (fun dt -> List.init 4 (fun i -> add dt (4 + i)))
    Dtype.(fp8s @ [ Float16; Bfloat16; Int64 ])

let () =
  exit
    (run "Tolk.Decomp_dtype, compiled in parallel"
       [
         test
           "kernels of emulated data types compile on several domains at once"
           (fun () ->
             is_true (Setting.value Setting.parallel >= 4);
             let programs =
               Worker.map (fun k -> Codegen.to_program k clang) kernels
             in
             equal int (List.length kernels) (List.length programs));
       ])
