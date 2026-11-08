(* Kernels of emulated data types compiled on several domains at once, the first
   compilations of the process, and kernels for targets of different operations
   lowered on several domains at once, each with its own target's rules. *)

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
  let r = Ops.range ~axis_type:Weak (Int n) [ 0 ] in
  let at slot = Ops.index (Shape.param ~shape:[ Int n ] slot dt) [ r ] in
  let sum = Ops.add (Ops.load (at 1) []) (Ops.load (at 2) []) in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (at 0) sum) [ r ] ]

let kernels =
  List.concat_map
    (fun dt -> List.init 4 (fun i -> add dt (4 + i)))
    Dtype.(fp8s @ [ Float16; Bfloat16; Int64 ])

(* Targets of different operations: [shifts] has right shifts and [exp2], and
   [plain] neither. The rendering functions are never called. *)
let code o = (o, fun _ _ -> invalid_arg "not rendered")
let shifts = Renderer.v ~code_for_op:[ code Op.Shr; code Op.Exp2 ] target
let plain = Renderer.v ~code_for_op:[ code Op.Add ] target

(* [unary dt f] stores into slot 0 [f] of slot 1, of 4 elements of [dt]. *)
let unary dt f =
  let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
  let at slot = Ops.index (Shape.param ~shape:[ Int 4 ] slot dt) [ r ] in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (at 0) (f (Ops.load (at 1) []))) [ r ] ]

let quotient =
  unary Dtype.Int32 (fun x ->
      Ops.div ~rounding:`Floor x (Ops.const (`Int (Bigint.of_int 8))))

let power = unary Dtype.Float32 Ops.exp2

let has o u =
  List.exists (fun n -> Ops.op n = o) (Ops.toposort ~calls:Enter u)

(* Each lowering, its target first made on whichever domain gets to it, keeps
   the operations its target has: a floor division by 8 is a right shift and
   [exp2] stays for [shifts], and neither for [plain]. *)
let targets_apart =
  test "targets lowered on several domains at once get their own rules"
    (fun () ->
      let lowerings =
        List.concat
          (List.init 8 (fun _ ->
               [ (shifts, true); (plain, false) ]))
      in
      let lowered =
        Worker.map
          (fun (r, has_them) ->
            ( has_them,
              Codegen.full_rewrite_to_sink ~optimize:false quotient r,
              Codegen.full_rewrite_to_sink ~optimize:false power r ))
          lowerings
      in
      List.iter
        (fun (has_them, q, p) ->
          equal bool has_them (has Op.Shr q);
          equal bool has_them (has Op.Exp2 p))
        lowered)

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
         targets_apart;
       ])
