(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Uniform → Ptree.S bridge: differentiation and JIT through Ptree.Make. *)

open Windtrap
open Rune_test_support.Support

(* Hand-written uniform structure, copied from the nx test so this test is
   self-contained. *)
module U = struct
  type 'a t = { w : 'a; b : 'a }

  let map (f : 'a -> 'b) { w; b } = { w = f w; b = f b }
  let map2 (f : 'a -> 'b -> 'c) a b = { w = f a.w b.w; b = f a.b b.b }

  let iter (f : 'a -> unit) { w; b } =
    f w;
    f b

  let fold (f : string -> 'acc -> 'a -> 'acc) acc { w; b } =
    f "b" (f "w" acc w) b

  let fold2 (f : string -> 'acc -> 'a -> 'b -> 'acc) acc a b =
    f "b" (f "w" acc a.w b.w) a.b b.b
end

module T = Nx.Ptree.Make (U)

let pack x = Nx.Ptree.P x

let params () =
  { U.w = pack (vec32 [| 1.0; -2.0; 3.0 |]); b = pack (vec32 [| 0.5 |]) }

(* Loss: sum(w^2) + 2*sum(b). Scalar output to use Rune.grad. *)
let loss (t : T.t) : Nx.float32_t =
  let { U.w = Nx.Ptree.P w; b = Nx.Ptree.P b } = t in
  let w = as_f32 w and b = as_f32 b in
  Nx.add (Nx.sum (Nx.mul w w)) (Nx.mul_s (Nx.sum b) 3.0)

let test_grad_over_uniform () =
  let p : T.t = params () in
  let g = Rune.grad (module T) loss p in
  match g with
  | { U.w = Nx.Ptree.P gw; b = Nx.Ptree.P gb } ->
      check_arr ~msg:"dw" [| 2.0; -4.0; 6.0 |] (as_f32 gw);
      check_arr ~msg:"db" [| 3.0 |] (as_f32 gb)

let test_value_and_grad () =
  let p : T.t = params () in
  let v, g = Rune.value_and_grad (module T) loss p in
  (* loss = 1+4+9 + 3*0.5 = 14 + 1.5 = 15.5 *)
  equal ~msg:"value" (float 1e-5) 15.5 (scalar v);
  match g with
  | { U.w = Nx.Ptree.P gw; _ } ->
      check_arr ~msg:"dw" [| 2.0; -4.0; 6.0 |] (as_f32 gw)

let test_jit_cache () =
  let p : T.t = params () in
  let f_jit = Rune.jit (module T) loss in
  let v1 = f_jit p in
  let v2 = f_jit p in
  equal ~msg:"cached jit" (float 1e-5) 15.5 (scalar v1);
  equal ~msg:"cached jit again" (float 1e-5) 15.5 (scalar v2)

let test_uniform_fold_paths_in_grad () =
  (* Fold over the gradient tree with paths, verifying the paths are the same as
     the structure's own fold would give. *)
  let g = Rune.grad (module T) loss (params ()) in
  let paths = ref [] in
  let _ =
    U.fold
      (fun path acc _ ->
        paths := path :: !paths;
        acc)
      0 g
  in
  let paths = List.rev !paths in
  equal ~msg:"leaf count" int 2 (List.length paths);
  equal ~msg:"first path" string "w" (List.nth paths 0);
  equal ~msg:"second path" string "b" (List.nth paths 1)

let tests =
  [
    group "differentiation"
      [
        test "grad works over a uniform-backed Ptree.Make"
          test_grad_over_uniform;
        test "value_and_grad returns correct value and gradient"
          test_value_and_grad;
      ];
    group "JIT"
      [ test "jit-cached loss function returns correct values" test_jit_cache ];
    group "fold"
      [
        test "fold over gradient tree produces correct paths"
          test_uniform_fold_paths_in_grad;
      ];
  ]

let () = run "rune uniform" tests
