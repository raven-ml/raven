(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Second-order transformations: Newton's method on the Rosenbrock function with
   hessian', a matrix-free Hessian-vector product with hvp', and a
   finite-difference sanity check with check_grads. *)

(* Rosenbrock: f(x) = (1 - x0)^2 + 100 (x1 - x0^2)^2, minimized at (1, 1). *)
let rosenbrock x =
  let x0 = Nx.slice [ Nx.I 0 ] x and x1 = Nx.slice [ Nx.I 1 ] x in
  let a = Nx.square (Nx.sub_s x0 1.0) in
  let b = Nx.square (Nx.sub x1 (Nx.square x0)) in
  Nx.add a (Nx.mul_s b 100.0)

(* A single float64 vector as a one-leaf Ptree.S structure, for check_grads. *)
module Vec = struct
  type t = Nx.float64_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) x = f x
  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) = f
  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) x = f x
end

let () =
  let x0 = Nx.create Nx.float64 [| 2 |] [| -1.2; 1.0 |] in

  (* Newton iteration: x <- x - H^-1 g, with the gradient from grad' and the
     full Hessian (shape [2; 2]) from hessian' (forward over reverse). *)
  Printf.printf "Newton's method on Rosenbrock, from (-1.2, 1):\n";
  let x = ref x0 in
  for i = 1 to 8 do
    let g = Rune.grad' rosenbrock !x in
    let h = Rune.hessian' rosenbrock !x in
    x := Nx.sub !x (Nx.solve h g);
    Printf.printf "  iter %d  f = %.3e\n" i (Nx.item [] (rosenbrock !x))
  done;
  Printf.printf "  minimum at %s\n\n" (Nx.to_string !x);

  (* Hessian-vector products never materialize the Hessian: hvp' f x v equals
     (hessian' f x) @ v. *)
  let v = Nx.create Nx.float64 [| 2 |] [| 0.5; -1.0 |] in
  let hv = Rune.hvp' rosenbrock x0 v in
  let hv' = Nx.matmul (Rune.hessian' rosenbrock x0) v in
  Printf.printf "hvp:           %s\n" (Nx.to_string hv);
  Printf.printf "hessian @ v:   %s\n\n" (Nx.to_string hv');

  (* check_grads compares reverse-mode gradients against central differences
     along deterministic directions. *)
  match Rune.check_grads (module Vec) rosenbrock x0 with
  | Ok () ->
      Printf.printf "check_grads: reverse mode agrees with finite differences\n"
  | Error msg -> Printf.printf "check_grads: %s\n" msg
