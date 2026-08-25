(* A minimal RNN exercising the staged [Rune.scan] under jit.

   x_(t+1) = x_t *@ A + u[t] *@ B

   u is the [T; m] tensor whose leading axis the scan folds over, x is the
   carry, and the per-step output is x itself. Compares eager execution, a
   jitted forward pass, and a jitted gradient against each other. *)

open Nx

let f32 = float32

(* Single-tensor carry (the RNN state x). *)
module Car = struct
  type t = float32_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

(* Scan output: final carry and stacked states. *)
module Out = struct
  type t = { c : Nx.float32_t; xs : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { c; xs } =
    { c = f c; xs = f xs }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { c = f p.c q.c; xs = f p.xs q.xs }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { c; xs } =
    f c;
    f xs
end

(* RNN parameters: recurrence matrix, input matrix, initial state. *)
module Params = struct
  type t = { a : Nx.float32_t; b : Nx.float32_t; x0 : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { a; b; x0 } =
    { a = f a; b = f b; x0 = f x0 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { a = f p.a q.a; b = f p.b q.b; x0 = f p.x0 q.x0 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { a; b; x0 } =
    f a;
    f b;
    f x0
end

(* Parameters and the input sequence together, so everything can be an input of
   a jitted function. *)
module All = struct
  type t = {
    a : Nx.float32_t;
    b : Nx.float32_t;
    x0 : Nx.float32_t;
    u : Nx.float32_t;
  }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { a; b; x0; u } =
    { a = f a; b = f b; x0 = f x0; u = f u }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { a = f p.a q.a; b = f p.b q.b; x0 = f p.x0 q.x0; u = f p.u q.u }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { a; b; x0; u } =
    f a;
    f b;
    f x0;
    f u
end

(* One RNN step. *)
let step ~a ~b x ut =
  let x = add (matmul x a) (matmul ut b) in
  (x, x)

let rnn ~a ~b ~x0 u = Rune.scan (module Car) ~f:(step ~a ~b) ~init:x0 u

(* Loss: mean square of the accumulated states. *)
let loss ~a ~b ~x0 u =
  let _, xs = rnn ~a ~b ~x0 u in
  mean (square xs)

let to_arr t = to_array (reshape [| -1 |] (contiguous t))

let check ?(eps = 1e-5) ~msg expected actual =
  let e = to_arr expected and a = to_arr actual in
  let ok =
    Array.length e = Array.length a
    && Array.for_all2 (fun e a -> Float.abs (e -. a) <= eps) e a
  in
  Printf.printf "[%s] %s\n%!" (if ok then "OK " else "FAIL") msg;
  if not ok then
    Printf.printf "  expected: %s\n  actual:   %s\n%!"
      (String.concat "; " (Array.to_list (Array.map string_of_float e)))
      (String.concat "; " (Array.to_list (Array.map string_of_float a)))

let check_leaf ~msg f expected actual = check ~msg (f expected) (f actual)

let loss_all (p : All.t) =
  let _, xs = rnn ~a:p.a ~b:p.b ~x0:p.x0 p.u in
  mean (square xs)

let main () =
  let device = if Array.length Sys.argv > 1 then Sys.argv.(1) else "CPU" in
  Printf.printf "device: %s\n%!" device;
  let t, d, m = (5, 4, 3) in
  let a =
    create f32 [| d; d |]
      [|
        0.5;
        0.1;
        0.2;
        0.3;
        0.1;
        0.6;
        0.0;
        0.2;
        0.0;
        0.3;
        0.7;
        0.1;
        0.2;
        0.0;
        0.1;
        0.8;
      |]
  in
  let b =
    create f32 [| m; d |]
      [| 1.0; 0.0; 0.5; -0.2; 0.0; -1.0; 0.3; 0.1; 0.5; 0.5; 0.0; 1.0 |]
  in
  let x0 = create f32 [| d |] [| 0.1; -0.2; 0.3; 0.0 |] in
  let u =
    create f32 [| t; m |]
      [|
        1.0;
        0.0;
        -1.0;
        0.5;
        0.5;
        0.0;
        -0.5;
        0.2;
        0.7;
        0.3;
        -0.3;
        0.1;
        0.0;
        0.1;
        -0.1;
      |]
  in

  (* Forward: eager vs jitted. *)
  let _, eager_xs = rnn ~a ~b ~x0 u in
  let jit_rnn =
    Rune.jit2 ~device
      (module Car)
      (module Out)
      (fun u ->
        let c, xs = rnn ~a ~b ~x0 u in
        { c; xs })
  in
  let { Out.xs = jit_xs; _ } = jit_rnn u in
  check ~msg:"forward: jitted scan matches eager" eager_xs jit_xs;

  (* Gradient of the loss w.r.t. u: eager vs jitted. *)
  let loss_u u = loss ~a ~b ~x0 u in
  let jit_grad = Rune.jit' ~device (fun u -> Rune.grad' loss_u u) in
  check ~msg:"grad wrt u: jitted scan matches eager" (Rune.grad' loss_u u)
    (jit_grad u);

  (* Replay the compiled functions on fresh inputs: the loop trip count is part
     of the compiled signature, and inputs must rebind correctly. *)
  let u2 =
    create f32 [| 3; m |] [| 0.2; -0.4; 0.6; 1.0; 0.0; 0.5; -0.1; 0.1; 0.9 |]
  in
  let _, eager_xs2 = rnn ~a ~b ~x0 u2 in
  let { Out.xs = jit_xs2; _ } = jit_rnn u2 in
  check ~msg:"replay (different n): jitted scan matches eager" eager_xs2 jit_xs2;
  check ~msg:"replay grad (different n): jitted scan matches eager"
    (Rune.grad' loss_u u2) (jit_grad u2);

  (* Gradients w.r.t. the RNN parameters a, b, x0 (u captured as a constant). *)
  let params = { Params.a; b; x0 } in
  let loss_ab (p : Params.t) =
    loss_all { All.a = p.a; b = p.b; x0 = p.x0; u }
  in
  let grads_ab = Rune.grad (module Params) loss_ab params in
  let jit_grad_ab =
    Rune.jit2 ~device
      (module Params)
      (module Params)
      (fun p -> Rune.grad (module Params) loss_ab p)
  in
  let jit_grads_ab = jit_grad_ab params in
  check_leaf ~msg:"grad wrt a (u captured): jit matches eager"
    (fun g -> g.Params.a)
    grads_ab jit_grads_ab;
  check_leaf ~msg:"grad wrt b (u captured): jit matches eager"
    (fun g -> g.Params.b)
    grads_ab jit_grads_ab;
  check_leaf ~msg:"grad wrt x0 (u captured): jit matches eager"
    (fun g -> g.Params.x0)
    grads_ab jit_grads_ab;

  (* Gradients w.r.t. everything at once: a, b, x0 and u as jit inputs. *)
  let all = { All.a; b; x0; u } in
  let grads_all = Rune.grad (module All) loss_all all in
  let jit_grad_all =
    Rune.jit2 ~device
      (module All)
      (module All)
      (fun p -> Rune.grad (module All) loss_all p)
  in
  let jit_grads_all = jit_grad_all all in
  check_leaf ~msg:"grad wrt a: jit matches eager"
    (fun g -> g.All.a)
    grads_all jit_grads_all;
  check_leaf ~msg:"grad wrt b: jit matches eager"
    (fun g -> g.All.b)
    grads_all jit_grads_all;
  check_leaf ~msg:"grad wrt x0: jit matches eager"
    (fun g -> g.All.x0)
    grads_all jit_grads_all;
  check_leaf ~msg:"grad wrt u: jit matches eager"
    (fun g -> g.All.u)
    grads_all jit_grads_all;

  (* Replay with a different input sequence (different trip count). *)
  let all2 = { All.a; b; x0; u = u2 } in
  let grads_all2 = Rune.grad (module All) loss_all all2 in
  let jit_grads_all2 = jit_grad_all all2 in
  check_leaf ~msg:"replay grad wrt a (different n): jit matches eager"
    (fun g -> g.All.a)
    grads_all2 jit_grads_all2;
  check_leaf ~msg:"replay grad wrt b (different n): jit matches eager"
    (fun g -> g.All.b)
    grads_all2 jit_grads_all2;
  check_leaf ~msg:"replay grad wrt x0 (different n): jit matches eager"
    (fun g -> g.All.x0)
    grads_all2 jit_grads_all2;
  check_leaf ~msg:"replay grad wrt u (different n): jit matches eager"
    (fun g -> g.All.u)
    grads_all2 jit_grads_all2

let () = main ()
