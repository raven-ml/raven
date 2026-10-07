(* Derivatives, compilation and batching.

   Every jera program is eager and runs unchanged under rune: [grad]
   differentiates a solve through its answer's equation, [jit] compiles it, and
   [vmap] gives each lane its own problem and status. *)

open Jera

let f64 = Nx.float64
let s x = Nx.scalar f64 x

let row x =
  String.concat " "
    (Array.to_list (Array.map (Printf.sprintf "%.12f") (Nx.to_array x)))

(* y' = -k y from y(0) = 1, solved to t = 1: y(1) = exp(-k). *)
let decay k =
  Ode.solve Nx.Ptree.tensor Ode.tsit5
    ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12)
    ~budget:200
    (fun _t y -> Nx.mul (Nx.neg k) y)
    ~t0:(s 0.) ~t1:(s 1.) (s 1.)
  |> Solution.get

(* The integral of exp(a x) over [0, 1], (exp a - 1) / a. *)
let integral a =
  Quad.adaptive (Quad.Rule.kronrod 7)
    ~tol:(Tol.v ~rel:1e-12 ~abs:1e-14)
    ~budget:50
    (fun x -> Nx.exp (Nx.mul a x))
    (Quad.Range.v (s 0.) (s 1.))
  |> Solution.get

let () =
  let k = s 0.7 in
  Printf.printf "%-34s %s\n" "d exp(-k) / dk, through the ODE"
    (row (Rune.grad' decay k));
  Printf.printf "%-34s %s\n" "  exact, -exp(-k)" (row (s (-.exp (-0.7))));

  let a = s 2. in
  Printf.printf "%-34s %s\n" "d integral / da, by quadrature"
    (row (Rune.grad' integral a));
  Printf.printf "%-34s %s\n" "  exact"
    (row (s (((exp 2. *. (2. -. 1.)) +. 1.) /. 4.)));

  (* Compiled: the first call traces and compiles, later ones replay. *)
  let compiled = Rune.jit Nx.Ptree.(tensor @-> returns tensor) decay in
  Printf.printf "%-34s %s\n" "compiled y(1) at k = 0.7" (row (compiled k));
  Printf.printf "%-34s %s\n" "compiled y(1) at k = 1.5" (row (compiled (s 1.5)));

  (* Batched: one system x^3 + x = c per lane, each with its own status. A solve
     returns its whole answer, statuses included, as a structure. *)
  let solve c =
    System.solve Nx.Ptree.tensor System.broyden ~linear:Linear.dense
      ~tol:(Tol.rel 1e-12) ~budget:30
      (fun x -> Nx.sub (Nx.add (Nx.mul x (Nx.square x)) x) c)
      (Nx.zeros f64 [| 1 |])
  in
  let c = Nx.create f64 [| 4; 1 |] [| 0.; 2.; 10.; 30. |] in
  let sol =
    Rune.vmap Nx.Ptree.(tensor @-> returns (Solution.ptree tensor)) solve c
  in
  Printf.printf "%-34s %s\n" "x^3 + x = c for c = 0 2 10 30"
    (row (Solution.get sol));
  Format.printf "%-34s %a@." "evaluations per lane" Nx.pp
    (Solution.evaluations sol)
