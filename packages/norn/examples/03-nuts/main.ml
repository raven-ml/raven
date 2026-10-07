(* The No-U-Turn sampler.

   A density maps a position, a value of your own structure whose tensors have a
   leading chain axis, to one log density per chain. NUTS runs many chains in
   lock step, warms up each chain's step size and geometry, and returns draws of
   your structure with [chain; draw] axes. *)

module D = Norn.Dist

let f64 = Nx.float64

(* The unknowns: a location and the log of a scale. *)
type 'a params = { loc : 'a; log_scale : 'a }

module Params = struct
  type 'a t = 'a params

  let walk c { loc; log_scale } =
    let open Nx.Ptree.Walk in
    let loc = field c "loc" leaf loc in
    let log_scale = field c "log_scale" leaf log_scale in
    { loc; log_scale }
end

let params : Nx.float64_t params Nx.Ptree.t =
  Nx.Ptree.instantiate (module Params)

(* Twenty observations from a normal of location 3 and scale 2. *)
let data =
  D.sample (Nx.Rng.key 0)
    (D.iid [| 20 |]
       (D.normal ~loc:(Nx.scalar f64 3.) ~scale:(Nx.scalar f64 2.)))

(* Priors loc ~ N(0, 10), log_scale ~ N(0, 2), and the likelihood. Positions
   have shape [chains]; [factors] of a [chains; 1] normal at [20] data has shape
   [chains; 20]. *)
let log_density p =
  let prior =
    Nx.add
      (D.factors
         (D.normal ~loc:(Nx.scalar f64 0.) ~scale:(Nx.scalar f64 10.))
         p.loc)
      (D.factors
         (D.normal ~loc:(Nx.scalar f64 0.) ~scale:(Nx.scalar f64 2.))
         p.log_scale)
  in
  let column x = Nx.reshape [| -1; 1 |] x in
  let likelihood =
    D.factors
      (D.normal ~loc:(column p.loc) ~scale:(Nx.exp (column p.log_scale)))
      data
  in
  Nx.add prior (Nx.sum ~axes:[ 1 ] likelihood)

let () =
  let chains = 4 in
  let start =
    { loc = Nx.zeros f64 [| chains |]; log_scale = Nx.zeros f64 [| chains |] }
  in
  let key = Nx.Rng.key 1 in
  let state = Norn.Nuts.init params log_density start in
  let state = Norn.Nuts.warmup params log_density key ~steps:200 state in
  Format.printf "step sizes after warmup: %a@." Nx.pp state.step_size;
  let _, draws, stats =
    Norn.Nuts.sample params log_density key ~draws:300 state
  in

  (* Draws have your structure, with [chain; draw] axes. *)
  let d = (draws :> Nx.float64_t params) in
  Printf.printf "posterior mean of loc %.3f, of exp log_scale %.3f\n"
    (Nx.item [] (Nx.mean d.loc))
    (Nx.item [] (Nx.mean (Nx.exp d.log_scale)));
  Printf.printf "data mean %.3f, data sd %.3f\n"
    (Nx.item [] (Nx.mean data))
    (Nx.item [] (Nx.std data));

  (* Diagnostics are values of the same structure. *)
  let rhat = Norn.Diag.rhat params draws in
  Printf.printf "R-hat of loc %.4f\n" (Nx.item [] rhat.loc);

  (* A summary table, with the transitions' statistics for its findings. *)
  Format.printf "%a@." Norn.Summary.pp (Norn.Summary.v params ~stats draws)
