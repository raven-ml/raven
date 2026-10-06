(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Six posteriordb posteriors as models, with their data and reference moments
   from golden/posteriordb.golden. Each model's latent values are a list of
   tensors, named in the order of the list by posteriordb's parameter names. *)

module M = Norn_model
module D = Norn.Dist

type reference = {
  param : string;
  index : int;
  mean : float;
  sd : float;
  mcse_mean : float;
  mcse_sd : float;
}

type t =
  | Posterior : {
      name : string;
      params : string list;
      model : (Nx.float64_t list, 'y, Nx.float64_elt) M.t;
      y : 'y;
      refs : reference list;
    }
      -> t

let latent : Nx.float64_t list Nx.Ptree.t = Nx.Ptree.list Nx.Ptree.tensor
let f64 = Nx.scalar Nx.float64

(* Golden *)

type block = { data : (string * Nx.float64_t) list; rows : reference list }

let read path =
  let ic = open_in path in
  let blocks = Hashtbl.create 8 and current = ref None in
  let close () =
    match !current with
    | None -> ()
    | Some (name, b) ->
        Hashtbl.replace blocks name
          { data = List.rev b.data; rows = List.rev b.rows }
  in
  let add f =
    match !current with
    | Some (name, b) -> current := Some (name, f b)
    | None -> failwith "posteriordb.golden: a line before any posterior"
  in
  (try
     while true do
       let l = input_line ic in
       match String.split_on_char ' ' l with
       | "#" :: _ | [ "" ] -> ()
       | [ "posterior"; name ] ->
           close ();
           current := Some (name, { data = []; rows = [] })
       | "data" :: name :: rank :: rest ->
           let rank = int_of_string rank in
           let dims =
             Array.of_list
               (List.map int_of_string
                  (List.filteri (fun i _ -> i < rank) rest))
           in
           let values =
             Array.of_list
               (List.map float_of_string
                  (List.filteri (fun i _ -> i >= rank) rest))
           in
           add (fun b ->
               {
                 b with
                 data = (name, Nx.create Nx.float64 dims values) :: b.data;
               })
       | [ "ref"; param; index; mean; sd; mcse_mean; mcse_sd ] ->
           let r =
             {
               param;
               index = int_of_string index;
               mean = float_of_string mean;
               sd = float_of_string sd;
               mcse_mean = float_of_string mcse_mean;
               mcse_sd = float_of_string mcse_sd;
             }
           in
           add (fun b -> { b with rows = r :: b.rows })
       | _ -> failwith ("posteriordb.golden: unexpected line " ^ l)
     done
   with End_of_file -> close_in ic);
  close ();
  blocks

(* Models *)

(* Eight schools, non-centred: theta ~ N(mu, tau), y ~ N(theta, sigma). *)
let eight_schools data =
  let sigma = List.assoc "sigma" data in
  let model =
    M.noncentre
      (fun p -> List.nth p 2)
      ( M.v Nx.float64 latent Nx.Ptree.tensor @@ fun () ->
        let mu = M.sample (D.normal ~loc:(f64 0.) ~scale:(f64 5.)) in
        let tau = M.sample (D.half_cauchy ~scale:(f64 5.)) in
        let theta = M.sample (D.iid [| 8 |] (D.normal ~loc:mu ~scale:tau)) in
        let y = M.sample (D.normal ~loc:theta ~scale:sigma) in
        ([ mu; tau; theta ], y) )
  in
  (model, List.assoc "y" data, [ "mu"; "tau"; "theta" ])

(* An autoregression of order 5: y_t ~ N(alpha + Σ_k beta_k y_(t-k), sigma) for
   t > 5, the earlier values given. *)
let ar_k data =
  let k = 5 in
  let y = List.assoc "y" data in
  let n = (Nx.shape y).(0) in
  let ys = Nx.to_array y in
  let lags =
    Nx.create Nx.float64
      [| n - k; k |]
      (Array.init ((n - k) * k) (fun i -> ys.(k + (i / k) - (i mod k) - 1)))
  in
  let model =
    M.v Nx.float64 latent Nx.Ptree.tensor @@ fun () ->
    let alpha = M.sample (D.normal ~loc:(f64 0.) ~scale:(f64 10.)) in
    let beta =
      M.sample (D.iid [| k |] (D.normal ~loc:(f64 0.) ~scale:(f64 10.)))
    in
    let sigma = M.sample (D.half_cauchy ~scale:(f64 2.5)) in
    let loc = Nx.add alpha (Nx.dot lags beta) in
    let y = M.sample (D.normal ~loc ~scale:sigma) in
    ([ alpha; beta; sigma ], y)
  in
  (model, Nx.shrink [| (k, n) |] y, [ "alpha"; "beta"; "sigma" ])

(* Dogs avoiding shocks: the logit of avoiding is linear in the earlier
   avoidances and shocks. *)
let dogs data =
  let y = List.assoc "y" data in
  let trials = (Nx.shape y).(1) in
  (* The counts before each trial: a cumulative sum shifted by one. *)
  let before x =
    let c = Nx.cumsum ~axis:1 x in
    Nx.pad
      [| (0, 0); (1, 0) |]
      0.
      (Nx.shrink [| (0, (Nx.shape y).(0)); (0, trials - 1) |] c)
  in
  let shocks = before y and avoided = before (Nx.rsub_s 1. y) in
  let model =
    M.v Nx.float64 latent Nx.Ptree.tensor @@ fun () ->
    let beta =
      M.sample (D.iid [| 3 |] (D.normal ~loc:(f64 0.) ~scale:(f64 100.)))
    in
    let b i = Nx.get [ i ] beta in
    let logits =
      Nx.add (b 0) (Nx.add (Nx.mul (b 1) avoided) (Nx.mul (b 2) shocks))
    in
    let y = M.sample (D.bernoulli ~logits) in
    ([ beta ], y)
  in
  (model, Nx.cast Nx.bool y, [ "beta" ])

(* Two normals mixed with weight theta on the first, of ordered means. *)
let gauss_mix data =
  let y = List.assoc "y" data in
  let n = (Nx.shape y).(0) in
  let model =
    M.v Nx.float64 latent Nx.Ptree.tensor @@ fun () ->
    let mu = M.sample (D.sorted 2 (D.normal ~loc:(f64 0.) ~scale:(f64 2.))) in
    let sigma = M.sample (D.iid [| 2 |] (D.half_normal ~scale:(f64 2.))) in
    let theta = M.sample (D.beta ~a:(f64 5.) ~b:(f64 5.)) in
    let logits = Nx.stack [ Nx.log theta; Nx.log1p (Nx.neg theta) ] in
    let y =
      M.sample
        (D.iid [| n |] (D.mixture ~logits (D.normal ~loc:mu ~scale:sigma)))
    in
    ([ mu; sigma; theta ], y)
  in
  (model, y, [ "mu"; "sigma"; "theta" ])

(* A linear regression on five correlated predictors. *)
let blr data =
  let x = List.assoc "X" data in
  let d = (Nx.shape x).(1) in
  let model =
    M.v Nx.float64 latent Nx.Ptree.tensor @@ fun () ->
    let beta =
      M.sample (D.iid [| d |] (D.normal ~loc:(f64 0.) ~scale:(f64 10.)))
    in
    let sigma = M.sample (D.half_normal ~scale:(f64 10.)) in
    let y = M.sample (D.normal ~loc:(Nx.dot x beta) ~scale:sigma) in
    ([ beta; sigma ], y)
  in
  (model, List.assoc "y" data, [ "beta"; "sigma" ])

(* Poisson counts whose log rate is a Gaussian process with a squared
   exponential kernel, non-centred. *)
let gp_pois_regr data =
  let x = List.assoc "x" data in
  let n = (Nx.shape x).(0) in
  let sq_dist =
    Nx.square (Nx.sub (Nx.reshape [| n; 1 |] x) (Nx.reshape [| 1; n |] x))
  in
  let jitter = Nx.mul_s (Nx.eye Nx.float64 n) 1e-10 in
  let model =
    M.noncentre
      (fun p -> List.nth p 2)
      ( M.v Nx.float64 latent Nx.Ptree.tensor @@ fun () ->
        let rho = M.sample (D.gamma ~concentration:(f64 25.) ~rate:(f64 4.)) in
        let alpha = M.sample (D.half_normal ~scale:(f64 2.)) in
        let cov =
          Nx.add jitter
            (Nx.mul (Nx.square alpha)
               (Nx.exp (Nx.div sq_dist (Nx.mul_s (Nx.square rho) (-2.)))))
        in
        let f =
          M.sample
            (D.mvn
               ~loc:(Nx.zeros Nx.float64 [| n |])
               ~scale_tril:(Nx.cholesky cov))
        in
        let k = M.sample (D.poisson ~rate:(Nx.exp f)) in
        ([ rho; alpha; f ], k) )
  in
  (model, Nx.cast Nx.int32 (List.assoc "k" data), [ "rho"; "alpha"; "f" ])

let all path =
  let blocks = read path in
  let posterior name build =
    let b = Hashtbl.find blocks name in
    let model, y, params = build b.data in
    Posterior { name; params; model; y; refs = b.rows }
  in
  [
    posterior "eight_schools-eight_schools_noncentered" eight_schools;
    posterior "arK-arK" ar_k;
    posterior "dogs-dogs" dogs;
    posterior "low_dim_gauss_mix-low_dim_gauss_mix" gauss_mix;
    posterior "sblrc-blr" blr;
    posterior "gp_pois_regr-gp_pois_regr" gp_pois_regr;
  ]
