(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type finding =
  | Rhat_high of { path : Nx.Ptree.Path.t; index : int array; rhat : float }
  | Ess_low of {
      path : Nx.Ptree.Path.t;
      index : int array;
      bulk : float;
      tail : float;
    }
  | Not_finite of { path : Nx.Ptree.Path.t; index : int array; count : int }
  | Constant of { path : Nx.Ptree.Path.t; index : int array }
  | Divergent of { count : int; total : int; regions : region list }
  | Saturated of { count : int; total : int }
  | Ebfmi_low of { chain : int; ebfmi : float }
  | Pareto_k of { khat : float; threshold : float }

and region = { path : Nx.Ptree.Path.t; index : int array; shift : float }

(* A summary holds its columns as host arrays, one element per label; [columns]
   makes them tensors. *)
type t = {
  labels : string array;
  columns : (string * float array) list;
  findings : finding list;
}

let rhat_max = 1.01
let ess_per_chain = 100.
let ess_total = 400.
let ebfmi_min = 0.3
let quantiles = [ ("q5", 0.05); ("median", 0.5); ("q95", 0.95) ]

(* Elements *)

(* One element of one tensor, with its chains. *)
type element = {
  path : Nx.Ptree.Path.t;
  index : int array;
  chains : float array array;
  draws : float array; (* pooled *)
}

let label path index =
  let name = Nx.Ptree.Path.to_string path in
  if Array.length index = 0 then if name = "" then "value" else name
  else
    let idx =
      String.concat "," (Array.to_list (Array.map string_of_int index))
    in
    Printf.sprintf "%s[%s]" name idx

(* [elements u d] is each element of [d], in walk order. *)
let elements (type u) (u : (u, 'd) Nx.Ptree.t) (d : u Draws.t) =
  let leaf path x acc =
    let one (index, chains) =
      { path; index; chains; draws = Chains.pooled chains }
    in
    List.rev_append (Array.to_list (Array.map one (Chains.leaf x))) acc
  in
  List.rev (Nx.Ptree.fold u leaf (d :> u) [])

(* [values u x] is the elements of [x], a value of [u], in walk order. *)
let values (type u) (u : (u, 'd) Nx.Ptree.t) (x : u) =
  Array.concat
    (List.rev
       (Nx.Ptree.fold u
          (fun _ x acc -> Nx.to_array (Nx.cast Nx.float64 x) :: acc)
          x []))

let sd xs = Float.sqrt (Chains.var1 xs)
let constant e = Array.for_all (fun x -> Float.equal x e.draws.(0)) e.draws

let not_finite e =
  Array.fold_left (fun n x -> if Float.is_finite x then n else n + 1) 0 e.draws

(* Findings *)

let element_findings ~ess_min ~rhat ~bulk ~tail elements =
  List.concat
    (List.mapi
       (fun i e ->
         let count = not_finite e in
         if count > 0 then
           [ Not_finite { path = e.path; index = e.index; count } ]
         else if constant e then [ Constant { path = e.path; index = e.index } ]
         else
           let r =
             if rhat.(i) > rhat_max then
               [ Rhat_high { path = e.path; index = e.index; rhat = rhat.(i) } ]
             else []
           in
           let s =
             if bulk.(i) < ess_min || tail.(i) < ess_min then
               [
                 Ess_low
                   {
                     path = e.path;
                     index = e.index;
                     bulk = bulk.(i);
                     tail = tail.(i);
                   };
               ]
             else []
           in
           r @ s)
       elements)

let transition_findings (type u f) (u : (u, 'd) Nx.Ptree.t) (s : (f, 'd) Stats.t Draws.t)
    (d : u Draws.t) elements =
  let st = (s : (f, 'd) Stats.t Draws.t :> (f, 'd) Stats.t) in
  let count b =
    Array.fold_left (fun n b -> if b then n + 1 else n) 0 (Nx.to_array b)
  in
  let total = Nx.numel st.diverging in
  let divergent =
    let n = count st.diverging in
    if n = 0 then []
    else
      let shifts = values u (Diag.divergent_shift u s d) in
      let regions =
        List.concat
          (List.mapi
             (fun i e ->
               if Float.abs shifts.(i) > 1. then
                 [ { path = e.path; index = e.index; shift = shifts.(i) } ]
               else [])
             elements)
      in
      [ Divergent { count = n; total; regions } ]
  in
  let depth =
    let n = count st.saturated in
    if n = 0 then [] else [ Saturated { count = n; total } ]
  in
  let ebfmi =
    if (Nx.shape st.energy).(1) < 2 then []
    else
      let e = Nx.to_array (Nx.cast Nx.float64 (Diag.ebfmi s)) in
      List.concat
        (List.mapi
           (fun chain v ->
             if v < ebfmi_min then [ Ebfmi_low { chain; ebfmi = v } ] else [])
           (Array.to_list e))
  in
  divergent @ depth @ ebfmi

let minimum_draws = 4

let v (type u) (u : (u, 'd) Nx.Ptree.t) ?stats ?superchains (d : u Draws.t) =
  let chains, draws =
    Nx.Ptree.fold u
      (fun _ x _ -> ((Nx.shape x).(0), (Nx.shape x).(1)))
      (d :> u)
      (0, 0)
  in
  if draws < minimum_draws then
    invalid_arg
      (Printf.sprintf
         "Norn.Summary.v: chains of %d draws; a summary needs at least %d" draws
         minimum_draws);
  (match superchains with
  | Some k when k < 2 || chains mod k <> 0 ->
      invalid_arg
        (Printf.sprintf "Norn.Summary.v: %d superchains do not divide %d chains"
           k chains)
  | _ -> ());
  let elements = elements u d in
  (* Each diagnostic of each element at float64 on the host; NaN where the
     element has none. *)
  let stat f =
    Array.of_list
      (List.map
         (fun e -> if Chains.undefined e.chains then Float.nan else f e.chains)
         elements)
  in
  let rhat =
    match superchains with
    | None -> stat (Chains.rank_rhat Chains.rhat_of)
    | Some k -> stat (Chains.nested_rank_rhat k)
  in
  let bulk = stat Chains.ess_bulk_of and tail = stat Chains.ess_tail_of in
  let column f = Array.of_list (List.map (fun e -> f e.draws) elements) in
  let columns =
    [ ("mean", column Chains.mean); ("sd", column sd) ]
    @ List.map
        (fun (name, q) -> (name, column (fun xs -> Chains.quantile xs q)))
        quantiles
    @ [
        ("mcse", stat Chains.mcse_mean_of);
        ("ess_bulk", bulk);
        ("ess_tail", tail);
        ("rhat", rhat);
      ]
  in
  let ess_min =
    match superchains with
    | None -> ess_per_chain *. float_of_int chains
    | Some _ -> ess_total
  in
  let findings =
    element_findings ~ess_min ~rhat ~bulk ~tail elements
    @
    match stats with
    | None -> []
    | Some s -> transition_findings u s d elements
  in
  {
    labels = Array.of_list (List.map (fun e -> label e.path e.index) elements);
    columns;
    findings;
  }

(* Weighted draws *)

(* [weighted_quantile xs w q] is the [q] quantile of the draws [xs] of positive
   normalised weight [w]: in order, draw [j] stands at the middle of its
   weight's span, [C_(j-1) + w_j / 2], and [q] interpolates linearly between
   the draws beside it, clamped to the first and last. Unlike the inverse of
   the cumulative weights, it moves little when the weights do. *)
let weighted_quantile xs w q =
  let order =
    Array.of_list
      (List.filter (fun i -> w.(i) > 0.) (List.init (Array.length xs) Fun.id))
  in
  Array.stable_sort (fun i j -> Float.compare xs.(i) xs.(j)) order;
  let n = Array.length order in
  let x j = xs.(order.(j)) in
  let mid = Array.make n 0. in
  let acc = ref 0. in
  Array.iteri
    (fun j i ->
      mid.(j) <- !acc +. (w.(i) /. 2.);
      acc := !acc +. w.(i))
    order;
  if n = 0 then Float.nan
  else if q <= mid.(0) then x 0
  else if q >= mid.(n - 1) then x (n - 1)
  else
    let j = ref 1 in
    while mid.(!j) < q do
      incr j
    done;
    let j = !j in
    let f = (q -. mid.(j - 1)) /. (mid.(j) -. mid.(j - 1)) in
    x (j - 1) +. (f *. (x j -. x (j - 1)))

let of_weighted (type u f) (u : (u, 'd) Nx.Ptree.t)
    (wd : (u, f, 'd) Weighted.t) =
  let lw = wd.log_weights in
  if Array.length (Nx.shape lw) <> 1 then
    invalid_arg
      (Printf.sprintf
         "Norn.Summary.of_weighted: log weights of shape [%s], not [n]"
         (String.concat "; "
            (Array.to_list (Array.map string_of_int (Nx.shape lw)))));
  let n = (Nx.shape lw).(0) in
  (* Algorithm 1 whole: the statistics weigh by the smoothed weights, and the
     raw weights' k̂ judges them. *)
  let lw, khat =
    Pareto.smooth
      ~tail:(Pareto.tail_length ~r_eff:1. n)
      (Nx.to_array (Nx.cast Nx.float64 lw))
  in
  let total = Pareto.log_sum_exp lw in
  let w = Array.map (fun l -> Float.exp (l -. total)) lw in
  let leaf path x acc =
    let s = Nx.shape x in
    if Array.length s = 0 || s.(0) <> n then
      invalid_arg
        (Printf.sprintf
           "Norn.Summary.of_weighted: %s: %s draws for %d weights"
           (Nx.Ptree.Path.to_string path)
           (if Array.length s = 0 then "no axis of" else string_of_int s.(0))
           n);
    let x = Nx.reshape (Array.append [| 1 |] s) x in
    let one (index, chains) =
      { path; index; chains; draws = Chains.pooled chains }
    in
    List.rev_append (Array.to_list (Array.map one (Chains.leaf x))) acc
  in
  let elements = List.rev (Nx.Ptree.fold u leaf wd.values []) in
  let dot f xs =
    let s = ref 0. in
    Array.iteri (fun i x -> s := !s +. f w.(i) x) xs;
    !s
  in
  (* About the first draw, so that a constant element's mean is exact and its
     spread 0 however the weights round. *)
  let mean xs =
    if Array.length xs = 0 then Float.nan
    else xs.(0) +. dot (fun w x -> w *. (x -. xs.(0))) xs
  in
  (* The weighted variance, and the variance of the weighted mean of
     independent draws (Vehtari et al. 2024, equation 5). Their ratio, the
     [ess] column, is Var_p(h) / Var(Î): the draws a plain Monte Carlo mean
     needs for the same error, which for an [h] whose spread is unrelated to
     the weights is Kish's 1 / Σ w̃² (their equation 8). Their equation 7
     puts the spread of [h] over the unweighted draws in the numerator, the
     proposal's and not the target's: on 4 draws it can give 14.5. *)
  let spread xs =
    let m = mean xs in
    ( dot (fun w x -> w *. (x -. m) *. (x -. m)) xs,
      dot (fun w x -> w *. w *. (x -. m) *. (x -. m)) xs )
  in
  let column f = Array.of_list (List.map (fun e -> f e.draws) elements) in
  let columns =
    [
      ("mean", column mean);
      ("sd", column (fun xs -> Float.sqrt (fst (spread xs))));
    ]
    @ List.map
        (fun (name, q) -> (name, column (fun xs -> weighted_quantile xs w q)))
        quantiles
    @ [
        ("mcse", column (fun xs -> Float.sqrt (snd (spread xs))));
        ( "ess",
          column (fun xs ->
              let v, e = spread xs in
              v /. e) );
      ]
  in
  let element e =
    let count = not_finite e in
    if count > 0 then [ Not_finite { path = e.path; index = e.index; count } ]
    else if n > 0 && constant e then
      [ Constant { path = e.path; index = e.index } ]
    else []
  in
  let threshold = Pareto.threshold n in
  let weights =
    if khat <= threshold then [] else [ Pareto_k { khat; threshold } ]
  in
  {
    labels = Array.of_list (List.map (fun e -> label e.path e.index) elements);
    columns;
    findings = List.concat_map element elements @ weights;
  }

let concat ss =
  match ss with
  | [] -> { labels = [||]; columns = []; findings = [] }
  | first :: _ ->
      let names s = List.map fst s.columns in
      if List.exists (fun s -> names s <> names first) ss then
        invalid_arg
          "Norn.Summary.concat: summaries of chains and of weighted draws \
           have different columns";
      {
        labels = Array.concat (List.map (fun s -> s.labels) ss);
        columns =
          List.map
            (fun (name, _) ->
              ( name,
                Array.concat (List.map (fun s -> List.assoc name s.columns) ss)
              ))
            first.columns;
        findings = List.concat_map (fun s -> s.findings) ss;
      }

let findings s = s.findings
let labels s = Array.copy s.labels

let columns s =
  List.map
    (fun (name, xs) -> (name, Nx.create Nx.float64 [| Array.length xs |] xs))
    s.columns

(* Formatting *)

let pp_finding ppf = function
  | Rhat_high { path; index; rhat } ->
      Format.fprintf ppf "R-hat of %s is %.3f, above %.2f" (label path index)
        rhat rhat_max
  | Ess_low { path; index; bulk; tail } ->
      Format.fprintf ppf
        "effective sample size of %s is %.0f in the bulk and %.0f in the \
         tails, too few"
        (label path index) bulk tail
  | Not_finite { path; index; count } ->
      Format.fprintf ppf "%s has %d draws that are not finite"
        (label path index) count
  | Constant { path; index } ->
      Format.fprintf ppf
        "%s is constant: it has no R-hat nor effective sample size"
        (label path index)
  | Divergent { count; total; regions } ->
      Format.fprintf ppf "%d of %d transitions diverged" count total;
      if regions <> [] then
        Format.fprintf ppf "; they gather at %s"
          (String.concat ", "
             (List.map
                (fun (r : region) ->
                  Printf.sprintf "%s (%+.1f sd)" (label r.path r.index) r.shift)
                regions))
  | Saturated { count; total } ->
      Format.fprintf ppf "%d of %d transitions reached their maximum length"
        count total
  | Ebfmi_low { chain; ebfmi } ->
      Format.fprintf ppf "E-BFMI of chain %d is %.2f, below %.1f" chain ebfmi
        ebfmi_min
  | Pareto_k { khat; threshold } ->
      if Float.is_nan khat then
        Format.fprintf ppf
          "the weights have no Pareto k-hat: estimates from them are not judged"
      else
        Format.fprintf ppf
          "Pareto k-hat of the weights is %.2f, above %.2f: estimates from \
           them are unreliable"
          khat threshold

let cell name x =
  if Float.is_nan x then "nan"
  else
    match name with
    | "ess_bulk" | "ess_tail" | "ess" -> Printf.sprintf "%.0f" x
    | "rhat" -> Printf.sprintf "%.2f" x
    | _ -> Printf.sprintf "%.3g" x

let pp ppf s =
  let names = List.map fst s.columns in
  let cells =
    Array.mapi
      (fun i _ -> List.map (fun (name, xs) -> cell name xs.(i)) s.columns)
      s.labels
  in
  let label_width =
    Array.fold_left (fun w l -> max w (String.length l)) 0 s.labels
  in
  let widths =
    List.mapi
      (fun j name ->
        Array.fold_left
          (fun w row -> max w (String.length (List.nth row j)))
          (String.length name) cells)
      names
  in
  let row label xs =
    Format.fprintf ppf "%-*s" label_width label;
    List.iter2 (fun w x -> Format.fprintf ppf "  %*s" w x) widths xs
  in
  Format.fprintf ppf "@[<v>";
  row "" names;
  Array.iteri
    (fun i l ->
      Format.fprintf ppf "@,";
      row l cells.(i))
    s.labels;
  List.iter (fun f -> Format.fprintf ppf "@,%a" pp_finding f) s.findings;
  Format.fprintf ppf "@]"
