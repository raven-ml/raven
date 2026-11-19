(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* npz.exe COMMAND FILE...: what main.exe wrote.

   [entries F] prints each entry's name, dtype and shape; [ints F NAME] an
   int64 entry's values, a line per vector of its last axis; [equal F G NAME]
   whether the int64 entries NAME of F and G are equal; [sums F] whether each
   object's probabilities over each token's classes, p and p_1024 where there
   is one, sum to 1 within 1e-6; [same NAME F I G J [TOL]] whether row I of
   F's entry NAME equals row J of G's within TOL (1e-5 by default) of its
   largest magnitude; [upper F
   CACHE KEY] whether F's upper is, for each class k of each channel i, edge k
   of codebook i in the Hub cache CACHE, through the inverse of KEY's transform
   (tok_z's the identity, an XP spectrum's sinh twice), within 1e-6 relative;
   [values F CACHE KEY] whether F's values are its tokens decoded so, NaN for a
   token past the codebook's edges; [posterior F I R] whether object I's p, and
   p_1024 where there is one, are R's p, the softmax of the decoder's logits,
   within 1e-6; [chi2 G P T] whether the one-token draws of G, made at
   temperature T, have the frequencies that P's posterior of the same objects
   gives, by Pearson's test.

   The test pools each object's classes, in order, into bins of expected count
   at least 5, the last bin taking what remains, and sums the statistic and its
   degrees of freedom over the objects. A draw at temperature T has
   probability proportional to p^(1/T). The draws pass below the statistic's
   1 - 1e-3 quantile, so a correct sampler fails for one seed in a thousand:
   the seed is fixed, so the test is deterministic. *)

open Aion

let load = Nx_io.load_npz
let get dt name a = Nx.unpack dt (Option.get (Nx_io.Archive.find name a))

let row x i =
  let x = Nx.cast Nx.float64 (Nx.slice [ Nx.I i ] x) in
  Nx.to_array x

(* Codebook i's edges of KEY in the Hub cache [cache], as values: through the
   inverse of KEY's transform. *)
let codebooks cache key =
  let m, inverse =
    match key with
    | "tok_z" -> (Modality.Z, Fun.id)
    | "tok_xp_bp" -> (Modality.Xp_bp, fun e -> sinh (sinh e))
    | "tok_xp_rp" -> (Modality.Xp_rp, fun e -> sinh (sinh e))
    | k -> failwith ("no transform for " ^ k)
  in
  let file = Hub.file (Scalar m) in
  let a =
    Nx_io.load_safetensors
      (Kaun_hf.cache_path ~cache_dir:cache ~revision:Hub.revision ~file Hub.repo)
  in
  let books =
    if m = Modality.Z then [ "_quantizer.buckets" ]
    else List.init 55 (Printf.sprintf "_quantizer.quantizers.%d._index_to_val")
  in
  List.map
    (fun b ->
      Array.map inverse (Nx.to_array (Nx.cast Nx.float64 (get Nx.float32 b a))))
    books
  |> Array.of_list

let close got want =
  Float.abs (got -. want) <= 1e-6 *. Float.max 1. (Float.abs want)

(* The upper [alpha] quantile of chi-squared with [df] degrees of freedom: twice
   that of the gamma distribution of shape [df / 2]. *)
let chi2_upper alpha df =
  let shape = Nx.scalar Nx.float64 (Float.of_int df /. 2.) in
  2. *. Nx.item [] (Nx.gammainccinv shape (Nx.scalar Nx.float64 alpha))

(* Pearson's statistic of [draws] against the probabilities [p] and its degrees
   of freedom, the classes pooled in order into bins of expected count at least
   5. *)
let pearson p draws =
  let m = Float.of_int (Array.length draws) in
  let counts = Array.make (Array.length p) 0. in
  Array.iter (fun d -> counts.(d) <- counts.(d) +. 1.) draws;
  let bins = ref [] and e = ref 0. and o = ref 0. in
  Array.iteri
    (fun v pv ->
      e := !e +. (m *. pv);
      o := !o +. counts.(v);
      if !e >= 5. then begin
        bins := (!o, !e) :: !bins;
        e := 0.;
        o := 0.
      end)
    p;
  let bins =
    match !bins with
    | (o', e') :: rest when !e > 0. || !o > 0. -> (o' +. !o, e' +. !e) :: rest
    | l -> l
  in
  ( List.fold_left (fun a (o, e) -> a +. ((o -. e) *. (o -. e) /. e)) 0. bins,
    List.length bins - 1 )

let () =
  match Array.to_list Sys.argv |> List.tl with
  | [ "entries"; f ] ->
      let a = load f in
      List.iter
        (fun n ->
          let (Nx.P x) = Option.get (Nx_io.Archive.find n a) in
          Format.printf "%s %a %a@." n Nx.pp_dtype (Nx.dtype x) Nx.pp_shape
            (Nx.shape x))
        (Nx_io.Archive.names a)
  | [ "ints"; f; name ] ->
      let x = get Nx.int64 name (load f) in
      let w = Nx.dim (-1) x in
      let x = Nx.reshape [| -1; w |] x in
      let v = Nx.to_array x in
      for i = 0 to Nx.dim 0 x - 1 do
        print_endline
          (String.concat " "
             (List.init w (fun j -> Int64.to_string v.((i * w) + j))))
      done
  | [ "sums"; f ] ->
      let a = load f in
      let p = Nx.cast Nx.float64 (get Nx.float32 "p" a) in
      let s = Nx.sum ~axes:[ -1 ] p in
      let s =
        match Nx_io.Archive.find "p_1024" a with
        | Some q -> Nx.add s (Nx.cast Nx.float64 (Nx.unpack Nx.float32 q))
        | None -> s
      in
      let worst =
        Array.fold_left
          (fun w v -> Float.max w (Float.abs (v -. 1.)))
          0. (Nx.to_array s)
      in
      if worst <= 1e-6 then
        print_endline "each object's probabilities sum to 1 within 1e-6"
      else Printf.printf "an object's probabilities sum to 1 + %g\n" worst
  | "same" :: name :: f :: i :: g :: j :: tol ->
      let tol = match tol with [ t ] -> t | _ -> "1e-5" in
      let x = row (get Nx.float32 name (load f)) (int_of_string i)
      and y = row (get Nx.float32 name (load g)) (int_of_string j) in
      let big = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. x in
      let d = ref 0. in
      Array.iteri (fun k v -> d := Float.max !d (Float.abs (v -. y.(k)))) x;
      if !d <= float_of_string tol *. big then
        Printf.printf "equal within %s\n" tol
      else Printf.printf "differ by %g of %g\n" !d big
  | [ "upper"; f; cache; key ] -> (
      let want = Array.concat (Array.to_list (codebooks cache key)) in
      let got =
        Nx.to_array (Nx.cast Nx.float64 (get Nx.float32 "upper" (load f)))
      in
      let bad = ref None in
      Array.iteri
        (fun i w -> if !bad = None && not (close got.(i) w) then bad := Some i)
        want;
      match !bad with
      | _ when Array.length got <> Array.length want ->
          Printf.printf "upper holds %d values, %d expected\n"
            (Array.length got) (Array.length want)
      | None ->
          print_endline "upper is each codebook's edges, decoded, within 1e-6"
      | Some i ->
          Printf.printf "upper's value %d is %g, %g expected\n" i got.(i)
            want.(i))
  | [ "values"; f; cache; key ] -> (
      let books = codebooks cache key in
      let c = Array.length books in
      let a = load f in
      let tokens = Nx.to_array (get Nx.int64 "tokens" a) in
      let got = Nx.to_array (Nx.cast Nx.float64 (get Nx.float32 "values" a)) in
      let bad = ref None in
      Array.iteri
        (fun i t ->
          let book = books.(i mod c) and t = Int64.to_int t in
          let ok =
            if t >= Array.length book then Float.is_nan got.(i)
            else close got.(i) book.(t)
          in
          if !bad = None && not ok then bad := Some (i, t))
        tokens;
      match !bad with
      | _ when Array.length got <> Array.length tokens ->
          Printf.printf "values holds %d values for %d tokens\n"
            (Array.length got) (Array.length tokens)
      | None ->
          print_endline
            "values are the tokens' edges, decoded, within 1e-6, NaN past the \
             edges"
      | Some (i, t) ->
          Printf.printf "value %d is %g for the token %d\n" i got.(i) t)
  | [ "equal"; f; g; name ] ->
      let x = Nx.to_array (get Nx.int64 name (load f))
      and y = Nx.to_array (get Nx.int64 name (load g)) in
      if x = y then Printf.printf "%s equal\n" name
      else if Array.length x <> Array.length y then
        Printf.printf "%s of %d and %d values\n" name (Array.length x)
          (Array.length y)
      else
        Printf.printf "%s differ in %d of %d values\n" name
          (Array.fold_left ( + ) 0
             (Array.mapi (fun i v -> if v <> y.(i) then 1 else 0) x))
          (Array.length x)
  | [ "chi2"; g; p; t ] ->
      let g = load g and pa = load p and t = float_of_string t in
      if
        Nx.to_array (get Nx.int64 "rows" g)
        <> Nx.to_array (get Nx.int64 "rows" pa)
      then failwith "the draws and the posterior are of different objects";
      let tokens = get Nx.int64 "tokens" g in
      let probs = get Nx.float32 "p" pa in
      let n = Nx.dim 0 probs in
      let rest =
        Option.map (Nx.unpack Nx.float32) (Nx_io.Archive.find "p_1024" pa)
      in
      let stat = ref 0. and df = ref 0 in
      for i = 0 to n - 1 do
        let p = row probs i in
        let p =
          match rest with
          | Some q -> Array.append p [| (row q i).(0) |]
          | None -> p
        in
        (* p^(1/T), normalised. *)
        let p = Array.map (fun x -> Float.pow x (1. /. t)) p in
        let s = Array.fold_left ( +. ) 0. p in
        let p = Array.map (fun x -> x /. s) p in
        let draws =
          Array.map Int64.to_int (Nx.to_array (Nx.slice [ Nx.I i ] tokens))
        in
        let x, d = pearson p draws in
        stat := !stat +. x;
        df := !df + d
      done;
      let q = chi2_upper 1e-3 !df in
      if !stat <= q then
        print_endline
          "the draws' frequencies agree with p: chi2 below its 0.999 quantile"
      else
        print_endline
          "the draws' frequencies differ from p: chi2 above its 0.999 quantile"
  | [ "posterior"; f; i; r ] ->
      let a = load f and i = int_of_string i in
      let want = get Nx.float64 "p" (load r) in
      let c = Nx.dim 0 want and v = Nx.dim 1 want in
      let p =
        Nx.reshape
          [| -1; c; Nx.dim (-1) (get Nx.float32 "p" a) |]
          (get Nx.float32 "p" a)
      in
      let k = Nx.dim 2 p in
      let got = row p i in
      let w = Nx.to_array want in
      let d = ref 0. in
      Array.iteri
        (fun j g ->
          d := Float.max !d (Float.abs (g -. w.((j / k * v) + (j mod k)))))
        got;
      (match Nx_io.Archive.find "p_1024" a with
      | Some q ->
          let q = row (Nx.unpack Nx.float32 q) i in
          let s = ref 0. in
          for j = k to v - 1 do
            s := !s +. w.(j)
          done;
          d := Float.max !d (Float.abs (q.(0) -. !s))
      | None -> ());
      if !d <= 1e-6 then
        print_endline "p is the softmax of the decoder's logits within 1e-6"
      else Printf.printf "p differs from the softmax by %g\n" !d
  | _ ->
      prerr_endline
        "usage: npz.exe \
         entries|ints|equal|sums|same|upper|values|posterior|chi2 FILE...";
      exit 2
