(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Random-order generation with AION-1's weights against upstream
   (PolymathicAI/AION db7fba6df53b). It lives outside runtest: the weights are
   1.27 GB, the references 0.6 GB, and a run takes minutes. The alias parity
   runs it as [roar_parity.exe check]; [posterior] and [time] are run by hand:

   roar_parity.exe [check | posterior | time]

   It reads AION_PARITY's huggingface, holding aion-base's model.safetensors and
   its redshift codec (codecs/Z) at Hub.revision; reference/f64 and
   reference/f32 (REF64 and REF32), the references of transformer_parity.exe;
   and roar (P4REF), which gen/roar_reference.py writes, with upstream's code at
   db7fba6df53b and torch 2.14.1 on the CPU: g1.safetensors (upstream's recorded
   states of a ROAR run and their logits) and z.COND.safetensors (redshift
   posteriors and upstream's draws).

   [check] prints three things, and gates none.

   Step logits (G1): each of upstream's recorded states is decoded by the port
   at the state's positions, with the state's revealed tokens present: the first
   and fifth steps of a linear-8 run from four Gaia stars' XP coefficients to a
   DESI spectrum, and the one-token steps of redshift from photometry with the
   image and from a DESI spectrum (transformer_parity's sets, eight objects).
   For the port in float32 and cast to float64 it prints the largest difference
   from upstream's float64 logits, that over their largest magnitude, the
   argmaxes that differ, and PyTorch float32's numbers beside them. The rounding
   ensemble of these values is not yet run.

   The redshift posterior (AION-1 §7.1.1) of 64 objects from three
   conditionings, the port's against upstream's float32: the distance between
   the two, the mass on redshift's class 1024, which the decoded draws leave
   out, and from 50 draws per object of each (upstream's seeds 10^6 apart), the
   posterior medians' NMAD, bias and outliers against the DESI redshift and per
   object a two-sample Kolmogorov-Smirnov test of the two sets of draws.

   The encoder past 256 present tokens: one DESI token's distribution given a
   star's 110 XP tokens and m revealed DESI tokens, against the same with 20 of
   them dropped, and for m past 146 with the revealed set cut to 146.

   [posterior] prints the second alone.

   [time] times one §7.1.2 draw, the first Gaia star of REF64's tokens to a DESI
   spectrum at upstream's linear 12, float32: one untimed draw, then three. Run
   it under the machine's timing lock. *)

open Aion

(* The inputs, under AION_PARITY as the README's Tests section lays them out. *)
let dir =
  match Sys.getenv_opt "AION_PARITY" with
  | Some d when d <> "" -> d
  | _ ->
      prerr_endline
        "roar_parity: AION_PARITY is unset: set it to the directory of the \
         README's Tests section";
      exit 1

let path = Filename.concat dir
let printf = Printf.printf

let get dt name a =
  match Nx_io.Archive.find name a with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no tensor " ^ name)

let tokens_of a name =
  match Nx_io.Archive.find name a with
  | Some (Nx.P t) -> Nx.cast Nx.int64 t
  | None -> failwith ("no tokens " ^ name)

let first n t = Nx.slice [ Nx.R (0, n) ] t

let input m tokens : _ Transformer.input =
  { modality = m; tokens; present = None }

let target_input m b : _ Transformer.input =
  let n = Modality.tokens m in
  {
    modality = m;
    tokens = Nx.zeros Nx.int64 [| b; n |];
    present = Some (Nx.zeros Nx.bool [| b; n |]);
  }

(* The port's logits of a generation step: [inputs], then [target] present where
   [revealed], decoded at [positions]. *)
let step t inputs target ~tokens ~revealed ~positions =
  let ti : _ Transformer.input =
    { modality = target; tokens; present = Some revealed }
  in
  List.hd
    (Transformer.forward t (inputs @ [ ti ])
       [ { modality = target; positions = Some positions } ])

(* The first step of a one-token target, nothing revealed. *)
let one_token t inputs target =
  let b = Nx.dim 0 (List.hd inputs : _ Transformer.input).tokens in
  let ti = target_input target b in
  step t inputs target ~tokens:ti.tokens ~revealed:(Option.get ti.present)
    ~positions:(Nx.zeros Nx.int64 [| b; 1 |])

let max_abs x = Nx.item [] (Nx.max (Nx.abs x))

let error x r =
  let d = max_abs (Nx.sub (Nx.cast Nx.float64 x) r) in
  (d, d /. max_abs r)

let differing x r =
  let a = Nx.argmax ~axis:(-1) x and b = Nx.argmax ~axis:(-1) r in
  ( Int64.to_int (Nx.item [] (Nx.sum (Nx.cast Nx.int64 (Nx.not_equal a b)))),
    Nx.numel a )

let report name l ~r64 ~p32 =
  let d, nm = error l r64 and pd, pn = error p32 r64 in
  let k, total = differing (Nx.cast Nx.float64 l) r64 in
  let pk, _ = differing (Nx.cast Nx.float64 p32) r64 in
  printf
    "%-24s abs %.3g norm %.3g, argmax %d/%d differ | torch f32 abs %.3g norm \
     %.3g, argmax %d differ\n\
     %!"
    name d nm k total pd pn pk

let ls_phot = Modality.[ Flux_g; Flux_r; Flux_i; Flux_z ]

(* G1 *)

let g1 (type b) name (t : (b, Nx.host) Transformer.t) ~codecs ~r64 ~r32 ~g =
  printf "\n== G1 step logits, port %s\n%!" name;
  let gaia m =
    input (Scalar m)
      (first 4
         (tokens_of codecs ("gaia." ^ Modality.key (Scalar m) ^ ".tokens")))
  in
  let xp = [ gaia Xp_bp; gaia Xp_rp ] in
  List.iter
    (fun s ->
      let l =
        step t xp (Spectrum Desi)
          ~tokens:(get Nx.int64 (s ^ ".tokens") g)
          ~revealed:(Nx.not_equal_s (get Nx.uint8 (s ^ ".revealed") g) 0)
          ~positions:(get Nx.int64 (s ^ ".positions") g)
      in
      report ("desi_from_xp." ^ s) l
        ~r64:(get Nx.float64 ("f64." ^ s) g)
        ~p32:(get Nx.float32 ("f32." ^ s) g))
    [ "s0"; "s4" ];
  let ls m =
    input m
      (first 8 (tokens_of codecs ("ls_desi." ^ Modality.key m ^ ".tokens")))
  in
  List.iter
    (fun (set, ms) ->
      let l = one_token t (List.map ls ms) (Scalar Z) in
      let key = "logits." ^ set ^ ".upstream" in
      report set l ~r64:(get Nx.float64 key r64) ~p32:(get Nx.float32 key r32))
    Modality.
      [
        ("z_from_phot", Image :: List.map (fun s -> Scalar s) ls_phot);
        ("z_from_desi", [ Spectrum Desi ]);
      ]

(* The redshift posterior *)

let rows x =
  let s = Nx.shape x in
  let v = s.(Array.length s - 1) in
  let a = Nx.to_array (Nx.cast Nx.float64 x) in
  Array.init (Array.length a / v) (fun i -> Array.sub a (i * v) v)

let softmax l =
  let m = Array.fold_left Float.max Float.neg_infinity l in
  let e = Array.map (fun x -> exp (x -. m)) l in
  let s = Array.fold_left ( +. ) 0. e in
  Array.map (fun x -> x /. s) e

let kl p q =
  let a = ref 0. in
  Array.iteri
    (fun i pi -> if pi > 0. then a := !a +. (pi *. log (pi /. q.(i))))
    p;
  !a

let median a =
  let a = Array.copy a in
  Array.sort compare a;
  let n = Array.length a in
  if n = 0 then Float.nan
  else if n mod 2 = 1 then a.(n / 2)
  else (a.((n / 2) - 1) +. a.(n / 2)) /. 2.

let maximum a = Array.fold_left Float.max Float.neg_infinity a

(* Q_KS, the Kolmogorov distribution's tail, by its alternating series, summed
   until a term is negligible; where the series does not converge, near [l = 0],
   the tail is 1 (Numerical Recipes, probks). *)
let q_ks l =
  let rec go k sign sum prev =
    if k > 100 then 1.
    else
      let k' = Float.of_int k in
      let term = sign *. 2. *. exp (-2. *. k' *. k' *. l *. l) in
      let sum = sum +. term in
      if Float.abs term <= 1e-3 *. prev || Float.abs term <= 1e-8 *. sum then
        sum
      else go (k + 1) (-.sign) sum (Float.abs term)
  in
  Float.min 1. (Float.max 0. (go 1 1. 0. 0.))

(* Two-sample Kolmogorov-Smirnov: the statistic and its asymptotic p-value
   (Numerical Recipes' Q_KS with the effective size's correction). *)
let ks a b =
  let a = Array.copy a and b = Array.copy b in
  Array.sort compare a;
  Array.sort compare b;
  let na = Array.length a and nb = Array.length b in
  let i = ref 0 and j = ref 0 and d = ref 0. in
  while !i < na && !j < nb do
    let x = Float.min a.(!i) b.(!j) in
    while !i < na && a.(!i) <= x do
      incr i
    done;
    while !j < nb && b.(!j) <= x do
      incr j
    done;
    d :=
      Float.max !d
        (Float.abs
           ((Float.of_int !i /. Float.of_int na)
           -. (Float.of_int !j /. Float.of_int nb)))
  done;
  let ne = Float.of_int (na * nb) /. Float.of_int (na + nb) in
  (!d, q_ks ((sqrt ne +. 0.12 +. (0.11 /. sqrt ne)) *. !d))

(* The posterior medians' NMAD, bias and outlier fraction against [truth]. *)
let metrics ~truth med =
  let e =
    Array.mapi (fun i m -> (m -. truth.(i)) /. (1. +. truth.(i))) med
    |> Array.to_list
    |> List.filter (fun x -> not (Float.is_nan x))
    |> Array.of_list
  in
  let m = median e in
  let nmad = 1.4826 *. median (Array.map (fun x -> Float.abs (x -. m)) e) in
  let out =
    Array.fold_left (fun a x -> if Float.abs x > 0.15 then a + 1 else a) 0 e
  in
  (nmad, m, out, Array.length e)

let draws_port l ~count =
  (* Roar.sample of one token at temperature 1: step 0 draws from [fold_in key
     1] the logits less each row's largest. *)
  let d = Nx.sub l (Nx.max ~axes:[ -1 ] ~keepdims:true l) in
  List.init count (fun j ->
      let key = Nx.Rng.fold_in (Nx.Rng.key j) 1 in
      Nx.to_array (Nx.reshape [| -1 |] (Nx.Rng.categorical ~key d)))
  |> Array.of_list

let posterior t ~hub ~codecs ~p4 =
  printf "\n== Redshift posterior (§7.1.1), 64 objects, 50 draws each\n%!";
  let codec = Scalar.load ~cache_dir:hub ~offline:true Z in
  let decode k =
    if k = 1024L then Float.nan
    else
      Nx.item [ 0 ]
        (Nx.cast Nx.float64
           (Scalar.decode codec (Nx.create Nx.int64 [| 1 |] [| k |])))
  in
  let n = 64 and chunk = 8 in
  let ls lo m =
    let key = "ls_desi." ^ Modality.key m ^ ".tokens" in
    input m (Nx.slice [ Nx.R (lo, lo + chunk) ] (tokens_of codecs key))
  in
  List.iter
    (fun (c, ms) ->
      let z =
        Nx_io.load_safetensors (Filename.concat p4 ("z." ^ c ^ ".safetensors"))
      in
      let truth =
        Nx.to_array
          (Nx.cast Nx.float64
             (Scalar.decode codec (get Nx.int64 "z_true_token" z)))
      in
      let l =
        Nx.concatenate ~axis:0
          (List.init (n / chunk) (fun i ->
               one_token t (List.map (ls (i * chunk)) ms) (Scalar Z)))
      in
      (* One Roar.sample of the first objects, to tie these draws to it. *)
      let s0 =
        Roar.sample ~key:(Nx.Rng.key 0) ~reveal:[ 1 ] t
          (List.map (ls 0) ms)
          (Scalar Z)
      in
      let port = draws_port l ~count:50 in
      let same =
        Nx.to_array (Nx.reshape [| -1 |] s0) = Array.sub port.(0) 0 chunk
      in
      let pp = Array.map softmax (rows l) in
      let pu = Array.map softmax (rows (get Nx.float32 "f32" z)) in
      let dp =
        maximum
          (Array.mapi
             (fun i p ->
               maximum (Array.mapi (fun v x -> Float.abs (x -. pu.(i).(v))) p))
             pp)
      in
      let kls = Array.mapi (fun i p -> kl pu.(i) p) pp in
      let up = rows (Nx.cast Nx.float64 (get Nx.int64 "draws" z)) in
      let port_i i = Array.init 50 (fun j -> port.(j).(i)) in
      let up_i i = Array.map Int64.of_float up.(i) in
      let c1024 f =
        Array.fold_left (fun a k -> if k = 1024L then a + 1 else a) 0 f
      in
      let count1024 g =
        List.fold_left ( + ) 0 (List.init n (fun i -> c1024 (g i)))
      in
      let med g =
        Array.init n (fun i ->
            Array.map decode (g i)
            |> Array.to_list
            |> List.filter (fun x -> not (Float.is_nan x))
            |> Array.of_list |> median)
      in
      let mass p = Array.map (fun r -> r.(1024)) p in
      printf "-- %s: Roar.sample's draw is the first draw: %b\n" c same;
      printf
        "   posterior, port against upstream float32: max |dp| %.3g, KL(up || \
         port) median %.3g max %.3g\n"
        dp (median kls) (maximum kls);
      printf
        "   class 1024: mass median %.3g max %.3g (upstream %.3g %.3g); draws \
         of it port %d, upstream %d of %d\n"
        (median (mass pp))
        (maximum (mass pp))
        (median (mass pu))
        (maximum (mass pu))
        (count1024 port_i) (count1024 up_i) (50 * n);
      List.iter
        (fun (who, g) ->
          let nmad, bias, out, m = metrics ~truth (med g) in
          printf
            "   %-8s posterior median vs DESI z: NMAD %.4f bias %.4f outliers \
             %d/%d\n"
            who nmad bias out m)
        [ ("port", port_i); ("upstream", up_i) ];
      let ps =
        Array.init n (fun i ->
            let a = Array.map (fun k -> Int64.to_float k) (port_i i) in
            let b = Array.map (fun k -> Int64.to_float k) (up_i i) in
            snd (ks a b))
      in
      let below x =
        Array.fold_left (fun a p -> if p < x then a + 1 else a) 0 ps
      in
      printf
        "   two-sample KS over tokens per object: p < 0.05 for %d/%d, p < 0.01 \
         for %d, smallest p %.3g\n\
         %!"
        (below 0.05) n (below 0.01)
        (Array.fold_left Float.min 1. ps))
    Modality.
      [
        ("phot", List.map (fun s -> Scalar s) ls_phot);
        ("image_phot", Image :: List.map (fun s -> Scalar s) ls_phot);
        ("desi", [ Spectrum Desi ]);
      ]

(* The encoder past 256 present tokens *)

let probe t ~codecs ~g =
  printf
    "\n== Encoder past 256 present tokens: 4 stars, 110 XP tokens + m DESI\n%!";
  let b = 4 in
  let gaia m =
    input (Scalar m)
      (first b
         (tokens_of codecs ("gaia." ^ Modality.key (Scalar m) ^ ".tokens")))
  in
  let xp = [ gaia Xp_bp; gaia Xp_rp ] in
  let desi = get Nx.int64 "final" g in
  let perm = Nx.to_array (Nx.Rng.permutation ~key:(Nx.Rng.key 7) 273) in
  let target = perm.(272) in
  let revealed set =
    let a = Array.make 273 false in
    List.iter (fun p -> a.(Int64.to_int p) <- true) set;
    Nx.broadcast_to [| b; 273 |] (Nx.create Nx.bool [| 1; 273 |] a)
  in
  let dist set =
    let l =
      step t xp (Spectrum Desi) ~tokens:desi ~revealed:(revealed set)
        ~positions:(Nx.full Nx.int64 [| b; 1 |] target)
    in
    Array.map softmax (rows l)
  in
  let prefix k = List.init k (fun i -> perm.(i)) in
  let compare name m k =
    let p = dist (prefix m) and q = dist (prefix k) in
    let kls = Array.mapi (fun i pi -> kl pi q.(i)) p in
    let agree =
      Array.mapi
        (fun i pi ->
          let am a =
            snd
              (Array.fold_left
                 (fun (j, (bj, bv)) x ->
                   (j + 1, if x > bv then (j, x) else (bj, bv)))
                 (0, (0, Float.neg_infinity))
                 a)
            |> fst
          in
          am pi = am q.(i))
        p
      |> Array.fold_left (fun a x -> if x then a + 1 else a) 0
    in
    printf
      "   m %3d (encoder %3d) %-10s KL median %.3g max %.3g, argmax agree %d/%d\n\
       %!"
      m (110 + m) name (median kls) (maximum kls) agree b
  in
  List.iter
    (fun m ->
      compare "drop 20" m (m - 20);
      if m > 146 then compare "cut to 146" m 146)
    [ 100; 130; 146; 170; 200; 240; 272 ]

(* Timing one §7.1.2 draw *)

let linear12 = [ 23; 23; 23; 23; 23; 23; 23; 23; 23; 22; 22; 22 ]

let time t ~codecs =
  let gaia m =
    input (Scalar m)
      (first 1
         (tokens_of codecs ("gaia." ^ Modality.key (Scalar m) ^ ".tokens")))
  in
  let xp = [ gaia Xp_bp; gaia Xp_rp ] in
  let draw seed =
    let t0 = Unix.gettimeofday () in
    ignore
      (Roar.sample ~key:(Nx.Rng.key seed) ~reveal:linear12 t xp (Spectrum Desi));
    Unix.gettimeofday () -. t0
  in
  ignore (draw 0);
  let ts = List.map draw [ 1; 2; 3 ] in
  printf "one star to DESI at linear 12, float32: %s s; median %.2f s\n%!"
    (String.concat ", " (List.map (Printf.sprintf "%.2f") ts))
    (median (Array.of_list ts))

let () =
  let model dir =
    Nx_io.load_safetensors (Filename.concat dir "model.safetensors")
  in
  let hub = path "huggingface" and p4 = path "roar" in
  let ref64 = path "reference/f64" and ref32 = path "reference/f32" in
  match Sys.argv with
  | [| _; "check" |] ->
      let r64 = model ref64 and r32 = model ref32 in
      let codecs =
        Nx_io.load_safetensors (Filename.concat ref64 "codecs.safetensors")
      in
      let g = Nx_io.load_safetensors (Filename.concat p4 "g1.safetensors") in
      let t = Transformer.load ~cache_dir:hub ~offline:true () in
      g1 "float32" t ~codecs ~r64 ~r32 ~g;
      g1 "float64" (Transformer.cast Nx.float64 t) ~codecs ~r64 ~r32 ~g;
      posterior t ~hub ~codecs ~p4;
      probe t ~codecs ~g
  | [| _; "posterior" |] ->
      let codecs =
        Nx_io.load_safetensors (Filename.concat ref64 "codecs.safetensors")
      in
      posterior
        (Transformer.load ~cache_dir:hub ~offline:true ())
        ~hub ~codecs ~p4
  | [| _; "time" |] ->
      let codecs =
        Nx_io.load_safetensors (Filename.concat ref64 "codecs.safetensors")
      in
      time (Transformer.load ~cache_dir:hub ~offline:true ()) ~codecs
  | _ ->
      prerr_endline "usage: roar_parity.exe [check | posterior | time]";
      exit 2
