(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

type mask = Unmasked | Causal | Window | Documents | Blocks | Opaque

type t = {
  dtype : Nx.Dtype.any;
  lead : int;
  hq : int;
  hk : int;
  tq : int;
  tk : int;
  dk : int;
  dv : int;
  page : int option;
  holes : bool;
  lengths : int array option;
  mask : mask;
  alibi : bool;
  rising : bool;
  positive : bool;
  nans : bool;
  seed : int;
}

let mask_name = function
  | Unmasked -> "none"
  | Causal -> "causal"
  | Window -> "window"
  | Documents -> "documents"
  | Blocks -> "blocks"
  | Opaque -> "opaque"

let pp ppf c =
  let (Nx.Dtype.Any dt) = c.dtype in
  Format.fprintf ppf
    "%s lead %d hq %d hk %d tq %d tk %d dk %d dv %d page %s holes %b \
     lengths %s mask %s alibi %b rising %b positive %b nans %b seed %d"
    (Nx.Dtype.name dt) c.lead c.hq c.hk c.tq c.tk c.dk c.dv
    (match c.page with Some p -> string_of_int p | None -> "none")
    c.holes
    (match c.lengths with
    | Some l -> String.concat "," (Array.to_list (Array.map string_of_int l))
    | None -> "none")
    (mask_name c.mask) c.alibi c.rising c.positive c.nans c.seed

let numel = Array.fold_left ( * ) 1
let edges = [ 0; 1; 2; 15; 16; 17; 31; 32; 33; 63; 64; 65 ]

let gen dtypes =
  let open Gen in
  let* dtype = of_list dtypes in
  let* lead = int_range 1 3 in
  let* hq, hk = of_list [ (1, 1); (8, 8); (8, 2); (8, 1) ] in
  let* tq = of_list edges in
  let* page = frequency [ (2, constant None); (1, constant (Some 16)); (1, constant (Some 5)) ] in
  let* tk = of_list edges in
  let tk = match page with Some p -> max p (tk / p * p) | None -> tk in
  let* holes = bool in
  let* lengths =
    frequency
      [
        (2, constant None);
        ( 2,
          map Option.some
            (array ~size:(constant lead)
               (frequency
                  [ (1, constant 0); (1, constant 1); (2, constant tk); (2, int_range 0 tk) ])) );
      ]
  in
  let* mask = of_list [ Unmasked; Causal; Window; Documents; Blocks; Opaque ] in
  let* alibi = bool in
  let* rising = frequency [ (3, constant false); (1, constant true) ] in
  let* positive = frequency [ (3, constant false); (1, constant true) ] in
  let* nans = bool in
  let* dk = of_list [ 1; 8; 16; 24; 64 ] in
  let* dv = of_list [ 1; 8; 16; 64; 72 ] in
  let+ seed = int_range 0 1000 in
  {
    dtype;
    lead;
    hq;
    hk;
    tq;
    tk;
    dk;
    dv;
    page;
    holes;
    lengths;
    mask;
    alibi;
    rising;
    positive;
    nans = nans && page = None;
    seed;
  }

let base dtype =
  {
    dtype;
    lead = 1;
    hq = 8;
    hk = 2;
    tq = 300;
    tk = 300;
    dk = 64;
    dv = 64;
    page = None;
    holes = false;
    lengths = None;
    mask = Causal;
    alibi = false;
    rising = false;
    positive = false;
    nans = false;
    seed = 3;
  }

let examples dtypes =
  let b = base (List.hd dtypes) in
  let cases =
    [
      { b with hq = 64; hk = 8; tq = 1; tk = 1000; lengths = Some [| 777 |] };
      { b with mask = Unmasked; rising = true };
      { b with mask = Window; positive = true };
      { b with mask = Documents };
      { b with mask = Blocks; alibi = true };
      { b with mask = Opaque; lengths = Some [| 290 |]; nans = true };
      { b with tq = 0; tk = 5 };
      { b with tq = 5; tk = 0 };
      { b with dv = 0; tq = 3; tk = 3 };
      { b with page = Some 16; holes = true; tq = 40; tk = 64 };
      { b with dk = 768; tq = 20; tk = 40 };
    ]
  in
  let n = List.length dtypes in
  List.mapi (fun i c -> { c with dtype = List.nth dtypes (i mod n) }) cases

(* Inputs *)

let draw c salt shape =
  let n = numel shape in
  let st = Random.State.make [| c.seed; salt |] in
  Nx.create Nx.float32 shape
    (Array.init n (fun _ -> Random.State.float st 2. -. 1.))

(* The case's call over [q], [k] and [v], of any device, in [acc]. *)
let call (type a s d) (acc : (float, a) Nx.dtype) c ~table ~lengths ~docs
    ~blocks ~slopes (q : (float, s, d) Nx.t) k v =
  let place x = Nx.place (Nx.beside q) x in
  let mask : (d Nx.pair -> d Nx.bool_t) option =
    match c.mask with
    | Unmasked -> None
    | Causal -> Some Nx.causal
    | Window ->
        Some
          (fun p -> Nx.logical_and (Nx.causal p) (Nx.less p.q (Nx.add_s p.k 8L)))
    | Documents ->
        let dq, dk = docs in
        let dq = place dq and dk = place dk in
        Some
          (fun _ ->
            Nx.equal (Nx.reshape [| c.tq; 1 |] dq) (Nx.reshape [| 1; c.tk |] dk))
    | Blocks ->
        let blocks = place blocks in
        Some (fun _ -> blocks)
    | Opaque ->
        Some
          (fun p ->
            let f x = Nx.cast Nx.float32 x in
            Nx.less_equal (f p.k) (f (Nx.add p.q (Nx.sub p.len (Nx.scalar_like p.len (Int64.of_int c.tq))))))
  in
  let score =
    match (c.alibi, c.rising) with
    | false, false -> None
    | _ ->
        let slopes = place (Nx.cast acc slopes) in
        Some
          (fun (p : d Nx.pair) s ->
            let s =
              if c.alibi then Nx.(Infix.(s - (cast acc (p.q - p.k) * slopes)))
              else s
            in
            if c.rising then Nx.add s (Nx.mul_s (Nx.cast acc p.k) 0.25) else s)
  in
  let kv : (float, s, d) Nx.kv =
    match (c.page, table) with
    | Some _, Some t -> Paged { k; v; table = place t }
    | _ -> Dense { k; v }
  in
  Nx.attention ?lengths:(Option.map place lengths) ?mask ?score ~acc ~kv q

let unit_roundoff (Nx.Dtype.Any dt) =
  match Nx.Dtype.name dt with
  | "float64" -> 0x1p-53
  | "float32" -> 0x1p-24
  | "float16" -> 0x1p-11
  | _ -> 0x1p-8

let gamma n u = Float.of_int n *. u /. (1. -. (Float.of_int n *. u))

type float_dtype = F : (float, 's) Nx.dtype -> float_dtype

let float_dtype (Nx.Dtype.Any dt) =
  match Nx.Dtype.name dt with
  | "float16" -> F Nx.float16
  | "bfloat16" -> F Nx.bfloat16
  | "float64" -> F Nx.float64
  | _ -> F Nx.float32

let law (module S : Nx.Devices) c =
  cover "paged" (c.page <> None);
  cover "lengths" (c.lengths <> None);
  cover (mask_name c.mask) true;
  let (F dt) = float_dtype c.dtype in
  let wide = Nx.Dtype.name dt = "float64" in
  let lead = [| c.lead |] in
  let pages = match c.page with Some p -> c.tk / p | None -> 0 in
  let kshape, vshape =
    match c.page with
    | None -> ([| c.lead; c.hk; c.tk; c.dk |], [| c.lead; c.hk; c.tk; c.dv |])
    | Some p ->
        let pool = (c.lead * pages) + 1 in
        ([| pool; c.hk; p; c.dk |], [| pool; c.hk; p; c.dv |])
  in
  (* Values rounded to the case's dtype, as float32 and float64. *)
  let rounded x = Nx.cast Nx.float32 (Nx.cast dt x) in
  let q = rounded (draw c 1 [| c.lead; c.hq; c.tq; c.dk |]) in
  let k = rounded (draw c 2 kshape) in
  let v = draw c 3 vshape in
  let v = rounded (if c.positive then Nx.add_s (Nx.abs v) 0.01 else v) in
  let table =
    match c.page with
    | None -> None
    | Some _ ->
        let pool = (c.lead * pages) + 1 in
        Some
          (Nx.create Nx.int64 [| c.lead; pages |]
             (Array.init (c.lead * pages) (fun i ->
                  if c.holes && i mod 3 = 1 then (if i mod 2 = 0 then -1L else Int64.of_int pool)
                  else Int64.of_int ((i * 7 + c.seed) mod pool))))
  in
  let lengths =
    Option.map
      (fun l -> Nx.create Nx.int64 lead (Array.map Int64.of_int l))
      c.lengths
  in
  let docs =
    let ids n = Nx.create Nx.int64 [| n |] (Array.init n (fun i -> Int64.of_int (i * 3 / 40))) in
    (ids c.tq, ids c.tk)
  in
  let blocks =
    let st = Random.State.make [| c.seed; 9 |] in
    let b = Array.init (((c.tq / 4) + 1) * ((c.tk / 4) + 1)) (fun _ -> Random.State.bool st) in
    Nx.create Nx.bool [| c.tq; c.tk |]
      (Array.init (c.tq * c.tk) (fun i ->
           b.((i / c.tk / 4 * ((c.tk / 4) + 1)) + (i mod c.tk / 4))))
  in
  let slopes = Nx.abs (draw c 8 [| c.hq; 1; 1 |]) in
  (* NaN at keys past each row's length, which no row reads. *)
  let poisoned x =
    match (c.nans, c.lengths) with
    | true, Some ls ->
        let s = Nx.shape x in
        let a = Nx.to_array x in
        let per = numel (Array.sub s 1 (Array.length s - 1)) in
        let d = s.(Array.length s - 1) in
        Array.iteri
          (fun i _ ->
            let b = i / per and key = i mod per / d mod c.tk in
            if key >= ls.(b) then a.(i) <- Float.nan)
          a;
        Nx.create Nx.float32 s a
    | _ -> x
  in
  let compiled (type a) (acc : (float, a) Nx.dtype) =
    let place x = Nx.place S.on (Nx.cast dt x) in
    let f q k v =
      let o, lse = call acc c ~table ~lengths ~docs ~blocks ~slopes q k v in
      (o, Nx.cast Nx.float64 lse)
    in
    let g =
      Rune.jit Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns (pair tensor tensor)) f
    in
    let before = Rune.evaluations () in
    let o, lse = g (place q) (place (poisoned k)) (place (poisoned v)) in
    equal ~msg:"compiled, not evaluated" Testable.int 0 (Rune.evaluations () - before);
    let host x = Nx.to_array (Nx.cast Nx.float64 (Nx.place Nx.Host.on x)) in
    (host o, host lse)
  in
  let f64 x = Nx.cast Nx.float64 x in
  let reference v =
    let o, lse =
      call Nx.float64 c ~table ~lengths ~docs ~blocks ~slopes (f64 q) (f64 k) (f64 v)
    in
    (Nx.to_array o, Nx.to_array lse)
  in
  let o_ref, lse_ref = reference v in
  let m, _ = reference (Nx.abs v) in
  let o, lse = if wide then compiled Nx.float64 else compiled Nx.float32 in
  (* The bound, nx.mli's: u the accumulator's, u_q the dtype's, values and
     keys within [-1, 1], [a] the greatest |scale Σ q k|, [b] the greatest
     |score|. *)
  let u = if wide then 0x1p-53 else 0x1p-24 in
  let uq = unit_roundoff c.dtype in
  let narrow = uq > 0x1p-20 in
  let scale = 1. /. Float.sqrt (Float.of_int c.dk) in
  let a = scale *. Float.of_int c.dk in
  let b =
    a
    +. (if c.alibi then Float.of_int (c.tq + c.tk) else 0.)
    +. if c.rising then 0.25 *. Float.of_int c.tk else 0.
  in
  let eta = (Float.of_int ((2 * c.dk) + 4) *. u *. a) +. (2. *. u *. b) in
  let ln = Float.log (Float.of_int (max 1 c.tk)) in
  let vmax = if c.positive then 1.01 else 1. in
  Array.iteri
    (fun i l ->
      let at = Printf.sprintf "row %d" i in
      if l = Float.neg_infinity then
        equal ~msg:("an empty row's lse at " ^ at) float_exact l lse.(i)
      else begin
        let eps =
          (((1.37 *. Float.of_int (c.tk - 1)) +. 19. +. (3. *. ln)) *. u)
          +. (u *. Float.abs l)
        in
        (* The float64 reference's own error, nx.cpu's, where it is as wide
           as the compiled call. *)
        let n = (c.tk + 63) / 64 in
        let eps_ref =
          if wide then
            ((Float.of_int ((13 * n) + 9) +. (7. *. ln)) *. u) +. (u *. Float.abs l)
          else 0.
        in
        let w = ((1. +. u) *. ((2. *. eta) +. eps)) +. (((2. *. b) +. ln +. 19.) *. u) in
        let w_ref = if wide then (2. *. eta) +. eps_ref +. (((2. *. b) +. ln +. 5.) *. u) else 0. in
        less ~msg:("lse at " ^ at) float_exact
          ~than:(eta +. eps +. (if wide then eta +. eps_ref else 0.) +. 1e-300)
          (Float.abs (l -. lse.(i)));
        for d = 0 to c.dv - 1 do
          let j = (i * c.dv) + d in
          let mj = m.(j) in
          let bound =
            (uq *. Float.abs o_ref.(j))
            +. (1. +. uq)
               *. (Float.expm1 w +. (gamma (c.tk + 1) (2. *. u) *. Float.exp w))
               *. mj
            +. (if wide then
                  (Float.expm1 w_ref +. (gamma (c.tk + 1) (2. *. u) *. Float.exp w_ref)) *. mj
                else 0.)
            +. (if narrow then uq *. mj else 0.)
            +. (if uq = 0x1p-11 then 0x1p-25 *. vmax *. Float.of_int c.tk else 0.)
          in
          less
            ~msg:(Printf.sprintf "o at row %d column %d" i d)
            float_exact ~than:(bound +. 1e-300)
            (Float.abs (o.(j) -. o_ref.(j)))
        done
      end)
    lse_ref
