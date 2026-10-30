(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx's attention against its definition computed in float64: the softmax of
   each row's visible scaled scores, rewritten by its score, times the values,
   over dense and paged keys, grouped heads, lead axes, lengths, masks as
   programs and as captured arrays, empty sequences, and float16 and bfloat16
   accumulating at float32. A hidden finite value never changes the result,
   and a hidden NaN or infinity within the lengths makes its row NaN. *)

open Windtrap
module D = Nx_array.Dtype

let same_bits =
  array
    (Testable.make
       ~pp:(fun ppf x -> Format.fprintf ppf "%h" x)
       ~equal:(fun a b ->
         (Float.is_nan a && Float.is_nan b)
         || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)))

let invalid f = raises_match (Exn.invalid_arg ~sub:"Nx.attention") f
let numel s = Array.fold_left ( * ) 1 s
let u32 = 0x1p-23

(* A case *)

type mask = Unmasked | Causal | Window | Captured of bool array
type score = Identity | Alibi of float array | Bias of float array

type case = {
  lead : int array;
  hk : int;
  groups : int;
  tq : int;
  tk : int;
  dk : int;
  dv : int;
  lengths : int array option;  (** One per lead index. *)
  scale : float option;
  mask : mask;  (** A captured mask is [[hq; tq; tk]]. *)
  score : score;
      (** ALiBi's slopes, one per query head, or a bias [[hq; tq; tk]]. *)
  q : float array;
  k : float array;
  v : float array;
}

let hq c = c.hk * c.groups
let rows c = numel c.lead

let pp_case ppf c =
  let ints ppf a =
    Format.fprintf ppf "[%s]"
      (String.concat "; " (Array.to_list (Array.map string_of_int a)))
  in
  Format.fprintf ppf
    "lead %a hk %d groups %d tq %d tk %d dk %d dv %d lengths %s %s %s" ints
    c.lead c.hk c.groups c.tq c.tk c.dk c.dv
    (match c.lengths with None -> "none" | Some l -> Format.asprintf "%a" ints l)
    (match c.mask with
    | Unmasked -> "unmasked"
    | Causal -> "causal"
    | Window -> "window"
    | Captured _ -> "captured")
    (match c.score with
    | Identity -> "identity"
    | Alibi _ -> "alibi"
    | Bias _ -> "bias");
  let floats name xs =
    Format.fprintf ppf "@ %s [%s]" name
      (String.concat "; " (Array.to_list (Array.map (Printf.sprintf "%h") xs)))
  in
  Option.iter (floats "scale") (Option.map (fun s -> [| s |]) c.scale);
  floats "q" c.q;
  floats "k" c.k;
  floats "v" c.v;
  match c.score with
  | Identity -> ()
  | Alibi xs -> floats "slopes" xs
  | Bias xs -> floats "bias" xs

let element =
  Gen.frequency
    [ (8, Gen.float_range (-2.) 2.); (1, Gen.constant 0.); (1, Gen.constant (-0.)) ]

let case =
  let open Gen in
  let* lead = array ~size:(int_range 0 2) (int_range 0 2) in
  let* hk = int_range 1 2 in
  let* groups = int_range 1 3 in
  let* tq = frequency [ (1, constant 0); (4, int_range 1 4) ] in
  let* tk = frequency [ (1, constant 0); (4, int_range 1 5) ] in
  let* dk = int_range 1 4 in
  let* dv = int_range 1 3 in
  let* lengths =
    frequency
      [
        (1, constant None);
        (1, map Option.some (array ~size:(constant (numel lead)) (int_range (-1) (tk + 1))));
      ]
  in
  let* scale = of_list [ None; Some 0.7; Some (-1.5) ] in
  let hq = hk * groups in
  let* mask =
    frequency
      [
        (1, constant Unmasked);
        (1, constant Causal);
        (1, constant Window);
        (1, map (fun m -> Captured m) (array ~size:(constant (hq * tq * tk)) bool));
      ]
  in
  let* score =
    frequency
      [
        (2, constant Identity);
        (1, map (fun s -> Alibi s) (array ~size:(constant hq) (float_range 0. 0.5)));
      ]
  in
  let n = numel lead in
  let* q = array ~size:(constant (n * hq * tq * dk)) element in
  let* k = array ~size:(constant (n * hk * tk * dk)) element in
  let+ v = array ~size:(constant (n * hk * tk * dv)) element in
  { lead; hk; groups; tq; tk; dk; dv; lengths; scale; mask; score; q; k; v }

(* The definition *)

(* The row [b]'s length: keys at or past it are hidden. *)
let length c b = match c.lengths with None -> c.tk | Some l -> l.(b)

let visible c b h i j =
  let len = length c b in
  let causal = j <= i + (len - c.tq) in
  j < len
  &&
  match c.mask with
  | Unmasked -> true
  | Causal -> causal
  | Window -> causal && i < j + 2
  | Captured m -> m.((((h * c.tq) + i) * c.tk) + j)

let rewritten c h i j s =
  match c.score with
  | Identity -> s
  | Alibi slopes -> s -. (Float.of_int (i - j) *. slopes.(h))
  | Bias bias -> s +. bias.((((h * c.tq) + i) * c.tk) + j)

(* What a case reaches, at a row that has a visible key: a finite and a
   non-finite value hidden from it within its length, and a NaN value at a
   visible key whose weight is zero. *)
type reached = {
  mutable hidden_finite : bool;
  mutable hidden_nonfinite : bool;
  mutable zero_weight_nan : bool;
}

(* Per row [(b, h, i)]: [o] and the mass [Σ P |v|] at each [d], [lse], and
   the greatest [|scale| Σ |q k|] or rewritten score, the size of a score's
   rounding; then what the case reaches. [key b kh j d] and [value b kh j d]
   read key [j] of key head [kh]. Weights are float32's: one below half its
   least subnormal is zero. A value hidden from a row within its length is
   read at weight zero. *)
let definition c ~visible ~key ~value q =
  let hq = hq c in
  let scale =
    match c.scale with
    | Some s -> s
    | None -> 1. /. Float.sqrt (Float.of_int c.dk)
  in
  let n = rows c * hq * c.tq in
  let o = Array.make (n * c.dv) 0. and mass = Array.make (n * c.dv) 0. in
  let lse = Array.make n neg_infinity and big = Array.make n 0. in
  let reached =
    { hidden_finite = false; hidden_nonfinite = false; zero_weight_nan = false }
  in
  for b = 0 to rows c - 1 do
    for h = 0 to hq - 1 do
      let kh = h / c.groups in
      for i = 0 to c.tq - 1 do
        let row = (((b * hq) + h) * c.tq) + i in
        let s =
          Array.init c.tk (fun j ->
              let dot = ref 0. and size = ref 0. in
              for d = 0 to c.dk - 1 do
                let x = q.((row * c.dk) + d) and y = key b kh j d in
                dot := !dot +. (x *. y);
                size := !size +. Float.abs (x *. y)
              done;
              let s = rewritten c h i j (scale *. !dot) in
              let size = Float.abs scale *. !size in
              big.(row) <- Float.max big.(row) (Float.max size (Float.abs s));
              s)
        in
        let seen j = visible b h i j in
        let m = ref neg_infinity in
        Array.iteri (fun j x -> if seen j then m := Float.max !m x) s;
        if !m > neg_infinity then begin
          let sum = ref 0. in
          Array.iteri (fun j x -> if seen j then sum := !sum +. Float.exp (x -. !m)) s;
          lse.(row) <- !m +. Float.log !sum;
          Array.iteri
            (fun j x ->
              let p = Float.exp (x -. lse.(row)) in
              let p = if p < 0x1p-150 then 0. else p in
              for d = 0 to c.dv - 1 do
                let y = value b kh j d in
                if not (seen j) then begin
                  if j < length c b then
                    if Float.is_finite y then reached.hidden_finite <- true
                    else begin
                      reached.hidden_nonfinite <- true;
                      o.((row * c.dv) + d) <- nan
                    end
                end
                else begin
                  if p = 0. && Float.is_nan y then reached.zero_weight_nan <- true;
                  o.((row * c.dv) + d) <- o.((row * c.dv) + d) +. (p *. y);
                  if p > 0. then
                    mass.((row * c.dv) + d) <- mass.((row * c.dv) + d) +. (p *. Float.abs y)
                end
              done)
            s
        end
      done
    done
  done;
  ((o, mass, lse, big), reached)

(* Nx's call *)

let ints s xs = Nx.create Nx.int64 s (Array.map Int64.of_int xs)

let mask_of c : (Nx.host Nx.pair -> Nx.host Nx.bool_t) option =
  match c.mask with
  | Unmasked -> None
  | Causal -> Some Nx.causal
  | Window ->
      Some (fun p -> Nx.(logical_and (causal p) (less p.q (add_s p.k 2L))))
  | Captured m -> Some (fun _ -> Nx.create Nx.bool [| hq c; c.tq; c.tk |] m)

let score_of c : (Nx.host Nx.pair -> Nx.host Nx.float32_t -> Nx.host Nx.float32_t) option =
  match c.score with
  | Identity -> None
  | Alibi slopes ->
      let slopes = Nx.create Nx.float32 [| hq c; 1; 1 |] slopes in
      Some (fun p s -> Nx.(Infix.(s - (cast float32 (p.q - p.k) * slopes))))
  | Bias bias ->
      let bias = Nx.create Nx.float32 [| hq c; c.tq; c.tk |] bias in
      Some (fun _ s -> Nx.add s bias)

let lengths_of c = Option.map (ints c.lead) c.lengths

(* [o] and [lse] against the definition: [o] within [q]'s rounding [u] of the
   value and float32's of the scores, sums and weights, relative to the mass;
   [lse] within float32's. [tiny] is [q]'s least subnormal. *)
let agrees c ~u ~tiny (o, lse) (o', mass, lse', big) =
  let o = Nx.to_array o and lse = Nx.to_array lse in
  equal ~msg:"o's elements" int (Array.length o') (Array.length o);
  equal ~msg:"lse's elements" int (Array.length lse') (Array.length lse);
  Array.iteri
    (fun row l' ->
      let l = lse.(row) in
      let rounds = Float.of_int (c.tk + 16) +. (4. *. Float.of_int (c.dk + 2) *. big.(row)) in
      if l' = neg_infinity then begin
        if l <> neg_infinity then failf "lse at %d: %h, by definition -inf" row l
      end
      else if not (Float.abs (l -. l') <= rounds *. u32 *. (1. +. Float.abs l')) then
        failf "lse at %d: %h, by definition %h" row l l';
      for d = 0 to c.dv - 1 do
        let i = (row * c.dv) + d in
        (* Products below float32's normal range round absolutely. *)
        let under = Float.of_int (c.tk + 1) *. 0x1p-149 in
        let tol = ((u +. (rounds *. u32)) *. mass.(i)) +. tiny +. under in
        let x = o.(i) and x' = o'.(i) in
        let agree =
          if Float.is_nan x' then Float.is_nan x
          else if Float.is_finite x' && mass.(i) = 0. then
            (* An exact zero sum is +0. *)
            Int64.equal (Int64.bits_of_float x) 0L
          else if Float.is_finite x' then Float.abs (x -. x') <= tol
          else x = x'
        in
        if not agree then failf "o at %d: %h, by definition %h" i x x'
      done)
    lse'

let shape c tail = Array.append c.lead tail

(* The dense call of [c] at [dt] against the definition; what [c] reaches. *)
let dense (type s) (dt : (float, s) D.t) u tiny c =
  let hq = hq c in
  let q = Nx.create dt (shape c [| hq; c.tq; c.dk |]) c.q in
  let k = Nx.create dt (shape c [| c.hk; c.tk; c.dk |]) c.k in
  let v = Nx.create dt (shape c [| c.hk; c.tk; c.dv |]) c.v in
  let qs = Nx.to_array q and ks = Nx.to_array k and vs = Nx.to_array v in
  let key b kh j d = ks.((((((b * c.hk) + kh) * c.tk) + j) * c.dk) + d) in
  let value b kh j d = vs.((((((b * c.hk) + kh) * c.tk) + j) * c.dv) + d) in
  let result =
    Nx.attention ?scale:c.scale ?lengths:(lengths_of c) ?mask:(mask_of c)
      ?score:(score_of c) ~kv:(Nx.Dense { k; v }) q
  in
  equal ~msg:"o's shape" (array int) (shape c [| hq; c.tq; c.dv |]) (Nx.shape (fst result));
  equal ~msg:"lse's shape" (array int) (shape c [| hq; c.tq |]) (Nx.shape (snd result));
  let expected, reached = definition c ~visible:(visible c) ~key ~value qs in
  agrees c ~u ~tiny result expected;
  reached

let law_dense dt u tiny c =
  cover "causal" (c.mask = Causal);
  cover "unmasked" (c.mask = Unmasked);
  cover "a window" (c.mask = Window);
  cover "a captured mask" (match c.mask with Captured _ -> true | _ -> false);
  cover "alibi" (c.score <> Identity);
  cover "lengths" (c.lengths <> None);
  cover "grouped heads" (c.groups > 1);
  cover "lead axes" (Array.length c.lead > 0);
  cover "no query" (c.tq = 0 && rows c > 0);
  cover "no key" (c.tk = 0 && c.tq > 0 && rows c > 0);
  cover "a row with no visible key"
    (rows c > 0 && c.tq > 0
    && not (List.exists (visible c 0 0 0) (List.init c.tk Fun.id)));
  ignore (dense dt u tiny c : reached)

(* Values with NaN, infinities and [-0.], and a bias of [0] or [-200], whose
   weights underflow to zero at float32. *)
let unusual =
  let open Gen in
  let* c = case in
  let* v =
    array ~size:(constant (Array.length c.v))
      (frequency
         [
           (4, element);
           (1, constant nan);
           (1, constant infinity);
           (1, constant neg_infinity);
           (1, constant (-0.));
         ])
  in
  let+ bias = array ~size:(constant (hq c * c.tq * c.tk)) (of_list [ 0.; -200. ]) in
  let mask = if c.mask = Unmasked then Causal else c.mask in
  { c with v; mask; score = Bias bias }

(* A finite value hidden from a row never changes its [o]; a NaN or an
   infinity hidden within its length, or a NaN at a visible key whose weight
   is zero, makes it NaN. *)
let law_unusual dt u tiny c =
  let reached = dense dt u tiny c in
  cover "a finite value hidden from a row" reached.hidden_finite;
  cover "a non-finite value hidden from a row" reached.hidden_nonfinite;
  cover "a NaN at a visible key of zero weight" reached.zero_weight_nan

(* Paged keys: the case's keys and values are the pools, [[pages; hk; page;
   d]], read through a table of entries in [[-1, pages]]. *)

let paged =
  let open Gen in
  let* c = case in
  let* pages = int_range 1 3 in
  let* page = int_range 1 3 in
  let* max_pages = int_range 0 3 in
  let* table =
    array ~size:(constant (numel c.lead * max_pages)) (int_range (-1) pages)
  in
  let tk = max_pages * page in
  let* k = array ~size:(constant (pages * c.hk * page * c.dk)) element in
  let* v = array ~size:(constant (pages * c.hk * page * c.dv)) element in
  let* lengths =
    frequency
      [
        (1, constant None);
        (1, map Option.some (array ~size:(constant (numel c.lead)) (int_range (-1) (tk + 1))));
      ]
  in
  let+ mask =
    frequency
      [
        (2, constant (match c.mask with Captured _ -> Causal | m -> m));
        (1, map (fun m -> Captured m) (array ~size:(constant (hq c * c.tq * tk)) bool));
      ]
  in
  ({ c with tk; k; v; lengths; mask }, (pages, page, max_pages, table))

let pp_paged ppf (c, (pages, page, _, table)) =
  Format.fprintf ppf "%a pages %d page %d table [%s]" pp_case c pages page
    (String.concat "; " (Array.to_list (Array.map string_of_int table)))

let law_paged (c, (pages, page, max_pages, table)) =
  let hq = hq c in
  let q = Nx.create Nx.float32 (shape c [| hq; c.tq; c.dk |]) c.q in
  let k = Nx.create Nx.float32 [| pages; c.hk; page; c.dk |] c.k in
  let v = Nx.create Nx.float32 [| pages; c.hk; page; c.dv |] c.v in
  let entry b j = table.((b * max_pages) + (j / page)) in
  let in_range b j = entry b j >= 0 && entry b j < pages in
  cover "an entry out of range" (Array.exists (fun e -> e < 0 || e >= pages) table);
  cover "an entry in range" (Array.exists (fun e -> e >= 0 && e < pages) table);
  let qs = Nx.to_array q and ks = Nx.to_array k and vs = Nx.to_array v in
  (* An entry out of range reads zero, never seen. *)
  let pool xs d b kh j i =
    if not (in_range b j) then 0.
    else
      let p = entry b j in
      xs.((((((p * c.hk) + kh) * page) + (j mod page)) * d) + i)
  in
  let visible b h i j = in_range b j && visible c b h i j in
  let result =
    Nx.attention ?scale:c.scale ?lengths:(lengths_of c) ?mask:(mask_of c)
      ?score:(score_of c)
      ~kv:(Nx.Paged { k; v; table = ints (shape c [| max_pages |]) table })
      q
  in
  agrees c ~u:u32 ~tiny:0x1p-149 result
    (fst (definition c ~visible ~key:(pool ks c.dk) ~value:(pool vs c.dv) qs))

(* Cases *)

let f32 s xs = Nx.create Nx.float32 s xs

(* Past the length, and where the causal mask hides a pair, a NaN key, value
   and score bias reach neither [o] nor [lse]. *)
let test_hidden_nan () =
  let q = f32 [| 1; 2; 1 |] [| 1.; -1. |] in
  let k = f32 [| 1; 3; 1 |] [| 0.5; 2.; nan |] in
  let v = f32 [| 1; 3; 1 |] [| 3.; -1.; nan |] in
  let bias = f32 [| 2; 3 |] [| 0.; nan; nan; 0.25; 0.; nan |] in
  let o, lse =
    Nx.attention ~scale:1. ~lengths:(Nx.scalar Nx.int64 2L) ~mask:Nx.causal
      ~score:(fun _ s -> Nx.add s bias)
      ~kv:(Nx.Dense { k; v }) q
  in
  (* Row 0 sees key 0 alone; row 1 sees keys 0 and 1, scores -0.25 and -2. *)
  let s0 = -0.5 +. 0.25 and s1 = -2. in
  let l1 = s0 +. Float.log1p (Float.exp (s1 -. s0)) in
  let p0 = Float.exp (s0 -. l1) and p1 = Float.exp (s1 -. l1) in
  equal ~msg:"o" (array (float 1e-6)) [| 3.; (3. *. p0) -. p1 |] (Nx.to_array o);
  equal ~msg:"lse" (array (float 1e-6)) [| 0.5; l1 |] (Nx.to_array lse)

(* [q k] is [2049] for the first key and [2048] for the second, which float16
   and bfloat16 round to one value: summed at float32, the weights are
   [sigmoid 1] and [1 - sigmoid 1]. *)
let test_scores_at_float32 () =
  let check (type s) (dt : (float, s) D.t) u =
    let q = Nx.create dt [| 1; 1; 2 |] [| 1.; 1. |] in
    let k = Nx.create dt [| 1; 2; 2 |] [| 2048.; 1.; 2048.; 0. |] in
    let v = Nx.create dt [| 1; 2; 1 |] [| 1.; 0. |] in
    let o, lse = Nx.attention ~scale:1. ~kv:(Nx.Dense { k; v }) q in
    let p = 1. /. (1. +. Float.exp (-1.)) in
    equal ~msg:(D.name dt ^ " o") (array (float (u *. p))) [| p |] (Nx.to_array o);
    equal ~msg:(D.name dt ^ " lse") (array (float 1e-3))
      [| 2048. +. Float.log1p (Float.exp 1.) |]
      (Nx.to_array lse)
  in
  check Nx.float16 0x1p-11;
  check Nx.bfloat16 0x1p-8

let test_no_visible_key () =
  let q = f32 [| 2; 1; 2; 1 |] [| 1.; 2.; 3.; 4. |] in
  let k = f32 [| 2; 1; 2; 1 |] [| 1.; 1.; 1.; 1. |] in
  let v = f32 [| 2; 1; 2; 1 |] [| nan; nan; 5.; 5. |] in
  let o, lse =
    Nx.attention ~lengths:(ints [| 2 |] [| 0; 2 |]) ~kv:(Nx.Dense { k; v }) q
  in
  equal ~msg:"o" (array (float 1e-6)) [| 0.; 0.; 5.; 5. |] (Nx.to_array o);
  equal ~msg:"lse" (array (float 1e-6))
    [| neg_infinity; neg_infinity; 3. +. Float.log 2.; 4. +. Float.log 2. |]
    (Nx.to_array lse);
  let o, lse =
    Nx.attention ~mask:(fun _ -> Nx.scalar Nx.bool false) ~kv:(Nx.Dense { k; v }) q
  in
  equal ~msg:"masked o" (array float_exact) [| 0.; 0.; 0.; 0. |] (Nx.to_array o);
  equal ~msg:"masked lse" (array float_exact)
    [| neg_infinity; neg_infinity; neg_infinity; neg_infinity |]
    (Nx.to_array lse)

(* The last row sees every key before the length, each row before it one
   fewer. *)
let test_causal () =
  let p =
    Nx.
      {
        q = ints [| 2; 1 |] [| 0; 1 |];
        k = ints [| 1; 3 |] [| 0; 1; 2 |];
        len = Nx.scalar Nx.int64 2L;
        tq = 2;
      }
  in
  equal (array bool)
    [| true; false; false; true; true; false |]
    (Nx.to_array (Nx.causal p))

(* A row whose visible scores are all [-∞] has [o = 0] and [lse = -∞], a NaN
   value at a visible key included. *)
let test_infinite_scores () =
  let q = f32 [| 1; 2; 1 |] [| 1.; 2. |] in
  let k = f32 [| 1; 2; 1 |] [| 1.; 1. |] in
  let v = f32 [| 1; 2; 1 |] [| nan; 3. |] in
  let o, lse =
    Nx.attention
      ~score:(fun _ _ -> Nx.scalar Nx.float32 neg_infinity)
      ~kv:(Nx.Dense { k; v }) q
  in
  equal ~msg:"o" (array float_exact) [| 0.; 0. |] (Nx.to_array o);
  equal ~msg:"lse" (array float_exact) [| neg_infinity; neg_infinity |]
    (Nx.to_array lse)

(* Zero heads, and no key dimension under a given scale, are empty and zero
   scores. *)
let test_empty_extents () =
  let o, lse =
    Nx.attention
      ~kv:(Nx.Dense { k = f32 [| 0; 3; 1 |] [||]; v = f32 [| 0; 3; 2 |] [||] })
      (f32 [| 0; 2; 1 |] [||])
  in
  equal ~msg:"o's shape" (array int) [| 0; 2; 2 |] (Nx.shape o);
  equal ~msg:"lse's shape" (array int) [| 0; 2 |] (Nx.shape lse);
  let o, lse =
    Nx.attention ~scale:2.
      ~kv:(Nx.Dense { k = f32 [| 1; 2; 0 |] [||]; v = f32 [| 1; 2; 1 |] [| 1.; 3. |] })
      (f32 [| 1; 1; 0 |] [||])
  in
  equal ~msg:"o" (array (float 1e-6)) [| 2. |] (Nx.to_array o);
  equal ~msg:"lse" (array (float 1e-6)) [| Float.log 2. |] (Nx.to_array lse)

(* A sum of zeros is +0, here beside a value the causal mask hides: finite,
   it changes nothing; NaN, it makes the row NaN. *)
let test_zero () =
  let q = f32 [| 1; 2; 1 |] [| 1.; 1. |] in
  let k = f32 [| 1; 2; 1 |] [| 1.; 1. |] in
  let o v =
    let v = f32 [| 1; 2; 1 |] v in
    Nx.to_array (fst (Nx.attention ~mask:Nx.causal ~kv:(Nx.Dense { k; v }) q))
  in
  equal ~msg:"finite" same_bits [| 0.; 1. |] (o [| -0.; 2. |]);
  equal ~msg:"NaN" same_bits [| nan; nan |] (o [| -0.; nan |])

let test_rejections () =
  let q = f32 [| 2; 3; 4 |] (Array.make 24 1.) in
  let k = f32 [| 1; 5; 4 |] (Array.make 20 1.) in
  let v = f32 [| 1; 5; 2 |] (Array.make 10 1.) in
  let kv = Nx.Dense { k; v } in
  let q64 = Nx.create Nx.float64 [| 2; 3; 4 |] (Array.make 24 1.) in
  let k64 = Nx.create Nx.float64 [| 1; 5; 4 |] (Array.make 20 1.) in
  let v64 = Nx.create Nx.float64 [| 1; 5; 2 |] (Array.make 10 1.) in
  invalid (fun () -> Nx.attention ~kv:(Nx.Dense { k = k64; v = v64 }) q64);
  invalid (fun () -> Nx.attention ~kv (f32 [| 3; 4 |] (Array.make 12 1.)));
  invalid (fun () ->
      Nx.attention ~kv:(Nx.Dense { k = f32 [| 2; 5; 4 |] (Array.make 40 1.); v }) q);
  invalid (fun () ->
      Nx.attention ~kv:(Nx.Dense { k; v = f32 [| 1; 4; 2 |] (Array.make 8 1.) }) q);
  invalid (fun () ->
      Nx.attention ~kv:(Nx.Dense { k = f32 [| 1; 5; 3 |] (Array.make 15 1.); v }) q);
  invalid (fun () ->
      Nx.attention ~kv:(Nx.Dense { k = f32 [| 3; 5; 4 |] (Array.make 60 1.); v = f32 [| 3; 5; 2 |] (Array.make 30 1.) }) q);
  invalid (fun () -> Nx.attention ~lengths:(ints [| 1 |] [| 3 |]) ~kv q);
  invalid (fun () -> Nx.attention ~mask:(fun p -> Nx.(less p.k (sum p.q))) ~kv q);
  invalid (fun () ->
      Nx.attention ~mask:(fun p -> Nx.(less p.k (scalar int64 (item [] p.q)))) ~kv q);
  invalid (fun () ->
      Nx.attention ~mask:(fun p -> Nx.(less (reshape [| 1; 1 |] p.k) p.q)) ~kv q);
  invalid (fun () ->
      Nx.attention ~mask:(fun _ -> Nx.create Nx.bool [| 7 |] (Array.make 7 true)) ~kv q);
  invalid (fun () ->
      Nx.attention ~mask:(fun p -> Nx.(less (broadcast_to [| 7 |] p.k) p.q)) ~kv q);
  let q0 = f32 [| 2; 3; 0 |] [||] and k0 = f32 [| 1; 5; 0 |] [||] in
  invalid (fun () -> Nx.attention ~kv:(Nx.Dense { k = k0; v }) q0);
  let pool = f32 [| 2; 1; 3; 4 |] (Array.make 24 1.) in
  let table = ints [| 2; 2 |] [| 0; 1; 1; 0 |] in
  invalid (fun () -> Nx.attention ~kv:(Nx.Paged { k = pool; v = pool; table }) q)

let () =
  exit
    (run "nx attention"
       [
         group "dense"
           [
             prop "float32 by its definition" (Gen.with_pp pp_case case) (law_dense D.Float32 u32 0x1p-149);
             prop "float16 by its definition" (Gen.with_pp pp_case case) (law_dense D.Float16 0x1p-11 0x1p-24);
             prop "bfloat16 by its definition" (Gen.with_pp pp_case case) (law_dense D.Bfloat16 0x1p-8 0x1p-133);
             prop "float32, non-finite and hidden values"
               (Gen.with_pp pp_case unusual) (law_unusual D.Float32 u32 0x1p-149);
             prop "float16, non-finite and hidden values"
               (Gen.with_pp pp_case unusual) (law_unusual D.Float16 0x1p-11 0x1p-24);
           ];
         prop "paged by its definition" (Gen.with_pp pp_paged paged) law_paged;
         test "a hidden NaN" test_hidden_nan;
         test "scores at float32" test_scores_at_float32;
         test "no visible key" test_no_visible_key;
         test "causal" test_causal;
         test "only infinite scores" test_infinite_scores;
         test "empty extents" test_empty_extents;
         test "a sum of zeros" test_zero;
         test "rejections" test_rejections;
       ])
