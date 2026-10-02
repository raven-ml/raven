(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type tick = { position : float; label : string; context : string option }
type t = { major : tick list; minor : float list; note : string option }

let err fn fmt =
  Format.kasprintf (fun s -> invalid_arg ("Ticks." ^ fn ^ ": " ^ s)) fmt

let decimal_notation : Number.notation -> Decimal.notation = function
  | Plain -> Plain
  | Exponent -> Exponent
  | Si -> Si
  | Percent -> Percent

let check_notation (type d) fn notation (s : d Scale.t) =
  match (notation, Scale.kind s) with
  | None, _ | Some _, Scale.Quantitative -> ()
  | Some _, (Scale.Temporal | Scale.Categorical) ->
      err fn "a notation labels quantitative scales only"

(* Quantities *)

let ten_thousand = Decimal.pow10 false 4

let is_grouped ds =
  Array.exists (fun d -> Decimal.compare_mag d ten_thousand >= 0) ds

let write locale ~group ~trim notation decimals d =
  Decimal.write locale ~group ~trim notation (Decimals decimals) d

(* [p ds] is the exponent of the greatest power of ten every decimal of [ds] is
   a multiple of, [0] if every one is zero, and [e ds] the exponent of their
   largest magnitude, [0] if every one is zero. *)
let p ds =
  Array.fold_left
    (fun p (d : Decimal.t) ->
      if Decimal.is_zero d then p
      else Some (Int.min d.exp (Option.value ~default:d.exp p)))
    None ds
  |> Option.value ~default:0

let e ds =
  Array.fold_left
    (fun e d ->
      if Decimal.is_zero d then e
      else
        Some (Int.max (Decimal.mag d) (Option.value ~default:(Decimal.mag d) e)))
    None ds
  |> Option.value ~default:0

(* [offset ds e p] is the offset of the distinct decimals [ds], in increasing or
   decreasing order, if any. *)
let offset ds e p =
  let n = Array.length ds in
  let positive =
    Array.for_all (fun (d : Decimal.t) -> not (Decimal.is_zero d || d.neg)) ds
  and negative = Array.for_all (fun (d : Decimal.t) -> d.neg) ds in
  if n < 2 || (not (positive || negative)) || e - p + 1 <= 7 then None
  else
    let lo, hi =
      if Decimal.compare ds.(0) ds.(n - 1) < 0 then (ds.(0), ds.(n - 1))
      else (ds.(n - 1), ds.(0))
    in
    let q = Decimal.mag (Decimal.sub hi lo) + 1 in
    let o = Decimal.round Toward_zero q (if positive then lo else hi) in
    if Decimal.is_zero o then None else Some o

(* [exactly locale ~group notation d] writes [d] with the fewest digits that
   write it exactly in [notation]. *)
let exactly locale ~group notation (d : Decimal.t) =
  let decimals =
    if Decimal.is_zero d then 0
    else
      match notation with
      | Decimal.Plain -> Int.max 0 (-d.exp)
      | Percent -> Int.max 0 (-d.exp - 2)
      | Exponent -> String.length d.digits - 1
      | Si ->
          let k =
            Int.max (-10) (Int.min 10 (Decimal.floor_div (Decimal.mag d) 3))
          in
          Int.max 0 ((3 * k) - d.exp)
  in
  write locale ~group ~trim:true notation decimals d

let quantities locale notation ds =
  let p = p ds in
  match notation with
  | Some (Decimal.Plain as n) ->
      let group = is_grouped ds in
      (Array.map (write locale ~group ~trim:false n (Int.max 0 (-p))) ds, None)
  | Some (Percent as n) ->
      let group = is_grouped (Array.map (Decimal.shift 2) ds) in
      ( Array.map (write locale ~group ~trim:false n (Int.max 0 (-p - 2))) ds,
        None )
  | Some ((Exponent | Si) as n) ->
      let label (d : Decimal.t) =
        if Decimal.is_zero d then write locale ~group:false ~trim:true n 0 d
        else
          let at =
            match n with
            | Exponent -> Decimal.mag d
            | _ ->
                3
                * Int.max (-10)
                    (Int.min 10 (Decimal.floor_div (Decimal.mag d) 3))
          in
          write locale ~group:false ~trim:true n (Int.max 0 (at - p)) d
      in
      (Array.map label ds, None)
  | None ->
      let e0 = e ds in
      let o = offset ds e0 p in
      let ds =
        match o with
        | Some o -> Array.map (fun d -> Decimal.sub d o) ds
        | None -> ds
      in
      let e = match o with Some _ -> e ds | None -> e0 in
      let k = if e < -4 || e > 5 then 3 * Decimal.floor_div e 3 else 0 in
      let ds = Array.map (Decimal.shift (-k)) ds in
      let group = is_grouped ds in
      let labels =
        Array.map (write locale ~group ~trim:false Plain (Int.max 0 (k - p))) ds
      in
      let factor =
        if k = 0 then None else Some ("×10" ^ Decimal.superscript k)
      in
      let offset =
        Option.map
          (fun (o : Decimal.t) ->
            let a = Decimal.abs o in
            let small = Decimal.pow10 false (-4)
            and large = Decimal.pow10 false 6 in
            let n =
              if
                Decimal.compare_mag a small >= 0
                && Decimal.compare_mag a large < 0
              then Decimal.Plain
              else Exponent
            in
            (if o.neg then Locale.minus locale else "+")
            ^ exactly locale ~group:true n a)
          o
      in
      let note =
        match (factor, offset) with
        | Some f, Some o -> Some (f ^ " " ^ o)
        | (Some _ as n), None | None, (Some _ as n) -> n
        | None, None -> None
      in
      (labels, note)

(* Logarithms *)

(* [log_parts b x] is [Some (n, i)] if [x > 0.] is [n × b^i] as the kit computes
   it, for [n] from [1] to [b - 1] in an integer base, [n = 1] otherwise. *)
let log_parts b x =
  let i0 = Float.to_int (Float.floor (Float.log x /. Float.log b)) in
  let nmax = if Steps.is_integer_base b then Float.to_int b - 1 else 1 in
  let rec find i n =
    if i > i0 + 1 then None
    else if n > nmax then find (i + 1) 1
    else if Float.equal (Steps.multiple b n i) x then Some (n, i)
    else find i (n + 1)
  in
  find (i0 - 1) 1

let is_signed_power x =
  x = 0.
  || match log_parts 10. (Float.abs x) with Some (1, _) -> true | _ -> false

let is_logarithms tf vs =
  match (tf : Scale.transform) with
  | Log b -> Array.for_all (fun x -> Option.is_some (log_parts b x)) vs
  | Symlog _ -> Array.for_all is_signed_power vs
  | Linear | Pow _ | Custom _ -> false

let logarithms ~shortest locale notation base vs =
  match notation with
  | Some n -> Array.map (fun x -> exactly locale ~group:false n (shortest x)) vs
  | None when base = 10. ->
      let ds = Array.map shortest vs in
      let in_range (d : Decimal.t) =
        Decimal.is_zero d
        || Decimal.compare_mag d (Decimal.pow10 false (-3)) >= 0
           && Decimal.compare_mag d ten_thousand <= 0
      in
      if Array.for_all in_range ds then
        let group = is_grouped ds in
        Array.map (exactly locale ~group Plain) ds
      else Array.map (exactly locale ~group:false Exponent) ds
  | None ->
      let b =
        if base = Float.exp 1. then "e"
        else exactly locale ~group:false Plain (shortest base)
      in
      Array.map
        (fun x ->
          match log_parts base x with
          | Some (1, i) -> b ^ Decimal.superscript i
          | Some (n, i) -> string_of_int n ^ "×" ^ b ^ Decimal.superscript i
          | None -> assert false)
        vs

(* Time *)

(* The coarsest unit whose starts include every instant labelled. *)
type part = Years | Months | Days | Minutes | Seconds | Fraction of int

let time_unit tz (ts : Time.t array) =
  let fields t = (Time.to_date_time ~tz_offset_s:tz t, t.Time.nsec) in
  let all p = Array.for_all (fun t -> p (fields t)) ts in
  if
    all (fun (((_, m, d), (hh, mm, ss)), ns) ->
        m = 1 && d = 1 && hh = 0 && mm = 0 && ss = 0 && ns = 0)
  then Years
  else if
    all (fun (((_, _, d), (hh, mm, ss)), ns) ->
        d = 1 && hh = 0 && mm = 0 && ss = 0 && ns = 0)
  then Months
  else if
    all (fun ((_, (hh, mm, ss)), ns) -> hh = 0 && mm = 0 && ss = 0 && ns = 0)
  then Days
  else if all (fun ((_, (_, _, ss)), ns) -> ss = 0 && ns = 0) then Minutes
  else if all (fun (_, ns) -> ns = 0) then Seconds
  else
    let rec digits n =
      if
        n = 9
        || all (fun (_, ns) -> ns mod Int.of_float (Steps.pow10 (9 - n)) = 0)
      then n
      else digits (n + 1)
    in
    Fraction (digits 1)

let year locale y =
  if y < 0 then Locale.minus locale ^ string_of_int (-y) else string_of_int y

let times locale tz ts =
  let unit = time_unit tz ts in
  let label t =
    let (y, m, d), (hh, mm, ss) = Time.to_date_time ~tz_offset_s:tz t in
    let month = Locale.month locale m in
    match unit with
    | Years -> (year locale y, None)
    | Months -> (month, Some (year locale y))
    | Days -> (string_of_int d, Some (month ^ " " ^ year locale y))
    | Minutes ->
        (Printf.sprintf "%02d:%02d" hh mm, Some (string_of_int d ^ " " ^ month))
    | Seconds ->
        (Printf.sprintf ":%02d" ss, Some (Printf.sprintf "%02d:%02d" hh mm))
    | Fraction n ->
        ( Locale.decimal locale
          ^ String.sub (Printf.sprintf "%09d" t.Time.nsec) 0 n,
          Some (Printf.sprintf "%02d:%02d:%02d" hh mm ss) )
  in
  let labels = Array.map label ts in
  Array.mapi
    (fun i (l, c) ->
      match c with
      | Some _ when i > 0 && Option.equal String.equal c (snd labels.(i - 1)) ->
          (l, None)
      | _ -> (l, c))
    labels

(* Labelling *)

(* [label ~shortest locale notation s vs] is the labels, contexts and note of
   the values [vs] of [s], in their order, [shortest] computing
   [Decimal.shortest]. *)
let label : type d.
    shortest:(float -> Decimal.t) ->
    Locale.t ->
    Decimal.notation option ->
    d Scale.t ->
    d array ->
    (string * string option) array * string option =
 fun ~shortest locale notation s vs ->
  match Scale.kind s with
  | Quantitative ->
      let tf = Scale.transform s in
      if is_logarithms tf vs then
        let base = match tf with Log b -> b | _ -> 10. in
        ( Array.map
            (fun l -> (l, None))
            (logarithms ~shortest locale notation base vs),
          None )
      else
        let labels, note = quantities locale notation (Array.map shortest vs) in
        (Array.map (fun l -> (l, None)) labels, note)
  | Temporal -> (times locale (Scale.tz_offset_s s) vs, None)
  | Categorical ->
      let (Categories c) = Scale.domain s in
      let text =
        match c with
        | Labels _ -> Fun.id
        | Indices ix ->
            let texts = Hashtbl.create (Array.length ix) in
            Array.iter
              (fun (i, t) -> Hashtbl.replace texts (string_of_int i) t)
              ix;
            Hashtbl.find texts
      in
      (Array.map (fun v -> (text v, None)) vs, None)

(* [arrange s vs] is the values of [vs] in the domain of [s] and not missing for
   it, each once, with their positions, in increasing order of positions. *)
let arrange : type d. d Scale.t -> d array -> d array * float array =
 fun s vs ->
  let norm = Scale.normalize s in
  let ((inside, compare) : (d -> bool) * (d -> d -> int)) =
    match Scale.domain s with
    | Floats (a, b) -> ((fun v -> a <= v && v <= b), Float.compare)
    | Instants (a, b) ->
        ((fun v -> Time.compare a v <= 0 && Time.compare v b <= 0), Time.compare)
    | Categories _ -> ((fun _ -> true), String.compare)
  in
  let vs =
    List.filter
      (fun v -> inside v && not (Float.is_nan (norm v)))
      (Array.to_list vs)
  in
  let rec dedup = function
    | v :: (w :: _ as rest) ->
        if compare v w = 0 then dedup (v :: List.tl rest) else v :: dedup rest
    | l -> l
  in
  let vs = dedup (List.sort compare vs) in
  let placed =
    List.stable_sort
      (fun (_, p) (_, q) -> Float.compare p q)
      (List.map (fun v -> (v, norm v)) vs)
  in
  (Array.of_list (List.map fst placed), Array.of_list (List.map snd placed))

let ticks_of positions labels note minor =
  let major =
    List.init (Array.length positions) (fun i ->
        let label, context = labels.(i) in
        { position = positions.(i); label; context })
  in
  { major; minor; note }

let of_values ?(locale = Locale.default) ?notation s vs =
  check_notation "of_values" notation s;
  let vs, positions = arrange s vs in
  let notation = Option.map decimal_notation notation in
  let labels, note = label ~shortest:Decimal.shortest locale notation s vs in
  ticks_of positions labels note []

(* Choosing *)

type 'd best = {
  score : float;
  n : int;
  key : int list;
  positions : float array;
  labels : (string * string option) array;
  note : string option;
  minor : unit -> 'd array;
}

type 'd search = {
  s : 'd Scale.t;
  norm : 'd -> float;
  length : float;
  rho_t : float;
  locale : Locale.t;
  notation : Decimal.notation option;
  extent : string -> float;
  shortest : float -> Decimal.t;
  value : Steps.step -> int -> float;
  mutable best : 'd best option;
}

let tick_extent st (l, c) =
  match c with
  | None -> st.extent l
  | Some c -> Float.max (st.extent l) (st.extent c)

(* [beats st score n key] is [true] iff a candidate of [n] ticks, [score] and
   order [key] is better than the best so far. *)
let beats st score n key =
  match st.best with
  | None -> true
  | Some b ->
      score > b.score
      || score = b.score
         && (n < b.n || (n = b.n && List.compare Int.compare key b.key < 0))

(* [can_beat st ub] is [true] iff a candidate whose score is at most [ub] may be
   chosen. *)
let can_beat st ub = match st.best with None -> true | Some b -> ub >= b.score

(* [density_bound st n] bounds the density of a candidate of at least [n] ticks,
   whose positions span at most [1]. *)
let density_bound st n =
  if n -. 1. > st.rho_t then 2. -. ((n -. 1.) /. st.rho_t) else 1.

(* The weights of simplicity, coverage and density in a score, after Talbot, Lin
   and Hanrahan. *)
let w_simplicity = 0.25
let w_coverage = 0.2
let w_density = 0.5
let score s c d = (w_simplicity *. s) +. (w_coverage *. c) +. (w_density *. d)

(* [worth st s_max n] is [true] iff a candidate of simplicity at most [s_max]
   and at least [n] ticks may be chosen: its coverage is at most [1]. *)
let worth st s_max n = can_beat st (score s_max 1. (density_bound st n))

(* [skips st ~simplest ~last f] calls [f j] for the skips [j] from [1], up to
   [last], while a candidate of simplicity [simplest - j] may be chosen. *)
let skips st ~simplest ?(last = max_int) f =
  let rec loop j =
    if j <= last && worth st (simplest -. Float.of_int j) 0. then begin
      f j;
      loop (j + 1)
    end
  in
  loop 1

let overlaps st positions labels =
  let rec loop k =
    k + 1 < Array.length positions
    && ((positions.(k + 1) -. positions.(k)) *. st.length
        < (tick_extent st labels.(k) +. tick_extent st labels.(k + 1)) /. 2.
       || loop (k + 1))
  in
  loop 0

let sq x = x *. x

(* [increasing st vs] is the values [vs], increasing or decreasing, and their
   positions, both in increasing order of the positions. *)
let increasing st vs =
  let n = Array.length vs in
  let positions = Array.map st.norm vs in
  if n > 0 && positions.(0) > positions.(n - 1) then
    let rev a = Array.init n (fun i -> a.(n - 1 - i)) in
    (rev vs, rev positions)
  else (vs, positions)

(* [consider st ~s_term ~key ~minor vs] scores the candidate of the distinct
   values [vs], increasing, and keeps it if it is the best so far. Labels are
   made and measured only for a candidate that would be. *)
let consider st ~s_term ~key ~minor vs =
  let n = Array.length vs in
  if n > 0 then begin
    let vs, positions = increasing st vs in
    let p1 = positions.(0) and pn = positions.(n - 1) in
    let c = 1. -. (50. *. (sq (1. -. pn) +. sq p1)) in
    let rho = if n = 1 then 1. else Float.of_int (n - 1) /. (pn -. p1) in
    let d = 2. -. Float.max (rho /. st.rho_t) (st.rho_t /. rho) in
    let score = score s_term c d in
    if beats st score n key then
      let labels, note =
        label ~shortest:st.shortest st.locale st.notation st.s vs
      in
      if not (overlaps st positions labels) then
        st.best <- Some { score; n; key; positions; labels; note; minor }
  end

(* [residue i j] is [i] modulo [j > 0], in \[[0];[j - 1]\]. *)
let residue i j = ((i mod j) + j) mod j

(* Decimal steps: [q] of [Q] as a step [m × 10^(z + dk)] at the exponent [z],
   and the step of its minor ticks. *)
let q_steps = [| (1, 0); (5, 0); (2, 0); (25, -1) |]
let q_minor_steps = [| (2, -1); (1, 0); (5, -1); (5, -1) |]

(* [q_rank i] is the simplicity [q_steps.(i)] loses to the first. *)
let q_rank i = Float.of_int i /. Float.of_int (Array.length q_steps - 1)

(* [skips_to_q j i] is [true] iff [j] times the step [q_steps.(i)] is a step of
   [Q] at some power of ten: the skip [2] turns the step [0.1] into [0.2], a
   step of [Q], and the skip [3] into [0.3], which is not. *)
let skips_to_q j i =
  let rec mantissa n = if n mod 10 = 0 then mantissa (n / 10) else n in
  Array.exists (fun (m, _) -> m = mantissa (j * fst q_steps.(i))) q_steps

(* [decimal_steps st ~v (a, b)] is the search of the decimal steps in two parts:
   a candidate of one tick, then every other. *)
let decimal_steps st ~v (a, b) =
  let mag = Float.max (Float.abs a) (Float.abs b) in
  let z_top = Float.to_int (Float.floor (Float.log10 mag)) + 2 in
  let half = (b /. 2.) -. (a /. 2.) in
  let ulp = mag -. Float.pred mag in
  let step (m, dk) z = { Steps.m; k = z + dk } in
  let candidate j i z r =
    let s = step q_steps.(i) z in
    let vs = Steps.multiples ~value:st.value ~skip:j ~offset:r s a b in
    let minor () =
      Steps.multiples ~value:st.value
        (if j > 1 then s else step q_minor_steps.(i) z)
        a b
    in
    let s_term = 1. -. q_rank i -. Float.of_int j +. v vs in
    consider st ~s_term ~key:[ 0; j; i; -z; r ] ~minor vs
  in
  (* A lone tick: [j] beyond the multiples of the coarsest power of ten that has
     one in the domain. *)
  let rec seed z =
    let s = step q_steps.(0) z in
    match Steps.multiples ~value:st.value s a b with
    | [||] -> seed (z - 1)
    | vs ->
        let first = Float.to_int (Float.round (Steps.index s vs.(0))) in
        let j =
          Float.to_int (Float.round (Steps.index s vs.(Array.length vs - 1)))
          - first + 1
        in
        candidate j 0 z (residue first j)
  in
  let rec descend j i z =
    let s = step q_steps.(i) z in
    let delta = Float.of_int j *. Steps.width s in
    (* Points [delta] apart rounded to floats [ulp] apart at most, over the
       domain less [delta] at each end. *)
    let n_lb =
      ((half -. delta -. (ulp /. 2.)) /. ((ulp +. delta) /. 2.)) +. 1.
    in
    let s_max = 2. -. q_rank i -. Float.of_int j in
    if worth st s_max n_lb then
      if not (Steps.is_fine s a b) then begin
        for r = 0 to j - 1 do
          candidate j i z r
        done;
        descend j i (z - 1)
      end
  in
  let search () =
    skips st ~simplest:2. (fun j ->
        for i = 0 to Array.length q_steps - 1 do
          let s_max = 2. -. q_rank i -. Float.of_int j in
          if skips_to_q j i && worth st s_max 0. then descend j i z_top
        done)
  in
  ((fun () -> seed z_top), search)

let powers_family st ~v base (a, b) =
  let lb = Float.log base in
  let e_lo = Float.to_int (Float.floor (Float.log a /. lb)) - 1
  and e_hi = Float.to_int (Float.ceil (Float.log b /. lb)) + 1 in
  (* [multiples ns keep] is the [n × base^i] inside the domain for [n] in [ns]
     and the exponents [i] that [keep] keeps. *)
  let multiples ns keep =
    Steps.collect (fun push ->
        for i = e_lo to e_hi do
          if keep i then
            List.iter
              (fun n ->
                let x = Steps.multiple base n i in
                if a <= x && x <= b then push x)
              ns
        done)
  in
  let all _ = true in
  let integer = Steps.is_integer_base base in
  let by =
    if integer then List.init (Float.to_int base - 2) (fun n -> n + 2) else []
  in
  (* At least [inside] consecutive powers lie in the domain, and a candidate
     whose least number of ticks [n] already scores too low is not made. *)
  let inside =
    Float.floor (Float.log b /. lb) -. Float.ceil (Float.log a /. lb) -. 1.
  in
  let form i ns ~minor =
    let per = Float.of_int (List.length ns) in
    if worth st (2. -. (Float.of_int i /. 2.) -. 1.) (per *. (inside -. 1.))
    then
      let vs = multiples ns all in
      let s_term = 1. -. (Float.of_int i /. 2.) -. 1. +. v vs in
      consider st ~s_term ~key:[ 1; 1; i; 0; 0 ] ~minor vs
  in
  let powers j =
    if worth st (2. -. Float.of_int j) (Float.floor (inside /. Float.of_int j))
    then
      for r = 0 to j - 1 do
        let vs = multiples [ 1 ] (fun i -> residue i j = r) in
        let minor () =
          if j > 1 then multiples [ 1 ] all else multiples by all
        in
        consider st
          ~s_term:(1. -. Float.of_int j +. v vs)
          ~key:[ 1; j; 0; 0; r ] ~minor vs
      done
  in
  if base = 10. then
    form 1 [ 1; 2; 5 ] ~minor:(fun () -> multiples [ 3; 4; 6; 7; 8; 9 ] all);
  if by <> [] then form 2 (1 :: by) ~minor:(fun () -> [||]);
  (* Beyond a skip of [e_hi - e_lo + 1], each offset keeps one power at most, as
     a smaller skip did. *)
  skips st ~simplest:2. ~last:(e_hi - e_lo + 1) powers

(* [signed_exponents c (a, b) sign] is the exponents [i], increasing, of the
   powers of ten of magnitude at least [c] whose values [sign × 10^i] lie in
   \[[a];[b]\]. *)
let signed_exponents c (a, b) sign =
  List.filter
    (fun i ->
      let p = Steps.pow10 i in
      p >= c && a <= sign *. p && sign *. p <= b)
    (List.init 632 (fun i -> i - 323))

(* [signed_powers (a, b) ~pos ~neg keep] is, increasing, the negated powers of
   the exponents [neg], [0.] if it lies in \[[a];[b]\], and the powers of the
   exponents [pos], of the exponents that [keep] keeps. *)
let signed_powers (a, b) ~pos ~neg keep =
  Array.of_list
    (List.concat
       [
         List.rev_map (fun i -> -.Steps.pow10 i) (List.filter keep neg);
         (if a <= 0. && 0. <= b then [ 0. ] else []);
         List.map Steps.pow10 (List.filter keep pos);
       ])

let signed_family st ~v c (a, b) =
  let pos = signed_exponents c (a, b) 1. in
  let neg = signed_exponents c (a, b) (-1.) in
  let width =
    match List.sort Int.compare (pos @ neg) with
    | [] -> 1
    | first :: _ as l -> List.nth l (List.length l - 1) - first + 1
  in
  let values = signed_powers (a, b) ~pos ~neg in
  skips st ~simplest:2. ~last:width (fun j ->
      for r = 0 to j - 1 do
        let vs = values (fun i -> residue i j = r) in
        let minor () = if j > 1 then values (fun _ -> true) else [||] in
        consider st
          ~s_term:(1. -. Float.of_int j +. v vs)
          ~key:[ 2; j; 0; 0; r ] ~minor vs
      done)

let calendar_family st tz ((a : Time.t), (b : Time.t)) =
  let steps = Steps.time_steps in
  let count = Array.length steps in
  let length = Steps.ns_diff b a in
  let first (e : Steps.time_step) =
    match Time.ceil ~tz_offset_s:tz e.interval a with
    | f when Time.compare f b <= 0 -> Some f
    | _ | (exception Invalid_argument _) -> None
  in
  (* [pick e f j r] is every [j]th boundary of [e] from the [(r + 1)]th, [f]
     being the first in the domain. *)
  let pick (e : Steps.time_step) f j r =
    let rec loop acc k =
      match Time.add ~tz_offset_s:tz e.interval k f with
      | x when Time.compare x b <= 0 -> loop (x :: acc) (k + j)
      | _ | (exception Invalid_argument _) -> Array.of_list (List.rev acc)
    in
    loop [] r
  in
  let candidate j idx f r =
    let e = steps.(idx) in
    let vs = pick e f j r in
    let minor () =
      if j > 1 then pick e f 1 0
      else
        match Steps.minor_step e with
        | Some m -> Time.range ~tz_offset_s:tz m.interval a b
        | None -> [||]
    in
    let s_term =
      1. -. (Float.of_int (e.i - 1) /. Float.of_int (e.n - 1)) -. Float.of_int j
    in
    consider st ~s_term ~key:[ 3; j; e.i; count - 1 - idx; r ] ~minor vs;
    Array.length vs > 0
  in
  (* A lone tick: the first boundary of the coarsest interval that has one,
     skipping as many as it has. *)
  let rec seed idx =
    match first steps.(idx) with
    | None -> seed (idx - 1)
    | Some f ->
        ignore (candidate (Array.length (pick steps.(idx) f 1 0)) idx f 0)
  in
  let intervals j =
    for rank = 1 to 4 do
      for idx = count - 1 downto 0 do
        let e = steps.(idx) in
        let s_max =
          1.
          -. (Float.of_int (e.i - 1) /. Float.of_int (e.n - 1))
          -. Float.of_int j
        in
        let span = Float.of_int j *. e.longest in
        let n_lb = ((length -. (2. *. span)) /. span) +. 1. in
        if e.i = rank && worth st s_max n_lb then
          match first e with
          | None -> ()
          | Some f ->
              let rec offsets r =
                if r < j && candidate j idx f r then offsets (r + 1)
              in
              offsets 0
      done
    done
  in
  seed (count - 1);
  skips st ~simplest:1. intervals

(* Density references *)

(* About ten values of the scale's own family, whose labels set the density
   target of [choose]. *)
let reference_count = 10

(* The multiples of the decimal step for a tenth of \[[a];[b]\], the step a nice
   domain rounds to. *)
let decimal_reference a b = Steps.multiples (Steps.step a b reference_count) a b

(* In an integer base over fewer than ten powers, the powers with their
   multiples; otherwise the powers whose exponents are multiples of the decimal
   step for a tenth of the exponents' span, at least [1]; and the decimal
   reference if that gives fewer than half of ten values. *)
let log_reference base a b =
  let lb = Float.log base in
  let ea = Float.log a /. lb and eb = Float.log b /. lb in
  let i0 = Float.to_int (Float.floor ea) - 1
  and i1 = Float.to_int (Float.ceil eb) + 1 in
  let inside v = a <= v && v <= b in
  let vs =
    if Steps.is_integer_base base && eb -. ea < Float.of_int reference_count
    then
      Steps.collect (fun push ->
          for i = i0 to i1 do
            for j = 1 to Float.to_int base - 1 do
              let v = Steps.multiple base j i in
              if inside v then push v
            done
          done)
    else
      let st = Steps.step ea eb reference_count in
      let st = if st.k < 0 then { Steps.m = 1; k = 0 } else st in
      Steps.collect (fun push ->
          Array.iter
            (fun e ->
              let v = Steps.power base (Float.to_int e) in
              if inside v then push v)
            (Steps.multiples st (Float.of_int i0) (Float.of_int i1)))
  in
  if 2 * Array.length vs < reference_count then decimal_reference a b else vs

(* The signed powers whose exponents are multiples of the least stride [k] that
   gives at most ten values, or of the one that gives the fewest; and the
   decimal reference if that gives fewer than two values. *)
let symlog_reference c a b =
  let pos = signed_exponents c (a, b) 1. in
  let neg = signed_exponents c (a, b) (-1.) in
  let zero = a <= 0. && 0. <= b in
  let span = List.fold_left (fun m i -> Int.max m (Int.abs i)) 0 (pos @ neg) in
  let size k =
    let n l = List.length (List.filter (fun i -> i mod k = 0) l) in
    (if zero then 1 else 0) + n pos + n neg
  in
  let rec stride r best =
    let k = Steps.stride r in
    let n = size k in
    if n <= reference_count then k
    else
      let best =
        match best with Some (_, m) when m <= n -> best | _ -> Some (k, n)
      in
      if k > span then fst (Option.get best) else stride (r + 1) best
  in
  let k = stride 0 None in
  let vs = signed_powers (a, b) ~pos ~neg (fun i -> i mod k = 0) in
  if Array.length vs < 2 then decimal_reference a b else vs

(* The most ticks [choose] aims for, however long the axis, and the most a band
   axis shows. A reader takes in no more on one axis, and the cap bounds the
   search: the density bounds it by the target, and strides start from the least
   that gives no more ticks. Without the cap, an axis of billions of points
   would make candidates of billions of ticks. *)
let most_ticks = 100.

(* [strides st names] chooses the least stride of at most [most_ticks] ticks
   whose labels do not overlap. A reader cannot place a category between two
   labels, so a band axis labels as many categories as have room. *)
let strides st names =
  let n = Array.length names in
  let count k = (n + k - 1) / k in
  let rec go j =
    let k = Steps.stride (j - 1) in
    let vs, positions =
      increasing st (Array.init (count k) (fun i -> names.(i * k)))
    in
    let labels, note =
      label ~shortest:st.shortest st.locale st.notation st.s vs
    in
    if k < n && overlaps st positions labels then go (j + 1)
    else
      let minor () = [||] in
      let n = Array.length vs in
      st.best <-
        Some { score = 0.; n; key = []; positions; labels; note; minor }
  in
  let rec least j =
    let k = Steps.stride (j - 1) in
    if Float.of_int (count k) > most_ticks then least (j + 1) else j
  in
  go (least 1)

let choose (type d) ?(locale = Locale.default) ?notation ?(spacing = 0.) ~length
    ~measure (s : d Scale.t) : t =
  if not (Float.is_finite length && length > 0.) then
    err "choose" "length %g is not finite and positive" length;
  if not (Float.is_finite spacing && spacing >= 0.) then
    err "choose" "spacing %g is not finite and non-negative" spacing;
  check_notation "choose" notation s;
  let notation = Option.map decimal_notation notation in
  let extents = Hashtbl.create 64 in
  let extent l =
    match Hashtbl.find_opt extents l with
    | Some e -> e
    | None ->
        let e = measure l in
        if not (Float.is_finite e && e > 0.) then
          err "choose" "the extent %g of %S is not finite and positive" e l;
        Hashtbl.add extents l e;
        e
  in
  (* The exact values and decimals one search reads again and again. *)
  let memo f =
    let table = Hashtbl.create 64 in
    fun x ->
      match Hashtbl.find_opt table x with
      | Some y -> y
      | None ->
          let y = f x in
          Hashtbl.add table x y;
          y
  in
  let shortest = memo Decimal.shortest in
  let value =
    let exact = memo (fun (m, k, i) -> Steps.value { m; k } i) in
    fun (s : Steps.step) i -> exact (s.m, s.k, i)
  in
  let st =
    {
      s;
      norm = Scale.normalize s;
      length;
      rho_t = 1.;
      locale;
      notation;
      extent;
      shortest;
      value;
      best = None;
    }
  in
  let chosen st =
    match st.best with
    | None -> assert false
    | Some best ->
        let _, positions = arrange s (best.minor ()) in
        let minor =
          List.filter
            (fun p -> not (Array.exists (Float.equal p) best.positions))
            (Array.to_list positions)
        in
        ticks_of best.positions best.labels best.note
          (List.sort_uniq Float.compare minor)
  in
  (* [continuous a reference search] is the ticks of a continuous domain from
     [a]. The labels of the values [reference ()], those a nice domain rounds
     to, set the density target: [m] ticks of their mean extent fill half the
     axis, and [ρt] is the lesser of their [m - 1] intervals and the [length /
     spacing] intervals that fit at the spacing. *)
  let continuous a reference search =
    let u = st.norm a in
    if Float.is_nan u then ticks_of [||] [||] None []
    else if Float.equal u 0.5 then
      (* Only a constant domain normalises an end to [0.5]. *)
      let labels, note = label ~shortest locale notation s [| a |] in
      ticks_of [| 0.5 |] labels note []
    else
      (* Every reference holds a value of a domain that is not constant, so
         [mean] is a number. *)
      let reference, _ = arrange s (reference ()) in
      let labels, _ = label ~shortest locale notation s reference in
      let mean =
        Array.fold_left (fun m l -> m +. tick_extent st l) 0. labels
        /. Float.of_int (Array.length labels)
      in
      let m = Float.min (length /. (2. *. mean)) most_ticks in
      let rho_t = Float.min (m -. 1.) (length /. spacing) in
      let st = { st with rho_t = Float.max 1. rho_t } in
      search st;
      chosen st
  in
  match Scale.domain s with
  | Floats (a, b) ->
      let tf = Scale.transform s in
      let one = match tf with Log _ -> 1. | _ -> 0. in
      let v vs = if Array.exists (Float.equal one) vs then 1. else 0. in
      let reference () =
        match tf with
        | Linear | Pow _ | Custom _ -> decimal_reference a b
        | Log base -> log_reference base a b
        | Symlog c -> symlog_reference c a b
      in
      continuous a reference (fun st ->
          (* A lone tick, then the powers, bound the decimal steps; the order
             keys keep ties as the order of candidates states. *)
          let seed, search = decimal_steps st ~v (a, b) in
          seed ();
          (match tf with
          | Linear | Pow _ | Custom _ -> ()
          | Log base -> powers_family st ~v base (a, b)
          | Symlog c -> signed_family st ~v c (a, b));
          search ())
  | Instants (a, b) ->
      let tz = Scale.tz_offset_s s in
      (* The boundaries of the nice interval, or of the next finer one that has
         one in the domain: the finest has one in every domain that is not
         constant. *)
      let rec reference i () =
        let i' = Steps.time_steps.(i).interval in
        match Time.range ~tz_offset_s:tz i' a b with
        | [||] -> reference (i - 1) ()
        | vs -> vs
      in
      let tenth = Steps.ns_diff b a /. Float.of_int reference_count in
      continuous a
        (reference (Steps.nearest_time_step tenth))
        (fun st -> calendar_family st tz (a, b))
  | Categories (Labels names) ->
      strides st names;
      chosen st
  | Categories (Indices ix) ->
      strides st (Array.map (fun (i, _) -> string_of_int i) ix);
      chosen st

(* Comparing and formatting *)

let equal_tick t t' =
  Float.equal t.position t'.position
  && String.equal t.label t'.label
  && Option.equal String.equal t.context t'.context

let equal t t' =
  List.equal equal_tick t.major t'.major
  && List.equal Float.equal t.minor t'.minor
  && Option.equal String.equal t.note t'.note

(* [pp_text] formats a string between quotes, its UTF-8 as it is. *)
let pp_text ppf s = Format.fprintf ppf "\"%s\"" s

let pp ppf t =
  let pp_tick ppf tk =
    Format.fprintf ppf "@[<1>(%g %a%a)@]" tk.position pp_text tk.label
      (fun ppf -> Option.iter (Format.fprintf ppf "@ %a" pp_text))
      tk.context
  in
  let pp_minor ppf = function
    | [] -> ()
    | m ->
        Format.fprintf ppf "@ @[<1>(minor%a)@]"
          (fun ppf -> List.iter (Format.fprintf ppf "@ %g"))
          m
  in
  Format.fprintf ppf "@[<1>(ticks%a%a%a)@]"
    (fun ppf -> List.iter (Format.fprintf ppf "@ %a" pp_tick))
    t.major pp_minor t.minor
    (fun ppf -> Option.iter (Format.fprintf ppf "@ @[<1>(note %a)@]" pp_text))
    t.note
