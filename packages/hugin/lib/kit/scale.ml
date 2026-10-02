(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_gg

type _ kind =
  | Quantitative : float kind
  | Temporal : Time.t kind
  | Categorical : string kind

type categories = Labels of string array | Indices of (int * string) array

type _ domain =
  | Floats : float * float -> float domain
  | Instants : Time.t * Time.t -> Time.t domain
  | Categories : categories -> string domain

(* The transform of a quantitative scale, a custom one with its functions. *)
type tf =
  | Linear
  | Log of float
  | Symlog of float
  | Pow of float
  | Custom of {
      name : string;
      forward : float -> float;
      inverse : float -> float;
    }

(* Each [option] field is a property, [None] when unset. Temporal and band
   scales have the transform [Linear], which they never read. *)
type 'd t = {
  kind : 'd kind;
  transform : tf;
  name : string option;
  domain : 'd domain option;
  nice : bool option;
  zero : bool option;
  clamp : bool option;
  reverse : bool option;
  stepped : bool option;
  ticks : 'd array option;
  notation : Number.notation option;
  padding : float option;
  wrap : int option;
  tz_offset_s : int option;
  scheme : Scheme.t option;
  areas : (float * float) option;
  symbols : Symbol.t array option;
  dashes : Dash.t array option;
  unknown : Color.t option;
}

type property =
  | Name
  | Transform
  | Domain
  | Nice
  | Zero
  | Clamp
  | Reverse
  | Stepped
  | Ticks
  | Notation
  | Padding
  | Wrap
  | Tz_offset_s
  | Scheme
  | Areas
  | Symbols
  | Dashes
  | Unknown

let err fn fmt =
  Format.kasprintf (fun s -> invalid_arg ("Scale." ^ fn ^ ": " ^ s)) fmt

let is_set = function Some true -> true | Some false | None -> false

(* Transforms *)

let symlog c x =
  let y = Float.abs x /. c in
  let v =
    if Float.is_finite y then Float.log1p y
    else Float.log (Float.abs x) -. Float.log c
  in
  Float.copy_sign v x

let pow e x = Float.copy_sign (Float.pow (Float.abs x) e) x

(* [missing tf x] is [true] iff [x] is missing for a quantitative scale of
   transform [tf]. *)
let missing tf x =
  (not (Float.is_finite x))
  ||
  match tf with
  | Log _ -> x <= 0.
  | Custom c -> not (Float.is_finite (c.forward x))
  | Linear | Symlog _ | Pow _ -> false

(* [forward tf a b] is the transform of a quantitative scale over
   \[[a];[b]\]. *)
let forward tf a b =
  match tf with
  | Linear -> Fun.id
  | Log base ->
      let lb = Float.log base in
      fun x -> Float.log x /. lb
  | Symlog c -> symlog c
  | Pow e ->
      let m = Float.max (Float.abs a) (Float.abs b) in
      if m = 0. then pow e else fun x -> pow e (x /. m)
  | Custom c -> c.forward

let backward tf a b =
  match tf with
  | Linear -> Fun.id
  | Log base ->
      let lb = Float.log base in
      fun v -> Float.exp (v *. lb)
  | Symlog c -> fun v -> Float.copy_sign (c *. Float.expm1 (Float.abs v)) v
  | Pow e ->
      let m = Float.max (Float.abs a) (Float.abs b) in
      fun v -> m *. pow (1. /. e) v
  | Custom c -> c.inverse

let is_constant tf a b =
  let t = forward tf a b in
  Float.equal (t a) (t b)

(* Validation *)

let check_floats fn tf (a, b) =
  if not (Float.is_finite a && Float.is_finite b && a <= b) then
    err fn "domain (%g, %g) is not finite and ordered" a b;
  if missing tf a || missing tf b then
    err fn "domain (%g, %g) has a missing end" a b

let check_categories fn = function
  | Labels l ->
      let seen = Hashtbl.create (Array.length l) in
      Array.iter
        (fun x ->
          if Hashtbl.mem seen x then err fn "label %S repeats" x;
          Hashtbl.add seen x ())
        l
  | Indices ix ->
      for k = 1 to Array.length ix - 1 do
        if fst ix.(k) <= fst ix.(k - 1) then
          err fn "indices are not strictly increasing"
      done

let copy_categories = function
  | Labels l -> Labels (Array.copy l)
  | Indices ix -> Indices (Array.copy ix)

let check_domain : type d. string -> tf -> d domain -> d domain =
 fun fn tf d ->
  match d with
  | Floats (a, b) ->
      check_floats fn tf (a, b);
      d
  | Instants (a, b) ->
      if Time.compare a b > 0 then err fn "domain ends are not ordered";
      d
  | Categories c ->
      check_categories fn c;
      Categories (copy_categories c)

let check_tz fn tz =
  if tz <= -86_400 || tz >= 86_400 then
    err fn "offset %d s not in ]-86400;86400[" tz

let check_areas fn (a0, a1) =
  if not (Float.is_finite a0 && Float.is_finite a1 && a0 >= 0. && a1 >= 0.) then
    err fn "areas (%g, %g) are not finite and not negative" a0 a1

(* Constructors *)

let make fn kind transform ?name ?domain ?nice ?zero ?clamp ?reverse ?stepped
    ?ticks ?notation ?padding ?wrap ?tz_offset_s ?scheme ?areas ?symbols ?dashes
    ?unknown () =
  Option.iter (check_areas fn) areas;
  Option.iter
    (fun ss -> if Array.length ss = 0 then err fn "no symbols")
    symbols;
  Option.iter (fun ds -> if Array.length ds = 0 then err fn "no dashes") dashes;
  Option.iter (check_tz fn) tz_offset_s;
  Option.iter
    (fun p ->
      if not (p >= 0. && p <= 1.) then err fn "padding %g not in [0;1]" p)
    padding;
  Option.iter (fun w -> if w < 1 then err fn "wrap %d below 1" w) wrap;
  let domain = Option.map (check_domain fn transform) domain in
  {
    kind;
    transform;
    name;
    domain;
    nice;
    zero;
    clamp;
    reverse;
    stepped;
    ticks = Option.map Array.copy ticks;
    notation;
    padding;
    wrap;
    tz_offset_s;
    scheme;
    areas;
    symbols = Option.map Array.copy symbols;
    dashes = Option.map Array.copy dashes;
    unknown;
  }

let floats = Option.map (fun (a, b) -> Floats (a, b))

let linear ?name ?domain ?nice ?zero ?clamp ?reverse ?stepped ?ticks ?notation
    ?scheme ?areas ?unknown () =
  make "linear" Quantitative Linear ?name ?domain:(floats domain) ?nice ?zero
    ?clamp ?reverse ?stepped ?ticks ?notation ?scheme ?areas ?unknown ()

let log ?(base = 10.) ?name ?domain ?nice ?clamp ?reverse ?stepped ?ticks
    ?notation ?scheme ?areas ?unknown () =
  if not (Float.is_finite base && base > 1.) then
    err "log" "base %g is not finite above 1" base;
  make "log" Quantitative (Log base) ?name ?domain:(floats domain) ?nice ?clamp
    ?reverse ?stepped ?ticks ?notation ?scheme ?areas ?unknown ()

let symlog ?(constant = 1.) ?name ?domain ?nice ?zero ?clamp ?reverse ?stepped
    ?ticks ?notation ?scheme ?areas ?unknown () =
  if not (Float.is_finite constant && constant > 0.) then
    err "symlog" "constant %g is not finite and positive" constant;
  make "symlog" Quantitative (Symlog constant) ?name ?domain:(floats domain)
    ?nice ?zero ?clamp ?reverse ?stepped ?ticks ?notation ?scheme ?areas
    ?unknown ()

let pow ~exponent ?name ?domain ?nice ?zero ?clamp ?reverse ?stepped ?ticks
    ?notation ?scheme ?areas ?unknown () =
  if not (Float.is_finite exponent && exponent > 0.) then
    err "pow" "exponent %g is not finite and positive" exponent;
  make "pow" Quantitative (Pow exponent) ?name ?domain:(floats domain) ?nice
    ?zero ?clamp ?reverse ?stepped ?ticks ?notation ?scheme ?areas ?unknown ()

let custom ~transform ~forward ~inverse ?name ?domain ?nice ?zero ?clamp
    ?reverse ?stepped ?ticks ?notation ?scheme ?areas ?unknown () =
  make "custom" Quantitative
    (Custom { name = transform; forward; inverse })
    ?name ?domain:(floats domain) ?nice ?zero ?clamp ?reverse ?stepped ?ticks
    ?notation ?scheme ?areas ?unknown ()

let time ?name ?domain ?nice ?clamp ?reverse ?tz_offset_s ?ticks ?scheme
    ?unknown () =
  make "time" Temporal Linear ?name
    ?domain:(Option.map (fun (a, b) -> Instants (a, b)) domain)
    ?nice ?clamp ?reverse ?tz_offset_s ?ticks ?scheme ?unknown ()

let band ?name ?domain ?padding ?reverse ?wrap ?ticks ?scheme ?symbols ?dashes
    ?unknown () =
  make "band" Categorical Linear ?name
    ?domain:(Option.map (fun c -> Categories c) domain)
    ?padding ?reverse ?wrap ?ticks ?scheme ?symbols ?dashes ?unknown ()

(* Properties *)

let kind s = s.kind

let equal_kind : type a b. a kind -> b kind -> (a, b) Type.eq option =
 fun k k' ->
  match (k, k') with
  | Quantitative, Quantitative -> Some Type.Equal
  | Temporal, Temporal -> Some Type.Equal
  | Categorical, Categorical -> Some Type.Equal
  | (Quantitative | Temporal | Categorical), _ -> None

let name s = s.name

let default_domain : type d. d t -> d domain =
 fun s ->
  match s.kind with
  | Quantitative -> (
      match s.transform with Log b -> Floats (1., b) | _ -> Floats (0., 1.))
  | Temporal -> Instants (Time.epoch, Time.add (Time.days 1) 1 Time.epoch)
  | Categorical -> Categories (Labels [||])

(* [domain_of s] is the domain of [s], categories not copied. *)
let domain_of s = match s.domain with None -> default_domain s | Some d -> d

let domain : type d. d t -> d domain =
 fun s ->
  match domain_of s with
  | Categories c -> Categories (copy_categories c)
  | d -> d

let padding s = Option.value ~default:0. s.padding

let names = function
  | Labels l -> l
  | Indices ix -> Array.map (fun (i, _) -> string_of_int i) ix

let size = function Labels l -> Array.length l | Indices ix -> Array.length ix

let bandwidth s =
  let (Categories c) = domain_of s in
  let n = Float.of_int (size c) and p = padding s in
  if n = 0. then 0. else (1. -. p) /. (n +. p)

let length : type d. d t -> float =
 fun s ->
  match domain_of s with
  | Floats (a, b) ->
      let tf = s.transform in
      if missing tf a || missing tf b then Float.nan
      else
        let t = forward tf a b in
        Float.abs (t b -. t a)
  | Instants (a, b) -> Float.abs (Steps.ns_diff b a) /. 1e9
  | Categories c ->
      let n = size c in
      if n = 0 then 0. else Float.of_int n +. padding s

let wrap s = s.wrap
let scheme s = s.scheme
let stepped s = is_set s.stepped
let ticks s = Option.map Array.copy s.ticks
let notation s = s.notation
let areas s = s.areas
let symbols s = Option.map Array.copy s.symbols
let dashes s = Option.map Array.copy s.dashes
let unknown s = s.unknown
let tz_offset_s s = Option.value ~default:0 s.tz_offset_s

(* Instants as floats *)

(* [add_sec s d] is [s + d], or [None] if it overflows. *)
let add_sec s d =
  let r = Int64.add s d in
  (* The sum overflowed iff [s] and [d] have one sign and [r] the other. *)
  if Int64.compare (Int64.logand (Int64.logxor s r) (Int64.logxor d r)) 0L < 0
  then None
  else Some r

(* [ns_add a d] is [a] moved by the integer [d] nanoseconds, if representable.
   [d] is [s] seconds and the exact remainder [r], within a few seconds of [0],
   and the seconds are added in two halves of one sign, each in int64 range. *)
let ns_add (a : Time.t) d =
  let s = Float.floor (d /. 1e9) in
  if not (Float.abs s < 0x1p64) then None
  else
    let r = Float.to_int (Float.fma (-.s) 1e9 d) + a.nsec in
    let s1 = Float.trunc (s /. 2.) in
    match add_sec a.sec (Int64.of_float s1) with
    | None -> None
    | Some sec -> (
        match add_sec sec (Int64.of_float (s -. s1)) with
        | None -> None
        | Some sec -> (
            match Time.add (Time.nanoseconds 1) r (Time.v S sec) with
            | t -> Some t
            | exception Invalid_argument _ -> None))

(* Normalising and inverting *)

(* [clamp u] is [u] clamped into \[[0];[1]\], [-0.] made [0.]. *)
let clamp u = if u <= 0. then 0. else Float.min u 1.

let finish s =
  let reverse = is_set s.reverse and clamps = is_set s.clamp in
  fun u ->
    let u = if reverse then 1. -. u else u in
    if clamps then clamp u else u

let unfinish s u =
  let u = if is_set s.clamp then clamp u else u in
  if is_set s.reverse then 1. -. u else u

(* [band_index c] maps the name of each category of [c] to its index. *)
let band_index c =
  let names = names c in
  let index = Hashtbl.create (Array.length names) in
  Array.iteri (fun i x -> Hashtbl.replace index x i) names;
  index

let normalize_index s =
  let finish = finish s in
  let (Categories c) = domain_of s in
  let k = size c in
  let n = Float.of_int k and p = padding s in
  fun i ->
    if i < 0 || i >= k then Float.nan
    else finish ((Float.of_int i +. ((1. +. p) /. 2.)) /. (n +. p))

let normalize : type d. d t -> d -> float =
 fun s ->
  let finish = finish s in
  match domain_of s with
  | Floats (a, b) ->
      let tf = s.transform in
      let t = forward tf a b in
      let ta = t a and tb = t b in
      if Float.equal ta tb then fun x ->
        if missing tf x then Float.nan else finish 0.5
      else
        let w = tb -. ta in
        if Float.is_finite w then fun x ->
          if missing tf x then Float.nan
          else
            let tx = t x in
            let d = tx -. ta in
            if Float.is_finite d then finish (d /. w)
            else finish (((tx /. 2.) -. (ta /. 2.)) /. (w /. 2.))
        else
          let ha = ta /. 2. in
          let w = (tb /. 2.) -. ha in
          fun x ->
            if missing tf x then Float.nan else finish (((t x /. 2.) -. ha) /. w)
  | Instants (a, b) ->
      if Time.equal a b then fun _ -> finish 0.5
      else
        let w = Steps.ns_diff b a in
        fun x -> finish (Steps.ns_diff x a /. w)
  | Categories c -> (
      let index = band_index c and at = normalize_index s in
      fun x ->
        match Hashtbl.find_opt index x with None -> Float.nan | Some i -> at i)

let invert : type d. d t -> float -> d option =
 fun s u ->
  if not (Float.is_finite u) then None
  else
    let u = unfinish s u in
    match domain_of s with
    | Floats (a, b) ->
        let tf = s.transform in
        let t = forward tf a b in
        let ta = t a and tb = t b in
        if Float.equal ta tb then Some a
        else
          let x = backward tf a b (((1. -. u) *. ta) +. (u *. tb)) in
          if missing tf x then None else Some x
    | Instants (a, b) ->
        if Time.equal a b then Some a
        else
          (* From the nearer end, so that both ends invert exactly. *)
          let w = Steps.ns_diff b a in
          if u <= 0.5 then ns_add a (Float.round (u *. w))
          else ns_add b (-.Float.round ((1. -. u) *. w))
    | Categories c ->
        let n = size c in
        let p = padding s in
        let j = (u *. (Float.of_int n +. p)) -. (p /. 2.) in
        if n = 0 || j < 0. || j > Float.of_int n then None
        else
          let i =
            if Float.is_integer j && j > 0. then Float.to_int j - 1
            else Float.to_int j
          in
          Some
            (match c with
            | Labels l -> l.(i)
            | Indices ix -> string_of_int (fst ix.(i)))

(* Specifications and fitting *)

(* [below b x] and [above b x] are the greatest power of [b] not above [x > 0]
   and the least not below it. *)
let below base x =
  let i = ref (Float.to_int (Float.floor (Float.log x /. Float.log base))) in
  while Steps.power base !i > x do
    decr i
  done;
  while Steps.power base (!i + 1) <= x do
    incr i
  done;
  Steps.power base !i

let above base x =
  let i = ref (Float.to_int (Float.ceil (Float.log x /. Float.log base))) in
  while Steps.power base !i < x do
    incr i
  done;
  while Steps.power base (!i - 1) >= x do
    decr i
  done;
  Steps.power base !i

let nice_floats tf a b =
  let keep x x' = if missing tf x' then x else x' in
  match tf with
  | Linear | Pow _ | Custom _ ->
      let rec loop rounds prev a b =
        let st = Steps.step a b 10 in
        if rounds = 10 || prev = Some st then (a, b)
        else
          loop (rounds + 1) (Some st)
            (keep a (Steps.floor_multiple st a))
            (keep b (Steps.ceil_multiple st b))
      in
      loop 0 None a b
  | Log base -> (keep a (below base a), keep b (above base b))
  | Symlog c ->
      let lower x =
        if x > c then Float.max c (below 10. x)
        else if x < -.c then -.above 10. (-.x)
        else if x >= c then c
        else if x >= 0. then 0.
        else -.c
      in
      let upper x = 0. -. lower (-.x) in
      (keep a (lower a), keep b (upper b))

let nice_instants tz a b =
  let keep x f =
    match f x with x' -> x' | exception Invalid_argument _ -> x
  in
  let rec loop rounds prev a b =
    let tenth = Steps.ns_diff b a /. 10. in
    let i = Steps.time_steps.(Steps.nearest_time_step tenth).interval in
    let same =
      match prev with Some p -> Time.equal_interval p i | None -> false
    in
    if rounds = 10 || same then (a, b)
    else
      loop (rounds + 1) (Some i)
        (keep a (Time.floor ~tz_offset_s:tz i))
        (keep b (Time.ceil ~tz_offset_s:tz i))
  in
  loop 0 None a b

let fit_domain : type d. d t -> d domain -> d domain =
 fun s d ->
  let nice = Option.value ~default:true s.nice in
  match check_domain "fit" s.transform d with
  | Floats (a, b) ->
      let tf = s.transform in
      let a, b =
        if is_set s.zero && not (missing tf 0.) then
          (Float.min a 0., Float.max b 0.)
        else (a, b)
      in
      if nice && not (is_constant tf a b) then
        let a, b = nice_floats tf a b in
        Floats (a, b)
      else Floats (a, b)
  | Instants (a, b) ->
      if nice && not (Time.equal a b) then
        let a, b = nice_instants (tz_offset_s s) a b in
        Instants (a, b)
      else Instants (a, b)
  | Categories c -> Categories c

let fit observed s =
  let domain =
    match (s.domain, observed) with
    | (Some _ as d), _ -> d
    | None, None -> Some (default_domain s)
    | None, Some d -> Some (fit_domain s d)
  in
  { s with domain; nice = None; zero = None }

let with_domain d s =
  { s with domain = Some (check_domain "with_domain" s.transform d) }

(* Merging *)

let equal_opt eq x y =
  match (x, y) with
  | None, None -> true
  | Some x, Some y -> eq x y
  | None, Some _ | Some _, None -> false

let compatible eq x y =
  match (x, y) with Some x, Some y -> eq x y | None, _ | _, None -> true

let equal_pair (a, b) (a', b') = Float.equal a a' && Float.equal b b'

let equal_transform tf tf' =
  match (tf, tf') with
  | Linear, Linear -> true
  | Log b, Log b' | Symlog b, Symlog b' | Pow b, Pow b' -> Float.equal b b'
  | Custom c, Custom c' ->
      String.equal c.name c'.name
      && c.forward == c'.forward && c.inverse == c'.inverse
  | (Linear | Log _ | Symlog _ | Pow _ | Custom _), _ -> false

let equal_categories c c' =
  match (c, c') with
  | Labels l, Labels l' ->
      Array.length l = Array.length l' && Array.for_all2 String.equal l l'
  | Indices ix, Indices ix' ->
      Array.length ix = Array.length ix'
      && Array.for_all2
           (fun (i, t) (i', t') -> i = i' && String.equal t t')
           ix ix'
  | (Labels _ | Indices _), _ -> false

let equal_domain : type d. d domain -> d domain -> bool =
 fun d d' ->
  match (d, d') with
  | Floats (a, b), Floats (a', b') -> equal_pair (a, b) (a', b')
  | Instants (a, b), Instants (a', b') -> Time.equal a a' && Time.equal b b'
  | Categories c, Categories c' -> equal_categories c c'

let equal_array eq a a' =
  Array.length a = Array.length a' && Array.for_all2 eq a a'

let equal_symbols = equal_array Symbol.equal

let equal_value : type d. d kind -> d -> d -> bool = function
  | Quantitative -> Float.equal
  | Temporal -> Time.equal
  | Categorical -> String.equal

let equal_notation (n : Number.notation) n' = n = n'

(* A relation between the values of a property set or unset in two scales. *)
type relation = {
  holds : 'a. ('a -> 'a -> bool) -> 'a option -> 'a option -> bool;
}

(* [agreements r s s'] is, in the order of [property], whether [r] holds on each
   property of [s] and [s'], values compared as {!equal} compares them. *)
let agreements r s s' =
  [
    (Name, r.holds String.equal s.name s'.name);
    (Transform, equal_transform s.transform s'.transform);
    (Domain, r.holds equal_domain s.domain s'.domain);
    (Nice, r.holds Bool.equal s.nice s'.nice);
    (Zero, r.holds Bool.equal s.zero s'.zero);
    (Clamp, r.holds Bool.equal s.clamp s'.clamp);
    (Reverse, r.holds Bool.equal s.reverse s'.reverse);
    (Stepped, r.holds Bool.equal s.stepped s'.stepped);
    (Ticks, r.holds (equal_array (equal_value s.kind)) s.ticks s'.ticks);
    (Notation, r.holds equal_notation s.notation s'.notation);
    (Padding, r.holds Float.equal s.padding s'.padding);
    (Wrap, r.holds Int.equal s.wrap s'.wrap);
    (Tz_offset_s, r.holds Int.equal s.tz_offset_s s'.tz_offset_s);
    (Scheme, r.holds Scheme.equal s.scheme s'.scheme);
    (Areas, r.holds equal_pair s.areas s'.areas);
    (Symbols, r.holds equal_symbols s.symbols s'.symbols);
    (Dashes, r.holds (equal_array Dash.equal) s.dashes s'.dashes);
    (Unknown, r.holds Color.equal s.unknown s'.unknown);
  ]

let first o o' = match o with Some _ -> o | None -> o'

let union s s' =
  {
    s with
    name = first s.name s'.name;
    domain = first s.domain s'.domain;
    nice = first s.nice s'.nice;
    zero = first s.zero s'.zero;
    clamp = first s.clamp s'.clamp;
    reverse = first s.reverse s'.reverse;
    stepped = first s.stepped s'.stepped;
    ticks = first s.ticks s'.ticks;
    notation = first s.notation s'.notation;
    padding = first s.padding s'.padding;
    wrap = first s.wrap s'.wrap;
    tz_offset_s = first s.tz_offset_s s'.tz_offset_s;
    scheme = first s.scheme s'.scheme;
    areas = first s.areas s'.areas;
    symbols = first s.symbols s'.symbols;
    dashes = first s.dashes s'.dashes;
    unknown = first s.unknown s'.unknown;
  }

let merge s s' =
  match
    List.find_opt
      (fun (_, agree) -> not agree)
      (agreements { holds = compatible } s s')
  with
  | Some (p, _) -> Error p
  | None -> Ok (union s s')

(* A domain [i] sets was checked against the transform of [i]; of the
   constraints on domains, only missing ends depend on the transform. *)
let imply : type d. d t -> d t -> d t =
 fun i s ->
  let domain =
    match i.domain with
    | Some (Floats (a, b)) when missing s.transform a || missing s.transform b
      ->
        None
    | d -> d
  in
  union s { i with name = None; domain }

let equal s s' = List.for_all snd (agreements { holds = equal_opt } s s')

(* Missing values *)

let is_real (type a b) (dtype : (a, b) Nx.dtype) =
  match dtype with
  | Complex64 | Complex128 | Bool -> false
  | Float16 | Float32 | Float64 | BFloat16 | Float8_e4m3 | Float8_e5m2 | Int4
  | UInt4 | Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 | Int64 | UInt64 ->
      true

let missing s x =
  if not (is_real (Nx.dtype x)) then err "missing" "the tensor is not real";
  let xf = Nx.cast Nx.float64 x in
  match s.transform with
  | Custom _ ->
      let xs = Nx.to_array xf in
      Nx.create Nx.bool (Nx.shape x) (Array.map (missing s.transform) xs)
  | Log _ ->
      Nx.logical_not (Nx.logical_and (Nx.isfinite xf) (Nx.greater_s xf 0.))
  | Linear | Symlog _ | Pow _ -> Nx.logical_not (Nx.isfinite xf)

(* Formatting *)

let property_name = function
  | Name -> "name"
  | Transform -> "transform"
  | Domain -> "domain"
  | Nice -> "nice"
  | Zero -> "zero"
  | Clamp -> "clamp"
  | Reverse -> "reverse"
  | Stepped -> "stepped"
  | Ticks -> "ticks"
  | Notation -> "notation"
  | Padding -> "padding"
  | Wrap -> "wrap"
  | Tz_offset_s -> "tz_offset_s"
  | Scheme -> "scheme"
  | Areas -> "areas"
  | Symbols -> "symbols"
  | Dashes -> "dashes"
  | Unknown -> "unknown"

let pp_property ppf p = Format.pp_print_string ppf (property_name p)

(* [pp_float] formats the fewest significant digits from 15 that read back. *)
let pp_float ppf x =
  let digits = List.map (fun p -> Printf.sprintf "%.*g" p x) [ 15; 16; 17 ] in
  Format.pp_print_string ppf
    (List.find (fun s -> Float.equal (float_of_string s) x) digits)

let pp_transform ppf = function
  | Linear -> Format.pp_print_string ppf "linear"
  | Log b -> Format.fprintf ppf "log %a" pp_float b
  | Symlog c -> Format.fprintf ppf "symlog %a" pp_float c
  | Pow e -> Format.fprintf ppf "pow %a" pp_float e
  | Custom c -> Format.fprintf ppf "custom %s" c.name

let pp_categories ppf = function
  | Labels l ->
      Format.fprintf ppf "@[<1>(labels%a)@]"
        (fun ppf -> Array.iter (Format.fprintf ppf "@ %S"))
        l
  | Indices ix ->
      Format.fprintf ppf "@[<1>(indices%a)@]"
        (fun ppf ->
          Array.iter (fun (i, t) -> Format.fprintf ppf "@ (%d %S)" i t))
        ix

let pp_domain : type d. Format.formatter -> d domain -> unit =
 fun ppf -> function
  | Floats (a, b) -> Format.fprintf ppf "%a %a" pp_float a pp_float b
  | Instants (a, b) -> Format.fprintf ppf "%a %a" Time.pp a Time.pp b
  | Categories c -> pp_categories ppf c

let pp (type d) ppf (s : d t) =
  let head ppf (s : d t) =
    match s.kind with
    | Quantitative -> pp_transform ppf s.transform
    | Temporal -> Format.pp_print_string ppf "time"
    | Categorical -> Format.pp_print_string ppf "band"
  in
  let field name pp_v ppf = function
    | None -> ()
    | Some v -> Format.fprintf ppf "@ @[<1>(%s %a)@]" name pp_v v
  in
  let bool ppf b = Format.pp_print_bool ppf b in
  let value : Format.formatter -> d -> unit =
    match s.kind with
    | Quantitative -> pp_float
    | Temporal -> Time.pp
    | Categorical -> fun ppf -> Format.fprintf ppf "%S"
  in
  let values = Format.pp_print_array ~pp_sep:Format.pp_print_space value in
  Format.fprintf ppf "@[<1>(%a%a%a%a%a%a%a%a%a%a%a%a%a%a%a%a%a%a)@]" head s
    (field "name" (fun ppf -> Format.fprintf ppf "%S"))
    s.name (field "domain" pp_domain) s.domain (field "nice" bool) s.nice
    (field "zero" bool) s.zero (field "clamp" bool) s.clamp
    (field "reverse" bool) s.reverse (field "stepped" bool) s.stepped
    (field "ticks" values) s.ticks
    (field "notation" Number.pp_notation)
    s.notation (field "padding" pp_float) s.padding
    (field "wrap" Format.pp_print_int)
    s.wrap
    (field "tz_offset_s" Format.pp_print_int)
    s.tz_offset_s (field "scheme" Scheme.pp) s.scheme
    (field "areas" (fun ppf (a, b) ->
         Format.fprintf ppf "%a %a" pp_float a pp_float b))
    s.areas
    (field "symbols"
       (Format.pp_print_array ~pp_sep:Format.pp_print_space Symbol.pp))
    s.symbols
    (field "dashes"
       (Format.pp_print_array ~pp_sep:Format.pp_print_space (fun ppf d ->
            Format.fprintf ppf "@[<1>(%a)@]" Dash.pp d)))
    s.dashes (field "unknown" Color.pp) s.unknown

(* Observers *)

let sets (type d) p (s : d t) =
  match p with
  | Name -> Option.is_some s.name
  | Transform -> true
  | Domain -> Option.is_some s.domain
  | Nice -> Option.is_some s.nice
  | Zero -> Option.is_some s.zero
  | Clamp -> Option.is_some s.clamp
  | Reverse -> Option.is_some s.reverse
  | Stepped -> Option.is_some s.stepped
  | Ticks -> Option.is_some s.ticks
  | Notation -> Option.is_some s.notation
  | Padding -> Option.is_some s.padding
  | Wrap -> Option.is_some s.wrap
  | Tz_offset_s -> Option.is_some s.tz_offset_s
  | Scheme -> Option.is_some s.scheme
  | Areas -> Option.is_some s.areas
  | Symbols -> Option.is_some s.symbols
  | Dashes -> Option.is_some s.dashes
  | Unknown -> Option.is_some s.unknown

(* Defined last, so that the constructors above are those of [tf]. *)
type transform =
  | Linear
  | Log of float
  | Symlog of float
  | Pow of float
  | Custom of string

let transform s : transform =
  match s.transform with
  | Linear -> Linear
  | Log b -> Log b
  | Symlog c -> Symlog c
  | Pow e -> Pow e
  | Custom c -> Custom c.name
