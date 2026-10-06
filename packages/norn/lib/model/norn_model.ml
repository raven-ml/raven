(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Norn.Dist
module Bij = Norn.Bij
module Path = Nx.Ptree.Path

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let shape_string s =
  String.concat "; " (Array.to_list (Array.map string_of_int s))

(* Effects *)

type _ Effect.t +=
  | Sample : ('x, 'f) D.t -> 'x Effect.t
  | Factor : (float, 'f) Nx.t -> unit Effect.t

let sample d = Effect.perform (Sample d)
let factor l = Effect.perform (Factor l)

(* Sites *)

type role = Latent | Observed | Fixed

type site = {
  name : string;
  role : role;
  family : string;
  shape : int array;
  points : int option;
  coordinates : string;
}

type values = Reals | Counts | Categories | Booleans

(* What [v] learned of a site, beyond its description: the index of its tensor
   in its side's walk, its coordinates' shape, its value's dtype and its
   support, read when printed. *)
type learned = {
  site : site;
  leaf : int;
  coord_shape : int array;
  value_dtype : string;
  values : values; (* the kind of its values *)
  support : unit -> Norn.Support.t;
}

type ('p, 'y, 'f) t = {
  dtype : (float, 'f) Nx.dtype;
  latent : 'p Nx.Ptree.t;
  observed : 'y Nx.Ptree.t;
  gen : unit -> 'p * 'y;
  sites : learned array;
  like : 'p; (* the latent value [v] learned, a skeleton for coordinates *)
  reparams : (int * (((float, 'f) Nx.t, 'f) D.t -> 'f Bij.t)) list;
  fixed : (int * Nx.packed) list;
}

type 'p coords = 'p

(* Values of any kind *)

let pack : type x f. (x, f) D.t -> x -> Nx.packed =
 fun d x ->
  match D.kind d with
  | D.Continuous -> Nx.P x
  | D.Counts -> Nx.P x
  | D.Categories -> Nx.P x
  | D.Booleans -> Nx.P x

let unpack : type x f. (x, f) D.t -> Nx.packed -> x =
 fun d p ->
  match D.kind d with
  | D.Continuous -> Nx.unpack (D.dtype d) p
  | D.Counts -> Nx.unpack Nx.int32 p
  | D.Categories -> Nx.unpack Nx.int64 p
  | D.Booleans -> Nx.unpack Nx.bool p

(* [fresh d x] is [x], or a copy of it if it is [source]: a handed-out value is
   distinct from every coordinate and datum. *)
let fresh : type x f. (x, f) D.t -> x -> Nx.packed -> x =
 fun d x source ->
  let (Nx.P s) = source in
  if Obj.repr x != Obj.repr s then x
  else
    match D.kind d with
    | D.Continuous -> Nx.copy x
    | D.Counts -> Nx.copy x
    | D.Categories -> Nx.copy x
    | D.Booleans -> Nx.copy x

let zeros_of : type x f. (x, f) D.t -> int array -> x =
 fun d s ->
  match D.kind d with
  | D.Continuous -> Nx.zeros (D.dtype d) s
  | D.Counts -> Nx.zeros Nx.int32 s
  | D.Categories -> Nx.zeros Nx.int64 s
  | D.Booleans -> Nx.zeros Nx.bool s

let same (Nx.P a) (Nx.P b) = Obj.repr a == Obj.repr b
let packed_shape (Nx.P x) = Nx.shape x
let packed_dtype (Nx.P x) = Nx_dtype.to_string (Nx.dtype x)

let leaves s x =
  Array.of_list
    (List.rev (Nx.Ptree.fold s (fun p t acc -> (p, Nx.P t) :: acc) x []))

let describe family shape =
  Printf.sprintf "a %s of shape [%s]" family (shape_string shape)

let nth i =
  let suffix =
    match i mod 100 with
    | 11 | 12 | 13 -> "th"
    | _ -> (
        match i mod 10 with 1 -> "st" | 2 -> "nd" | 3 -> "rd" | _ -> "th")
  in
  Printf.sprintf "%d%s" i suffix

let site_name side path = match Path.to_string path with "" -> side | s -> s

(* Learning *)

type seen = {
  value : Nx.packed;
  s_family : string;
  s_shape : int array;
  discrete : bool;
  s_coord_shape : int array;
  s_coordinates : string;
  s_dtype : string;
  s_values : values;
  s_points : int;
  s_support : unit -> Norn.Support.t;
}

(* [learn d] answers a site at [v]: a continuous site with its bijector's image
   of zero coordinates, a discrete one with zeros. *)
let learn : type x f. (x, f) D.t -> x * seen =
 fun d ->
  let shape = D.shape d in
  let points x =
    let f = D.factors (D.check ~unless:(Nx.scalar Nx.bool true) "" d) x in
    if Nx.ndim f = 0 then 1 else (Nx.shape f).(0)
  in
  let seen value ~discrete ~coord_shape ~coordinates x =
    {
      value;
      s_family = D.family d;
      s_shape = shape;
      discrete;
      s_coord_shape = coord_shape;
      s_coordinates = coordinates;
      s_dtype = packed_dtype value;
      s_values =
        (match D.kind d with
        | D.Continuous -> Reals
        | D.Counts -> Counts
        | D.Categories -> Categories
        | D.Booleans -> Booleans);
      s_points = points x;
      s_support = (fun () -> D.support d);
    }
  in
  match D.kind d with
  | D.Continuous ->
      let b = D.coords d in
      let cs = Bij.shape b shape in
      let x, _ = Bij.forward b (Nx.zeros (D.dtype d) cs) in
      ( x,
        seen (Nx.P x) ~discrete:false ~coord_shape:cs
          ~coordinates:(Format.asprintf "%a" Bij.pp b)
          x )
  | _ ->
      let x = zeros_of d shape in
      (x, seen (pack d x) ~discrete:true ~coord_shape:[| 0 |] ~coordinates:"" x)

let v dtype latent observed gen =
  let seen = ref [] in
  let effc : type b. b Effect.t -> ((b, _) Effect.Deep.continuation -> _) option
      = function
    | Sample d ->
        Some
          (fun k ->
            let x, s = learn d in
            seen := s :: !seen;
            Effect.Deep.continue k x)
    | Factor _ -> Some (fun k -> Effect.Deep.continue k ())
    | _ -> None
  in
  (* A failure of the model's own code names the sites learned so far. *)
  let learned () =
    let n = List.length !seen in
    Printf.sprintf "Norn_model.v: the model raised after %d site%s (%s): " n
      (if n = 1 then "" else "s")
      (String.concat ", "
         (List.rev_map (fun s -> describe s.s_family s.s_shape) !seen))
  in
  let p, y =
    match Effect.Deep.try_with gen () { effc } with
    | r -> r
    | exception Invalid_argument msg -> invalid_arg (learned () ^ msg)
    | exception Failure msg -> failwith (learned () ^ msg)
  in
  let seen = Array.of_list (List.rev !seen) in
  let n = Array.length seen in
  let found = Array.make n None in
  let place side role values =
    Array.iteri
      (fun j (path, t) ->
        let name = site_name side path in
        let rec find i =
          if i = n then
            invalid_argf
              "Norn_model.v: the %s result at %s is not a sampled value; a \
               model returns every sampled value unchanged"
              (if role = Latent then "latent" else "observed")
              name
          else if same seen.(i).value t then i
          else find (i + 1)
        in
        let i = find 0 in
        (match found.(i) with
        | Some (other, _, _) ->
            invalid_argf
              "Norn_model.v: site %s appears twice in the result, also at %s"
              other name
        | None -> ());
        found.(i) <- Some (name, role, j))
      values
  in
  place "p" Latent (leaves latent p);
  place "y" Observed (leaves observed y);
  let sites =
    Array.mapi
      (fun i s ->
        match found.(i) with
        | None ->
            invalid_argf
              "Norn_model.v: the %s sample, %s, is not in the result; a model \
               returns every sampled value"
              (nth (i + 1))
              (describe s.s_family s.s_shape)
        | Some (name, role, leaf) ->
            if role = Latent && s.discrete then
              invalid_argf
                "Norn_model.v: site %s: %s is over %a, and a latent site needs \
                 a continuous support"
                name s.s_family
                (fun () s -> Format.asprintf "%a" Norn.Support.pp s)
                (s.s_support ());
            {
              site =
                {
                  name;
                  role;
                  family = s.s_family;
                  shape = s.s_shape;
                  points = (if role = Observed then Some s.s_points else None);
                  coordinates = (if role = Latent then s.s_coordinates else "");
                };
              leaf;
              coord_shape = s.s_coord_shape;
              value_dtype = s.s_dtype;
              values = s.s_values;
              support = s.s_support;
            })
      seen
  in
  { dtype; latent; observed; gen; sites; like = p; reparams = []; fixed = [] }

(* Running a model *)

(* An answer to a site: [i] its index, [l] what [v] learned of it. *)
type answer = { answer : 'x 'f. int -> learned -> ('x, 'f) D.t -> 'x }

(* An answer to a factor. *)
type factor = { factor : 'f. (float, 'f) Nx.t -> unit }

let no_factor = { factor = (fun _ -> ()) }

let static_check context l d =
  let family = D.family d and shape = D.shape d in
  if family <> l.site.family || shape <> l.site.shape then
    invalid_argf
      "%s: site %s: the model is not static: %s when it was built, %s now"
      context l.site.name
      (describe l.site.family l.site.shape)
      (describe family shape)

(* [run m context a ~factor] runs [m]'s model, answering each site with [a] and
   each factor with [factor], and checks the result rule. *)
let run m context a ~factor =
  let n = Array.length m.sites in
  let handed = Array.make n None in
  let next = ref 0 in
  let effc : type b. b Effect.t -> ((b, _) Effect.Deep.continuation -> _) option
      = function
    | Sample d ->
        Some
          (fun k ->
            let i = !next in
            if i >= n then
              invalid_argf
                "%s: the model is not static: it performs a %s sample, %s, \
                 when it was built with %d"
                context
                (nth (i + 1))
                (describe (D.family d) (D.shape d))
                n;
            let l = m.sites.(i) in
            static_check context l d;
            next := i + 1;
            let x = a.answer i l d in
            handed.(i) <- Some (pack d x);
            Effect.Deep.continue k x)
    | Factor l ->
        Some
          (fun k ->
            factor.factor l;
            Effect.Deep.continue k ())
    | _ -> None
  in
  let p, y = Effect.Deep.try_with m.gen () { effc } in
  if !next < n then
    invalid_argf
      "%s: the model is not static: it performs %d samples, when it was built \
       with %d"
      context !next n;
  let lat = leaves m.latent p and obs = leaves m.observed y in
  if Array.length lat + Array.length obs <> n then
    invalid_argf
      "%s: the model is not static: its result holds %d tensors, when it was \
       built with %d"
      context
      (Array.length lat + Array.length obs)
      n;
  Array.iteri
    (fun i l ->
      let side = if l.site.role = Observed then obs else lat in
      let ok =
        match handed.(i) with
        | Some h -> l.leaf < Array.length side && same h (snd side.(l.leaf))
        | None -> false
      in
      if not ok then
        invalid_argf
          "%s: site %s: the result holds another value at %s; a model returns \
           every sampled value unchanged, outside any loop or map"
          context l.site.name l.site.name)
    m.sites;
  (p, y)

let reparam_of m i = List.assoc_opt i m.reparams
let fixed_of m i = List.assoc_opt i m.fixed

(* [bijector m i d] is the coordinates' bijector of the continuous site [i]. *)
let bijector (type f g) (m : (_, _, f) t) i (d : ((float, g) Nx.t, g) D.t) :
    g Bij.t =
  match reparam_of m i with
  | None -> D.coords d
  | Some b -> (
      match Nx_dtype.equal_witness (D.dtype d) m.dtype with
      | Some Type.Equal -> b d
      | None ->
          invalid_argf "Norn_model.reparam: site %s is %s, the model %s"
            m.sites.(i).site.name
            (Nx_dtype.to_string (D.dtype d))
            (Nx_dtype.to_string m.dtype))

(* [forward m i d u] is the continuous site [i]'s value at coordinates [u],
   fresh, and its log-determinant. *)
let forward : type x g.
    (_, _, _) t -> int -> (x, g) D.t -> Nx.packed -> x * (float, g) Nx.t =
 fun m i d u ->
  match D.kind d with
  | D.Continuous ->
      let x, ld = Bij.forward (bijector m i d) (unpack d u) in
      (fresh d x u, ld)
  | _ -> assert false (* [v] refuses a discrete latent site *)

let inverse : type x g. (_, _, _) t -> int -> (x, g) D.t -> x -> Nx.packed =
 fun m i d x ->
  match D.kind d with
  | D.Continuous -> Nx.P (Bij.inverse (bijector m i d) x)
  | _ -> assert false

(* Scoring *)

(* A site's term, and what explains it when it is not finite: its factors'
   shape, the first element whose factor is not finite (flat, or 0), whether
   every factor is finite, and the bounds of the support at that element. *)
type 'f term = {
  name : string;
  value : (float, 'f) Nx.t;
  shape : int array;
  outside : Nx.int32_t;
  finite : Nx.bool_t;
  lo : (float, 'f) Nx.t;
  hi : (float, 'f) Nx.t;
}

type 'f score = {
  mutable total : (float, 'f) Nx.t;
  mutable terms : 'f term list;
  mutable points : (float, 'f) Nx.t list;
  mutable factors : (float, 'f) Nx.t option;
}

(* What a scoring run counts: the latent sites' prior, their log-determinants,
   the observed sites' likelihood and the factors. *)
type parts = { prior : bool; jacobian : bool; likelihood : bool }

let all_parts = { prior = true; jacobian = true; likelihood = true }

(* What a site does with a computed parameter outside its domain. [Total]: its
   term is -inf, as a density a sampler explores must be. [Refuse site]: it
   raises with [site name] as the context, except where the density so far is
   -inf or [unless] holds. *)
type params =
  | Total
  | Refuse of { site : string -> string; unless : Nx.bool_t option }

(* The latent sites' answer: coordinates, or values. *)
type latent = Coords of Nx.packed array | Values of Nx.packed array

let neg_inf_like x = Nx.scalar (Nx.dtype x) Float.neg_infinity

(* [points x] is the log density of each point of [x]: axis 0, every other axis
   summed; a scalar is one point. *)
let points x =
  if Nx.ndim x = 0 then Nx.reshape [| 1 |] x
  else if Nx.ndim x = 1 then x
  else Nx.sum ~axes:(List.init (Nx.ndim x - 1) (fun a -> a + 1)) x

let score m context params parts latent data =
  let s =
    { total = Nx.zeros m.dtype [||]; terms = []; points = []; factors = None }
  in
  let dead () = Nx.equal s.total (neg_inf_like s.total) in
  (* A term enters where the density so far is finite; where it is -inf the term
     is -inf. A term that is NaN or +inf raises naming its source. *)
  let enter source term =
    let term = Nx.cast m.dtype term in
    let ok =
      Nx.logical_or (dead ())
        (Nx.logical_not
           (Nx.logical_or (Nx.isnan term)
              (Nx.equal term (Nx.scalar m.dtype Float.infinity))))
    in
    Nx.check Nx.Ptree.tensor ok term (fun _ v ->
        Invalid_argument
          (Printf.sprintf "%s: %s: its log density is %s" context source
             (Nx.to_string v)));
    s.total <- Nx.where (dead ()) s.total (Nx.add s.total term);
    term
  in
  (* [valid] is whether the site's parameters are in their domain: where they
     are not, the term is -inf. *)
  let add : type x g.
      string ->
      Nx.bool_t ->
      (x, g) D.t ->
      (float, g) Nx.t ->
      (float, g) Nx.t ->
      unit =
   fun name valid d f term ->
    let term = Nx.where valid term (neg_inf_like term) in
    let value = enter ("site " ^ name) term in
    let flat = Nx.reshape [| Nx.numel f |] (Nx.cast m.dtype f) in
    let bad = Nx.logical_not (Nx.isfinite flat) in
    let outside =
      if Nx.numel f = 0 then Nx.scalar Nx.int32 0l
      else Nx.cast Nx.int32 (Nx.argmax ~axis:0 (Nx.cast Nx.int32 bad))
    in
    let lo, hi = D.bounds d in
    let lo = Nx.cast m.dtype lo and hi = Nx.cast m.dtype hi in
    (* An elementwise family's bounds at the element, the hull otherwise. *)
    let lo, hi =
      if Nx.shape f = Nx.shape lo && Nx.numel f > 0 then
        let at x =
          Nx.reshape [||]
            (Nx.take
               ~indices:(Nx.reshape [| 1 |] (Nx.cast Nx.int64 outside))
               (Nx.reshape [| Nx.numel x |] x))
        in
        (at lo, at hi)
      else (Nx.min lo, Nx.max hi)
    in
    s.terms <-
      {
        name;
        value;
        shape = Nx.shape f;
        outside;
        finite = Nx.logical_not (Nx.any bad);
        lo;
        hi;
      }
      :: s.terms
  in
  (* [checked name d] is [d] with its parameters checked as [params] says, and
     whether they are in their domain. *)
  let checked name d =
    let valid = D.valid d in
    match params with
    | Total -> (D.check ~unless:(Nx.scalar Nx.bool true) "" d, valid)
    | Refuse { site; unless } ->
        let unless =
          match unless with
          | None -> dead ()
          | Some u -> Nx.logical_or (dead ()) u
        in
        (D.check ~unless (site name) d, valid)
  in
  let answer i l d =
    let name = l.site.name in
    match (l.site.role, fixed_of m i) with
    | _, Some v -> fresh d (unpack d v) v
    | (Latent | Fixed), None -> (
        match latent with
        | Coords cs ->
            let x, ld = forward m i d cs.(l.leaf) in
            if parts.prior || parts.jacobian then begin
              let d, valid = checked name d in
              let f =
                if parts.prior then D.factors d x else Nx.zeros (D.dtype d) [||]
              in
              let t = Nx.sum f in
              let t = if parts.jacobian then Nx.add t (Nx.sum ld) else t in
              add name valid d f t
            end;
            x
        | Values vs ->
            let x = fresh d (unpack d vs.(l.leaf)) vs.(l.leaf) in
            if parts.prior then begin
              let d, valid = checked name d in
              let f = D.factors d x in
              add name valid d f (Nx.sum f)
            end;
            x)
    | Observed, None -> (
        match data with
        | None -> zeros_of d l.site.shape
        | Some ys ->
            let x = fresh d (unpack d ys.(l.leaf)) ys.(l.leaf) in
            if parts.likelihood then begin
              let d, valid = checked name d in
              let f = D.factors d x in
              s.points <- Nx.cast m.dtype (points f) :: s.points;
              add name valid d f (Nx.sum f)
            end;
            x)
  in
  let factor : type g. (float, g) Nx.t -> unit =
   fun l ->
    if parts.likelihood then begin
      let l = Nx.cast m.dtype l in
      s.points <- points l :: s.points;
      let t = enter "a factor" (Nx.sum l) in
      s.factors <-
        Some (match s.factors with None -> t | Some f -> Nx.add f t)
    end
  in
  ignore
    (run m context { answer = (fun i l d -> answer i l d) } ~factor:{ factor });
  s.terms <- List.rev s.terms;
  s

let coords m = m.latent

let check_data context m y =
  let ys = leaves m.observed y in
  let expected =
    Array.to_list m.sites |> List.filter (fun l -> l.site.role = Observed)
  in
  if List.length expected <> Array.length ys then
    invalid_argf
      "%s: the observations do not match the model: %d tensors, the model \
       observes %d sites"
      context (Array.length ys) (List.length expected);
  List.iter
    (fun l ->
      let path, t = ys.(l.leaf) in
      let at = site_name "y" path in
      if packed_shape t <> l.site.shape then
        invalid_argf
          "%s: the observations do not match the model: at %s, data of shape \
           [%s], the site's shape [%s]"
          context at
          (shape_string (packed_shape t))
          (shape_string l.site.shape);
      if packed_dtype t <> l.value_dtype then
        invalid_argf
          "%s: the observations do not match the model: at %s, data of dtype \
           %s, the site's %s"
          context at (packed_dtype t) l.value_dtype)
    expected;
  Array.map snd ys

(* [check_shapes context what m ~lead x] refuses latent tensors [x] whose shapes
   are not the sites' [shape l], after [lead] leading axes of one length. *)
let check_shapes context what m ~lead shape x =
  let refuse at s expected chains =
    if lead = 0 then
      invalid_argf
        "%s: the %s do not match the model: at %s, %s of shape [%s], the \
         site's [%s]"
        context what at what (shape_string s) (shape_string expected)
    else
      invalid_argf
        "%s: the position does not match the model: at %s, %s of shape [%s], a \
         position of %d chains has [%s]"
        context at what (shape_string s) chains (shape_string expected)
  in
  let xs = leaves m.latent x in
  let first = ref None in
  Array.iter
    (fun l ->
      if l.site.role <> Observed then begin
        let path, t = xs.(l.leaf) in
        let s = packed_shape t in
        let prefix =
          if lead = 0 then [||]
          else
            match !first with
            | Some p -> p
            | None ->
                let p = Array.sub s 0 (min lead (Array.length s)) in
                first := Some p;
                p
        in
        let expected = Array.append prefix (shape l) in
        let chains = if Array.length prefix > 0 then prefix.(0) else 0 in
        if s <> expected then refuse (site_name "p" path) s expected chains
      end)
    m.sites

let check_coords context m ~lead c =
  check_shapes context "coordinates" m ~lead (fun l -> l.coord_shape) c

(* [batched m context f] is [f] mapped over the chain axis of positions. *)
let batched m context f c =
  check_coords context m ~lead:1 c;
  Rune.vmap
    Nx.Ptree.(m.latent @-> returns tensor)
    (fun c -> f (Coords (Array.map snd (leaves m.latent c))))
    c

let log_density m y =
  let context = "Norn_model.log_density" in
  let ys = check_data context m y in
  batched m context (fun c ->
      (score m context Total all_parts c (Some ys)).total)

let log_prior m =
  let context = "Norn_model.log_prior" in
  let parts = { prior = true; jacobian = true; likelihood = false } in
  batched m context (fun c -> (score m context Total parts c None).total)

let log_likelihood m y =
  let context = "Norn_model.log_likelihood" in
  let ys = check_data context m y in
  let parts = { prior = false; jacobian = false; likelihood = true } in
  batched m context (fun c -> (score m context Total parts c (Some ys)).total)

let terms m y c =
  let context = "Norn_model.terms" in
  let ys = check_data context m y in
  check_coords context m ~lead:1 c;
  let names = Array.to_list (Array.map (fun l -> l.site.name) m.sites) in
  let sites, factors =
    Rune.vmap
      Nx.Ptree.(m.latent @-> returns (pair (list tensor) tensor))
      (fun c ->
        let s =
          score m context Total all_parts
            (Coords (Array.map snd (leaves m.latent c)))
            (Some ys)
        in
        let term name =
          match List.find_opt (fun t -> t.name = name) s.terms with
          | Some t -> t.value
          | None -> Nx.zeros m.dtype [||]
        in
        let factors =
          match s.factors with Some f -> f | None -> Nx.zeros m.dtype [||]
        in
        (List.map term names, factors))
      c
  in
  (List.combine names sites, factors)

(* One instance's interpreters refuse a parameter outside its domain. *)
let refuse context =
  Refuse { site = (fun name -> context ^ ": site " ^ name); unless = None }

let log_joint m y p =
  let context = "Norn_model.log_joint" in
  let ys = check_data context m y in
  let parts = { prior = true; jacobian = false; likelihood = true } in
  (score m context (refuse context) parts
     (Values (Array.map snd (leaves m.latent p)))
     (Some ys))
    .total

let pointwise m y p =
  let context = "Norn_model.pointwise" in
  let ys = check_data context m y in
  let parts = { prior = false; jacobian = false; likelihood = true } in
  let s =
    score m context (refuse context) parts
      (Values (Array.map snd (leaves m.latent p)))
      (Some ys)
  in
  match List.rev s.points with
  | [] -> Nx.zeros m.dtype [| 0 |]
  | ps -> Nx.concatenate ~axis:0 ps

(* Running forward *)

let draw context i d k =
  let d = D.check (Printf.sprintf "%s: site %s" context i) d in
  D.sample k d

(* [coordinates m like us] is the coordinates [us] of each latent site, in the
   order of [m]'s latent walk, as a value of [m]'s structure. *)
let coordinates m like us = Nx.Ptree.rebuild m.latent ~like (Array.to_list us)
let empty_coords (Nx.P x) = Nx.P (Nx.zeros (Nx.dtype x) [| 0 |])

let prior_draw m context k =
  let n_latent = Array.length (leaves m.latent m.like) in
  let us = Array.make n_latent (Nx.P (Nx.zeros Nx.float32 [||])) in
  let answer i l d =
    let x =
      match fixed_of m i with
      | Some v -> fresh d (unpack d v) v
      | None -> draw context l.site.name d (Nx.Rng.fold_in k i)
    in
    (match l.site.role with
    | Latent -> us.(l.leaf) <- inverse m i d x
    | Fixed -> us.(l.leaf) <- empty_coords (pack d x)
    | Observed -> ());
    x
  in
  let p, y =
    run m context { answer = (fun i l d -> answer i l d) } ~factor:no_factor
  in
  (p, y, us)

let from_prior m ~n k =
  let context = "Norn_model.from_prior" in
  if n < 1 then invalid_argf "%s: n = %d is not positive" context n;
  Rune.vmap
    Nx.Ptree.(Nx.Rng.ptree @-> returns m.latent)
    (fun k ->
      let p, _, us = prior_draw m context k in
      coordinates m p us)
    (Nx.Rng.split_batch ~n k)

let simulate m k =
  let p, y, _ = prior_draw m "Norn_model.simulate" k in
  (p, y)

let predict m k p =
  let context = "Norn_model.predict" in
  let vs = Array.map snd (leaves m.latent p) in
  let answer i l d =
    match (l.site.role, fixed_of m i) with
    | _, Some v -> fresh d (unpack d v) v
    | (Latent | Fixed), None -> fresh d (unpack d vs.(l.leaf)) vs.(l.leaf)
    | Observed, None -> draw context l.site.name d (Nx.Rng.fold_in k i)
  in
  snd (run m context { answer = (fun i l d -> answer i l d) } ~factor:no_factor)

let constrain m c =
  let context = "Norn_model.constrain" in
  check_coords context m ~lead:0 c;
  let cs = Array.map snd (leaves m.latent c) in
  let answer i l d =
    match (l.site.role, fixed_of m i) with
    | _, Some v -> fresh d (unpack d v) v
    | (Latent | Fixed), None -> fst (forward m i d cs.(l.leaf))
    | Observed, None -> zeros_of d l.site.shape
  in
  fst (run m context { answer = (fun i l d -> answer i l d) } ~factor:no_factor)

let unconstrain m p =
  let context = "Norn_model.unconstrain" in
  check_shapes context "values" m ~lead:0 (fun l -> l.site.shape) p;
  let vs = Array.map snd (leaves m.latent p) in
  let us = Array.copy vs in
  let answer i l d =
    match (l.site.role, fixed_of m i) with
    | _, Some v ->
        us.(l.leaf) <- empty_coords v;
        fresh d (unpack d v) v
    | (Latent | Fixed), None ->
        let x = fresh d (unpack d vs.(l.leaf)) vs.(l.leaf) in
        us.(l.leaf) <- inverse m i d x;
        x
    | Observed, None -> zeros_of d l.site.shape
  in
  let p, _ =
    run m context { answer = (fun i l d -> answer i l d) } ~factor:no_factor
  in
  coordinates m p us

(* [zero_coords m] is the coordinates whose every element is zero. *)
let zero_coords m =
  let shapes = Array.make (Array.length (leaves m.latent m.like)) [||] in
  Array.iter
    (fun l -> if l.site.role <> Observed then shapes.(l.leaf) <- l.coord_shape)
    m.sites;
  let j = ref (-1) in
  Nx.Ptree.map m.latent
    (fun _ x ->
      incr j;
      Nx.zeros (Nx.dtype x) shapes.(!j))
    m.like

(* Initialisation *)

module Init = struct
  type 'p t = Uniform | Prior | Near of 'p

  let uniform = Uniform
  let prior = Prior
  let near p = Near p
end

let candidates = 100

(* [jittered c k half] is [candidates] copies of [c], each element moved
   uniformly in [(-half, half)] by draws from [k]. *)
let jittered (type a b) (c : (a, b) Nx.t) k half : (a, b) Nx.t =
  let move (type f) (c : (float, f) Nx.t) =
    let shape = Array.append [| candidates |] (Nx.shape c) in
    let u = Nx.Rng.uniform k (Nx.dtype c) shape in
    Nx.add c (Nx.mul_s (Nx.sub_s u 0.5) (2. *. half))
  in
  match Nx_dtype.kind (Nx.dtype c) with
  | Float -> move c
  | _ ->
      invalid_argf "Norn_model.init: a latent site of dtype %s"
        (Nx_dtype.to_string (Nx.dtype c))

(* [unravel shape k] is the index of the [k]-th element of [shape] in C
   order. *)
let unravel shape k =
  let index = Array.make (Array.length shape) 0 in
  let k = ref k in
  for d = Array.length shape - 1 downto 0 do
    index.(d) <- !k mod shape.(d);
    k := !k / shape.(d)
  done;
  index

(* The data of a refusal to start: each site's term at the last candidate, what
   explains it, and the factors' sum. *)
let explain_ptree () =
  Nx.Ptree.(
    pair
      (list (pair tensor (pair tensor (pair tensor (pair tensor tensor)))))
      tensor)

(* [support_of values lo hi] is the support of values of a kind between the
   bounds [lo] and [hi]. *)
let support_of values lo hi =
  match values with
  | Booleans -> Norn.Support.Boolean
  | Categories -> Norn.Support.Integer_interval (truncate lo, truncate hi)
  | Counts when Float.is_finite hi ->
      Norn.Support.Integer_interval (truncate lo, truncate hi)
  | Counts -> Norn.Support.Integers_from (truncate lo)
  | Reals when not (Float.is_finite lo) -> Norn.Support.Real
  | Reals when Float.is_finite hi -> Norn.Support.Interval (lo, hi)
  | Reals -> Norn.Support.Greater lo

(* [no_start m context terms data] refuses a start whose last candidate's terms
   [data] name no finite density. *)
let no_start m context (terms : (string * int array) list) (sites, factors) =
  let value t = Nx.item [] t in
  let parts =
    List.map2
      (fun (name, _) (t, _) -> Printf.sprintf "%s %g" name (value t))
      terms sites
  in
  let factors = value factors in
  let parts =
    if factors = 0. then parts
    else parts @ [ Printf.sprintf "the factors %g" factors ]
  in
  let rec first = function
    | [] -> ""
    | ((name, shape), (t, (outside, (finite, (lo, hi))))) :: rest ->
        if Float.is_finite (value t) then first rest
        else
          let why =
            if Nx.item [] finite then
              ": its coordinates saturate: the log-determinant is -inf"
            else
              let l = Array.find_opt (fun l -> l.site.name = name) m.sites in
              let what =
                if shape = [||] then "it"
                else
                  let index =
                    unravel shape (Int32.to_int (Nx.item [] outside))
                  in
                  Printf.sprintf "element [%s]" (shape_string index)
              in
              match l with
              | Some l ->
                  Format.asprintf ": %s is outside the support of %s, %a" what
                    l.site.family Norn.Support.pp
                    (support_of l.values (value lo) (value hi))
              | None -> ""
          in
          Printf.sprintf "; at the last, site %s has log density %g%s" name
            (value t) why
  in
  Invalid_argument
    (Printf.sprintf
       "%s: no finite log density in %d candidates%s; the terms are %s" context
       candidates
       (first (List.combine terms sites))
       (String.concat ", " parts))

let init ?(from = Init.Uniform) m y ~chains k =
  let context = "Norn_model.init" in
  if chains < 1 then
    invalid_argf "%s: chains = %d is not positive" context chains;
  let ys = check_data context m y in
  let lp = log_density m y in
  (* [jitter k centre half] is [centre] moved [candidates] times, each tensor
     from its own key. *)
  let jitter k centre half =
    let j = ref (-1) in
    Nx.Ptree.map m.latent
      (fun _ c ->
        incr j;
        jittered c (Nx.Rng.fold_in k !j) half)
      centre
  in
  let centre =
    match from with
    | Init.Near p -> unconstrain m p
    | Init.Uniform | Init.Prior -> zero_coords m
  in
  let start k =
    let cands =
      match from with
      | Init.Prior -> from_prior m ~n:candidates k
      | Init.Uniform -> jitter k centre 2.
      | Init.Near _ -> jitter k centre 0.1
    in
    let finite = Nx.isfinite (lp cands) in
    (* The last candidate's terms explain a refusal. *)
    let last =
      Nx.Ptree.map m.latent
        (fun _ x -> Nx.slice [ Nx.I (candidates - 1) ] x)
        cands
    in
    (* Where no candidate is finite, a parameter outside its domain at the last
       is the refusal. *)
    let site name =
      Printf.sprintf
        "%s: no finite log density in %d candidates; at the last, site %s"
        context candidates name
    in
    let params = Refuse { site; unless = Some (Nx.any finite) } in
    let s =
      score m context params all_parts
        (Coords (Array.map snd (leaves m.latent last)))
        (Some ys)
    in
    let terms = List.map (fun t -> (t.name, t.shape)) s.terms in
    let sites =
      List.map
        (fun t -> (t.value, (t.outside, (t.finite, (t.lo, t.hi)))))
        s.terms
    in
    let factors =
      match s.factors with Some f -> f | None -> Nx.zeros m.dtype [||]
    in
    Nx.check (explain_ptree ()) (Nx.any finite) (sites, factors) (fun _ data ->
        no_start m context terms data);
    let first =
      Nx.reshape [| 1 |] (Nx.argmax ~axis:0 (Nx.cast Nx.int32 finite))
    in
    Nx.Ptree.map m.latent
      (fun _ x -> Nx.squeeze ~axes:[ 0 ] (Nx.take ~axis:0 ~indices:first x))
      cands
  in
  Rune.vmap
    Nx.Ptree.(Nx.Rng.ptree @-> returns m.latent)
    start
    (Nx.Rng.split_batch ~n:chains k)

(* Transformations *)

let select context m sel =
  let fresh =
    Nx.Ptree.map m.latent (fun _ x -> Nx.zeros (Nx.dtype x) [| 1 |]) m.like
  in
  let r = sel fresh in
  let leaves = leaves m.latent fresh in
  let rec find j =
    if j = Array.length leaves then
      invalid_argf
        "%s: the selector computes a value; a selector is a projection of the \
         latent values, such as (fun p -> p.theta)"
        context
    else if same (snd leaves.(j)) (Nx.P r) then j
    else find (j + 1)
  in
  let leaf = find 0 in
  let rec site i =
    let l = m.sites.(i) in
    if l.site.role <> Observed && l.leaf = leaf then i else site (i + 1)
  in
  site 0

let bijector_name : type x g. (_, _, _) t -> int -> (x, g) D.t -> string =
 fun m i d ->
  match D.kind d with
  | D.Continuous -> Format.asprintf "%a" Bij.pp (bijector m i d)
  | _ -> ""

let reparam sel b m =
  let context = "Norn_model.reparam" in
  let i = select context m sel in
  let m = { m with reparams = (i, b) :: List.remove_assoc i m.reparams } in
  (* The bijector depends on the site's distribution: name it from one run at
     zero coordinates. *)
  let name = ref "" in
  let cs = Array.map snd (leaves m.latent (zero_coords m)) in
  let answer j l d =
    match (l.site.role, fixed_of m j) with
    | _, Some v -> fresh d (unpack d v) v
    | (Latent | Fixed), None ->
        if j = i then name := bijector_name m j d;
        fst (forward m j d cs.(l.leaf))
    | Observed, None -> zeros_of d l.site.shape
  in
  ignore
    (run m context { answer = (fun j l d -> answer j l d) } ~factor:no_factor);
  let sites = Array.copy m.sites in
  let l = sites.(i) in
  sites.(i) <- { l with site = { l.site with coordinates = !name } };
  { m with sites }

let noncentre sel m = reparam sel D.standardize m

let fix sel v m =
  let context = "Norn_model.fix" in
  let i = select context m sel in
  let l = m.sites.(i) in
  if
    Nx.shape v <> l.site.shape
    || Nx_dtype.to_string (Nx.dtype v) <> l.value_dtype
  then
    invalid_argf "%s: site %s is %s [%s], the value %s [%s]" context l.site.name
      l.value_dtype
      (shape_string l.site.shape)
      (Nx_dtype.to_string (Nx.dtype v))
      (shape_string (Nx.shape v));
  let sites = Array.copy m.sites in
  sites.(i) <-
    {
      l with
      site = { l.site with role = Fixed; coordinates = "" };
      coord_shape = [| 0 |];
    };
  { m with sites; fixed = (i, Nx.P v) :: List.remove_assoc i m.fixed }

(* Sites *)

let pp ppf m =
  let role = function
    | Latent -> "latent"
    | Observed -> "observed"
    | Fixed -> "fixed"
  in
  let rows =
    List.map
      (fun l ->
        let s = l.site in
        [
          s.name;
          role s.role;
          s.family;
          "[" ^ shape_string s.shape ^ "]";
          (match s.points with Some n -> string_of_int n | None -> "");
          Format.asprintf "%a" Norn.Support.pp (l.support ());
          s.coordinates;
        ])
      (Array.to_list m.sites)
  in
  let header =
    [ "site"; "role"; "family"; "shape"; "points"; "support"; "coordinates" ]
  in
  let widths =
    List.mapi
      (fun j h ->
        List.fold_left
          (fun w r -> max w (String.length (List.nth r j)))
          (String.length h) rows)
      header
  in
  let line r =
    let cells = List.map2 (fun w c -> Printf.sprintf "%-*s" w c) widths r in
    let s = String.concat "  " cells in
    (* No trailing blanks. *)
    let n = ref (String.length s) in
    while !n > 0 && s.[!n - 1] = ' ' do
      decr n
    done;
    String.sub s 0 !n
  in
  Format.fprintf ppf "@[<v>%s" (line header);
  List.iter (fun r -> Format.fprintf ppf "@,%s" (line r)) rows;
  Format.fprintf ppf "@]"
