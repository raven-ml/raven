(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type order = [ `Lt | `Le | `Gt | `Ge ]

type atom =
  | Eq of string * string
  | Compare of order * string * string
  | Closest of {
      order : order;
      left : string;
      right : string;
      within : Expr.packed option;
    }
  | Nearest of { left : string; right : string; within : Expr.packed option }
  | Position

type cond = atom list
type kind = Inner | Left | Full | Semi | Anti
type count = Any | At_most_one | One | At_least_one

let err fmt = Format.kasprintf invalid_arg fmt

(* Formatting *)

let order_name : order -> string = function
  | `Lt -> "lt"
  | `Le -> "le"
  | `Gt -> "gt"
  | `Ge -> "ge"

let pp_within ppf (Expr.Packed w) =
  Format.fprintf ppf "~within:%a@ " Expr.pp_arg w

let pp_name = Type.pp_quoted

let pp ppf c =
  let atom ppf = function
    | Eq (l, r) ->
        Format.fprintf ppf "@[<hov 2>eq@ %a@ %a@]" pp_name l pp_name r
    | Compare (o, l, r) ->
        Format.fprintf ppf "@[<hov 2>%s@ %a@ %a@]" (order_name o) pp_name l
          pp_name r
    | Closest { order; left; right; within } ->
        Format.fprintf ppf "@[<hov 2>closest@ %a(%s %a %a)@]"
          (Format.pp_print_option pp_within)
          within (order_name order) pp_name left pp_name right
    | Nearest { left; right; within } ->
        Format.fprintf ppf "@[<hov 2>nearest@ %a%a@ %a@]"
          (Format.pp_print_option pp_within)
          within pp_name left pp_name right
    | Position -> Format.pp_print_string ppf "position"
  in
  (* Runs of [Eq (n, n)] atoms print as the [keys] that make them. *)
  let rec groups = function
    | [] -> []
    | Eq (l, r) :: _ as atoms when String.equal l r ->
        let rec run ns = function
          | Eq (l, r) :: atoms when String.equal l r -> run (l :: ns) atoms
          | atoms -> (List.rev ns, atoms)
        in
        let ns, rest = run [] atoms in
        (fun ppf ->
          Format.fprintf ppf "@[<hov 2>keys@ %a@]" (Type.pp_list pp_name) ns)
        :: groups rest
    | a :: atoms -> (fun ppf -> atom ppf a) :: groups atoms
  in
  match c with
  | [] -> Format.pp_print_string ppf "all"
  | c ->
      Format.fprintf ppf "@[<hov>%a@]"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf "@ && ")
           (fun ppf g -> g ppf))
        (groups c)

(* Conditions *)

let is_ordering = function Closest _ | Nearest _ -> true | _ -> false
let is_compare = function Compare _ -> true | _ -> false

(* [conjunction c0 c1] is [c0 @ c1] if an algorithm runs it. *)
let conjunction c0 c1 =
  let c = c0 @ c1 in
  let fail rule = err "Join.( && ): %a && %a: %s" pp c0 pp c1 rule in
  if
    List.exists (function Position -> true | _ -> false) c
    && List.compare_length_with c 1 > 0
  then fail "position joins row i with row i and takes no other atom";
  (match List.filter is_ordering c with
  | _ :: _ :: _ -> fail "a join takes one closest or nearest atom"
  | [ _ ] when List.exists is_compare c ->
      fail "a closest or nearest join takes no inequality: filter after it"
  | _ -> ());
  let rights =
    List.sort_uniq String.compare
      (List.filter_map (function Compare (_, _, r) -> Some r | _ -> None) c)
  in
  if List.compare_length_with rights 1 > 0 then
    fail
      "an inequality join compares to one right column: join on one, then \
       filter";
  let rec twice = function
    | [] -> ()
    | Eq (l, r) :: atoms
      when List.exists
             (function
               | Eq (l', r') -> String.equal l l' && String.equal r r'
               | _ -> false)
             atoms ->
        fail (Format.asprintf "%a appears twice" pp [ Eq (l, r) ])
    | _ :: atoms -> twice atoms
  in
  twice c;
  c

let keys ns =
  if List.is_empty ns then
    invalid_arg "Join.keys: no key; a join on no key is Join.all";
  (match Problem.repeated ns with
  | n :: _ -> err "Join.keys: %a is named twice" pp_name n
  | [] -> ());
  List.map (fun n -> Eq (n, n)) ns

let eq l r = [ Eq (l, r) ]
let lt l r = [ Compare (`Lt, l, r) ]
let le l r = [ Compare (`Le, l, r) ]
let gt l r = [ Compare (`Gt, l, r) ]
let ge l r = [ Compare (`Ge, l, r) ]
let packed w = Option.map (fun w -> Expr.Packed w) w

let closest ?within = function
  | [ Compare (order, left, right) ] ->
      [ Closest { order; left; right; within = packed within } ]
  | c -> err "Join.closest: %a is not one inequality atom" pp c

let nearest ?within left right =
  [ Nearest { left; right; within = packed within } ]

let position = [ Position ]
let all = []

(* Checking *)

let has_ext (Type.Any t) = Type.has_ext t

(* [common_type a0 a1] is the type at which a column of type [a0] meets one of
   type [a1]: their common type, or their one type if either holds an extension
   type. *)
let common_type (Type.Any t0 as a0) (Type.Any t1 as a1) =
  if has_ext a0 || has_ext a1 then if Type.equal t0 t1 then Some a0 else None
  else
    match Kind.equal_witness (Type.kind t0) (Type.kind t1) with
    | Some Equal -> Option.map (fun t -> Type.Any t) (Type.common [ t0; t1 ])
    | None -> None

(* [joined kind left right c] is the columns of the [kind] join on [c] of rows
   of the columns [left] and [right], two of which may have one name: [left]'s,
   then, for a join that keeps them, [right]'s but the right of an equality
   atom. A [Full] join's key takes the type at which its two columns meet. *)
let joined kind left right c =
  let eqs =
    List.filter_map (function Eq (l, r) -> Some (l, r) | _ -> None) c
  in
  let rights =
    List.filter
      (fun (n, _) -> not (List.exists (fun (_, r) -> String.equal n r) eqs))
      (Schema.columns right)
  in
  let key (n, t) =
    match Option.bind (List.assoc_opt n eqs) (Schema.find right) with
    | Some rt -> (n, Option.value ~default:t (common_type t rt))
    | None -> (n, t)
  in
  match kind with
  | Semi | Anti -> Schema.columns left
  | Inner | Left -> Schema.columns left @ rights
  | Full -> List.map key (Schema.columns left) @ rights

let columns kind left right c = Schema.v (joined kind left right c)

(* [difference t] is the type of the differences of values of [t], and whether
   they are whole days, or [None] if [t] has no difference. *)
let difference : type a. a Type.t -> (Type.any * bool) option = function
  | (Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64) as t ->
      Some (Any t, false)
  | (Float16 | Float32 | Float64) as t -> Some (Any t, false)
  | Decimal _ as t -> Some (Any t, false)
  | Datetime { unit_; _ } | Duration unit_ | Clock unit_ ->
      Some (Any (Type.duration unit_), false)
  | Date -> Some (Any (Type.duration Type.S), true)
  | Bool | String | Binary | Categorical _ | List _ | Record _ | Tensor _
  | Ext _ ->
      None

let is_negative : type a. a Kind.t -> a -> bool =
 fun k v ->
  match k with
  | Int -> v < 0
  | Float -> Float.is_nan v || v < 0.
  | Decimal -> Int64.compare (Decimal.unscaled v) 0L < 0
  | Span -> Int64.compare (Time.Span.to_ns v) 0L < 0
  | _ -> false

let whole_days : type a. a Kind.t -> a -> bool =
 fun k v ->
  match k with
  | Span ->
      Int64.equal
        (Int64.rem (Time.Span.to_ns v) (Time.Span.to_ns (Time.Span.days 1)))
        0L
  | _ -> true

let check kind left right c =
  let problems = ref [] in
  let report fmt =
    Format.kasprintf (fun s -> problems := Problem.v "%s" s :: !problems) fmt
  in
  let find side s n =
    let t = Schema.find s n in
    if Option.is_none t then
      report "%s: %a" side Problem.pp (Problem.missing n s);
    t
  in
  (* [meet l r] is the common type of the left column [l] and the right column
     [r]. *)
  let meet l r =
    match (find "left" left l, find "right" right r) with
    | Some (Type.Any tl as al), Some (Type.Any tr as ar) ->
        let common = common_type al ar in
        if Option.is_none common then
          report "%a is %a and %a is %a, which do not meet: cast one first."
            pp_name l Type.pp tl pp_name r Type.pp tr;
        common
    | _ -> None
  in
  let ordered l r =
    match meet l r with
    | Some (Type.Any (Ext _ as t)) ->
        report
          "%a and %a are %a, which orders only through its declaration: \
           compare their storage, derived first."
          pp_name l pp_name r Type.pp t;
        None
    | Some (Type.Any t) when Type.has_ext t ->
        report
          "%a and %a are %a, which holds an extension type and has no order."
          pp_name l pp_name r Type.pp t;
        None
    | t -> t
  in
  (* [within l r t w] is [w] bound to the difference of [t], the common type of
     [l] and [r] if they order. Whether [w] is a literal at least zero needs no
     type, so it is checked whatever [t] is. *)
  let within l r t = function
    | None -> None
    | Some (Expr.Packed w) -> (
        let literal =
          match Expr.node w with
          | Lit (k, v) when is_negative k v ->
              report "~within:%a is not at least zero." Expr.pp_arg w;
              false
          | Lit _ -> true
          | _ ->
              report "~within:%a is not a literal." Expr.pp_arg w;
              false
        in
        match (t, Expr.node w) with
        | Some (Type.Any t), Lit (k, v) when literal -> (
            match difference t with
            | None ->
                report "~within:%a: %a and %a are %a, which has no difference."
                  Expr.pp_arg w pp_name l pp_name r Type.pp t;
                None
            | Some (Type.Any dt, days) -> (
                match Kind.equal_witness k (Type.kind dt) with
                | None ->
                    report
                      "~within:%a is %a, where the difference of %a and %a is \
                       %a."
                      Expr.pp_arg w Kind.pp k pp_name l pp_name r Type.pp dt;
                    None
                | Some Equal ->
                    if not (Type.holds dt v) then
                      report "~within:%a: %a does not hold it." Expr.pp_arg w
                        Type.pp dt
                    else if days && not (whole_days k v) then
                      report "~within:%a is not a whole number of days."
                        Expr.pp_arg w;
                    Some (Expr.Packed (Expr.typed (Column dt) (Expr.node w)))))
        | _ -> None)
  in
  let bind = function
    | Eq (l, r) as a ->
        ignore (meet l r);
        a
    | Compare (_, l, r) as a ->
        ignore (ordered l r);
        a
    | Closest ({ left = l; right = r; within = w; _ } as a) ->
        Closest { a with within = within l r (ordered l r) w }
    | Nearest { left = l; right = r; within = w } ->
        let t =
          match ordered l r with
          | Some (Type.Any t) when Option.is_none (difference t) ->
              report "nearest %a %a: %a has no difference." pp_name l pp_name r
                Type.pp t;
              None
          | t -> t
        in
        Nearest { left = l; right = r; within = within l r t w }
    | Position -> Position
  in
  let bound = List.map bind c in
  if kind = Full then
    List.iter
      (report
         "%a is the left of two equality atoms, so a Full join cannot coalesce \
          it."
         pp_name)
      (Problem.repeated
         (List.filter_map (function Eq (l, _) -> Some l | _ -> None) c));
  (match (kind, Problem.repeated (List.map fst (joined kind left right c))) with
  | (Semi | Anti), _ | _, [] -> ()
  | _, [ n ] -> report "%a is on both sides: rename one side first." pp_name n
  | _, ns ->
      report "%a are on both sides: rename one side first."
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
           pp_name)
        ns);
  match !problems with [] -> Ok bound | ps -> Error (List.rev ps)

(* Comparing *)

let within_equal w0 w1 =
  Option.equal (fun (Expr.Packed a) (Expr.Packed b) -> Expr.same a b) w0 w1

let atom_equal a0 a1 =
  match (a0, a1) with
  | Eq (l0, r0), Eq (l1, r1) -> String.equal l0 l1 && String.equal r0 r1
  | Compare (o0, l0, r0), Compare (o1, l1, r1) ->
      o0 = o1 && String.equal l0 l1 && String.equal r0 r1
  | Closest c0, Closest c1 ->
      c0.order = c1.order
      && String.equal c0.left c1.left
      && String.equal c0.right c1.right
      && within_equal c0.within c1.within
  | Nearest n0, Nearest n1 ->
      String.equal n0.left n1.left
      && String.equal n0.right n1.right
      && within_equal n0.within n1.within
  | Position, Position -> true
  | (Eq _ | Compare _ | Closest _ | Nearest _ | Position), _ -> false

let equal c0 c1 = List.equal atom_equal c0 c1

(* Last, since it shadows [Stdlib.( && )]. *)
let ( && ) = conjunction
