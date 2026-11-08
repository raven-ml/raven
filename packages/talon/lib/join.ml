(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type compare = [ `Lt | `Le | `Gt | `Ge ]

type atom =
  | Eq of string * string
  | Cmp of compare * string * string
  | Overlaps of (string * string) * (string * string)
  | Position

type cond = atom list
type kind = Inner | Left | Full | Semi | Anti
type count = Any | At_most_one | One | At_least_one

let err fmt = Format.kasprintf invalid_arg fmt

(* Formatting *)

let pp_name = Type.pp_quoted

let compare_name : compare -> string = function
  | `Lt -> "lt"
  | `Le -> "le"
  | `Gt -> "gt"
  | `Ge -> "ge"

let pp ppf c =
  let period ppf (s, e) =
    Format.fprintf ppf "@[<hov 1>(%a,@ %a)@]" pp_name s pp_name e
  in
  let atom ppf = function
    | Eq (l, r) ->
        Format.fprintf ppf "@[<hov 2>eq@ %a@ %a@]" pp_name l pp_name r
    | Cmp (op, l, r) ->
        Format.fprintf ppf "@[<hov 2>%s@ %a@ %a@]" (compare_name op) pp_name l
          pp_name r
    | Overlaps (l, r) ->
        Format.fprintf ppf "@[<hov 2>overlaps@ %a@ %a@]" period l period r
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

(* Comparing *)

let atom_equal a0 a1 = a0 = (a1 : atom)
let equal c0 c1 = List.equal atom_equal c0 c1

(* Conditions *)

(* [conjunction c0 c1] is [c0 @ c1] if no atom of it appears twice and
   [Position] is alone. *)
let conjunction c0 c1 =
  let c = c0 @ c1 in
  let fail rule = err "Join.( && ): %a && %a: %s" pp c0 pp c1 rule in
  if
    List.exists (function Position -> true | _ -> false) c
    && List.compare_length_with c 1 > 0
  then fail "position joins row i with row i and takes no other atom";
  let rec twice = function
    | [] -> ()
    | a :: atoms when List.exists (atom_equal a) atoms ->
        fail (Format.asprintf "%a appears twice" pp [ a ])
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
let lt l r = [ Cmp (`Lt, l, r) ]
let le l r = [ Cmp (`Le, l, r) ]
let gt l r = [ Cmp (`Gt, l, r) ]
let ge l r = [ Cmp (`Ge, l, r) ]
let inside p (s, e) = conjunction (ge p s) (lt p e)
let overlaps l r = [ Overlaps (l, r) ]
let position = [ Position ]
let all = []

let reads c =
  List.fold_left
    (fun (ls, rs) -> function
      | Eq (l, r) | Cmp (_, l, r) -> (l :: ls, r :: rs)
      | Overlaps ((a, b), (c, d)) -> (b :: a :: ls, d :: c :: rs)
      | Position -> (ls, rs))
    ([], []) c

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

(* [ordered find ts] is the problem of the columns [ts] of a comparison or a
   period, found by [find], if they do not meet at a type that has an order. *)
let ordered find ts =
  let found = List.map (fun (side, s, n) -> (n, find side s n)) ts in
  if List.for_all (fun (_, t) -> Option.is_some t) found then
    let ts = List.map (fun (n, t) -> (n, Option.get t)) found in
    let pp_typed ppf (n, Type.Any t) =
      Format.fprintf ppf "%a is %a" pp_name n Type.pp t
    in
    (* "a is t, b is u and c is v". *)
    let pp_all ppf ts =
      let last = List.length ts - 1 in
      List.iteri
        (fun i t ->
          if i = last then Format.pp_print_string ppf " and "
          else if i > 0 then Format.pp_print_string ppf ", ";
          pp_typed ppf t)
        ts
    in
    match List.find_opt (fun (_, t) -> has_ext t) ts with
    | Some (n, Type.Any t) ->
        Some
          (Format.asprintf "%a is %a, which has no order of its own." pp_name n
             Type.pp t)
    | None when Option.is_none (Type.common_any (List.map snd ts)) ->
        Some
          (Format.asprintf "%a, which do not meet: cast one first." pp_all ts)
    | None -> None
  else None

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
  List.iter
    (function
      | Eq (l, r) -> (
          match (find "left" left l, find "right" right r) with
          | Some (Type.Any tl as al), Some (Type.Any tr as ar) ->
              if Option.is_none (common_type al ar) then
                report
                  "%a is %a and %a is %a, which do not meet: cast one first."
                  pp_name l Type.pp tl pp_name r Type.pp tr
          | _ -> ())
      | Cmp (_, l, r) ->
          Option.iter (report "%s")
            (ordered find [ ("left", left, l); ("right", right, r) ])
      | Overlaps ((ls, le), (rs, re)) ->
          Option.iter (report "%s")
          @@ ordered find
               [
                 ("left", left, ls);
                 ("left", left, le);
                 ("right", right, rs);
                 ("right", right, re);
               ]
      | Position -> ())
    c;
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
  List.rev !problems

(* Last, since it shadows [Stdlib.( && )]. *)
let ( && ) = conjunction
