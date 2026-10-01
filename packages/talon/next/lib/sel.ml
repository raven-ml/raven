(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | All
  | Names of string list
  | Prefix of string
  | Suffix of string
  | Of_kind : 'a Kind.t -> t
  | Where of (string -> Type.any -> bool)
  | Union of t * t
  | Diff of t * t
  | Inter of t * t

let all = All
let names ns = Names ns
let prefix p = Prefix p
let suffix s = Suffix s
let of_kind k = Of_kind k
let where p = Where p
let ( + ) s0 s1 = Union (s0, s1)
let ( - ) s0 s1 = Diff (s0, s1)
let inter s0 s1 = Inter (s0, s1)

(* Resolving *)

let binds k (Type.Any t) = Option.is_some (Kind.provably_equal k (Type.kind t))

let dedup names =
  let seen = Hashtbl.create 16 in
  List.filter
    (fun n ->
      (not (Hashtbl.mem seen n))
      &&
      (Hashtbl.add seen n ();
       true))
    names

let check sel schema =
  let columns = Schema.columns schema in
  let problems = ref [] in
  let filter p =
    List.filter_map (fun (n, t) -> if p n t then Some n else None) columns
  in
  let rec resolve = function
    | All -> List.map fst columns
    | Names ns ->
        let present n =
          Option.is_some (Schema.find schema n)
          ||
          (problems := Problem.missing n schema :: !problems;
           false)
        in
        dedup (List.filter present ns)
    | Prefix p -> filter (fun n _ -> String.starts_with ~prefix:p n)
    | Suffix s -> filter (fun n _ -> String.ends_with ~suffix:s n)
    | Of_kind k -> filter (fun _ t -> binds k t)
    | Where p -> filter p
    | Union (s0, s1) ->
        let ns0 = resolve s0 in
        let ns1 = resolve s1 in
        dedup (ns0 @ ns1)
    | Diff (s0, s1) ->
        let ns0 = resolve s0 in
        let ns1 = resolve s1 in
        List.filter (fun n -> not (List.mem n ns1)) ns0
    | Inter (s0, s1) ->
        let ns0 = resolve s0 in
        let ns1 = resolve s1 in
        List.filter (fun n -> List.mem n ns1) ns0
  in
  let names = resolve sel in
  (names, List.rev !problems)

(* Formatting *)

(* [pp_at level ppf s] formats [s] at [level]: [0] where any selector stands,
   [1] as the right operand of [+] or [-], [2] as a function's argument. *)
let rec pp_at level ppf s =
  let parens min ppf pp =
    if level >= min then Format.fprintf ppf "(%t)" pp else pp ppf
  in
  match s with
  | All -> Format.pp_print_string ppf "all"
  | Union (s0, s1) ->
      parens 1 ppf (fun ppf ->
          Format.fprintf ppf "%a +@ %a" (pp_at 0) s0 (pp_at 1) s1)
  | Diff (s0, s1) ->
      parens 1 ppf (fun ppf ->
          Format.fprintf ppf "%a -@ %a" (pp_at 0) s0 (pp_at 1) s1)
  | Names ns ->
      parens 2 ppf (fun ppf ->
          Format.fprintf ppf "names %a"
            (Type.pp_list (fun ppf -> Format.fprintf ppf "%S"))
            ns)
  | Prefix p -> parens 2 ppf (fun ppf -> Format.fprintf ppf "prefix %S" p)
  | Suffix x -> parens 2 ppf (fun ppf -> Format.fprintf ppf "suffix %S" x)
  | Of_kind k ->
      parens 2 ppf (fun ppf -> Format.fprintf ppf "of_kind %a" Kind.pp k)
  | Where _ -> parens 2 ppf (fun ppf -> Format.pp_print_string ppf "where <fn>")
  | Inter (s0, s1) ->
      parens 2 ppf (fun ppf ->
          Format.fprintf ppf "inter %a@ %a" (pp_at 2) s0 (pp_at 2) s1)

let pp_arg ppf s = Format.fprintf ppf "@[<hov 2>%a@]" (pp_at 2) s
