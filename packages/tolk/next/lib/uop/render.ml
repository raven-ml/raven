(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let strf = Printf.sprintf
let to_string pp x = Format.asprintf "%a" pp x
let str_const c = to_string Dtype.pp_const c

let repr_ints = function
  | [ n ] -> strf "(%d,)" n
  | ns -> "(" ^ String.concat ", " (List.map string_of_int ns) ^ ")"

(* Uop helpers *)

(* A float constant prints bare and a string without its quotes, as Python's
   [str] prints them. *)
let str_arg = function
  | Const c -> str_const c
  | String s | Device (Single s) -> s
  | a -> to_string pp_arg a

let pp_uops ppf uops =
  let index = Tbl.create 64 in
  List.iteri (fun i u -> Tbl.replace index u i) uops;
  let src_str x =
    match Tbl.find_opt index x with
    | None -> "'--'"
    | Some _ when op x = Op.Const -> "'" ^ str_const (value x) ^ "'"
    | Some i -> string_of_int i
  in
  let line i u =
    strf "%4d %-20s: %s %-40s %-32s %s" i
      (to_string Op.pp (op u))
      (multirange_str ~color:true ~pad:10 (Nodes.to_list (ranges u)))
      (to_string Dtype.pp (dtype u))
      ("[" ^ String.concat ", " (List.map src_str (src u)) ^ "]")
      (str_arg (arg u))
  in
  Format.fprintf ppf "@[<v>%a@]"
    (Format.pp_print_list ~pp_sep:Format.pp_print_cut Format.pp_print_string)
    (List.mapi line uops)

(* For debug *)

let syms =
  Op.
    [
      (Add, "+");
      (Sub, "-");
      (Floordiv, "//");
      (Floormod, "%");
      (Shl, "<<");
      (Shr, ">>");
      (Mul, "*");
      (Cmplt, "<");
      (Cmpne, "!=");
      (And, "&");
      (Or, "|");
      (Xor, "^");
    ]

(* Comparisons have no precedence: Python chains them rather than associating
   them to the left. *)
let precedence =
  Op.
    [
      (Mul, 1);
      (Floordiv, 1);
      (Floormod, 1);
      (Add, 2);
      (Sub, 2);
      (Shl, 3);
      (Shr, 3);
      (And, 4);
      (Xor, 5);
      (Or, 6);
    ]

let strip_binary_parens x left right =
  let code a b = strf "(%s%s%s)" a (List.assoc (op x) syms) b in
  match List.assoc_opt (op x) precedence with
  | None -> code left right
  | Some p ->
      let prec i =
        Option.value ~default:99 (List.assoc_opt (op (nth x i)) precedence)
      in
      code
        (if prec 0 <= p then Helpers.strip_parens left else left)
        (if prec 1 < p then Helpers.strip_parens right else right)

let tuple = function
  | [ p ] -> "(" ^ p ^ ",)"
  | ps -> "(" ^ String.concat "," ps ^ ")"

(* The context holds the text of each node rendered so far, and [render] for the
   sizes of a movement that the graph never contained: a movement's argument is
   simplified. *)
type ctx = { strs : string Tbl.t; render : t -> string }

let marg_str ctx = function
  | Int n -> string_of_int n
  | Sym a -> (
      match Tbl.find_opt ctx.strs a with Some s -> s | None -> ctx.render a)

let render_marg ctx x =
  match marg x with
  | Permute axes -> repr_ints axes
  | Flip flips ->
      repr_ints
        (List.concat (List.mapi (fun i f -> if f then [ i ] else []) flips))
  | Reshape s | Expand s -> tuple (List.map (marg_str ctx) s)
  | Pad b | Shrink b ->
      tuple
        (List.map
           (fun (a0, a1) -> strf "(%s, %s)" (marg_str ctx a0) (marg_str ctx a1))
           b)

let renderer =
  let open Pattern_matcher in
  let x o = Upat.op o ~name:"x" and xs os = Upat.v ~op:os ~name:"x" () in
  let r ctx u = Tbl.find ctx.strs u in
  let src ctx m i = r ctx (nth (m "x") i) in
  let named ~prefix m =
    match arg (m "x") with
    | Param { name = Some n; _ } -> Some n
    | Param p -> Some (strf "%s%d" prefix p.slot)
    | _ -> None
  in
  fold
    [
      rule (x Op.Param) (named ~prefix:"p");
      rule
        (xs (Op.Set.of_list [ Op.Buffer; Op.Alloc ]))
        (fun m -> named ~prefix:(if op (m "x") = Op.Alloc then "a" else "b") m);
      rule_ctx (x Op.After) (fun ctx m -> Some (src ctx m 0));
      rule (x Op.Special) (fun m ->
          match arg (m "x") with String s -> Some s | _ -> None);
      rule (Upat.op Op.Range ~dtype:[ Dtype.Void ] ~name:"x") (fun m ->
          Some (strf "loop%d" (List.hd (axis_id (m "x")))));
      rule (x Op.Range) (fun m -> Some ("r" ^ range_str (m "x")));
      rule (x Op.Const) (fun m -> Some (str_const (value (m "x"))));
      (* The cast states the width, the weak constant carries the value. *)
      rule
        (Upat.f (Upat.cvar "c") Op.Cast)
        (fun m -> Some (str_const (value (m "c"))));
      rule_ctx (x Op.Cast) (fun ctx m ->
          let dt = to_string Dtype.pp (dtype (m "x")) in
          let n = String.length "dtypes." in
          let name = String.sub dt n (String.length dt - n) in
          Some (strf "(%s)(%s)" name (src ctx m 0)));
      rule_ctx (x Op.Neg) (fun ctx m -> Some (strf "(-%s)" (src ctx m 0)));
      rule_ctx (x Op.Reciprocal) (fun ctx m ->
          Some (strf "(1/%s)" (src ctx m 0)));
      rule_ctx (x Op.Max) (fun ctx m ->
          Some (strf "max(%s, %s)" (src ctx m 0) (src ctx m 1)));
      rule_ctx (x Op.Mulacc) (fun ctx m ->
          Some (strf "(%s*%s+%s)" (src ctx m 0) (src ctx m 1) (src ctx m 2)));
      rule_ctx (x Op.Where) (fun ctx m ->
          Some
            (strf "(%s if %s else %s)" (src ctx m 1) (src ctx m 0) (src ctx m 2)));
      rule_ctx (x Op.Cdiv) (fun ctx m ->
          Some (strf "cdiv(%s, %s)" (src ctx m 0) (src ctx m 1)));
      rule_ctx (x Op.Cmod) (fun ctx m ->
          Some (strf "cmod(%s, %s)" (src ctx m 0) (src ctx m 1)));
      rule_ctx (xs Op.Set.movement) (fun ctx m ->
          let name = String.lowercase_ascii (Op.name (op (m "x"))) in
          Some (strf "%s.%s(%s)" (src ctx m 0) name (render_marg ctx (m "x"))));
      rule_ctx
        (xs (Op.Set.of_list (List.map fst syms)))
        (fun ctx m ->
          Some (strip_binary_parens (m "x") (src ctx m 0) (src ctx m 1)));
      rule_ctx
        (xs (Op.Set.of_list [ Op.Index; Op.Stage ]))
        (fun ctx m ->
          let bracket y = "[" ^ Helpers.strip_parens (r ctx y) ^ "]" in
          Some (String.concat "" (List.map bracket (List.tl (Ops.src (m "x"))))));
      rule_ctx
        (Upat.load (Upat.op Op.Index ~name:"idx") [])
        (fun ctx m -> Some (r ctx (nth (m "idx") 0) ^ r ctx (m "idx")));
      rule_ctx
        (Upat.load
           (Upat.op Op.Index ~name:"idx")
           [ Upat.var "alt"; Upat.var "gate" ])
        (fun ctx m ->
          Some
            (strf "(%s%s if %s else %s)"
               (r ctx (nth (m "idx") 0))
               (r ctx (m "idx"))
               (r ctx (m "gate"))
               (r ctx (m "alt"))));
      rule_ctx (x Op.Stack) (fun ctx m ->
          Some
            ("{" ^ String.concat "," (List.map (r ctx) (Ops.src (m "x"))) ^ "}"));
      rule (xs Op.Set.all) (fun m -> Some (to_string pp (m "x")));
    ]

let rec render ?(simplify = true) u =
  let s = if simplify then Ops.simplify u else u in
  let ctx = { strs = Tbl.create 64; render = (fun a -> render a) } in
  List.iter
    (fun u ->
      Tbl.replace ctx.strs u
        (Option.get (Pattern_matcher.rewrite renderer ctx u)))
    (toposort s);
  Tbl.find ctx.strs s

let srender = function Int n -> string_of_int n | Sym u -> render u
