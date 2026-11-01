(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

let strf = Printf.sprintf
let to_string pp x = Format.asprintf "%a" pp x
let str_const c = to_string Dtype.pp_const c

let repr_ints = function
  | [ n ] -> strf "(%d,)" n
  | ns -> "(" ^ String.concat ", " (List.map string_of_int ns) ^ ")"

(* Wire format *)

let rgb hex =
  Helpers.Rgb
    ( int_of_string ("0x" ^ String.sub hex 0 2),
      int_of_string ("0x" ^ String.sub hex 2 2),
      int_of_string ("0x" ^ String.sub hex 4 2) )

let uops_colors =
  Op.
    [
      (Load, "ffc0c0");
      (Store, "87CEEB");
      (Const, "e0e0e0");
      (Reduce, "FF5B5B");
      (Range, "c8a0e0");
      (Barrier, "ff8080");
      (If, "c8b0c0");
      (Special, "c0c0ff");
      (Index, "CEF9B7");
      (Stack, "D8F9E4");
      (Wmma, "efefc0");
      (Ins, "eec4ff");
      (Threefry, "ffff80");
      (Buffer, "B0BDFF");
      (Getaddr, "9DB1F0");
      (Copy, "ff90c0");
      (Custom_function, "bf71b6");
      (Call, "00B7C8");
      (Param, "14686F");
      (Source, "c0c0c0");
      (Binary, "404040");
      (Linear, "7DF4FF");
      (Alloc, "C07788");
      (Program, "a0a0a0");
      (Stage, "FFC14D");
      (After, "8A7866");
      (End, "524C46");
      (Backedge, "464752");
    ]

let op_color o =
  match List.assoc_opt o uops_colors with
  | Some hex -> Some (rgb hex)
  | None when Op.Set.mem o Op.Set.alu -> Some (rgb "ffffc0")
  | None when Op.Set.mem o Op.Set.movement -> Some (rgb "D8F9E4")
  | None -> None

(* An enumeration's name, without its type: [LOOP] for [AxisType.LOOP]. *)
let member s =
  let dot = String.index s '.' + 1 in
  String.sub s dot (String.length s - dot)

let addr_name a = member (to_string Dtype.pp_addr_space a)
let axis_name a = member (to_string Axis_type.pp a)

let word_device s =
  String.length s > 0
  && String.for_all
       (function
         | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' | ':' -> true
         | _ -> false)
       s

(* A constant is a literal, of a weak type or boolean: a constant of another
   type is a cast of one. *)
let const_arg u =
  match value u with
  | `Invalid -> "invalid"
  | `Bool b -> string_of_bool b
  | v -> str_const v

let storage_arg u (p : param_arg) =
  let opts =
    List.concat
      [
        (match p.vmin_vmax with
        | Some (lo, hi) ->
            [ strf " bounds=[%s,%s]" (str_const lo) (str_const hi) ]
        | None -> []);
        (match p.addrspace with
        | Dtype.Global -> []
        | a -> [ " addrspace=" ^ addr_name a ]);
        (match p.device with
        | Some s when word_device s -> [ " device=" ^ s ]
        | Some d -> [ " device=" ^ to_string pp_device d ]
        | None -> []);
        (if p.volatile then [ " volatile=true" ] else []);
      ]
  in
  let name = match p.name with Some n -> strf "\"%s\" " n | None -> "" in
  strf "%sdtype=%s slot=%d%s" name (Dtype.name (dtype u)) p.slot
    (String.concat "" opts)

(* Bare scalars, keyed fields, and the literal of any other argument. *)
let render_arg u =
  match (op u, arg u) with
  | Op.Const, _ -> const_arg u
  | (Op.Param | Op.Buffer | Op.Alloc), Param p -> storage_arg u p
  | Op.Range, _ ->
      String.concat " "
        (axis_name (axis_type u) :: List.map string_of_int (axis_id u))
  | Op.Sink, Kernel k -> k.name
  | Op.Sink, _ -> ""
  | Op.Call, Call c ->
      String.concat " "
        ((match c.name with
         | Some n -> [ "name=" ^ to_string pp_arg (String n) ]
         | None -> [])
        @ if c.precompile then [ "precompile=true" ] else [])
  | Op.Reduce, Reduce r ->
      strf "op=%s%s"
        (String.lowercase_ascii (Op.name r.op))
        (if r.num_axes <> 0 then strf " pop=%d" r.num_axes else "")
  | (Op.Cast | Op.Bitcast), Dtype dt -> Dtype.name dt
  | Op.Copy, Device s | Op.Special, String s -> s
  | _, No_arg -> ""
  | _, a -> to_string pp_arg a

(* Constants have no line: they are written in their users, and so is a stack
   of constants, as a tuple. *)
let inline u =
  op u = Op.Const
  || (op u = Op.Stack && List.for_all (fun s -> op s = Op.Const) (src u))

let render_uir uops =
  let nodes = List.filter (fun u -> not (inline u)) uops in
  let table = Tbl.create 64 in
  List.iteri (fun i u -> Tbl.replace table u i) nodes;
  let rec src_str u =
    if not (inline u) then strf "%%%d" (Tbl.find table u)
    else if op u = Op.Const then render_arg u
    else "(" ^ String.concat ", " (List.map src_str (src u)) ^ ")"
  in
  let line i u =
    let name = String.lowercase_ascii (Op.name (op u)) in
    let name =
      match op_color (op u) with
      | Some c -> Helpers.colored c name
      | None -> name
    in
    let srcs =
      match src u with
      | [] -> ""
      | s -> " " ^ String.concat ", " (List.map src_str s)
    in
    let a = match render_arg u with "" -> "" | a -> " : " ^ a in
    strf "%%%d = %s%s%s" i name srcs a
  in
  String.concat "\n" (List.mapi line nodes)

(* For debug *)

let syms =
  Op.
    [
      (Add, "+");
      (Sub, "-");
      (Div, "//");
      (Mod, "%");
      (Shl, "<<");
      (Shr, ">>");
      (Mul, "*");
      (Cmplt, "<");
      (Cmpne, "!=");
      (Cmpeq, "==");
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
      (Div, 1);
      (Mod, 1);
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
  | Reshape s | Expand s -> tuple (List.map (marg_str ctx) s)
  | Pad b | Shrink b ->
      tuple
        (List.map
           (fun (a0, a1) -> strf "(%s, %s)" (marg_str ctx a0) (marg_str ctx a1))
           b)

(* Each index of an access, in brackets: [[i][j + 1]]. *)
let render_index srcs =
  String.concat "" (List.map (fun s -> "[" ^ Helpers.strip_parens s ^ "]") srcs)

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
    (fun () -> [
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
      rule_ctx (x Op.Recip) (fun ctx m ->
          Some (strf "(1/%s)" (src ctx m 0)));
      rule_ctx
        (xs (Op.Set.of_list [ Op.Max; Op.Truncdiv; Op.Truncmod ]))
        (fun ctx m ->
          let name = String.lowercase_ascii (Op.name (op (m "x"))) in
          Some (strf "%s(%s, %s)" name (src ctx m 0) (src ctx m 1)));
      rule_ctx (x Op.Mulacc) (fun ctx m ->
          Some (strf "(%s*%s+%s)" (src ctx m 0) (src ctx m 1) (src ctx m 2)));
      rule_ctx (x Op.Where) (fun ctx m ->
          Some
            (strf "(%s if %s else %s)" (src ctx m 1) (src ctx m 0) (src ctx m 2)));
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
          Some (render_index (List.map (r ctx) (List.tl (Ops.src (m "x"))))));
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
    ])

let rec render ?(simplify = true) u =
  let s = if simplify then Shape.simplify u else u in
  let ctx = { strs = Tbl.create 64; render = (fun a -> render a) } in
  List.iter
    (fun u ->
      Tbl.replace ctx.strs u
        (Option.get (Pattern_matcher.rewrite renderer ctx u)))
    (toposort ~calls:Skip s);
  Tbl.find ctx.strs s

let srender = function Int n -> string_of_int n | Sym u -> render u
