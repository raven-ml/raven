(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type dtype_pat =
  | Dtype of Dtype.t
  | Any_dtype of dtype_pat list

type pos = string * int * int * int

let loc_of_pos (file, line, _, _) = (file, line)

(* Source-pattern layout. [op] and [ops] expose the positional [Fixed] form
   through [?src]; [op_src] and [ops_src] expose the full source-pattern
   vocabulary directly. *)
type src_pat =
  | Fixed of t list
  | Prefix of t list
  | Perms of t list
  | Rep of t

and arg_pat =
  | Any_arg
  | Eq_arg of Uop.Arg.t
  | Has_int of int
  | Has_const of Const.t
  | Has_op of Ops.t
  | Has_op_in of Ops.t list
  | Has_reduce_op of Ops.t
  | Has_reduce_op_in of Ops.t list

and t = {
  alts : t list option;
  ops : Ops.t list option;
  dtype : dtype_pat option;
  src : src_pat option;
  arg : arg_pat;
  tag : string list option;
  name : string option;
  early_reject : Ops.t list option;
  location : (string * int) option;
  allow_any_len : bool;
}

(* Internal constructor. *)
let mk ?alts ?ops ?dtype ?src ?(arg = Any_arg) ?tag ?name ?early_reject
    ?location ?(allow_any_len = false) () =
  (match name with
   | Some "ctx" -> invalid_arg "Upat: capture name \"ctx\" is reserved"
   | _ -> ());
  { alts; ops; dtype; src; arg; tag; name; early_reject; location; allow_any_len }

(* Constructors *)

let exact_dtype dtype = Dtype dtype
let any_dtype dtypes = Any_dtype dtypes

let fixed pats = Fixed pats
let prefix pats = Prefix pats
let perms pats = Perms pats
let repeat pat = Rep pat

let any = mk ()
let var name = mk ~name ()
let is_any pats = mk ~alts:pats ()
let var_dtype name dtype = mk ~name ~dtype ()
let tag s pat = { pat with tag = Some [ s ] }
let tags ss pat = { pat with tag = Some ss }
let early_reject ops pat = { pat with early_reject = Some ops }
let located ~file ~line pat = { pat with location = Some (file, line) }
let located_pos ~loc pat = { pat with location = Some (loc_of_pos loc) }
let location pat = pat.location

let with_loc ?loc pat =
  match loc with
  | None -> pat
  | Some loc -> located_pos ~loc pat

let op ?loc ?dtype ?src ?(arg = Any_arg) ?name ?(allow_any_len = false) o =
  let src = Option.map (fun l -> Fixed l) src in
  let dtype = Option.map (fun d -> Dtype d) dtype in
  with_loc ?loc (mk ~ops:[ o ] ?dtype ?src ~arg ?name ~allow_any_len ())

let ops ?loc ?dtype ?src ?(arg = Any_arg) ?name ?(allow_any_len = false) os =
  let src = Option.map (fun l -> Fixed l) src in
  let dtype = Option.map (fun d -> Dtype d) dtype in
  with_loc ?loc (mk ~ops:os ?dtype ?src ~arg ?name ~allow_any_len ())

let op_src ?loc ?dtype ?src ?(arg = Any_arg) ?name o =
  with_loc ?loc (mk ~ops:[ o ] ?dtype ?src ~arg ?name ())

let ops_src ?loc ?dtype ?src ?(arg = Any_arg) ?name os =
  with_loc ?loc (mk ~ops:os ?dtype ?src ~arg ?name ())

let const ?loc ?dtype ?name c =
  let dtype = Option.map (fun d -> Dtype d) dtype in
  with_loc ?loc (mk ~ops:[ Ops.Const ] ?dtype ~arg:(Has_const c) ?name ())

let const_int n = const (Const.int Dtype.index n)
let const_float x = const (Const.float Dtype.weakfloat x)

(* Bool literals keep a bool dtype constraint (the reference writes them as
   [UPat.const(dtypes.bool, ...)]); numeric literals match by value across
   dtypes. *)
let const_bool b =
  with_loc
    (mk ~ops:[ Ops.Const ] ~dtype:(Dtype Dtype.bool)
       ~arg:(Has_const (Const.bool b)) ())

let cvar ?loc ?name ?dtype ?arg () =
  let dtype = Option.map (fun d -> Dtype d) dtype in
  let arg = Option.fold ~none:Any_arg ~some:(fun c -> Has_const c) arg in
  with_loc ?loc (mk ~ops:[ Ops.Const ] ?name ?dtype ~arg ())

(* Literals *)

let zero = const_int 0
let one = const_int 1
let neg_one = const_int (-1)
let true_ = const_bool true
let false_ = const_bool false

(* Chainable builders *)

let load ?loc ?alt ?gate ?name p =
  let children = match alt, gate with
    | None, None -> [ p ]
    | Some a, Some g -> [ p; a; g ]
    | None, Some _ -> invalid_arg "Upat.load: gate requires alt"
    | Some _, None -> invalid_arg "Upat.load: alt requires gate"
  in
  with_loc ?loc (mk ~ops:[ Ops.Load ] ~src:(Fixed children) ?name ())

let store ?loc ?gate ?name dst value =
  let children = match gate with
    | None -> [ dst; value ]
    | Some g -> [ dst; value; g ]
  in
  with_loc ?loc (mk ~ops:[ Ops.Store ] ~src:(Fixed children) ?name ())

let index ?loc ?name idx ptr =
  with_loc ?loc (mk ~ops:[ Ops.Index ] ~src:(Fixed [ ptr; idx ]) ?name ())

let cast ?loc ?dtype ?name p =
  let dtype = Option.map (fun d -> Dtype d) dtype in
  with_loc ?loc (mk ~ops:[ Ops.Cast ] ?dtype ~src:(Fixed [ p ]) ?name ())

let bitcast ?loc ?dtype ?name p =
  let dtype = Option.map (fun d -> Dtype d) dtype in
  with_loc ?loc (mk ~ops:[ Ops.Bitcast ] ?dtype ~src:(Fixed [ p ]) ?name ())

let gep ?loc ?idx ?name p =
  let idx =
    match idx with None -> cvar ~name:"i" () | Some i -> const_int i
  in
  with_loc ?loc (mk ~ops:[ Ops.Index ] ~src:(Fixed [ p; idx ]) ?name ())

let sink ?loc ?name srcs =
  with_loc ?loc
    (mk ~ops:[ Ops.Sink ] ~src:(Fixed srcs) ~allow_any_len:true ?name ())

let where ?loc ?name cond then_ else_ =
  with_loc ?loc
    (mk ~ops:[ Ops.Where ] ~src:(Fixed [ cond; then_; else_ ]) ?name ())

let alu ?loc ?name args o =
  let src =
    if Ops.Group.is_commutative o && List.length args = 2
    then Perms args
    else Fixed args
  in
  with_loc ?loc (mk ~ops:[ o ] ~src ?name ())

(* Arg pattern constructors *)

let arg_any = Any_arg
let arg_eq a = Eq_arg a
let has_int n = Has_int n
let has_const c = Has_const c
let has_op o = Has_op o
let has_op_in os = Has_op_in os
let has_reduce_op o = Has_reduce_op o
let has_reduce_op_in os = Has_reduce_op_in os

(* Operators — scoped in a sub-module so that `let open Upat in ...` does
   not shadow int arithmetic in rule bodies. *)

module O = struct
  let ( + ) a b = alu [ a; b ] Ops.Add
  let ( * ) a b = alu [ a; b ] Ops.Mul
  let ( - ) a b = alu [ a; b ] Ops.Sub
  let ( / ) a b = alu [ a; b ] Ops.Fdiv
  let ( // ) a b = alu [ a; b ] Ops.Floordiv
  let ( mod ) a b = alu [ a; b ] Ops.Floormod
  let ( < ) a b = alu [ a; b ] Ops.Cmplt
  let cdiv a b = alu [ a; b ] Ops.Cdiv
  let cmod a b = alu [ a; b ] Ops.Cmod
  let floordiv a b = alu [ a; b ] Ops.Floordiv
  let floormod a b = alu [ a; b ] Ops.Floormod
  let ne a b = alu [ a; b ] Ops.Cmpne
end

(* Bindings *)

type bindings = (string * Uop.t) list

let empty_bindings : bindings = []
let find (b : bindings) k = List.assoc_opt k b
let get (b : bindings) k = List.assoc k b
let mem (b : bindings) k = List.mem_assoc k b
let ( $ ) = get

let add k v (b : bindings) : bindings = (k, v) :: b

let add_unique k v (b : bindings) =
  match find b k with
  | None -> Some (add k v b)
  | Some existing -> if Uop.equal existing v then Some b else None

let pp_bindings fmt (b : bindings) =
  Format.fprintf fmt "{";
  List.iteri (fun i (k, u) ->
    if i > 0 then Format.fprintf fmt ", ";
    Format.fprintf fmt "%s=%%%d" k (Uop.tag u)
  ) b;
  Format.fprintf fmt "}"

(* Matching *)

(* Literal patterns compare by numeric value across dtypes, mirroring the
   reference's Python [pat.arg == uop.arg] (where [-1 == -1.0] and
   [True == 1]). Two values of the same class compare exactly; mixed
   classes compare through float, which is lossless for the small literals
   patterns use. *)
let const_value_equal a b =
  let to_float = function
    | Const.Bool b -> Some (if b then 1.0 else 0.0)
    | Const.Int n -> Some (Int64.to_float n)
    | Const.Float f -> Some f
    | Const.Invalid -> None
  in
  match Const.view a, Const.view b with
  | Const.Invalid, Const.Invalid -> true
  | Const.Int x, Const.Int y -> Int64.equal x y
  | va, vb -> (
      match to_float va, to_float vb with
      | Some x, Some y -> Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
      | _ -> false)

let match_arg pat uop_arg =
  match pat with
  | Any_arg -> true
  | Eq_arg a -> Uop.Arg.equal a uop_arg
  | Has_int n ->
      (match uop_arg with Uop.Arg.Int m -> n = m | _ -> false)
  | Has_const c ->
      (match uop_arg with
       | Uop.Arg.Value v -> const_value_equal c v
       | _ -> false)
  | Has_op o ->
      (match uop_arg with
       | Uop.Arg.Op o' -> Ops.equal o o'
       | _ -> false)
  | Has_op_in os ->
      (match uop_arg with
       | Uop.Arg.Op o' -> List.exists (Ops.equal o') os
       | _ -> false)
  | Has_reduce_op o ->
      (match uop_arg with
       | Uop.Arg.Reduce_arg { op = o'; _ } -> Ops.equal o o'
       | _ -> false)
  | Has_reduce_op_in os ->
      (match uop_arg with
       | Uop.Arg.Reduce_arg { op = o'; _ } -> List.exists (Ops.equal o') os
       | _ -> false)

let rec match_dtype pat uop_dtype =
  match pat with
  | Dtype d -> Dtype.equal d uop_dtype
  | Any_dtype dtypes -> List.exists (fun dtype -> match_dtype dtype uop_dtype) dtypes

let permutations l =
  let rec insert x = function
    | [] -> [ [ x ] ]
    | y :: ys as l ->
        (x :: l) :: List.map (fun l' -> y :: l') (insert x ys)
  in
  let rec aux = function
    | [] -> [ [] ]
    | x :: xs ->
        List.concat_map (fun perm -> insert x perm) (aux xs)
  in
  aux l

let bindings_equal a b =
  let same_binding (k, v) =
    match find b k with
    | None -> false
    | Some v' -> Uop.equal v v'
  in
  List.length a = List.length b && List.for_all same_binding a

let add_binding_set bs sets =
  if List.exists (bindings_equal bs) sets then sets else bs :: sets

let dedup_bindings sets = List.rev (List.fold_left (fun acc bs -> add_binding_set bs acc) [] sets)

let rec match_one pat uop bs =
  match pat.alts with
  | Some pats -> List.concat_map (fun pat -> match_one pat uop bs) pats
  | None ->
  let op_ok = match pat.ops with
    | None -> true
    | Some os -> List.exists (Ops.equal (Uop.op uop)) os
  in
  if not op_ok then [] else
  let dt_ok = match pat.dtype with
    | None -> true
    | Some d -> match_dtype d (Uop.dtype uop)
  in
  if not dt_ok then [] else
  if not (match_arg pat.arg (Uop.arg uop)) then [] else
  let tag_ok = match pat.tag with
    | None -> true
    | Some tags ->
        (match Uop.node_tag uop with
         | None -> false
         | Some tag -> List.exists (String.equal tag) tags)
  in
  if not tag_ok then [] else
  let bs_list = match pat.name with
    | None -> [ bs ]
    | Some n ->
        (match add_unique n uop bs with
         | None -> []
         | Some bs' -> [ bs' ])
  in
  if bs_list = [] then [] else
  match pat.src with
  | None -> bs_list
  | Some src ->
      let uop_srcs = Uop.src uop in
      List.concat_map (fun bs -> match_src src uop_srcs bs pat.allow_any_len)
        bs_list

and match_src src_pat uop_srcs bs allow_any_len =
  let n_srcs = Array.length uop_srcs in
  match src_pat with
  | Fixed pats ->
      let n_pats = List.length pats in
      if n_pats > n_srcs then []
      else if (not allow_any_len) && n_pats < n_srcs then []
      else match_sequence pats uop_srcs 0 [ bs ]
  | Prefix pats ->
      if List.length pats > n_srcs then []
      else match_sequence pats uop_srcs 0 [ bs ]
  | Perms pats ->
      if List.length pats <> n_srcs && not allow_any_len then []
      else
        permutations pats
        |> List.concat_map (fun perm -> match_sequence perm uop_srcs 0 [ bs ])
        |> dedup_bindings
  | Rep pat ->
      Array.fold_left
        (fun acc uop_src ->
          List.concat_map (fun bs -> match_one pat uop_src bs) acc)
        [ bs ] uop_srcs

and match_sequence pats uop_srcs i bs_list =
  match pats with
  | [] -> bs_list
  | p :: ps' ->
      if i >= Array.length uop_srcs then []
      else
        let bs_list' =
          List.concat_map (fun bs -> match_one p uop_srcs.(i) bs) bs_list
        in
        match_sequence ps' uop_srcs (i + 1) bs_list'

let match_ pat uop = match_one pat uop empty_bindings

(* Rules *)

type 'ctx rule_with_ctx = t * ('ctx -> bindings -> Uop.t option)
type rule = unit rule_with_ctx

let with_ctx pat cb : 'ctx rule_with_ctx = (pat, cb)
let ( => ) pat cb : rule = with_ctx pat (fun () bs -> cb bs)

(* Variadic capture combinators.

   Each [rewriteN] builds the pattern by feeding the user-supplied
   pattern-builder fresh [var] captures, then extracts them from the
   bindings before invoking the user's callback. Fresh names are
   unique to this module so they cannot collide with user-chosen
   names. *)

let rewrite1 p k =
  let x = var "__upat_r0" in
  p x => fun b -> k (b $ "__upat_r0")

let rewrite2 p k =
  let x = var "__upat_r0" and y = var "__upat_r1" in
  p x y => fun b -> k (b $ "__upat_r0") (b $ "__upat_r1")

let rewrite3 p k =
  let x = var "__upat_r0" and y = var "__upat_r1"
  and z = var "__upat_r2" in
  p x y z => fun b ->
    k (b $ "__upat_r0") (b $ "__upat_r1") (b $ "__upat_r2")

let rewrite4 p k =
  let x = var "__upat_r0" and y = var "__upat_r1"
  and z = var "__upat_r2" and w = var "__upat_r3" in
  p x y z w => fun b ->
    k (b $ "__upat_r0") (b $ "__upat_r1")
      (b $ "__upat_r2") (b $ "__upat_r3")

(* Pattern matcher *)

let add_op op ops =
  if List.exists (Ops.equal op) ops then ops else op :: ops

let root_ops pat =
  match pat.ops with
  | Some (_ :: _ as ops) -> Some ops
  | _ -> None

let single_root_op pat =
  match pat.ops with
  | Some [ op ] -> Some op
  | _ -> None

let src_patterns = function
  | Fixed pats | Prefix pats | Perms pats -> pats
  | Rep pat -> [ pat ]

let inferred_early_reject pat =
  match pat.early_reject, pat.src with
  | Some ops, _ -> ops
  | None, None -> []
  | None, Some src ->
      List.fold_left
        (fun acc pat ->
          match single_root_op pat with
          | None -> acc
          | Some op -> add_op op acc)
        [] (src_patterns src)

let early_reject_matches reject src_ops =
  List.for_all (fun op -> List.exists (Ops.equal op) src_ops) reject

let rootless_message pat =
  match pat.location with
  | None -> "Upat.Pattern_matcher.make: root pattern has no op"
  | Some (file, line) ->
      Printf.sprintf
        "Upat.Pattern_matcher.make: root pattern at %s:%d has no op"
        file line

module Pattern_matcher = struct
  type 'ctx entry = {
    pat : t;
    cb : 'ctx -> bindings -> Uop.t option;
    reject : Ops.t list;
  }

  type 'ctx with_ctx = {
    rules : 'ctx entry list;
    dispatch : (Ops.t, 'ctx entry list) Hashtbl.t;
  }

  type t = unit with_ctx

  let entry (pat, cb) =
    { pat; cb; reject = inferred_early_reject pat }

  let rules_of_entries entries =
    List.map (fun entry -> entry.pat, entry.cb) entries

  let make_with_ctx rules =
    let entries = List.map entry rules in
    let dispatch = Hashtbl.create 16 in
    List.iter
      (fun entry ->
        match root_ops entry.pat with
        | None -> invalid_arg (rootless_message entry.pat)
        | Some ops ->
            List.iter
              (fun op ->
                let old =
                  match Hashtbl.find_opt dispatch op with
                  | None -> []
                  | Some entries -> entries
                in
                Hashtbl.replace dispatch op (entry :: old))
              ops)
      entries;
    (* Buckets accumulate in reverse rule order; restore it so dispatch keeps
       first-match-wins without reversing on every rewrite. *)
    Hashtbl.filter_map_inplace (fun _ es -> Some (List.rev es)) dispatch;
    { rules = entries; dispatch }

  let make rules = make_with_ctx rules

  let rewrite_with_ctx pm ~ctx u =
    match Hashtbl.find_opt pm.dispatch (Uop.op u) with
    | None -> None
    | Some entries ->
        let src_ops = Uop.child_ops u in
        let rec try_rules = function
          | [] -> None
          | entry :: rest ->
              if not (early_reject_matches entry.reject src_ops) then
                try_rules rest
              else
                let rec try_bindings = function
                  | [] -> try_rules rest
                  | bs :: more -> (
                      match entry.cb ctx bs with
                      | Some r ->
                          (* The callback commits the pattern on its first
                             non-None result: a rewrite to the same uop counts
                             as a no-op and falls through to the next rule, not
                             the next binding. *)
                          if Uop.equal r u then try_rules rest else Some r
                      | None -> try_bindings more)
                in
                try_bindings (match_ entry.pat u)
        in
        try_rules entries

  let rewrite pm u = rewrite_with_ctx pm ~ctx:() u

  let append_with_ctx a b =
    let rules = rules_of_entries a.rules @ rules_of_entries b.rules in
    make_with_ctx rules

  let ( ++ ) = append_with_ctx
  let compose = function
    | [] -> make []
    | pm :: pms -> List.fold_left append_with_ctx pm pms
end
