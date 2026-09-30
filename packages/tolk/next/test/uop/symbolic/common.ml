(* Builders and the law that a rewrite keeps the value, shared by the suite's
   groups. *)

open Windtrap
open Tolk_next
open Dtypes

let uop = Uops.uop

(* Builders *)

let i n = `Int (Z.of_int n)

let var ?dtype ?multiple_of name lo hi =
  Ops.variable ?dtype ?multiple_of name (i lo) (i hi)

let rewrite ?bottom_up m u = Ops.graph_rewrite ?bottom_up ~ctx:() u m
let simple u = rewrite Symbolic.symbolic_simple u
let symbolic u = rewrite Symbolic.symbolic u
let sym u = rewrite Symbolic.sym u

(* Values

   The law that a rewrite keeps the value: at bindings of an expression's
   leaves, the rewritten expression evaluates to what the expression does, and
   within the bounds it claims. The rules reason on bounds as integers do, so
   the law holds wherever no operation or cast of the expression leaves its
   type, which the machine would wrap. A graph of constants has no binding to
   choose: it folds to what the machine computes, wrapping included. *)

(* Exact graphs *)

(* The graphs the law covers: integer and boolean arithmetic over constants and
   leaves. Floats have their own law, at IEEE's special values. *)
let exact u =
  let node n =
    (not (Dtype.is_float (Ops.dtype n)))
    &&
    match Ops.op n with
    | Const | Param | Range | Special | Cast | Bitcast -> true
    | op -> Op.Set.mem op Op.Set.alu && op <> Op.Threefry
  in
  List.for_all node (Ops.toposort u)

(* Leaves *)

let is_leaf u =
  match Ops.arg u with
  | Param { bound = Some _; _ } -> false
  | _ -> Option.is_some (Interpreter.name u)

(* [draw rng k lo hi] is [lo] for the first binding, [hi] for the second, and a
   value between them after. *)
let draw rng k lo hi =
  let span = Z.(hi - lo) in
  match k with
  | 0 -> lo
  | 1 -> hi
  | _ when Z.fits_int span && Z.to_int span < 1 lsl 30 ->
      Z.add lo (Z.of_int (Random.State.int rng (Z.to_int span + 1)))
  | _ ->
      (match Random.State.int rng 4 with
        | 0 -> Z.(lo + of_int (Random.State.int rng 1000))
        | 1 -> Z.(hi - of_int (Random.State.int rng 1000))
        | 2 -> Z.(ediv (lo + hi) (of_int 2))
        | _ -> Z.zero)
      |> Z.min hi |> Z.max lo

let variable_value rng k (arg : Ops.param_arg) =
  match arg.vmin_vmax with
  | Some (`Bool lo, `Bool hi) -> (
      match k with
      | 0 -> `Bool lo
      | 1 -> `Bool hi
      | _ -> `Bool (Random.State.bool rng))
  | Some (lo, hi) ->
      let m = Z.of_int (Option.value arg.multiple_of ~default:1) in
      let lo = Z.(cdiv (Dtype.Value.to_z lo) m)
      and hi = Z.(fdiv (Dtype.Value.to_z hi) m) in
      `Int Z.(m * draw rng k lo hi)
  | None -> invalid_arg "a variable without bounds"

(* A range and a hardware index count from 0 below their end, which the binding
   so far gives. An empty one has no value, and neither has the binding. *)
let counter_value rng k env u =
  match Interpreter.eval ~vars:env (Ops.nth u 0) with
  | `Int n when Z.(n > zero) -> Some (`Int (draw rng k Z.zero Z.(n - one)))
  | _ -> None

(* [bindings rng k us] is the [k]th binding of the leaves of [us] by name, or
   [None] if a range it binds is empty or two leaves share a name. *)
let bindings rng k us =
  let leaves = List.filter is_leaf (Ops.toposort (Ops.sink us)) in
  let names = List.filter_map Interpreter.name leaves in
  if List.length (List.sort_uniq String.compare names) <> List.length names then
    None
  else
    List.fold_left2
      (fun env u name ->
        Option.bind env (fun env ->
            match Ops.arg u with
            | Param arg -> Some ((name, variable_value rng k arg) :: env)
            | _ ->
                Option.map
                  (fun v -> (name, v) :: env)
                  (counter_value rng k env u)))
      (Some []) leaves names

(* Division by zero is undefined, and so is an expression that divides by
   zero. *)
let defined env u =
  List.for_all
    (fun n ->
      match Ops.op n with
      | Floordiv | Floormod | Cdiv | Cmod -> (
          match Interpreter.eval ~vars:env (Ops.nth n 1) with
          | `Int d -> not (Z.equal d Z.zero)
          | _ -> true)
      | _ -> true)
    (Ops.toposort u)

let pp_binding ppf env =
  let pp_one ppf (name, v) = Format.fprintf ppf "%s=%a" name Dtype.pp_const v in
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
    pp_one ppf (List.rev env)

(* The law *)

let within_bounds ~msg u v =
  match (v : Dtype.const) with
  | `Invalid | `Float _ -> ()
  | (`Int _ | `Bool _) as v ->
      let lo = Ops.vmin u and hi = Ops.vmax u in
      if not Dtype.Value.(lo <= v && v <= hi) then
        failf "%s: %a is outside the bounds [%a, %a]" msg Dtype.pp_const v
          Dtype.pp_const lo Dtype.pp_const hi

(* [keeps_value ~name before after] checks that [after] has [before]'s value at
   [count] bindings of their leaves, their extremes first, where [before]
   divides by no zero and, unless it is a graph of constants, wraps nothing, and
   that the value is within [after]'s bounds. It checks nothing unless both are
   exact. *)
let keeps_value ?(count = 16) ~name before after =
  if exact before && exact after then begin
    let rng = Random.State.make [| Hashtbl.hash (name, Ops.key before) |] in
    let constants = not (List.exists is_leaf (Ops.toposort before)) in
    let defined env =
      defined env before
      && (constants || not (Interpreter.overflows ~vars:env before))
    in
    for k = 0 to count - 1 do
      match bindings rng k [ before; after ] with
      | Some env when defined env ->
          let msg = Format.asprintf "%s at %a" name pp_binding env in
          let v = Interpreter.eval ~vars:env before in
          equal ~msg const v (Interpreter.eval ~vars:env after);
          within_bounds ~msg after v
      | _ -> ()
    done
  end

(* Floats

   The law that a rewrite keeps a float value bit for bit: at bindings of an
   expression's scalar parameters to IEEE's special values, the rewritten
   expression evaluates to what the expression does, signed zeros, infinities
   and subnormals included; any NaN is every NaN, since neither nx nor a target
   pins a computed NaN's bits. *)

let float32 x = Dtype.truncate Float32 (`Float x)

let specials =
  List.map float32
    [
      0.;
      -0.;
      Float.infinity;
      Float.neg_infinity;
      Float.nan;
      Int32.float_of_bits 0x7f7fffffl;
      Int32.float_of_bits 0xff7fffffl;
      Int32.float_of_bits 1l;
      Int32.float_of_bits 0x80000001l;
      Int32.float_of_bits 0x007fffffl;
      1.;
      -1.;
      0.5;
      3.;
      1e30;
    ]

let same_float : Dtype.const Testable.t =
  let nan = function `Float x -> Float.is_nan x | _ -> false in
  Testable.make ~pp:Dtype.pp_const ~equal:(fun v0 v1 ->
      (nan v0 && nan v1) || Testable.equal Dtypes.const v0 v1)

(* [keeps_float_value ~name before after] checks that [after] has [before]'s
   value at [count] bindings of the scalar parameters of slots [0] to [2] to
   specials: the first binds each of them to each special in turn, the others at
   random. *)
let keeps_float_value ?(count = 24) ~name before after =
  let rng = Random.State.make [| Hashtbl.hash (name, Ops.key before) |] in
  let n = List.length specials in
  for k = 0 to count - 1 do
    let pick slot =
      if k < n then List.nth specials ((k + slot) mod n)
      else List.nth specials (Random.State.int rng n)
    in
    let params = List.init 3 (fun slot -> (slot, pick slot)) in
    let msg =
      Format.asprintf "%s at %a" name
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
           (fun ppf (s, v) -> Format.fprintf ppf "p%d=%a" s Dtype.pp_const v))
        params
    in
    equal ~msg same_float
      (Interpreter.eval ~params before)
      (Interpreter.eval ~params after)
  done
