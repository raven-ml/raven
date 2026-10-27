(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx.Dtype

type value = V : ('v, 's, 'd) Nx.t -> value
type form = F : ('v, 's, 'd) Nx.Prim.form -> form

let form (V x) = F (Nx.Prim.form x)

(* [v] at [f]'s type. A program binds each variable to a value its equation
   computed at the variable's type, but holds it as a [value], so the type is
   read back from the run-time facts nx checks: the dtype and the set. A value
   is read back at the form its stand-in was traced at, so a failure is a
   binding at the wrong variable, or a trace at other forms than the
   evaluation's. *)
let cast (type v s d) (f : (v, s, d) Nx.Prim.form) (V y) : (v, s, d) Nx.t =
  match Nx.Prim.coerce f y with
  | Some y -> y
  | None when D.equal (Nx.dtype y) f.dtype ->
      let pp ppf = function
        | None -> Format.pp_print_string ppf "every set"
        | Some p -> Nx.Placement.pp ppf p
      in
      invalid_arg
        (Format.asprintf "Rune: a %a value on %a where one on %a was expected"
           Nx.pp_dtype f.dtype pp (Nx.placement y) pp f.placement)
  | None ->
      invalid_arg
        (Format.asprintf "Rune: a %a value where %a was expected" Nx.pp_dtype
           (Nx.dtype y) Nx.pp_dtype f.dtype)

(* [v] at [x]'s type. *)
let like x v = cast (Nx.Prim.form x) v

(* Whether [a] and [b] are one value. *)
let same (V a) (V b) =
  match Nx.Prim.coerce (Nx.Prim.form a) b with Some b -> a == b | None -> false

(* Names *)

(* A map's name is a number no other name has. *)
type axis = int

let axes = Atomic.make 0
let axis () = Atomic.fetch_and_add axes 1
let same_axis = Int.equal

type ('v, 's, 'd) total = { id : ('v, 's, 'd) Nx.t Type.Id.t }

let total () = { id = Type.Id.make () }
let same_total t u = Type.Id.provably_equal t.id u.id

(* Programs *)

type ('v, 's, 'd) Nx.Prim.payload += Var : int -> ('v, 's, 'd) Nx.Prim.payload
type addition =
  | Addition : ('v, 's, 'd) total * ('v, 's, 'd) Nx.Prim.form -> addition

type t = {
  owner : Nx.Prim.interpretation;
  params : int array;
  eqns : eqn array;
  outputs : value array;
  adds : addition array;
  slots : int;
}

and eqn =
  | Op : { by : string; op : 'r Nx.Prim.t; base : int } -> eqn
  | Do : { c : construct; base : int } -> eqn

and construct =
  | Detach : ('v, 's, 'd) Nx.t -> construct
  | Lanes : axis * ('v, 's, 'd) Nx.t -> construct
  | Lane_index : axis -> construct
  | Add : ('v, 's, 'd) total * ('v, 's, 'd) Nx.t -> construct
  | Scan : {
      step : t;
      length : int;
      init : value array;
      xs : value array;
      captures : value array;
    }
      -> construct
  | Iterate : {
      name : string;
      step : t;
      until : t;
      max : int;
      init : value array;
      step_captures : value array;
      until_captures : value array;
    }
      -> construct
  | Call : {
      body : t;
      args : value array;
      compiled : (value array -> value array) option;
    }
      -> construct

let make ~owner ~params eqns ~outputs ~adds ~slots =
  { owner; params; eqns = Array.of_list eqns; outputs; adds; slots }

type _ Effect.t += Construct : construct -> value array Effect.t

exception Evaluate

type _ Effect.t += Bound : axis -> int option Effect.t

let bound a =
  match Effect.perform (Bound a) with
  | n -> n
  | exception Effect.Unhandled (Bound _) -> None

type _ Effect.t += Staging : bool Effect.t

let staging () =
  match Effect.perform Staging with
  | b -> b
  | exception Effect.Unhandled Staging -> false

(* Operands *)

type map = { f : 'v 's 'd. ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t }

let map_values m = Array.map (fun (V x) -> V (m.f x))

let map_operands m = function
  | Detach x -> Detach (m.f x)
  | Lanes (a, x) -> Lanes (a, m.f x)
  | Lane_index _ as c -> c
  | Add (t, x) -> Add (t, m.f x)
  | Scan s ->
      Scan
        {
          s with
          init = map_values m s.init;
          xs = map_values m s.xs;
          captures = map_values m s.captures;
        }
  | Iterate s ->
      Iterate
        {
          s with
          init = map_values m s.init;
          step_captures = map_values m s.step_captures;
          until_captures = map_values m s.until_captures;
        }
  | Call c -> Call { c with args = map_values m c.args }

let operands = function
  | Detach x -> [ V x ]
  | Lanes (_, x) -> [ V x ]
  | Add (_, x) -> [ V x ]
  | Lane_index _ -> []
  | Scan { init; xs; captures; _ } ->
      Array.to_list (Array.concat [ init; xs; captures ])
  | Iterate { init; step_captures; until_captures; _ } ->
      Array.to_list (Array.concat [ init; step_captures; until_captures ])
  | Call { args; _ } -> Array.to_list args

let map_programs f = function
  | (Detach _ | Lanes _ | Lane_index _ | Add _) as c -> c
  | Scan s -> Scan { s with step = f s.step }
  | Iterate s -> Iterate { s with step = f s.step; until = f s.until }
  | Call c -> Call { c with body = f c.body; compiled = None }

(* Forms *)

(* Outputs [i] to [j - 1]. *)
let slice a i j = Array.sub a i (j - i)
let contiguous shape = Nx_array.Layout.contiguous (Iarray.of_array shape)

let shape_of (f : _ Nx.Prim.form) =
  Iarray.to_array (Nx_array.Layout.shape f.layout)

(* The form of a value of [f]'s with a leading axis of [n]. *)
let stacked n (F f) =
  F
    {
      f with
      layout = contiguous (Array.append [| n |] (shape_of f));
      placement = Option.map Nx.Placement.with_leading_axis f.placement;
    }

let computed (F f) = F { f with layout = contiguous (shape_of f) }

(* The additions [p]'s outputs end with. *)
let tail p =
  let n = Array.length p.outputs in
  slice p.outputs (n - Array.length p.adds) n

(* A loop's sums: one per total and shape its programs add to, in the order
   their first additions are made, each the chain of that total's additions in
   the order they are made, as an eager loop would make them. With each
   program's additions, the sum each feeds. *)
let sums programs =
  let found = ref [] in
  (* A sum's addition and form are its first placed addition's, where one
     is: an addition of every set joins it at its placement, as nx adds a
     constant where its other operand lies. *)
  let index (Addition (t, _) as a) (F f) =
    let shape = shape_of f in
    let rec go k = function
      | [] ->
          found := !found @ [ (a, F f) ];
          k
      | (Addition (u, _), F g) :: rest ->
          if Option.is_some (same_total t u) && shape_of g = shape then begin
            if Option.is_none g.placement && Option.is_some f.placement then
              found :=
                List.mapi (fun j e -> if j = k then (a, F f) else e) !found;
            k
          end
          else go (k + 1) rest
    in
    go 0 !found
  in
  let feeds =
    List.map
      (fun p ->
        Array.mapi (fun k a -> index a (computed (form (tail p).(k)))) p.adds)
      programs
  in
  let adds, forms = List.split !found in
  (Array.of_list adds, Array.of_list forms, feeds)

(* The additions a construct's results end with: a loop's sums, and a call's
   additions. *)
let additions = function
  | Detach _ | Lanes _ | Lane_index _ | Add _ -> [||]
  | Scan { step; _ } ->
      let adds, _, _ = sums [ step ] in
      adds
  | Iterate { step; until; _ } ->
      let adds, _, _ = sums [ until; step ] in
      adds
  | Call { body; _ } -> body.adds

let no_map name =
  invalid_arg (name ^ ": no map named by the axis is around the call")

let forms = function
  | Detach x -> [| form (V x) |]
  | Lanes (a, x) -> (
      match bound a with
      | Some n -> [| stacked n (form (V x)) |]
      | None -> no_map "Rune.lanes")
  | Lane_index _ ->
      [| F { dtype = Nx.int32; layout = contiguous [||]; placement = None } |]
  | Add _ -> [||]
  | Scan { step; length; init; _ } ->
      let nc = Array.length init and no = Array.length step.outputs in
      let na = Array.length step.adds in
      let carry =
        if length = 0 then Array.map form init
        else Array.map form (slice step.outputs 0 nc)
      in
      let ys = Array.map form (slice step.outputs nc (no - na)) in
      let _, sums, _ = sums [ step ] in
      Array.concat [ carry; Array.map (stacked length) ys; sums ]
  | Iterate { step; until; init; _ } ->
      let _, sums, _ = sums [ until; step ] in
      (* [init] where no step runs, the step's carry otherwise: the placed of
         the two forms, where nx would compute a constant beside the other. *)
      let placed (F i as fi) (F o as fo) =
        match (i.placement, o.placement) with None, Some _ -> fo | _ -> fi
      in
      let carry = slice step.outputs 0 (Array.length init) in
      let carry = Array.map2 placed (Array.map form init) (Array.map form carry) in
      Array.append carry sums
  | Call { body; _ } -> Array.map form body.outputs

(* Values of a form *)

(* The additive identity that keeps every sign: [-0.], and [0] where a dtype has
   no negative zero. *)
let neg_zero (type v s) (dt : (v, s) Nx.dtype) : v =
  match D.kind dt with
  | Float -> -0.
  | Complex -> { Complex.re = -0.; im = -0. }
  | Signed | Unsigned | Boolean -> D.zero dt

let place (f : _ Nx.Prim.form) x =
  match f.placement with Some p -> Nx.place p x | None -> x

let filled ~zero (F f) =
  let v = if zero then D.zero f.dtype else neg_zero f.dtype in
  V (place f (Nx.full f.dtype (shape_of f) v))

let add (V a) b = V (Nx.add a (like a b))

(* Rows each read at [f], the form their output was traced at. *)
let stack (F f) vs = V (Nx.stack ~axis:0 (List.map (cast f) vs))

(* Evaluation *)

(* An unbound slot. *)
let hole = V (Nx.zeros Nx.bool [||] : (bool, Nx.bool_elt, Nx.host) Nx.t)

(* Whether [until]'s output, a boolean of one element at any set, holds. *)
let stops (V y) =
  match D.equal_witness (Nx.dtype y) Nx.bool with
  | Some Type.Equal -> (Nx.item (List.init (Nx.ndim y) (fun _ -> 0)) y : bool)
  | None ->
      invalid_arg
        (Format.asprintf "Rune: until gave a %a value, not a boolean"
           Nx.pp_dtype (Nx.dtype y))

let rec eval p args =
  if Array.length args <> Array.length p.params then
    invalid_arg
      (Printf.sprintf "Rune: a program of %d parameters applied to %d values"
         (Array.length p.params) (Array.length args));
  let env = Array.make p.slots hole in
  Array.iteri (fun k slot -> env.(slot) <- args.(k)) p.params;
  let owner = p.owner in
  let lookup (type v s d) (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
    match Nx.Prim.payload owner x with Some (Var n) -> like x env.(n) | _ -> x
  in
  let m = { f = lookup } in
  let at = ref 0 in
  let bind (type v s d) (y : (v, s, d) Nx.t) =
    env.(!at) <- V y;
    incr at
  in
  Array.iter
    (function
      | Op { by; op; base } ->
          let op = Nx.Prim.map lookup op in
          let r = Nx.Prim.eval ~by op in
          at := base;
          Nx.Prim.iter_results bind op r
      | Do { c; base } ->
          let r = perform (map_operands m c) in
          Array.blit r 0 env base (Array.length r))
    p.eqns;
  map_values m p.outputs

and perform c =
  match Effect.perform (Construct c) with
  | r -> r
  | exception Effect.Unhandled (Construct c') when c' == c -> default c
  | exception Evaluate -> default c

and default = function
  | Detach x -> [| V x |]
  | Lanes _ -> no_map "Rune.lanes"
  | Lane_index _ -> no_map "Rune.lane_index"
  | Add _ -> [||]
  | Scan { step; length; init; xs; captures } ->
      scan step ~length ~init ~xs ~captures
  | Iterate { name; step; until; max; init; step_captures; until_captures } ->
      iterate name step ~until ~max ~init ~step_captures ~until_captures
  | Call { body; args; compiled } -> (
      let untraced (V x) = Option.is_none (Nx.Prim.owner x) in
      match compiled with
      | Some run when Array.for_all untraced args && not (staging ()) ->
          run args
      | _ -> eval body args)

(* A loop's sums start at the identity, so that a loop that runs no step adds
   nothing a total can tell. *)
and accumulators forms = Array.map (filled ~zero:false) forms

(* [acc] plus the additions [p]'s outputs [out] end with, each to the sum it
   feeds. *)
and accumulate acc feeds out =
  let base = Array.length out - Array.length feeds in
  Array.iteri (fun k j -> acc.(j) <- add acc.(j) out.(base + k)) feeds

and scan step ~length ~init ~xs ~captures =
  let nc = Array.length init in
  let no = Array.length step.outputs and na = Array.length step.adds in
  let ny = no - nc - na in
  let _, forms, feeds = sums [ step ] in
  let feeds = List.hd feeds and acc = accumulators forms in
  let carry = ref init and ys = Array.make ny [] in
  for i = 0 to length - 1 do
    let row = Array.map (fun (V x) -> V (Nx.get [ i ] x)) xs in
    let out = eval step (Array.concat [ !carry; row; captures ]) in
    carry := slice out 0 nc;
    for j = 0 to ny - 1 do
      ys.(j) <- out.(nc + j) :: ys.(j)
    done;
    accumulate acc feeds out
  done;
  let ys =
    if length = 0 then
      Array.map
        (fun v -> filled ~zero:true (stacked 0 (form v)))
        (slice step.outputs nc (nc + ny))
    else
      Array.mapi
        (fun j l -> stack (form step.outputs.(nc + j)) (List.rev l))
        ys
  in
  Array.concat [ !carry; ys; acc ]

and iterate name step ~until ~max ~init ~step_captures ~until_captures =
  let _, forms, feeds = sums [ until; step ] in
  let until_feeds = List.nth feeds 0 and step_feeds = List.nth feeds 1 in
  let acc = accumulators forms in
  let rec go k carry =
    let out = eval until (Array.append carry until_captures) in
    accumulate acc until_feeds out;
    if stops out.(0) then
      carry
    else if k = max then
      invalid_arg
        (Printf.sprintf "%s: until is still false after max = %d steps" name max)
    else begin
      let out = eval step (Array.append carry step_captures) in
      accumulate acc step_feeds out;
      go (k + 1) (slice out 0 (Array.length carry))
    end
  in
  let carry = go 0 init in
  Array.append carry acc

let run c =
  let r = perform c in
  let adds = additions c in
  let base = Array.length r - Array.length adds in
  Array.iteri
    (fun k (Addition (t, f)) -> ignore (perform (Add (t, cast f r.(base + k)))))
    adds;
  r

(* Printing *)

(* A program's names: a variable is [%n], its number; a constant is [cK], in
   the order the program first reads it. *)
type names = { owner : Nx.Prim.interpretation; mutable constants : value list }

let pp_name names ppf (V x) =
  match Nx.Prim.payload names.owner x with
  | Some (Var n) -> Format.fprintf ppf "%%%d" n
  | _ ->
      let rec find k = function
        | [] ->
            names.constants <- names.constants @ [ V x ];
            k
        | v :: rest -> if same v (V x) then k else find (k + 1) rest
      in
      Format.fprintf ppf "c%d" (find 0 names.constants)

let pp_names names ppf vs =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
    (pp_name names) ppf vs

let pp_form ppf (F f) =
  let shape = Iarray.to_array (Nx_array.Layout.shape f.layout) in
  Format.fprintf ppf "%a %a" Nx.pp_dtype f.dtype Nx.pp_shape shape;
  match f.placement with
  | None -> Format.pp_print_string ppf " of every set"
  | Some p -> Format.fprintf ppf " at %a" Nx.Placement.pp p

(* The variables an equation's results bind: from [base] to the next
   equation's, those its program's parameters take aside. *)
let results (p : t) i base =
  let next =
    if i + 1 < Array.length p.eqns then
      match p.eqns.(i + 1) with Op { base; _ } | Do { base; _ } -> base
    else p.slots
  in
  List.filter
    (fun n -> not (Array.mem n p.params))
    (List.init (Int.max 0 (next - base)) (fun k -> base + k))

let rec pp_program ~indent ppf (p : t) =
  let names = { owner = p.owner; constants = [] } in
  let pad = String.make indent ' ' in
  let var ppf n = Format.fprintf ppf "%%%d" n in
  let vars ppf ns =
    Format.pp_print_list
      ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
      var ppf ns
  in
  let body = Format.asprintf "%a" (pp_eqns ~indent names) p in
  Format.fprintf ppf "%sprogram (%a)" pad vars (Array.to_list p.params);
  List.iteri
    (fun k v -> Format.fprintf ppf "@\n%s  c%d : %a" pad k pp_form (form v))
    names.constants;
  Format.fprintf ppf "%s" body;
  let outs = Array.to_list p.outputs in
  let n = List.length outs - Array.length p.adds in
  Format.fprintf ppf "@\n%s  return %a" pad (pp_names names)
    (List.filteri (fun i _ -> i < n) outs);
  if Array.length p.adds > 0 then
    Format.fprintf ppf "@\n%s  adds %a" pad (pp_names names)
      (List.filteri (fun i _ -> i >= n) outs)

and pp_eqns ~indent names ppf (p : t) =
  let pad = String.make (indent + 2) ' ' in
  let lhs i base =
    match results p i base with
    | [] -> ""
    | ns ->
        String.concat ", " (List.map (Printf.sprintf "%%%d") ns) ^ " = "
  in
  let operands ppf vs =
    Format.pp_print_list
      ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
      (fun ppf (i, v) -> Format.fprintf ppf "x%d = %a" i (pp_name names) v)
      ppf
      (List.mapi (fun i v -> (i, v)) vs)
  in
  Array.iteri
    (fun i -> function
      | Op { op; base; _ } ->
          let (Operands xs) = Nx.Prim.operands op in
          Format.fprintf ppf "@\n%s%s%a@\n%s  with %a" pad (lhs i base)
            Nx.Prim.pp op pad operands
            (List.map (fun (Nx.Prim.Any x) -> V x) xs)
      | Do { c; base } ->
          Format.fprintf ppf "@\n%s%s%a" pad (lhs i base)
            (pp_construct ~indent:(indent + 2) names)
            c)
    p.eqns

and pp_construct ~indent names ppf c =
  let pad = String.make indent ' ' in
  let values = pp_names names in
  let program ppf p = Format.fprintf ppf "@\n%a" (pp_program ~indent:(indent + 2)) p in
  match c with
  | Detach x -> Format.fprintf ppf "detach %a" (pp_name names) (V x)
  | Lanes (_, x) -> Format.fprintf ppf "lanes %a" (pp_name names) (V x)
  | Lane_index _ -> Format.pp_print_string ppf "lane_index"
  | Add (_, x) -> Format.fprintf ppf "add %a" (pp_name names) (V x)
  | Scan { step; length; init; xs; captures } ->
      Format.fprintf ppf "scan %d rows, init (%a), xs (%a), captures (%a)%a"
        length values (Array.to_list init) values (Array.to_list xs) values
        (Array.to_list captures) program step
  | Iterate { step; until; max; init; step_captures; until_captures; _ } ->
      Format.fprintf ppf
        "iterate at most %d, init (%a), captures (%a) and (%a)@\n%s  until%a@\n%s  step%a"
        max values (Array.to_list init) values
        (Array.to_list until_captures)
        values
        (Array.to_list step_captures)
        pad program until pad program step
  | Call { body; args; _ } ->
      Format.fprintf ppf "call (%a)%a" values (Array.to_list args) program body

let pp ppf p = pp_program ~indent:0 ppf p
