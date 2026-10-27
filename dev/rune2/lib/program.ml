(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx.Dtype

type value = V : ('v, 's, 'd) Nx.t -> value
type form = F : ('v, 's, 'd) Nx.Prim.form -> form

let form (V x) = F (Nx.Prim.form x)

(* A value at another set. nx's sets are phantom types: no value holds one at
   run time. A program binds each variable to a value its equation computed at
   the variable's own type, and a structure's arrays lie on its set, so the
   value read back is at its true type. OCaml cannot see it: a payload is
   covariant in its set, which a witness of the set would not be. The dtype is
   checked by its witness wherever a value is read back. *)
let rebrand : type v s d e. (v, s, d) Nx.t -> (v, s, e) Nx.t = Obj.magic

let unpack (type v s d) (dt : (v, s) Nx.dtype) (V y) : (v, s, d) Nx.t =
  match D.equal_witness (Nx.dtype y) dt with
  | Some Type.Equal -> rebrand y
  | None ->
      invalid_arg
        (Format.asprintf "Rune: a %a value where %a was expected" Nx.pp_dtype
           (Nx.dtype y) Nx.pp_dtype dt)

let packed (V x) = Nx.P (rebrand x)
let same (V a) (V b) = Obj.repr a == Obj.repr b

(* Names *)

(* A map's name is a number no other name has. *)
type axis = int

let axes = Atomic.make 0
let axis () = Atomic.fetch_and_add axes 1

type ('v, 's, 'd) total = { id : ('v, 's, 'd) Nx.t Type.Id.t }

let total () = { id = Type.Id.make () }
let same_total t u = Type.Id.provably_equal t.id u.id

(* Programs *)

type ('v, 's, 'd) Nx.Prim.payload += Var : int -> ('v, 's, 'd) Nx.Prim.payload
type addition = Addition : ('v, 's, 'd) total * ('v, 's) Nx.dtype -> addition

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
  | Call : { body : t; args : value array } -> construct

let make ~owner ~params eqns ~outputs ~adds ~slots =
  { owner; params; eqns = Array.of_list eqns; outputs; adds; slots }

type _ Effect.t += Construct : construct -> value array Effect.t
type _ Effect.t += Bound : axis -> int option Effect.t

let bound a =
  match Effect.perform (Bound a) with
  | n -> n
  | exception Effect.Unhandled (Bound _) -> None

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
  | Call c -> Call { c with body = f c.body }

(* The additions a construct's results end with. *)
let additions = function
  | Detach _ | Lanes _ | Lane_index _ | Add _ -> [||]
  | Scan { step; _ } -> step.adds
  | Iterate { step; until; _ } -> Array.append step.adds until.adds
  | Call { body; _ } -> body.adds

(* Forms *)

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

(* Outputs [i] to [j - 1]. *)
let slice a i j = Array.sub a i (j - i)

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
      let adds = Array.map form (slice step.outputs (no - na) no) in
      Array.concat
        [ carry; Array.map (stacked length) ys; Array.map computed adds ]
  | Iterate { step; until; init; _ } ->
      let tail p =
        let n = Array.length p.outputs in
        slice p.outputs (n - Array.length p.adds) n
      in
      Array.concat
        [
          Array.map form init;
          Array.map (fun v -> computed (form v)) (tail step);
          Array.map (fun v -> computed (form v)) (tail until);
        ]
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

let add (V a) b = V (Nx.add a (unpack (Nx.dtype a) b))

let stack = function
  | [] -> assert false
  | V y :: _ as vs ->
      let dt = Nx.dtype y in
      V (Nx.stack ~axis:0 (List.map (fun v -> rebrand (unpack dt v)) vs))

(* Evaluation *)

(* An unbound slot. *)
let hole = V (Nx.zeros Nx.bool [||] : (bool, Nx.bool_elt, Nx.host) Nx.t)

(* [x]'s binding [v], read at [x]'s type. *)
let get (type v s d) v (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
  unpack (Nx.dtype x) v

let rec eval p args =
  if Array.length args <> Array.length p.params then
    invalid_arg
      (Printf.sprintf "Rune: a program of %d parameters applied to %d values"
         (Array.length p.params) (Array.length args));
  let env = Array.make p.slots hole in
  Array.iteri (fun k slot -> env.(slot) <- args.(k)) p.params;
  let owner = p.owner in
  let lookup (type v s d) (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
    match Nx.Prim.payload owner x with Some (Var n) -> get env.(n) x | _ -> x
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

and default = function
  | Detach x -> [| V x |]
  | Lanes _ -> no_map "Rune.lanes"
  | Lane_index _ -> no_map "Rune.lane_index"
  | Add _ -> [||]
  | Scan { step; length; init; xs; captures } ->
      scan step ~length ~init ~xs ~captures
  | Iterate { name; step; until; max; init; step_captures; until_captures } ->
      iterate name step ~until ~max ~init ~step_captures ~until_captures
  | Call { body; args } -> eval body args

(* The sums of a construct's additions start at the identity, so that a loop
   that runs no step adds nothing a total can tell. *)
and accumulators p =
  let n = Array.length p.outputs in
  Array.map
    (fun v -> filled ~zero:false (computed (form v)))
    (slice p.outputs (n - Array.length p.adds) n)

(* [acc] plus the additions the outputs [out] end with. *)
and accumulate acc out =
  let n = Array.length out and na = Array.length acc in
  Array.iteri (fun k a -> acc.(k) <- add a out.(n - na + k)) acc

and scan step ~length ~init ~xs ~captures =
  let nc = Array.length init in
  let no = Array.length step.outputs and na = Array.length step.adds in
  let ny = no - nc - na in
  let acc = accumulators step in
  let carry = ref init and ys = Array.make ny [] in
  for i = 0 to length - 1 do
    let row = Array.map (fun (V x) -> V (Nx.get [ i ] x)) xs in
    let out = eval step (Array.concat [ !carry; row; captures ]) in
    carry := slice out 0 nc;
    for j = 0 to ny - 1 do
      ys.(j) <- out.(nc + j) :: ys.(j)
    done;
    accumulate acc out
  done;
  let ys =
    if length = 0 then
      Array.map
        (fun v -> filled ~zero:true (stacked 0 (form v)))
        (slice step.outputs nc (nc + ny))
    else Array.map (fun l -> stack (List.rev l)) ys
  in
  Array.concat [ !carry; ys; acc ]

and iterate name step ~until ~max ~init ~step_captures ~until_captures =
  let step_acc = accumulators step and until_acc = accumulators until in
  let rec go k carry =
    let out = eval until (Array.append carry until_captures) in
    accumulate until_acc out;
    let stop = unpack Nx.bool out.(0) in
    if Nx.item (List.init (Array.length (Nx.shape stop)) (fun _ -> 0)) stop then
      carry
    else if k = max then
      invalid_arg
        (Printf.sprintf "%s: until is still false after max = %d steps" name max)
    else begin
      let out = eval step (Array.append carry step_captures) in
      accumulate step_acc out;
      go (k + 1) (slice out 0 (Array.length carry))
    end
  in
  let carry = go 0 init in
  Array.concat [ carry; step_acc; until_acc ]

let run c =
  let r = perform c in
  let adds = additions c in
  let base = Array.length r - Array.length adds in
  Array.iteri
    (fun k (Addition (t, dt)) ->
      ignore (perform (Add (t, unpack dt r.(base + k)))))
    adds;
  r
