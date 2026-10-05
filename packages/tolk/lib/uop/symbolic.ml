(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Divandmod
module V = Dtype.Value

let num u = number (value u)

let pop_num ?op u =
  let x, c = pop_const ?op u in
  (x, number c)

let equals u v =
  match value u with #Dtype.value as x -> V.(x = v) | _ -> false

let pm = Pattern_matcher.v
let ops = Op.Set.of_list
let zero = V.of_int 0
let one = V.of_int 1
let lit (v : V.t) = const (v :> Dtype.const)
let const_v u (v : V.t) = const_like u (v :> Dtype.const)
let is_const u = op u = Op.Const
let sum_of = function u :: us -> usum u us | [] -> invalid_arg "empty sum"

let conj = function
  | u :: us -> uprod u us
  | [] -> invalid_arg "empty conjunction"

(* A NaN or an infinity has no value in an integer type: a rule that would
   convert one does not apply. *)
let convertible dt : V.t -> bool = function
  | `Float f -> Float.is_finite f || Dtype.is_float dt || Dtype.is_bool dt
  | _ -> true

(* [v] as a machine holds it in [dt]: converted, then wrapped to [dt]'s
   width. *)
let at dt v =
  if not (convertible dt v) then raise_notrace Not_a_number;
  Dtype.truncate dt (number (Dtype.const dt v))

let dedup l =
  Helpers.dedup
    (module struct
      type t = Ops.t

      let equal = ( == )
      let hash = hash
    end)
    l

(* Phase 1: the most generic folding rules *)

(* The reciprocal overflows only where a power of magnitude at least 1 does, and
   [sqrt] gives -0. and NaN at -0. and -inf, where a half-integer power is +0.
   and +inf. *)
let simplify_pow x c =
  let c = num c and pow x v = pow x (lit v) in
  let h = V.(c - `Float 0.5) in
  match c with
  | `Float f when not (Float.is_finite f) -> None
  | _ when V.(c <= `Float (-1.)) -> Some (pow (reciprocal x) V.(-c))
  | _ when V.(c < zero) -> None
  | _ when V.(c = zero) -> Some (const_v x one)
  | _ when V.(h < c && `Float (Float.trunc (to_float h) +. 0.5) = c) ->
      let p = mul (pow x h) (sqrt x) in
      if not (Dtype.is_float (dtype x)) then Some p
      else
        let special v r =
          where O.(x <> float v) r (const_like x (`Float (Float.abs v)))
        in
        Some (special Float.neg_infinity (special 0. p))
  | _ when V.(`Int (to_z c) = c) ->
      let y = pow x V.(c // of_int 2) in
      Some O.(y * y * if V.(c % of_int 2 = one) then x else int 1)
  | _ -> None

let fold_bitcast root c =
  let dt = dtype c in
  if Dtype.itemsize dt <> Dtype.itemsize (dtype root) then None
  else
    (* the value is read as [dt] stores it: an integer is mathematical and may
       not fit, so it wraps to the stated width, and a NaN keeps its bits, which
       a conversion would quiet *)
    let v =
      if Dtype.is_float dt then number (Dtype.const dt (num c))
      else Dtype.truncate dt (num c)
    in
    Some (const_v root (Dtype.bitcast dt (dtype root) v))

(* A committed integer holds its type's value, so a fold reads a committed
   operand, and a weak integer operand the operation commits, at the width of
   the operation's operands, and writes a committed integer result at its width;
   floats re-round in the mint. A stack folds lane by lane. A shift by a
   negative count has no value, and does not fold. *)
let fold_const_alu a =
  let alu args = exec_alu (op a) (dtype a) args in
  let operands =
    if Op.Set.mem (op a) Op.Set.comparison then promo_dtype (src a) else dtype a
  in
  let read s =
    match (op s, value s) with
    | Op.Cast, (#Dtype.value as v) -> (at (dtype s) v :> Dtype.const)
    | Op.Const, (`Int _ as v)
      when Dtype.equal (dtype s) Dtype.Weak_int && List.mem operands Dtype.ints
      ->
        (at operands v :> Dtype.const)
    | _, c -> c
  in
  let defined args =
    match (op a, args) with
    | (Op.Shl | Op.Shr), [ _; (#Dtype.value as n) ] -> V.(n >= zero)
    | _ -> true
  in
  let stack s = op s = Op.Stack in
  match List.filter stack (src a) with
  | [] ->
      let args = List.map read (src a) in
      if defined args then Some (const_like a (alu args)) else None
  | stacks ->
      let count =
        List.fold_left (fun n s -> max n (List.length (src s))) 0 stacks
      in
      let lane i s = read (if stack s then nth s i else s) in
      let lanes = List.init count (fun i -> List.map (lane i) (src a)) in
      if List.for_all defined lanes then
        Some (consts ~dtype:(dtype a) (List.map alu lanes))
      else None

(* the B with q == B//div and B%div == base%div, or None. only such congruence
   is needed to recombine, and canonicalization moves consts freely: the
   quotient may be merged ((x//c + a)//div -> (x + a*c)//(c*div) for div>0) and
   shifted ((y + k*D)//D == y//D + k) *)
let quotient_base q base div =
  let (q, s), (n, a) = (pop_num q, pop_num base) in
  if op q <> Op.Floordiv || not (is_const (nth q 1)) then None
  else
    let qd = num (nth q 1) in
    let merged =
      if V.(div > zero) && op n = Op.Floordiv && is_const (nth n 1) then
        let c = num (nth n 1) in
        if V.(qd = c * div) then Some (nth n 0, V.(a * c), V.(c * div))
        else None
      else None
    in
    let found =
      match merged with
      | Some _ -> merged
      | None -> if V.(qd = div) then Some (n, a, div) else None
    in
    Option.bind found (fun (n, a, d) ->
        let (x, xa), (p, pa) = (pop_num n, pop_num (nth q 0)) in
        let t = V.(xa + a - pa) in
        if p != x || V.(t % d <> zero) then None
        else
          let k = V.((t // d) - s) in
          Some (if V.(k = zero) then base else sub base (lit V.(k * div))))

(* a scaled mod (base%div)*mul recombines with a partner q*(div*mul) carrying
   the quotient of a b == base (mod div): fully into b*mul when q == b//div, and
   partially into the wider mod (b%(div*d))*mul when q == (b//div)%d, for d>0 *)
let fold_add_divmod_recombine x =
  let terms = List.mapi (fun i t -> (i, t)) (split_uop x Op.Add) in
  let rest i j =
    List.filter_map
      (fun (k, t) -> if k = i || k = j then None else Some t)
      terms
  in
  terms
  |> List.find_map (fun (i, u) ->
      let md, mul = pop_num ~op:Op.Mul u in
      if op md <> Op.Floormod || not (is_const (nth md 1)) then None
      else
        let base = nth md 0 and div = num (nth md 1) in
        terms
        |> List.find_map (fun (j, v) ->
            let q, scale = pop_num ~op:Op.Mul v in
            if i = j || V.(scale <> div * mul) then None
            else
              let recombine b = Some (usum O.(b * lit mul) (rest i j)) in
              match quotient_base q base div with
              | Some b -> recombine b
              | None when op q = Op.Floormod && is_const (nth q 1) ->
                  let d = num (nth q 1) in
                  if V.(d <= zero) then None
                  else
                    Option.bind
                      (quotient_base (nth q 0) base div)
                      (fun b -> recombine O.(b % lit V.(div * d)))
              | None -> None))

(* Invalid poisons the value: ops move inside the gate so the Invalid reaches
   the LOAD/STORE and folds there. this needs to be before symbolic so that
   0*something_that_might_be_invalid doesnt become 0 *)
let invalid_pat = Upat.op Op.Const ~arg:(Const `Invalid) ~name:"i"
let invalid_gate = Upat.(where (var "cond") (var "x") invalid_pat)

(* the two const spellings: Invalid carries no width, so it rides bare inside
   either *)
let bare_const = Upat.(any [ op Op.Const; op Op.Stack ~each:(op Op.Const) ])

let casted_const =
  let p = Upat.(op Op.Cast ~src:[ op Op.Const ]) in
  Upat.(
    any [ p; op Op.Stack ~each:(any [ p; op Op.Const ~arg:(Const `Invalid) ]) ])

(* a REDUCE moves inside the gate clauses without its ranges: they invalidate
   every lane at once, so that gate lifts out *)
let lift_reduce_gate red cond x i =
  match arg red with
  | Reduce { num_axes = 0; _ } -> (
      let ranges = List.tl (src red) in
      let in_reduce c =
        let crs = Ops.ranges c in
        List.exists
          (fun r ->
            List.exists
              (fun rr -> Nodes.mem rr crs)
              (Nodes.to_list (Ops.ranges r)))
          ranges
      in
      let keep, lift = List.partition in_reduce (split_uop cond Op.And) in
      let inner = match keep with [] -> x | _ -> where (conj keep) x i in
      match lift with
      | [] -> None
      | _ -> Some (where (conj lift) (replace red ~src:(inner :: ranges)) i))
  | _ -> None

let unary_or_cast = Op.Set.union Op.Set.unary (ops [ Op.Cast; Op.Bitcast ])

(* A stack of Invalid lanes stays a stack: one Invalid in its place would drop
   the width that the lanes' movements and devectorize read. *)
let pm_data_invalid =
  pm
    (fun () -> [
      rule (Upat.v ~op:unary_or_cast ~src:[ invalid_pat ] ()) (fun m ->
          Some (m "i"));
      rule (Upat.v ~op:unary_or_cast ~src:[ invalid_gate ] ~name:"op" ())
        (fun m ->
          Some (where (m "cond") (replace (m "op") ~src:[ m "x" ]) (m "i")));
      (* binary ops move inside the gate, with Invalid in the false branch *)
      rule
        (Upat.v ~op:Op.Set.binary
           ~src:[ invalid_gate; Upat.var "y" ]
           ~name:"alu" ())
        (fun m ->
          Some (where (m "cond") (alu (m "x") (op (m "alu")) [ m "y" ]) (m "i")));
      rule
        (Upat.v ~op:Op.Set.binary
           ~src:[ Upat.var "y"; invalid_gate ]
           ~name:"alu" ())
        (fun m ->
          Some (where (m "cond") (alu (m "y") (op (m "alu")) [ m "x" ]) (m "i")));
      rule
        (Upat.v
           ~op:(Op.Set.diff Op.Set.binary Op.Set.comparison)
           ~perm:[ invalid_pat; Upat.wild ] ())
        (fun m -> Some (m "i"));
      (* a multiply-add (D25) moves inside the gate of each operand in turn,
         and an Invalid operand makes it Invalid, as binary ops do *)
      rule (Upat.v ~op:(ops [ Op.Mulacc ]) ~name:"alu" ()) (fun m ->
          let a = m "alu" in
          let gated s = op s = Op.Where && is_invalid (nth s 2) in
          match List.find_opt (fun s -> is_invalid s || gated s) (src a) with
          | None -> None
          | Some s when is_invalid s -> Some s
          | Some g ->
              let inner s = if s == g then nth g 1 else s in
              Some
                (where (nth g 0)
                   (replace a ~src:(List.map inner (src a)))
                   (nth g 2)));
      rule (Upat.reduce ~name:"red" ~allow_any_len:true invalid_gate [])
        (fun m -> lift_reduce_gate (m "red") (m "cond") (m "x") (m "i"));
      (* an Invalid condition poisons the whole where; a gated Invalid condition
         lifts the gate out *)
      rule Upat.(where invalid_pat wild wild) (fun m -> Some (m "i"));
      rule
        Upat.(where invalid_gate (var "a") (var "b"))
        (fun m ->
          Some (where (m "cond") (where (m "x") (m "a") (m "b")) (m "i")));
      (* normalize where(cond, Invalid, val) -> where(~cond, val, Invalid) *)
      rule
        Upat.(where (var "cond") invalid_pat (var "val"))
        (fun m ->
          let v = m "val" and i = m "i" in
          Some (if is_invalid v then i else where (logical_not (m "cond")) v i));
      (* lift Invalid out: a.where(cond.where(x, Invalid), c) ->
         (~a|cond).where(a.where(x, c), Invalid) *)
      rule
        Upat.(where (var "a") invalid_gate (var "c"))
        (fun m ->
          let a = m "a" and c = m "c" in
          if is_invalid c then None
          else
            Some
              (where O.(logical_not a lor m "cond") (where a (m "x") c) (m "i")));
      rule
        Upat.(where (var "a") (var "b") invalid_gate)
        (fun m ->
          let a = m "a" and b = m "b" in
          if is_invalid b then None
          else Some (where O.(a lor m "cond") (where a b (m "x")) (m "i")));
      (* fold gated LOAD/STORE *)
      rule
        (Upat.op Op.Store
           ~src:
             [
               Upat.or_casted
                 (Upat.index ~allow_any_len:true Upat.wild [ invalid_pat ]);
               Upat.wild;
             ])
        (fun _ -> Some (v Op.Noop));
      rule
        (Upat.op Op.Load ~allow_any_len:true ~name:"x"
           ~src:
             [
               Upat.or_casted
                 (Upat.index ~allow_any_len:true Upat.wild [ invalid_pat ]);
             ])
        (fun m ->
          let x = m "x" in
          Some (match src x with _ :: alt :: _ -> alt | _ -> const_v x zero));
    ])

let pm_remove_invalid =
  pm
    (fun () -> [
      rule (Upat.named "w" invalid_gate) (fun m ->
          let w = m "w" in
          Some (replace w ~src:[ m "cond"; m "x"; const_v w zero ]));
      rule (Upat.op Op.Stack ~name:"s") (fun m ->
          let s = m "s" in
          if not (List.exists is_invalid (src s)) then None
          else
            let zero_invalid x =
              if is_invalid x then const ~dtype:(dtype s) (`Int Bigint.zero) else x
            in
            Some (replace s ~src:(List.map zero_invalid (src s))));
    ])

(* [broadcast_const u] is the constant [u] is through movements that keep each
   element's value, as a constant shaped like another node is ({!const_like}). *)
let rec broadcast_const u =
  match op u with
  | Op.Const -> Some u
  | Op.Reshape | Op.Expand | Op.Permute | Op.Shrink | Op.Flip ->
      broadcast_const (nth u 0)
  | _ -> None

(* folding a strong dtype WHERE to a weak const branch keeps the strong dtype *)
let fold_const_where gate c0 c1 w =
  let ret = if V.to_bool (num gate) then c0 else c1 in
  let weak u = List.mem (dtype u) Dtype.weaks in
  if is_const ret && weak ret && not (weak w) then ccast ret (dtype w) else ret

let boolean = [ Dtype.Bool ]
let int_like = Dtype.Weak_int :: Dtype.ints
let int_or_bool = Dtype.Bool :: int_like

let symbolic_simple =
  Pattern_matcher.concat
    [
      pm_data_invalid;
      pm
        (fun () -> [
          (* Self folding *)
          (* a float x + 0 is x only for -0., since -0. + +0. is +0. *)
          rule
            (Upat.v
               ~op:(ops [ Op.Add; Op.Xor; Op.Or ])
               ~perm:Upat.[ var "x"; named "c" (int 0) ]
               ~name:"a" ())
            (fun m ->
              let a = m "a" in
              let negative_zero =
                match value (m "c") with
                | `Float z -> Float.sign_bit z
                | _ -> false
              in
              if op a = Op.Add && Dtype.is_float (dtype a) && not negative_zero
              then None
              else Some (m "x"));
          rule
            (Upat.v
               ~op:(ops [ Op.Shl; Op.Shr ])
               ~src:Upat.[ var "x"; int 0 ]
               ())
            (fun m -> Some (m "x"));
          rule Upat.(var "x" * int 1) (fun m -> Some (m "x"));
          rule Upat.(var "x" // var "x") (fun m -> Some (const_v (m "x") one));
          rule Upat.(var "x" // int 1) (fun m -> Some (m "x"));
          rule Upat.(var "x" // int (-1)) (fun m -> Some (neg (m "x")));
          rule Upat.(var "x" lxor var "y" lxor var "y") (fun m -> Some (m "x"));
          (* (x%y)%y = -> x%y (rewritten with base for speed) *)
          rule
            Upat.(named "base" (wild % var "y") % var "y")
            (fun m -> Some (m "base"));
          (* variations of (x%c)+(x//c)*c = x *)
          rule (Upat.op Op.Add ~dtype:[ Dtype.Weak_int ] ~name:"x") (fun m ->
              fold_add_divmod_recombine (m "x"));
          rule
            Upat.(var ~dtype:boolean "x" land cvar "c")
            (fun m -> Some (if V.to_bool (num (m "c")) then m "x" else m "c"));
          rule
            Upat.(var ~dtype:boolean "x" lor cvar "c")
            (fun m -> Some (if V.to_bool (num (m "c")) then m "c" else m "x"));
          rule
            Upat.(var ~dtype:boolean "x" <> const ~dtype:boolean (`Bool false))
            (fun m -> Some (m "x"));
          rule
            (Upat.v ~op:Op.Set.idempotent ~src:Upat.[ var "x"; var "x" ] ())
            (fun m -> Some (m "x"));
          rule
            Upat.(logical_not (logical_not (var ~dtype:boolean "x")))
            (fun m -> Some (m "x"));
          rule
            Upat.(
              where (var ~dtype:boolean "x")
                (const ~dtype:boolean (`Bool true))
                (const ~dtype:boolean (`Bool false)))
            (fun m -> Some (m "x"));
          rule
            Upat.(
              where (var ~dtype:boolean "x")
                (const ~dtype:boolean (`Bool false))
                (const ~dtype:boolean (`Bool true)))
            (fun m -> Some (logical_not (m "x")));
          (* CAST(bool -> int) != const — CAST(True)=1, CAST(False)=0, so fold
             based on const value *)
          rule
            Upat.(
              f ~dtype:int_like (var ~dtype:boolean "x") Op.Cast <> cvar "c")
            (fun m ->
              let x = m "x" and c = m "c" in
              Some
                (if equals c zero then x
                 else if equals c one then logical_not x
                 else const_like x (`Bool true)));
          rule Upat.(trunc (var ~dtype:int_or_bool "x")) (fun m -> Some (m "x"));
          (* Zero folding *)
          rule
            Upat.(var "x" < var "x")
            (fun m -> Some (const_like ~dtype:Dtype.Bool (m "x") (`Bool false)));
          rule Upat.(var "x" % var "x") (fun m -> Some (const_v (m "x") zero));
          rule
            Upat.(var "x" lxor var "x")
            (fun m -> Some (const_v (m "x") zero));
          rule Upat.(var "x" land int 0) (fun m -> Some (const_v (m "x") zero));
          (* (x&mask)>>k -> x>>k when mask only clears bits below k *)
          rule
            Upat.((var "x" land cvar "mask") lsr cvar "k")
            (fun m ->
              let mask = V.to_z (num (m "mask"))
              and k = V.to_int (num (m "k")) in
              if
                (k >= 0)
                [@mutate off "a shift by 0 is the x >> 0 rule's, tried first"]
                && Bigint.(equal (logor mask (pred (shift_left one k))) minus_one)
              then Some (shr (m "x") (lit (`Int (Bigint.of_int k))))
              else None);
          rule
            Upat.(var "x" land cvar "mask" // cvar "c")
            (fun m ->
              let mask = V.to_z (num (m "mask")) and c = V.to_z (num (m "c")) in
              if
                Bigint.(
                  gt c zero
                  && equal (logand c (pred c)) zero
                  && equal (logor mask (pred c)) minus_one)
              then Some O.(m "x" // lit (`Int c))
              else None);
          (* x != x -> False (only ints) *)
          rule
            Upat.(var ~dtype:int_or_bool "x" <> var "x")
            (fun m -> Some (const_like ~dtype:Dtype.Bool (m "x") (`Bool false)));
          (* Constant folding *)
          (* canonicalize casted CONST *)
          rule
            (Upat.op Op.Cast ~dtype:Dtype.all ~name:"root"
               ~src:[ Upat.cvar "c" ])
            (fun m ->
              let root = m "root" in
              match value (m "c") with
              | #Dtype.value as v when not (convertible (dtype root) v) -> None
              | c -> Some (const_like root c));
          (* collapse committed const conversions, the inner one read at its
             width *)
          rule
            (Upat.op Op.Cast ~dtype:Dtype.all ~name:"root"
               ~src:
                 [
                   Upat.op Op.Cast ~dtype:Dtype.all ~name:"inner"
                     ~src:[ Upat.op Op.Const ];
                 ])
            (fun m ->
              let root = m "root" in
              let v = at (dtype (m "inner")) (num (m "inner")) in
              if convertible (dtype root) v then Some (const_v root v) else None);
          (* one rule per spelling: bare has no width, a pair evaluates at its
             stated width, mixed commits to the promotion. THREEFRY(const,const)
             folds via its decomposition *)
          rule
            (Upat.v
               ~op:(Op.Set.diff Op.Set.alu (ops [ Op.Threefry ]))
               ~each:bare_const ~name:"a" ())
            (fun m -> fold_const_alu (m "a"));
          rule
            (Upat.v
               ~op:(Op.Set.diff Op.Set.alu (ops [ Op.Threefry ]))
               ~each:casted_const ~name:"a" ())
            (fun m -> fold_const_alu (m "a"));
          rule
            (Upat.v
               ~op:(Op.Set.diff Op.Set.binary (ops [ Op.Threefry ]))
               ~perm:[ casted_const; bare_const ]
               ~name:"a" ())
            (fun m ->
              let a = m "a" in
              let dt = promo_dtype (src a) in
              let commit s =
                if List.mem (dtype s) Dtype.weaks then ccast s dt else s
              in
              if List.mem dt Dtype.weaks then None
              else Some (replace a ~src:(List.map commit (src a))));
          (* bool MUL is AND, ADD/MAX is OR. prevents other rules to rewrite
             bool ADD/MUL incorrectly *)
          rule
            Upat.(var ~dtype:boolean "x" * var ~dtype:boolean "y")
            (fun m -> Some O.(m "x" land m "y"));
          rule
            Upat.(var ~dtype:boolean "x" + var ~dtype:boolean "y")
            (fun m -> Some O.(m "x" lor m "y"));
          rule
            Upat.(maximum (var ~dtype:boolean "x") (var ~dtype:boolean "y"))
            (fun m -> Some O.(m "x" lor m "y"));
          (* Div rules *)
          rule
            Upat.(cvar ~arg:(`Int Bigint.zero) "x" / int 0)
            (fun m -> Some (const_like (m "x") (`Float Dtype.nan)));
          (* x*0 -> 0 or 0*x -> 0, for integers: a float product by zero is NaN
             at an infinity or a NaN, and -0. at a negative x *)
          rule
            Upat.(var ~dtype:int_or_bool "x" * int 0)
            (fun m -> Some (const_v (m "x") zero));
          (* Cast/bitcast *)
          rule
            (Upat.v ~op:(ops [ Op.Cast; Op.Bitcast ]) ~name:"root" ())
            (fun m ->
              let root = m "root" in
              if Dtype.equal (dtype root) (dtype (nth root 0)) then
                Some (nth root 0)
              else None);
          (* a BITCAST reads its operand at the width it states, so a weak const
             is nonsense here: the bare arm is bool only *)
          rule
            (Upat.op Op.Bitcast ~name:"root"
               ~src:
                 [
                   Upat.(
                     any
                       [
                         op Op.Const ~dtype:boolean ~name:"c";
                         op Op.Cast ~src:[ op Op.Const ] ~name:"c";
                       ]);
                 ])
            (fun m -> fold_bitcast (m "root") (m "c"));
          (* b.cast(a).cast(b) -> b if a preserves all values in b *)
          rule
            Upat.(f ~name:"b" (f ~name:"a" (var "x") Op.Cast) Op.Cast)
            (fun m ->
              let x = m "x" and b = dtype (m "b") in
              if
                Dtype.equal (dtype x) b
                && Dtype.can_lossless_cast b (dtype (m "a"))
              then Some x
              else None);
          (* bitcast twice *)
          rule
            (Upat.op Op.Bitcast ~name:"b" ~src:[ Upat.bitcast (Upat.var "x") ])
            (fun m -> Some (bitcast (m "x") (dtype (m "b"))));
          rule
            Upat.(cast (var "x") Dtype.Bool)
            (fun m -> Some O.(m "x" <> int 0));
          (* Pow *)
          rule
            Upat.(alu (var "x") Op.Pow [ cvar "c" ])
            (fun m -> simplify_pow (m "x") (m "c"));
          (* positive const ** x *)
          rule
            Upat.(alu (cvar "c") Op.Pow [ var "x" ])
            (fun m ->
              let c = m "c" in
              let cv = num c in
              if V.(cv = one) then Some c
              else if V.(cv > zero && cv < `Float Float.infinity) then
                Some (exp2 O.(m "x" * float (Float.log2 (V.to_float cv))))
              else None);
          (* unpack a uint64 packed from two uint32 (threefry) *)
          rule
            Upat.(
              cast
                ((v ~dtype:[ Dtype.Uint64 ] () lsl int 32)
                lor cast (var ~dtype:[ Dtype.Uint32 ] "y") Dtype.Uint64)
                Dtype.Uint32)
            (fun m -> Some (m "y"));
          rule
            Upat.(
              ((cast (var ~dtype:[ Dtype.Uint32 ] "x") Dtype.Uint64 lsl int 32)
              lor cast (v ~dtype:[ Dtype.Uint32 ] ()) Dtype.Uint64)
              lsr int 32)
            (fun m -> Some (cast (m "x") Dtype.Uint64));
          (* Simple where folding *)
          (* a conditional with the same results either way is a noop, also fold
             const conditionals, broadcast ones included *)
          rule
            Upat.(where wild (var "val") (var "val"))
            (fun m -> Some (m "val"));
          rule
            Upat.(named "w" (where (var "gate") (var "c0") (var "c1")))
            (fun m ->
              Option.map
                (fun gate -> fold_const_where gate (m "c0") (m "c1") (m "w"))
                (broadcast_const (m "gate")));
        ]);
      Movement.mop_cleanup;
    ]

(* Phase 2: rules that match deeper *)

let lt_folding x c =
  let p, np =
    List.partition
      (fun u -> Bigint.equal (const_factor u) Bigint.one)
      (split_uop x Op.Add)
  in
  let d = List.fold_left (fun d u -> Bigint.gcd d (const_factor u)) c np in
  let sum f = List.fold_left (fun s u -> V.(s + f u)) zero p in
  match np with
  | n :: ns when Bigint.gt d Bigint.one && V.(zero <= sum vmin && sum vmax < `Int d) ->
      Some O.(Option.get (divides (usum n ns) d) < lit (`Int (Bigint.fdiv c d)))
  | _ -> None

(* (X := a0*x0 + a1*x1 + ...) > 0 is equivalent to x0 + x1 + ... > 0 if xi >= 0
   and ai > 0 for ints. returns x0 + x1 + ... in such case, or None if not *)
let canonicalize_simplex x =
  (* assumed the const is the last src of MUL *)
  let strip u =
    if op u = Op.Mul && is_const (nth u 1) && V.(num (nth u 1) > zero) then
      (true, nth u 0)
    else (false, u)
  in
  let terms = List.map strip (split_uop x Op.Add) in
  let atom (_, u) =
    Op.Set.mem (op u) Op.Set.irreducible && V.(vmin u >= zero)
  in
  if List.for_all atom terms && List.exists fst terms then
    Some (sum_of (List.map snd terms))
  else None

let commutative =
  pm
    (fun () -> [
      (* COMMUTATIVE flipping (only for index) *)
      (* NOTE: this can break merging vector math by only flipping some of them *)
      rule
        (Upat.v ~op:Op.Set.commutative ~dtype:[ Dtype.Weak_int ] ~name:"x" ())
        (fun m ->
          let x = m "x" in
          if compare_structure (nth x 1) (nth x 0) < 0 then
            Some (replace x ~src:(List.rev (src x)))
          else None);
    ])

(* in cond.where(t, f), cond is True within t and False within f *)
let fold_where_closure cond t f =
  if not (Dtype.equal (dtype cond) Dtype.Bool) then None
  else if
    (* a constant condition, broadcast or not, assumes nothing: the same node
       is every other use of that constant *)
    is_const (base cond)
  then None
  else if
    (* INDEX gates are owned by the valid/store-coalescing machinery, leave them
       alone. Asked before the search below, which walks the branches. *)
    List.exists
      (fun u -> op_in_backward_slice_with_self u [ Op.Index ])
      [ cond; t; f ]
  then None
  else if not (reaches t cond || reaches f cond) then None
  else
    let assume b u = substitute u [ (cond, const_like cond (`Bool b)) ] in
    Some (where cond (assume true t) (assume false f))

let both_const u0 u1 = is_const u0 && is_const u1

let symbolic =
  Pattern_matcher.concat
    [
      symbolic_simple;
      commutative;
      pm
        (fun () -> [
           (* Boolean algebra *)
           rule
             Upat.(
               var ~dtype:boolean "x" lor logical_not (var ~dtype:boolean "x"))
             (fun m -> Some (const_like (m "x") (`Bool true)));
           (* Combine terms *)
           (* like terms combine for integers: in floats each product and
              sum rounds *)
           rule
             Upat.(
               (var ~dtype:int_or_bool "x" * cvar "c0") + (var "x" * cvar "c1"))
             (fun m -> Some O.(m "x" * (m "c0" + m "c1")));
           rule
             Upat.(
               var "y"
               + (var ~dtype:int_or_bool "x" * cvar "c0")
               + (var "x" * cvar "c1"))
             (fun m -> Some O.(m "y" + (m "x" * (m "c0" + m "c1"))));
           rule
             Upat.(var ~dtype:int_or_bool "x" + (var "x" * cvar "c"))
             (fun m -> Some O.(m "x" * (m "c" + int 1)));
           rule
             Upat.(var "y" + var ~dtype:int_or_bool "x" + (var "x" * cvar "c"))
             (fun m -> Some O.(m "y" + (m "x" * (m "c" + int 1))));
           rule
             Upat.(var "y" + (var ~dtype:int_or_bool "x" * cvar "c") + var "x")
             (fun m -> Some O.(m "y" + (m "x" * (m "c" + int 1))));
           rule Upat.(var "x" + var "x") (fun m -> Some O.(m "x" * int 2));
           rule
             Upat.(var "y" + var ~dtype:int_or_bool "x" + var "x")
             (fun m -> Some O.(m "y" + (m "x" * int 2)));
           (* -(x+c) -> -x + -c, for integers: -(x + c) is -0. at x = -c *)
           rule
             Upat.(int (-1) * (var ~dtype:int_or_bool "x" + cvar "c"))
             (fun m -> Some O.(-m "x" + -m "c"));
           rule
             Upat.(cvar "y" * (var ~dtype:[ Dtype.Weak_int ] "x" + cvar "c"))
             (fun m ->
               let y = m "y" in
               Some O.((y * m "x") + (y * m "c")));
           (* Where folding *)
           rule
             Upat.(
               where
                 (logical_not (var ~dtype:boolean "cond"))
                 (var "t") (var "f"))
             (fun m ->
               let f = m "f" in
               if is_invalid f then None else Some (where (m "cond") f (m "t")));
           (* in cond.where(t, f), uses of cond fold to True within t and False
              within f *)
           rule
             Upat.(where (var ~dtype:boolean "cond") (var "t") (var "f"))
             (fun m -> fold_where_closure (m "cond") (m "t") (m "f"));
           rule
             Upat.(where (var "gate") (var "x") (int 0) <> int 0)
             (fun m -> Some O.(m "gate" land (m "x" <> int 0)));
           (* a.where(b.where(c, d), d) -> (a & b).where(c, d) *)
           rule
             Upat.(
               where (var "a") (where (var "b") (var "c") (var "d")) (var "d"))
             (fun m -> Some (where O.(m "a" land m "b") (m "c") (m "d")));
           (* a.where(c, b.where(c, d)) -> (a | b).where(c, d) *)
           rule
             Upat.(
               where (var "a") (var "c") (where (var "b") (var "c") (var "d")))
             (fun m -> Some (where O.(m "a" lor m "b") (m "c") (m "d")));
           (* alu of two where with same conds can combine, only do if true
              branch or false branch is const *)
           rule
             (Upat.v ~op:Op.Set.binary ~name:"alu"
                ~src:
                  Upat.
                    [
                      where (var "c") (var "t") (var "f");
                      where (var "c") (var "tt") (var "ff");
                    ]
                ())
             (fun m ->
               let o = op (m "alu")
               and t = m "t"
               and tt = m "tt"
               and f = m "f"
               and ff = m "ff" in
               if both_const t tt || both_const f ff then
                 Some (where (m "c") (alu t o [ tt ]) (alu f o [ ff ]))
               else None);
           (* if its a plus we add the associative variation too, for integers:
              it reassociates the sum *)
           rule
             Upat.(
               var ~dtype:int_or_bool "y"
               + where (var "c") (var "t") (var "f")
               + where (var "c") (var "tt") (var "ff"))
             (fun m ->
               let t = m "t" and tt = m "tt" and f = m "f" and ff = m "ff" in
               if both_const t tt || both_const f ff then
                 Some O.(m "y" + where (m "c") (t + tt) (f + ff))
               else None);
           (* complementary zero branches under the same condition select
              directly, for integers: a float t + 0 is +0. at t = -0. *)
           rule
             Upat.(
               where (var "c") (var ~dtype:int_or_bool "t") (int 0)
               + where (var "c") (int 0) (var "f"))
             (fun m -> Some (where (m "c") (m "t") (m "f")));
           (* ALU/variable min==max -> CONST *)
           rule
             (Upat.v
                ~op:
                  (ops
                     [
                       Op.Cmplt;
                       Op.Cmpne;
                       Op.Floordiv;
                       Op.Floormod;
                       Op.Param;
                       Op.After;
                       Op.Special;
                     ])
                ~name:"x" ())
             (fun m ->
               let x = m "x" in
               if V.(vmin x = vmax x) then Some (const_v x (vmin x)) else None);
           rule
             (Upat.op Op.Range
                ~src:[ Upat.or_casted (Upat.op Op.Const) ]
                ~name:"x")
             (fun m ->
               let x = m "x" in
               if V.(vmin x = vmax x) then Some (const_v x (vmin x)) else None);
           (* max folding, for integers: a float selection keeps IEEE's NaN and
              signed zeros where a maximum does not *)
           rule
             Upat.(
               where
                 (cvar "a" < var ~dtype:int_or_bool "b")
                 (var "b") (cvar "c"))
             (fun m ->
               if V.(num (m "a") = num (m "c")) then
                 Some (maximum (m "a") (m "b"))
               else None);
           rule
             Upat.(
               where
                 (var ~dtype:int_or_bool "a" < cvar "b")
                 (cvar "c") (var "a"))
             (fun m ->
               if V.(num (m "b") = num (m "c")) then
                 Some (maximum (m "a") (m "b"))
               else None);
           (* a float maximum's bounds leave out NaN and the order of zeros *)
           rule
             Upat.(named "m" (maximum (var ~dtype:int_or_bool "x") (var "y")))
             (fun m ->
               let mx = m "m" and x = m "x" and y = m "y" in
               let (x0, x1), (y0, y1) =
                 (operand_bounds mx x, operand_bounds mx y)
               in
               (* the operand kept is committed to the maximum's type *)
               let keep u =
                 if List.mem (dtype u) Dtype.weaks then ccast u (dtype mx)
                 else u
               in
               if V.(x0 >= y1) then Some (keep x)
               else if V.(x1 <= y0) then Some (keep y)
               else None);
         ]
        (* Two stage ALU folding; sums, products and maxima for integers: in
           floats each step rounds, and a maximum keeps a NaN only as its first
           operand *)
        @ List.map
            (fun o ->
              let dtype =
                if List.mem o Op.[ Add; Mul; Max ] then Some int_or_bool
                else None
              in
              let x = Upat.var ?dtype "x" in
              rule
                Upat.(named "f" (alu (alu x o [ cvar "c1" ]) o [ cvar "c2" ]))
                (fun m ->
                  let f = m "f" in
                  let o = op f in
                  (* a sum, a product and a bitwise operation fold the same
                     before or after wrapping; a maximum orders wrapped values:
                     on uint8, max(1, -3) is 253, so its weak constants fold at
                     its width *)
                  let at_width c =
                    if o = Op.Max && List.mem (Ops.dtype c) Dtype.weaks then
                      ccast c (Ops.dtype f)
                    else c
                  in
                  let c = alu (at_width (m "c1")) o [ at_width (m "c2") ] in
                  Some (alu (m "x") o [ c ])))
            (Op.Set.to_list Op.Set.associative)
        @ [
            (* (x//c1)//c2 -> x//(c1*c2) for c2>0, where c1*c2 does not wrap *)
            rule
              Upat.(var "x" // cvar "c1" // cvar "c2")
              (fun m ->
                let c1 = vmin (m "c1") and c2 = vmin (m "c2") in
                if V.(c2 > zero) && exact (dtype (m "x")) V.[ c1; c2; c1 * c2 ]
                then Some O.(m "x" // (m "c1" * m "c2"))
                else None);
            (* Lt *)
            (* c0+x<c1 -> x < c1-c0, where neither side wraps *)
            rule
              Upat.(cvar "c0" + var ~dtype:int_like "x" < cvar "c1")
              (fun m ->
                let x = m "x" and c0 = vmin (m "c0") and c1 = vmin (m "c1") in
                if
                  exact (dtype x)
                    V.[ c0; c1; vmin x + c0; vmax x + c0; c1 - c0 ]
                then Some O.(x < m "c1" - m "c0")
                else None);
            (* c0*x<c1 -> sign(c0)*x < ceil(c1/abs(c0)) *)
            rule
              Upat.(cvar "c0" * var ~dtype:[ Dtype.Weak_int ] "x" < cvar "c1")
              (fun m ->
                let c0 = num (m "c0") and c1 = num (m "c1") and x = m "x" in
                let a = if V.(c0 < zero) then V.(-c0) else c0 in
                if V.(a > one) then
                  Some
                    O.((if V.(c0 > zero) then x else -x) < lit V.(-(-c1 // a)))
                else None);
            (* x//d<c -> x<c*d for d>0, and -> c*d<x for d<0 *)
            rule
              Upat.(var ~dtype:[ Dtype.Weak_int ] "x" // cvar "d" < cvar "c")
              (fun m ->
                let d = num (m "d")
                and cd = lit V.(num (m "c") * num (m "d"))
                and x = m "x" in
                if V.(d > zero) then Some O.(x < cd)
                else if V.(d < zero) then Some O.(x > cd)
                else None);
            (* Move add/mul consts to end (NOTE: this is still happening before
               constant folding), for integers: it reassociates *)
            rule
              Upat.(var ~dtype:int_or_bool "x" + cvar "c1" + var "y")
              (fun m ->
                let y = m "y" in
                if is_const y then None else Some O.(m "x" + y + m "c1"));
            rule
              Upat.(var ~dtype:int_or_bool "x" * cvar "c1" * var "y")
              (fun m ->
                let y = m "y" in
                if is_const y then None else Some O.(m "x" * y * m "c1"));
            (* Rules from symbolic *)
            (* generic lt folding *)
            rule
              Upat.(var ~dtype:[ Dtype.Weak_int ] "x" < cvar "c")
              (fun m ->
                match num (m "c") with
                | `Int c when Bigint.sign c > 0 -> lt_folding (m "x") c
                | _ -> None);
            rule
              Upat.(
                var ~dtype:[ Dtype.Weak_int ] "x" * int (-1)
                < var "y" * int (-1))
              (fun m -> Some O.(m "y" < m "x"));
            (* canonicalize a simplex with positive coefficients > 0. NOTE: not
               x < 1 means x > 0 *)
            rule
              Upat.(ne (var ~dtype:[ Dtype.Weak_int ] "x" < int 1) (bool true))
              (fun m ->
                Option.map
                  (fun x -> O.(x < int 1 <> bool true))
                  (canonicalize_simplex (m "x")));
            (* a range mod its own upper bound is just the range *)
            rule
              Upat.(op Op.Range ~each:(var "end") ~name:"r" % var "end")
              (fun m -> Some (m "r"));
            rule
              Upat.(op Op.Range ~each:(var "end") ~name:"r" // var "end")
              (fun m -> Some (const_v (m "r") zero));
            (* cast/long folding *)
            (* if the intermediate cast doesnt narrow we can do it in one cast *)
            rule
              Upat.(f ~name:"b" (f ~name:"a" (var "x") Op.Cast) Op.Cast)
              (fun m ->
                let x = m "x" in
                if Dtype.can_lossless_cast (dtype x) (dtype (m "a")) then
                  Some (cast x (dtype (m "b")))
                else None);
            rule
              Upat.(
                f ~name:"b"
                  (f ~dtype:int_like ~name:"a" (var ~dtype:int_like "x") Op.Cast)
                  Op.Cast)
              (fun m ->
                let x = m "x" in
                if overflows x (dtype (m "a")) then None
                else Some (ccast x (dtype (m "b"))));
            (* try to do math in int instead of long, keep weak const weak *)
            rule
              (Upat.v ~op:Op.Set.binary ~name:"u"
                 ~src:
                   Upat.
                     [
                       var ~dtype:[ Dtype.Int64; Dtype.Weak_int ] "x";
                       var ~dtype:[ Dtype.Int64; Dtype.Weak_int ] "y";
                     ]
                 ())
              (fun m ->
                let u = m "u" and x = m "x" and y = m "y" in
                let narrow w =
                  if is_const w then lit (num w) else cast w Dtype.Int32
                in
                if
                  List.exists
                    (fun w -> Dtype.equal (dtype w) Dtype.Int64)
                    [ x; y ]
                  && not
                       (List.exists
                          (fun w -> overflows w Dtype.Int32)
                          [ u; x; y ])
                then Some (cast (alu (narrow x) (op u) [ narrow y ]) (dtype u))
                else None);
            rule
              Upat.(
                f ~dtype:Dtype.sints ~name:"cast"
                  (var ~dtype:[ Dtype.Weak_int ] "x" + cvar "c")
                  Op.Cast)
              (fun m ->
                let c = m "cast" in
                Some O.(cast (m "x") (dtype c) + const_like c (value (m "c"))));
            (* an AFTER waits only on the effect ops listed here, any other dep
               is replaced by its srcs *)
            rule (Upat.op Op.After ~name:"x") (fun m ->
                let x = m "x" in
                let effects =
                  ops
                    [
                      Op.Range;
                      Op.Store;
                      Op.Call;
                      Op.Barrier;
                      Op.End;
                      Op.Backedge;
                      Op.Linear;
                      Op.Stage;
                    ]
                in
                let deps y =
                  if Op.Set.mem (op y) effects then [ y ] else src y
                in
                Some
                  (replace x
                     ~src:
                       (nth x 0
                       :: dedup (List.concat_map deps (List.tl (src x))))));
            (* after/end with 1 src is just src[0] *)
            rule
              (Upat.v ~op:(ops [ Op.After; Op.End ]) ~src:[ Upat.var "s" ] ())
              (fun m -> Some (m "s"));
            (* ranges can be subbed for CONSTs, remove them from ENDs. BACKEDGE
               conditions are never range selectors. *)
            rule (Upat.op Op.End ~name:"x") (fun m ->
                let x = m "x" in
                Some
                  (replace x
                     ~src:
                       (nth x 0
                       :: List.filter
                            (fun r -> not (is_const r))
                            (List.tl (src x)))));
          ]);
      Divandmod.div_and_mod_symbolic;
      (* the rules above key on bare CONSTs, so a redundantly committed const
         has to be uncast in the same fixpoint *)
      Uop_weak.pm_uncast_const;
    ]

(* Valids *)

(* if it's X <= c, returns X, true, c; if it's X >= c, returns X, false, c *)
let parse_valid v =
  let int_lt u = op u = Op.Cmplt && Dtype.is_int (dtype (nth u 0)) in
  if
    op v = Op.Cmpne
    && is_const (nth v 1)
    && equals (nth v 1) one
    && int_lt (nth v 0)
  then
    (* (X < c).ne(True) -> X >= c *)
    let s0 = nth v 0 in
    Some (nth s0 0, false, V.to_z (vmin (nth s0 1)))
    (* c < X -> X >= c+1 (a const on the left is a lower bound on the right),
       and X < c -> X <= c-1 *)
  else if int_lt v && is_const (nth v 0) then
    match value (nth v 0) with
    | #Dtype.value as c -> Some (nth v 1, false, Bigint.succ (V.to_z c))
    | `Invalid -> None
  else if int_lt v then Some (nth v 0, true, Bigint.pred (V.to_z (vmax (nth v 1))))
  else None

let uop_given_valid ?(try_simplex = true) valid u =
  (* first, parse valid into [expr, (lower bound, upper bound)] *)
  let bound bounds stmt =
    match parse_valid stmt with
    | None -> bounds
    | Some (e, upper, c) ->
        let lo, hi =
          Option.value (List.assq_opt e bounds) ~default:(None, None)
        in
        let b = if upper then (lo, Some c) else (Some c, hi) in
        if List.mem_assq e bounds then
          List.map (fun (k, v) -> (k, if k == e then b else v)) bounds
        else bounds @ [ (e, b) ]
  in
  let bounds = List.fold_left bound [] (split_uop valid Op.And) in
  let fake i e lo hi =
    variable ~dtype:(dtype e) ("fake" ^ string_of_int i) lo hi
  in
  let or_bound f e = function Some c -> `Int c | None -> f e in
  let exprs =
    List.mapi
      (fun i (e, (lo, hi)) -> (i, e, or_bound vmin e lo, or_bound vmax e hi))
      bounds
  in
  (* simplify uop given that valid is True *)
  let simplex u (i, e, lo, _) =
    let terms = split_uop e Op.Add in
    let irreducible t = Op.Set.mem (op t) Op.Set.irreducible in
    if
      not
        (try_simplex
        && op e = Op.Add
        && V.(lo = one)
        && List.for_all irreducible terms)
    then u
    else
      (* For X0 + X1 + ... > 0, check whether every Xi > 0 gives the same
         simplified output. *)
      let candidate = List.map (fun t -> (t, fake i t one (vmax t))) terms in
      let slice = backward_slice_with_self u in
      if List.exists (fun (t, _) -> not (Nodes.mem t slice)) candidate then u
      else
        let given (x, nx) =
          simplify
            (substitute (simplify (substitute u [ (x, nx) ])) [ (nx, x) ])
        in
        match List.map given candidate with
        | n :: news when List.for_all (( == ) n) news -> n
        | n :: _ as news
          when op u = Op.Stack && List.compare_length_with (src u) 2 = 0 ->
            let same k = List.for_all (fun w -> nth w k == nth n k) news in
            let u = if same 0 then replace u ~src:[ nth n 0; nth u 1 ] else u in
            if same 1 then replace u ~src:[ nth u 0; nth n 1 ] else u
        | _ -> u
  in
  let u = List.fold_left simplex u exprs in
  (* try all the valids together (but only the whole expressions) *)
  let subs = List.map (fun (i, e, lo, hi) -> (e, fake i e lo hi)) exprs in
  let s = substitute u subs in
  if s == u then u
  else simplify (substitute (simplify s) (List.map (fun (e, x) -> (x, e)) subs))

(* prioritize dependencies, then tighter bounds, so weaker clauses don't hide
   useful simplifications *)
let valid_priority v valids =
  match parse_valid v with
  | None -> (0, Bigint.zero)
  | Some (e, upper, c) ->
      let depends o = e == o || Nodes.mem e (backward_slice o) in
      (-List.length (List.filter depends valids), if upper then c else Bigint.neg c)

let simplify_valid valid =
  (* this should only be for indexing, skip if there's a INDEX *)
  if op_in_backward_slice_with_self valid [ Op.Index ] then None
  else
    let valids = split_uop valid Op.And in
    let keyed = List.map (fun v -> (valid_priority v valids, v)) valids in
    let order ((d0, c0), _) ((d1, c1), _) =
      match Int.compare d0 d1 with 0 -> Bigint.compare c0 c1 | c -> c
    in
    let valids = List.map snd (List.stable_sort order keyed) in
    let given ret stmt =
      (match ret with
      | [] -> stmt
      | _ -> uop_given_valid (conj (List.rev ret)) stmt)
      :: ret
    in
    let ret = List.rev (List.fold_left given [] (dedup valids)) in
    if List.equal ( == ) ret valids then None else Some (conj ret)

(* Phase 3: the complete symbolic *)

(* A float factor moved out of a sum changes its rounding, and out of a maximum
   its NaN and signed zeros. *)
let reduce_mul_chain r =
  match arg r with
  | Reduce { op = (Op.Add | Op.Max) as rop; _ }
    when not (Dtype.is_float (dtype r)) -> (
      let ranges = List.tl (src r) in
      let outside m =
        let parents = backward_slice m in
        (not (List.memq m ranges))
        && List.for_all (fun rg -> not (Nodes.mem rg parents)) ranges
        && (rop <> Op.Max || V.(vmin m >= zero))
      in
      let prod = List.fold_left mul (int 1) in
      match List.partition outside (split_uop (nth r 0) Op.Mul) with
      | [], _ -> None
      | out, inside ->
          let body =
            match inside with [] -> const_v (nth r 0) one | _ -> prod inside
          in
          Some (mul (replace r ~src:(body :: ranges)) (prod out)))
  | _ -> None

let drop_and_clauses cond x i =
  let xs = Ops.ranges x in
  let in_x c =
    List.exists (fun r -> Nodes.mem r xs) (Nodes.to_list (Ops.ranges c))
  in
  match List.partition in_x (split_uop cond Op.And) with
  | _, [] -> None
  | keep, _ -> Some (where (uprod (bool true) keep) x i)

let pm_drop_and_clauses =
  pm
    (fun () -> [ rule invalid_gate (fun m -> drop_and_clauses (m "cond") (m "x") (m "i")) ])

(* move conditions from where to load's valid, drop clauses already in load *)
let where_on_load cond buf idx or_cast =
  let where_clauses = split_uop cond Op.And and load_valid = get_valid idx in
  let in_load = split_uop load_valid Op.And in
  let idx_index =
    List.filter
      (fun u -> op u = Op.Index)
      (Nodes.to_list (backward_slice_with_self idx))
  in
  let idx_ranges = Ops.ranges idx in
  (* can move if: not a const, condition's ranges are subset of idx's ranges,
     and no data dependent INDEX (only idx's INDEX allowed) *)
  let can_move c =
    let own u = op u <> Op.Index || List.memq u idx_index in
    (not (is_const c))
    && List.for_all
         (fun r -> Nodes.mem r idx_ranges)
         (Nodes.to_list (Ops.ranges c))
    && List.for_all own (Nodes.to_list (backward_slice_with_self c))
  in
  let clauses =
    List.filter (fun c -> not (List.memq c in_load)) where_clauses
  in
  let moved, keep = List.partition can_move clauses in
  if List.compare_lengths keep where_clauses = 0 then None
  else
    let idx = index buf [ valid (get_idx idx) (uprod load_valid moved) ] in
    let ret = if op or_cast = Op.Cast then cast idx (dtype or_cast) else idx in
    Some (where (uprod (bool true) keep) ret (const_v ret zero))

(* where after gated load becomes alt value. A gated load reads +0. where its
   gate fails, so a selection of -0. stays. *)
let pm_move_where_on_load =
  let loaded =
    Upat.(or_casted ~name:"or_cast" (index (var "buf") [ var "idx" ]))
  in
  let zero = Upat.named "zero" (Upat.int 0) in
  let on_load m cond =
    match value (m "zero") with
    | `Float z when Float.sign_bit z -> None
    | _ -> where_on_load cond (m "buf") (m "idx") (m "or_cast")
  in
  pm
    (fun () -> [
      rule Upat.(where (var "cond") loaded zero) (fun m -> on_load m (m "cond"));
      rule
        Upat.(where (var "cond") zero loaded)
        (fun m -> on_load m (logical_not (m "cond")));
    ])

(* pure index math only: a LOAD in x executes even where cond is false, so its
   INDEX valid must survive the assumption *)
let gated_given_valid cond x i =
  if
    (not (Dtype.equal (dtype x) Dtype.Weak_int))
    || op_in_backward_slice_with_self x [ Op.Index ]
  then None
  else Some (where cond (uop_given_valid ~try_simplex:false cond x) i)

let pm_simplify_valid =
  pm
    (fun () -> [
      (* simplify valid *)
      rule (Upat.op Op.And ~dtype:boolean ~name:"valid") (fun m ->
          simplify_valid (m "valid"));
      rule invalid_gate (fun m -> gated_given_valid (m "cond") (m "x") (m "i"));
    ])

let remove_from_sink_like = ops [ Op.Noop; Op.Stack; Op.Sink; Op.Group ]

let pm_clean_up_group_sink =
  pm
    (fun () -> [
      (* clean up GROUP/SINK *)
      rule (Upat.op Op.Group ~src:[ Upat.var "x" ]) (fun m -> Some (m "x"));
      rule
        (Upat.v ~op:(ops [ Op.Sink; Op.Group ]) ~name:"root" ())
        (fun m ->
          let root = m "root" in
          let spliced x = Op.Set.mem (op x) remove_from_sink_like in
          if not (List.exists spliced (src root)) then None
          else
            let srcs =
              List.concat_map
                (fun x -> if spliced x then src x else [ x ])
                (src root)
            in
            Some (v (op root) ~src:srcs ~arg:(arg root)));
    ])

let sym =
  let indexed = Upat.op Op.Index ~name:"index" in
  let gated_store idx cond x =
    store (index (nth idx 0) [ valid (nth idx 1) cond ]) x
  in
  Pattern_matcher.concat
    [
      symbolic;
      pm_simplify_valid;
      pm
        (fun () -> [
          (* Pow *)
          rule (Upat.op Op.Pow ~name:"p") (fun m ->
              let p = m "p" in
              Some (Transcendental.xpow (nth p 0) (nth p 1)));
          (* Load/store folding *)
          rule
            (Upat.store indexed [ Upat.load indexed [] ])
            (fun _ -> Some (v Op.Noop));
          rule
            Upat.(
              store indexed [ where (var "gate") (var "alt") (load indexed []) ])
            (fun m -> Some (gated_store (m "index") (m "gate") (m "alt")));
          (* fold gated LOAD/STORE *)
          rule
            (Upat.op Op.Store ~src:[ Upat.wild; invalid_pat ])
            (fun _ -> Some (v Op.Noop));
          (* store of where with invalid -> gated store *)
          rule
            (Upat.op Op.Store
               ~src:Upat.[ indexed; where (var "cond") (var "val") invalid_pat ])
            (fun m -> Some (gated_store (m "index") (m "cond") (m "val")));
          (* reduce mul chain, move muls after the reduce *)
          rule
            (Upat.reduce ~name:"r" ~allow_any_len:true (Upat.op Op.Mul) [])
            (fun m -> reduce_mul_chain (m "r"));
          (* Combine terms (opinionated) *)
          rule
            Upat.(int (-1) * (var ~dtype:int_or_bool "x" + var "y"))
            (fun m -> Some O.(-m "x" + -m "y"));
          (* (x+y)*c -> x*c+y*c. only for int, float has inf*0=nan issue *)
          rule
            Upat.((var ~dtype:[ Dtype.Weak_int ] "x" + var "y") * cvar "c")
            (fun m ->
              let c = m "c" in
              Some O.((m "x" * c) + (m "y" * c)));
        ]);
      pm_clean_up_group_sink;
    ]

let () = Private.set_symbolic symbolic
