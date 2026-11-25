(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Operations on sums, through Nx: an operation linear in its partial
   operands keeps them sums, and closing its result equals closing them
   first, on data whose sums are exact; any other raises, naming the place
   that closes the operand. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype

let m = Nx_support.memory
let by = "test_sums"

module Mem = (val Nx.devices (List.init 4 m))

let line = Nx.Mesh.v Mem.v [ ("x", 4) ]
let grid = Nx.Mesh.v Mem.v [ ("a", 2); ("b", 2) ]

(* Partial placements of Mem, the number of terms of a group and the term
   each device holds, device [d] at [(d / 2, d mod 2)] over [a] and [b]. *)
type sum = Line | Over_b | Over_a of int

let pp_sum ppf = function
  | Line -> Format.pp_print_string ppf "line"
  | Over_b -> Format.pp_print_string ppf "b"
  | Over_a c -> Format.fprintf ppf "a, %d cut over b" c

let sum_at = function
  | Line -> Nx.Placement.partial line [ "x" ] []
  | Over_b -> Nx.Placement.partial grid [ "b" ] []
  | Over_a c -> Nx.Placement.partial grid [ "a" ] [ (c, [ "b" ]) ]

let terms = function Line -> 4 | Over_b | Over_a _ -> 2
let term_of w d = match w with Line -> d | Over_a _ -> d / 2 | Over_b -> d mod 2
let cut = function Over_a c -> Some c | Line | Over_b -> None

(* The value at the partial [sum_at w] whose term [k] is [terms.(k)]. *)
let partial_value (type v s) w (dt : (v, s) D.t) shape (terms : v array array)
    =
  let p = sum_at w in
  let pieces = Nx.Prim.pieces p shape in
  let array j =
    let here, win = pieces.(j) in
    let whole =
      Nx.Repr.of_array Nx.Host.v
        (A.of_array dt (Iarray.of_array shape) terms.(term_of w j))
    in
    let piece = Nx.Prim.eval ~by (Move (Slice (Iarray.of_array win), whole)) in
    require_some (Nx.Repr.array ~by:"test" (Nx.place here piece))
  in
  Nx.Repr.of_shards p (Iarray.init (Array.length pieces) array)

(* Dtypes, and their elements from small integers, so that every sum below
   is exact; a float's zero may be negative. *)
type kind =
  | K : {
      dt : ('v, 's) D.t;
      of_int : int -> 'v;
      zero : 'v;  (** Negative where the dtype has a negative zero. *)
    }
      -> kind

let same_float a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let kinds =
  [
    K { dt = D.Float32; of_int = Float.of_int; zero = -0. };
    K { dt = D.Float64; of_int = Float.of_int; zero = -0. };
    K { dt = D.Bfloat16; of_int = Float.of_int; zero = -0. };
    K { dt = D.Float16; of_int = Float.of_int; zero = -0. };
    K
      {
        dt = D.Complex64;
        of_int = (fun n -> { Complex.re = Float.of_int n; im = Float.of_int (-n) });
        zero = { Complex.re = -0.; im = -0. };
      };
    K { dt = D.Int32; of_int = Int32.of_int; zero = 0l };
    K { dt = D.Int8; of_int = Fun.id; zero = 0 };
  ]

let pp_kind ppf (K k) = Format.pp_print_string ppf (D.name k.dt)

(* Operations linear in a sum [p], beside [q], a sum over the same groups, and
   [w], no sum, of one shape, along an axis [axis] no device cuts. *)
type env = { axis : int; positions : int array }

type linear =
  | Add_pq
  | Add_pw
  | Add_wp
  | Sub_pw
  | Sub_wp
  | Sub_pq
  | Neg
  | Mul_pw
  | Mul_wp
  | Times_3
  | Over_4
  | Plus_1
  | Add_zeros
  | Copy
  | Cast_float64
  | Cast_complex128
  | Mean
  | Where_p_w
  | Where_w_p
  | Where_p_q
  | Sum
  | Sum_kept
  | Reshape
  | Transpose
  | Broadcast
  | Take

let linear =
  [
    Add_pq;
    Add_pw;
    Add_wp;
    Sub_pw;
    Sub_wp;
    Sub_pq;
    Neg;
    Mul_pw;
    Mul_wp;
    Times_3;
    Over_4;
    Plus_1;
    Add_zeros;
    Copy;
    Cast_float64;
    Cast_complex128;
    Mean;
    Where_p_w;
    Where_w_p;
    Where_p_q;
    Sum;
    Sum_kept;
    Reshape;
    Transpose;
    Broadcast;
    Take;
  ]

let linear_name = function
  | Add_pq -> "add p q"
  | Add_pw -> "add p w"
  | Add_wp -> "add w p"
  | Sub_pw -> "sub p w"
  | Sub_wp -> "sub w p"
  | Sub_pq -> "sub p q"
  | Neg -> "neg p"
  | Mul_pw -> "mul p w"
  | Mul_wp -> "mul w p"
  | Times_3 -> "p times 3"
  | Over_4 -> "p over 4"
  | Plus_1 -> "p plus 1, a whole scalar beside it"
  | Add_zeros -> "add p, zeros like p"
  | Copy -> "copy p"
  | Cast_float64 -> "cast p to float64 and back"
  | Cast_complex128 -> "cast p to complex128 and back"
  | Mean -> "mean p along an axis of 1, 2 or 4"
  | Where_p_w -> "where w is negative, p, else w"
  | Where_w_p -> "where w is negative, w, else p"
  | Where_p_q -> "where w is negative, p, else q"
  | Sum -> "sum p along an axis"
  | Sum_kept -> "sum p along an axis, kept"
  | Reshape -> "reshape p flat"
  | Transpose -> "transpose p"
  | Broadcast -> "broadcast p"
  | Take -> "take p at whole positions"

(* It negates or scales terms: a zero's sign may differ. *)
let signs = function
  | Sub_wp | Sub_pq | Neg | Mul_pw | Mul_wp | Times_3 | Over_4 -> true
  | _ -> false

let apply (type v s d) op (dt : (v, s) D.t) (of_int : int -> v) env
    (p : (v, s, d) Nx.t) (q : (v, s, d) Nx.t) (w : (v, s, d) Nx.t) :
    (v, s, d) Nx.t =
  let negative = Nx.less w (Nx.zeros_like w) in
  match op with
  | Add_pq -> Nx.add p q
  | Add_pw -> Nx.add p w
  | Add_wp -> Nx.add w p
  | Sub_pw -> Nx.sub p w
  | Sub_wp -> Nx.sub w p
  | Sub_pq -> Nx.sub p q
  | Neg -> Nx.neg p
  | Mul_pw -> Nx.mul p w
  | Mul_wp -> Nx.mul w p
  | Times_3 -> Nx.Infix.(p *$ of_int 3)
  | Over_4 -> Nx.div p (Nx.scalar_like p (of_int 4))
  | Plus_1 -> Nx.Infix.(p +$ of_int 1)
  | Add_zeros -> Nx.add p (Nx.zeros_like p)
  | Copy -> Nx.copy p
  | Cast_float64 -> Nx.cast dt (Nx.cast Nx.float64 p)
  | Cast_complex128 -> Nx.cast dt (Nx.cast Nx.complex128 p)
  | Mean -> Nx.mean ~axes:[ env.axis ] p
  | Where_p_w -> Nx.where negative p w
  | Where_w_p -> Nx.where negative w p
  | Where_p_q -> Nx.where negative p q
  | Sum -> Nx.sum ~axes:[ env.axis ] p
  | Sum_kept -> Nx.sum ~keepdims:true ~axes:[ env.axis ] p
  | Reshape -> Nx.reshape [| Array.fold_left ( * ) 1 (Nx.shape p) |] p
  | Transpose -> Nx.transpose p
  | Broadcast -> Nx.broadcast_to (Array.append [| 2 |] (Nx.shape p)) p
  | Take ->
      let idx =
        Nx.place (Nx.beside p)
          (Nx.create Nx.int64
             [| Array.length env.positions |]
             (Array.map Int64.of_int env.positions))
      in
      Nx.take ~axis:env.axis idx p

(* The dtypes [op] takes: [Over_4], [Cast_float64] and [Mean] floats,
   [Cast_complex128] complex numbers. *)
let takes_dtype (type v s) op (dt : (v, s) D.t) =
  match (op, D.kind dt) with
  | (Over_4 | Cast_float64 | Mean), D.Float -> true
  | (Over_4 | Cast_float64 | Mean), _ -> false
  | Cast_complex128, D.Complex -> true
  | Cast_complex128, _ -> false
  | _ -> true

(* Whether [op] takes the shape [s] at the sum [w], along [env]'s axis. *)
let takes_shape op w s env =
  let has_axis = env.axis < Array.length s in
  match op with
  | Mean -> has_axis && List.mem s.(env.axis) [ 1; 2; 4 ]
  | Sum | Sum_kept -> has_axis
  | Take -> has_axis && s.(env.axis) > 0
  | Reshape -> cut w = None
  | Transpose -> Array.length s >= 2
  | _ -> true

(* Cases *)

type case =
  | Case : {
      kind : kind;
      dt : ('v, 's) D.t;
      of_int : int -> 'v;
      w : sum;
      shape : int array;
      p : 'v array array;
      q : 'v array array;
      whole : 'v array;
      env : env;
      op : linear;
    }
      -> case

let pp_case ppf (Case c) =
  Format.fprintf ppf "%s, %a [%s] over %a, axis %d" (linear_name c.op) pp_kind
    c.kind
    (String.concat "; " (Array.to_list (Array.map string_of_int c.shape)))
    pp_sum c.w c.env.axis

(* The operation first, then a dtype and a shape it takes: a shape it does not
   take, about one case in twenty, is discarded. *)
let case =
  let open Gen in
  let* op = of_list linear in
  let* (K k as kind) =
    of_list ~pp:pp_kind (List.filter (fun (K k) -> takes_dtype op k.dt) kinds)
  in
  let* shape =
    array ~size:(int_range 1 3)
      (frequency [ (1, constant 0); (1, constant 1); (4, int_range 2 5) ])
  in
  let rank = Array.length shape in
  let* w =
    of_list ~pp:pp_sum ([ Line; Over_b ] @ List.init rank (fun c -> Over_a c))
  in
  let free = List.filter (fun a -> cut w <> Some a) (List.init rank Fun.id) in
  let* axis = if free = [] then constant rank else of_list free in
  let extent = if axis < rank then shape.(axis) else 0 in
  let* positions =
    array ~size:(int_range 0 3) (int_range 0 (Int.max 0 (extent - 1)))
  in
  let env = { axis; positions } in
  let n = Array.fold_left ( * ) 1 shape in
  let elt =
    frequency [ (1, constant k.zero); (4, map k.of_int (int_range (-8) 8)) ]
  in
  let values = array ~size:(constant n) elt in
  let* p = array ~size:(constant (terms w)) values in
  let* q = array ~size:(constant (terms w)) values in
  let+ whole = values in
  Case { kind; dt = k.dt; of_int = k.of_int; w; shape; p; q; whole; env; op }

let elements x =
  A.to_array (require_some (Nx.Repr.array ~by:"test" (Nx.place Nx.Host.on x)))

(* Elements equal as [Nx_support.elements] has them, but zeros of either sign
   alike. *)
let alike (type v s) (dt : (v, s) D.t) : v list testable =
  let zero (v : v) : v =
    match D.kind dt with
    | D.Float -> if v = 0. then 0. else v
    | D.Complex ->
        let z x = if x = 0. then 0. else x in
        { Complex.re = z v.Complex.re; im = z v.im }
    | D.Signed | D.Unsigned | D.Boolean -> v
  in
  Testable.contramap (List.map zero) (Nx_support.elements dt)

let law_kept (Case c) =
  assume (takes_shape c.op c.w c.shape c.env);
  cover "a cut beside the sum" (cut c.w <> None);
  List.iter (fun o -> cover (linear_name o) (o = c.op)) linear;
  cover "no element" (Array.mem 0 c.shape);
  let p = partial_value c.w c.dt c.shape c.p in
  let q = partial_value c.w c.dt c.shape c.q in
  let closed = Nx.Placement.closed (sum_at c.w) in
  let w =
    Nx.place closed
      (Nx.Repr.of_array Nx.Host.v
         (A.of_array c.dt (Iarray.of_array c.shape) c.whole))
  in
  let r = apply c.op c.dt c.of_int c.env p q w in
  let at = Nx.placement r in
  not_equal (Nx_support.placement ()) ~msg:"the result is a sum"
    (Nx.Placement.closed at) at;
  let e =
    apply c.op c.dt c.of_int c.env (Nx.place closed p) (Nx.place closed q) w
  in
  let same = if signs c.op then alike c.dt else Nx_support.elements c.dt in
  equal same ~msg:"closing first"
    (Array.to_list (elements e))
    (Array.to_list (elements r))

(* With NaN, infinities and values large enough to overflow: closing after
   an elementwise operation equals closing first, NaN as a class, but where
   nx.mli states otherwise: a term that overflows where the sum would not,
   and a NaN where a scaled infinity's terms meet. *)
type special = {
  sop : string;
  scales : bool;
  sf : 'd. (float, D.float64_elt, 'd) Nx.t -> (float, D.float64_elt, 'd) Nx.t
    -> (float, D.float64_elt, 'd) Nx.t -> (float, D.float64_elt, 'd) Nx.t;
}

let specials_ops =
  [
    { sop = "add p q"; scales = false; sf = (fun p q _ -> Nx.add p q) };
    { sop = "add p w"; scales = false; sf = (fun p _ w -> Nx.add p w) };
    { sop = "sub w p"; scales = false; sf = (fun p _ w -> Nx.sub w p) };
    { sop = "neg p"; scales = false; sf = (fun p _ _ -> Nx.neg p) };
    { sop = "mul p w"; scales = true; sf = (fun p _ w -> Nx.mul p w) };
    { sop = "where w < 0, p, w"; scales = false;
      sf = (fun p _ w -> Nx.where (Nx.less w (Nx.zeros_like w)) p w) };
  ]

let law_specials (sop, w, shape, p, q, whole) =
  let p' = partial_value w D.Float64 shape p
  and q' = partial_value w D.Float64 shape q in
  let closed = Nx.Placement.closed (sum_at w) in
  let wv =
    Nx.place closed
      (Nx.Repr.of_array Nx.Host.v (A.of_array D.Float64 (Iarray.of_array shape) whole))
  in
  let got = elements (sop.sf p' q' wv)
  and want = elements (sop.sf (Nx.place closed p') (Nx.place closed q') wv) in
  let large v = Float.is_finite v && Float.abs v > 1e300 in
  Array.iteri
    (fun i e ->
      let g = got.(i) in
      let terms =
        Array.append (Array.map (fun t -> t.(i)) p) (Array.map (fun t -> t.(i)) q)
      in
      let overflow = Array.exists large terms || large whole.(i) in
      let ok =
        if overflow then true
        else if Float.is_nan e then Float.is_nan g
        else if Float.is_integer e || e = 0. then
          same_float e g || (e = 0. && g = 0.)
          || (sop.scales && Float.is_nan g && not (Float.is_finite e))
        else (sop.scales && Float.is_nan g) || same_float e g
      in
      cover "an overflow" overflow;
      satisfies
        ~msg:(Printf.sprintf "element %d" i)
        (Testable.pair float_exact float_exact)
        ~claim:"closing after as closing first, but where nx.mli says"
        (fun _ -> ok)
        (e, g))
    want

let specials_case =
  let open Gen in
  let* shape = array ~size:(int_range 1 2) (int_range 1 4) in
  let rank = Array.length shape in
  let* w = of_list ~pp:pp_sum ([ Line; Over_b ] @ List.init rank (fun c -> Over_a c)) in
  let* sop = of_list specials_ops in
  let n = Array.fold_left ( * ) 1 shape in
  let elt =
    frequency
      [ (4, map Float.of_int (int_range (-8) 8));
        (1, of_list [ -0.; Float.nan; Float.infinity; Float.neg_infinity; 1e308; -1e308 ]) ]
  in
  let values = array ~size:(constant n) elt in
  let* p = array ~size:(constant (terms w)) values in
  let* q = array ~size:(constant (terms w)) values in
  let+ whole = values in
  (sop, w, shape, p, q, whole)

let pp_specials ppf (sop, w, shape, _, _, _) =
  Format.fprintf ppf "%s, [%s] over %a" sop.sop
    (String.concat "; " (Array.to_list (Array.map string_of_int shape)))
    pp_sum w

(* Contractions *)

(* Where a contraction's operands lie: [a] of [m; k] and [b] of [k; n] cut
   alike along [k] over the line or over [a] with [m] cut over [b]; or [a] a
   sum over the line and [b] whole. *)
type over = Cut_line | Cut_grid | Summed_a

let pp_over ppf = function
  | Cut_line -> Format.pp_print_string ppf "k cut over the line"
  | Cut_grid -> Format.pp_print_string ppf "k cut over a, m over b"
  | Summed_a -> Format.pp_print_string ppf "a a sum over the line"

type contraction =
  | Contraction : {
      kind : kind;
      dt : ('v, 's) D.t;
      over : over;
      m : int;
      k : int;
      n : int;
      a : 'v array array;  (** One per term where [a] is a sum. *)
      b : 'v array;
      init : 'v array option;
    }
      -> contraction

let pp_contraction ppf (Contraction c) =
  Format.fprintf ppf "%a [%d; %d] by [%d; %d]%s, %a" pp_kind c.kind c.m c.k c.k
    c.n
    (if Option.is_some c.init then " from init" else "")
    pp_over c.over

let contraction =
  let open Gen in
  let* (K k as kind) = of_list ~pp:pp_kind kinds in
  let* over = of_list ~pp:pp_over [ Cut_line; Cut_grid; Summed_a ] in
  let* m = int_range 0 4 in
  let* kk = int_range 0 6 in
  let* n = int_range 0 4 in
  (* Every sum is exact. Elements of magnitude [r], four terms of [a] and
     [k <= 6] give products within [24 r^2 + r] with init, every partial
     sum too; bfloat16 holds each integer up to 2^8, so [r = 3] there. *)
  let r = match k.dt with D.Bfloat16 -> 3 | _ -> 8 in
  let elt =
    frequency [ (1, constant k.zero); (4, map k.of_int (int_range (-r) r)) ]
  in
  let values c = array ~size:(constant c) elt in
  let* a =
    array ~size:(constant (if over = Summed_a then 4 else 1)) (values (m * kk))
  in
  let* b = values (kk * n) in
  let+ init = option (values (m * n)) in
  Contraction { kind; dt = k.dt; over; m; k = kk; n; a; b; init }

let law_contraction (Contraction c) =
  cover "a tile of no element" (c.over <> Summed_a && c.k < 4);
  cover "an init" (Option.is_some c.init);
  cover "a sum beside a whole operand" (c.over = Summed_a);
  let host s xs = Nx.Repr.of_array Nx.Host.v (A.of_array c.dt (Iarray.of_array s) xs) in
  let pattern = Nx.Pattern.v "m k, k n -> m n | k" in
  let a, b =
    match c.over with
    | Cut_line ->
        ( Nx.place (Nx.Placement.split ~axis:1 Mem.v) (host [| c.m; c.k |] c.a.(0)),
          Nx.place (Nx.Placement.split ~axis:0 Mem.v) (host [| c.k; c.n |] c.b) )
    | Cut_grid ->
        ( Nx.place (Nx.Placement.mesh grid [ (0, [ "b" ]); (1, [ "a" ]) ])
            (host [| c.m; c.k |] c.a.(0)),
          Nx.place (Nx.Placement.mesh grid [ (0, [ "a" ]) ]) (host [| c.k; c.n |] c.b) )
    | Summed_a ->
        ( partial_value Line c.dt [| c.m; c.k |] c.a,
          Nx.place Mem.on (host [| c.k; c.n |] c.b) )
  in
  let init = Option.map (fun i -> Nx.place (Nx.beside a) (host [| c.m; c.n |] i)) c.init in
  (* The sum the devices' products make, asked for. *)
  let y = Nx.contract ?init ~sum:true c.dt pattern a b in
  let at = Nx.placement y in
  (* A summed axis of no element is read whole: no device sums a term. *)
  if c.k > 0 || c.over = Summed_a then
    not_equal (Nx_support.placement ()) ~msg:"the result is a sum"
      (Nx.Placement.closed at) at;
  let closed x = Nx.place Nx.Host.on x in
  let want =
    Nx.contract ?init:(Option.map closed init) c.dt pattern (closed a) (closed b)
  in
  let got = Array.to_list (elements y) in
  let again =
    Array.to_list (elements (Nx.contract ?init ~sum:true c.dt pattern a b))
  in
  equal (Nx_support.elements c.dt) ~msg:"a second run" got again;
  equal (Nx_support.elements c.dt) ~msg:"the closed operands' contraction"
    (Array.to_list (elements want))
    got

(* A bfloat16 product over a cut k. Closed by default, it sums in float32 and
   rounds once: within the contraction's bound [(1 + u') γ(β(m) + t - 1, u')
   Σ|ab|], [u'] float32's unit roundoff, then [u |exact|], [u = 2^-8]
   bfloat16's. Asked for its sum, each of the [t] terms rounds into bfloat16 and
   the closing adds them there: [γ(t, u) Σ|term|] more, [Σ|term| <= Σ|ab|]. *)
type narrow = { nm : int; nk : int; nn : int; nseed : int; over : int }

let narrow =
  let open Gen in
  let* nm = int_range 1 4 in
  let* nk = int_range 1 40 in
  let* nn = int_range 1 4 in
  let* over = of_list [ 2; 4 ] in
  let+ nseed = int in
  { nm; nk; nn; nseed; over }

let pp_narrow ppf c =
  Format.fprintf ppf "[%d; %d] by [%d; %d] over %d, seed %d" c.nm c.nk c.nk
    c.nn c.over c.nseed

module Two = (val Nx.devices [ m 0; m 1 ])

let gamma n u = n *. u /. (1. -. (n *. u))

(* nx_kernel.mli's [β(n) = ⌈5 min(n, 256) / 4⌉ + ⌈n / 128⌉] for a sum in
   blocks. *)
let beta n = (((5 * min n 256) + 3) / 4) + ((n + 127) / 128)
let u32 = Float.ldexp 1. (-24)
let ub = Float.ldexp 1. (-8)

(* [a b] over [s]'s devices, [k] cut: closed, and asked for its sum then
   closed. *)
let narrow_products (type d) (s : d Nx.Devices.t) a b =
  let sum =
    Nx.Placement.partial
      (Nx.Mesh.v s [ ("x", List.length (Nx.Devices.to_list s)) ])
      [ "x" ] []
  in
  let a = Nx.place (Nx.Placement.split ~axis:1 s) a
  and b = Nx.place (Nx.Placement.split ~axis:0 s) b in
  let read y = Nx.to_array (Nx.cast Nx.float64 (Nx.place Nx.Host.on y)) in
  let closed = Nx.matmul a b and asked = Nx.matmul ~sum:true a b in
  equal (Nx_support.placement ()) ~msg:"closed by default"
    (Nx.Placement.closed sum) (Nx.placement closed);
  equal (Nx_support.placement ()) ~msg:"a sum where asked" sum
    (Nx.placement asked);
  (read closed, read asked)

let law_narrow c =
  let st = Random.State.make [| c.nseed |] in
  let draw s =
    Nx.cast Nx.bfloat16
      (Nx.init Nx.float64 s (fun _ -> Random.State.float st 2. -. 1.))
  in
  let a = draw [| c.nm; c.nk |] and b = draw [| c.nk; c.nn |] in
  let wide x = Nx.to_array (Nx.cast Nx.float64 x) in
  let ha = wide a and hb = wide b in
  let closed, asked =
    if c.over = 2 then narrow_products Two.v a b else narrow_products Mem.v a b
  in
  let t = Float.of_int (Int.min c.over c.nk) in
  let tiny = Float.ldexp 1. (-133) in
  for i = 0 to c.nm - 1 do
    for j = 0 to c.nn - 1 do
      let exact = ref 0. and mag = ref 0. in
      for k = 0 to c.nk - 1 do
        let p = ha.((i * c.nk) + k) *. hb.((k * c.nn) + j) in
        exact := !exact +. p;
        mag := !mag +. Float.abs p
      done;
      let m = (c.nk + c.over - 1) / c.over in
      let roundings = Float.of_int (beta m + Int.min c.over c.nk - 1) in
      let product = (1. +. u32) *. gamma roundings u32 *. !mag in
      let check what got bound =
        let g = got.((i * c.nn) + j) in
        at_most float_exact
          ~msg:(Printf.sprintf "%s (%d, %d): %h against %h" what i j g !exact)
          ~than:bound (Nx_support.gap g !exact)
      in
      check "closed" closed (product +. (ub *. Float.abs !exact) +. tiny);
      check "a sum" asked (product +. (gamma t ub *. !mag) +. (t *. tiny))
    done
  done

(* Terms [1; 2^-8; 2^-8; 2^-8] of one product over four devices: closed in
   float32 they sum to [1 + 3 2^-8], which rounds to bfloat16's
   [1 + 2^-6]; as a bfloat16 sum each addition of [2^-8] to [1] rounds to
   [1], an error of [3u], within [γ(4, u) Σ|term|]. *)
let test_rounded_terms () =
  let e = Float.ldexp 1. (-8) in
  let a = Nx.cast Nx.bfloat16 (Nx.create Nx.float64 [| 1; 4 |] [| 1.; e; e; e |])
  and b = Nx.ones Nx.bfloat16 [| 4; 1 |] in
  let closed, asked = narrow_products Mem.v a b in
  equal (array float_exact) ~msg:"closed in float32, rounded once"
    [| 1. +. Float.ldexp 1. (-6) |] closed;
  equal (array float_exact) ~msg:"a bfloat16 sum, closed" [| 1. |] asked;
  let sum = 1. +. (3. *. e) in
  less float_exact ~msg:"within γ(4, u) of the terms"
    ~than:(gamma 4. ub *. sum)
    (Nx_support.gap asked.(0) sum)

(* Creations at a sum: the value made where the sum lies closed, on each
   group's first device. *)
let test_creations () =
  let p = sum_at (Over_a 0) in
  let same ~msg host x =
    equal (array float_exact) ~msg (Nx.to_array host)
      (Nx.to_array (Nx.place Nx.Host.on x))
  in
  let closed = Nx.Placement.closed p in
  let check ~msg (made : 'd Nx.Placement.t -> (float, _, 'd) Nx.t) =
    let x = made p in
    equal (Nx_support.placement ()) ~msg:(msg ^ ": where") p (Nx.placement x);
    same ~msg (Nx.place Nx.Host.on (made closed)) x
  in
  check ~msg:"zeros_at" (fun p -> Nx.zeros_at p Nx.float32 [| 4; 3 |]);
  check ~msg:"ones_at" (fun p -> Nx.ones_at p Nx.float32 [| 4; 3 |]);
  check ~msg:"full_at" (fun p -> Nx.full_at p Nx.float32 [| 4; 3 |] 2.5);
  check ~msg:"arange_f_at" (fun p -> Nx.arange_f_at p Nx.float32 0. 4. 1.);
  check ~msg:"linspace_at" (fun p -> Nx.linspace_at p Nx.float32 0. 1. 5);
  check ~msg:"eye_at" (fun p -> Nx.eye_at p Nx.float32 4);
  let key = Nx.Rng.key 3 in
  check ~msg:"Rng.uniform_at" (fun p ->
      Nx.Rng.uniform_at p ~key:(Nx.Rng.place (Nx.Placement.uncut p) key)
        Nx.float32 [| 4; 3 |]);
  check ~msg:"Rng.normal_at" (fun p ->
      Nx.Rng.normal_at p ~key:(Nx.Rng.place (Nx.Placement.uncut p) key)
        Nx.float32 [| 4; 3 |]);
  let refused ~by f =
    raises_match (Exn.invalid_arg ~sub:(by ^ ": a ")) (fun () -> ignore (f ()))
  in
  refused ~by:"Nx.full_at" (fun () -> Nx.full_at p Nx.bool [| 2 |] true);
  refused ~by:"Nx.zeros_at" (fun () -> Nx.zeros_at p Nx.int4 [| 2 |])

(* A product over a cut summed axis closes unless asked for its sum, and so does
   every function built on one: a convolution over cut channels, einsum's tree
   of products and a chain. The data are multiples of 1/64 of few terms, so
   every sum is exact in float32. *)
let x_conv =
  lazy
    (Nx.init Nx.float32 [| 1; 5; 8 |] (fun i ->
         Float.of_int (i.(1) + i.(2)) /. 8.))

let w_conv =
  lazy
    (Nx.init Nx.float32 [| 2; 3; 8 |] (fun i ->
         Float.of_int (i.(0) - i.(2)) /. 8.))

let a_prod =
  lazy
    (Nx.init Nx.float32 [| 3; 8 |] (fun i ->
         Float.of_int (i.(0) * i.(1)) /. 16.))

let b_prod =
  lazy
    (Nx.init Nx.float32 [| 8; 4 |] (fun i -> Float.of_int (i.(0) - i.(1)) /. 8.))

let v_prod = lazy (Nx.init Nx.float32 [| 4 |] (fun i -> Float.of_int i.(0)))
let rows_of x = Nx.place (Nx.Placement.split ~axis:1 Mem.v) x
let cols_of x = Nx.place (Nx.Placement.split ~axis:0 Mem.v) x

let same_product ~msg want got =
  equal ~msg
    (Nx_support.elements Nx.float32)
    (Array.to_list (Nx.to_array (Nx.place Nx.Host.on want)))
    (Array.to_list (Nx.to_array (Nx.place Nx.Host.on got)))

let closes =
  [
    ( "a convolution over cut channels",
      fun () ->
        let x = Lazy.force x_conv and w = Lazy.force w_conv in
        let cut = Nx.Placement.split ~axis:2 Mem.v in
        (Nx.conv x w, Nx.conv (Nx.place cut x) (Nx.place cut w)) );
    ( "a three-operand einsum",
      fun () ->
        let pat = Nx.Pattern.v "i j, j k, k -> i | j k" in
        let a = Lazy.force a_prod and b = Lazy.force b_prod in
        let v = Lazy.force v_prod in
        ( Nx.einsum pat [ a; b; v ],
          Nx.einsum pat [ rows_of a; cols_of b; Nx.place Mem.on v ] ) );
    ( "a chain",
      fun () ->
        let a = Lazy.force a_prod and b = Lazy.force b_prod in
        let v = Nx.reshape [| 4; 1 |] (Lazy.force v_prod) in
        ( Nx.multi_dot [ a; b; v ],
          Nx.multi_dot [ rows_of a; cols_of b; Nx.place Mem.on v ] ) );
    ( "a matrix product",
      fun () ->
        let a = Lazy.force a_prod and b = Lazy.force b_prod in
        (Nx.matmul a b, Nx.matmul (rows_of a) (cols_of b)) );
  ]

let law_closes (name, f) =
  let want, got = f () in
  let p = Nx.placement got in
  equal (Nx_support.placement ()) ~msg:"closed" (Nx.Placement.closed p) p;
  same_product ~msg:name want got

let test_reduce_scattered () =
  let a = Lazy.force a_prod and b = Lazy.force b_prod in
  let rows = Nx.Placement.split ~axis:0 Mem.v in
  let y = Nx.place rows (Nx.matmul ~sum:true (rows_of a) (cols_of b)) in
  equal (Nx_support.placement ()) ~msg:"where it lies" rows (Nx.placement y);
  same_product ~msg:"its rows" (Nx.matmul a b) y

let test_einsum_keeps () =
  let a = Lazy.force a_prod and b = Lazy.force b_prod in
  let s = Nx.matmul ~sum:true (rows_of a) (cols_of b) in
  equal (Nx_support.placement ()) ~msg:"a sum" (sum_at Line) (Nx.placement s);
  let summed = Nx.einsum (Nx.Pattern.v "i j -> i | j") [ s ] in
  let at = Nx.placement summed in
  not_equal (Nx_support.placement ()) ~msg:"a sum" (Nx.Placement.closed at) at;
  same_product ~msg:"its sum" (Nx.sum ~axes:[ 1 ] (Nx.matmul a b)) summed

(* Every product asked for its sum over a cut summed axis leaves the devices'
   terms, which close to the product: small integers, so every sum is exact. *)
let products =
  let a =
    lazy
      (Nx.init Nx.float32 [| 3; 8 |] (fun i -> Float.of_int (i.(0) + i.(1) - 4)))
  in
  let b =
    lazy
      (Nx.init Nx.float32 [| 8; 3 |] (fun i ->
           Float.of_int (i.(0) - (2 * i.(1)))))
  in
  let v =
    lazy (Nx.init Nx.float32 [| 8 |] (fun i -> Float.of_int (i.(0) - 3)))
  in
  let on = cols_of in
  [
    ( "contract",
      fun sum ->
        let p = Nx.Pattern.v "m k, k n -> m n | k" in
        let f a b = Nx.contract ~sum Nx.float32 p a b in
        ( f (Lazy.force a) (Lazy.force b),
          f (rows_of (Lazy.force a)) (on (Lazy.force b)) ) );
    ( "einsum",
      fun sum ->
        let p = Nx.Pattern.v "m k, k n -> m n | k" in
        let f a b = Nx.einsum ~sum p [ a; b ] in
        ( f (Lazy.force a) (Lazy.force b),
          f (rows_of (Lazy.force a)) (on (Lazy.force b)) ) );
    ( "matmul",
      fun sum ->
        let f a b = Nx.matmul ~sum a b in
        ( f (Lazy.force a) (Lazy.force b),
          f (rows_of (Lazy.force a)) (on (Lazy.force b)) ) );
    ( "dot",
      fun sum ->
        let f a b = Nx.dot ~sum a b in
        ( f (Lazy.force a) (Lazy.force v),
          f (rows_of (Lazy.force a)) (on (Lazy.force v)) ) );
    ( "vdot",
      fun sum ->
        let f a b = Nx.vdot ~sum a b in
        ( f (Lazy.force v) (Lazy.force v),
          f (on (Lazy.force v)) (on (Lazy.force v)) ) );
    ( "tensordot",
      fun sum ->
        let f a b = Nx.tensordot ~axes:([ 1 ], [ 0 ]) ~sum a b in
        ( f (Lazy.force a) (Lazy.force b),
          f (rows_of (Lazy.force a)) (on (Lazy.force b)) ) );
    ( "inner",
      fun sum ->
        let f a b = Nx.inner ~sum a b in
        let bt = Nx.transpose (Lazy.force b) in
        (f (Lazy.force a) bt, f (rows_of (Lazy.force a)) (rows_of bt)) );
    ( "vecdot",
      fun sum ->
        let f a b = Nx.vecdot ~sum a b in
        let bt = Nx.transpose (Lazy.force b) in
        (f (Lazy.force a) bt, f (rows_of (Lazy.force a)) (rows_of bt)) );
  ]

let law_takes_sum (name, f) =
  let want, _ = f false in
  let _, got = f true in
  let at = Nx.placement got in
  not_equal (Nx_support.placement ()) ~msg:"a sum" (Nx.Placement.closed at) at;
  same_product ~msg:name want got

(* A product whose summed axes no device cuts is the same asked for its sum: [a]
   cut along its rows or whole, [b] whole on every device. *)
let law_uncut_sum (m, k, n, rows_cut, seed) =
  let st = Random.State.make [| seed |] in
  let draw s =
    Nx.init Nx.float32 s (fun _ ->
        Float.ldexp
          (Random.State.float st 2. -. 1.)
          (Random.State.int st 21 - 10))
  in
  let a = draw [| m; k |] and b = draw [| k; n |] in
  cover "a cut kept axis" rows_cut;
  let a =
    Nx.place (if rows_cut then Nx.Placement.split ~axis:0 Mem.v else Mem.on) a
  in
  let b = Nx.place Mem.on b in
  let without = Nx.matmul a b and asked = Nx.matmul ~sum:true a b in
  equal (Nx_support.placement ()) ~msg:"where it lies" (Nx.placement without)
    (Nx.placement asked);
  same_product ~msg:"its elements" without asked

let uncut_case =
  let open Gen in
  let* m = int_range 1 9 and+ k = int_range 0 9 and+ n = int_range 1 5 in
  let+ rows_cut = bool and+ seed = int in
  (m, k, n, rows_cut, seed)

let pp_uncut ppf (m, k, n, rows_cut, seed) =
  Format.fprintf ppf "[%d; %d] by [%d; %d]%s, seed %d" m k k n
    (if rows_cut then ", rows cut" else "")
    seed

(* Products over cut summed axes, at every float dtype, [k] cut along a line of
   two or four devices, or over a grid whose cut lists its mesh axes major first
   or minor first. Asked for its sum at the accumulator's dtype, the product
   without [init] closes to the bits of the product without [~sum:true]. With
   one window holding products it has one device's bits, and with [init] or
   without it stays within the bound {!Nx.contract} states over cut summed
   names. test_partial's contraction law holds the order in which the partials
   add. *)
type float_dt = F : (float, 's) D.t -> float_dt
type layout = Line_of of int | Grid_ab | Grid_ba

type ordered = {
  fdt : float_dt;
  layout : layout;
  om : int;
  ok : int;
  on : int;
  oinit : bool;
  oseed : int;
}

let pp_layout ppf = function
  | Line_of t -> Format.fprintf ppf "a line of %d" t
  | Grid_ab -> Format.pp_print_string ppf "a grid, a then b"
  | Grid_ba -> Format.pp_print_string ppf "a grid, b then a"

let pp_ordered ppf c =
  let (F dt) = c.fdt in
  Format.fprintf ppf "%s [%d; %d] by [%d; %d] over %a%s, seed %d" (D.name dt)
    c.om c.ok c.ok c.on pp_layout c.layout
    (if c.oinit then ", from init" else "")
    c.oseed

(* One window holding products, from init: drawn, a case in 24. *)
let ordered_examples =
  [ { fdt = F D.Float32; layout = Line_of 2; om = 2; ok = 1; on = 2;
      oinit = true; oseed = 0 } ]

let ordered =
  let open Gen in
  let* fdt =
    of_list [ F D.Float16; F D.Bfloat16; F D.Float32; F D.Float64 ]
  in
  let* layout = of_list [ Line_of 2; Line_of 4; Grid_ab; Grid_ba ] in
  let* om = int_range 1 3 in
  let* ok = int_range 0 11 in
  let* on = int_range 1 3 in
  let* oinit = bool in
  let+ oseed = int in
  { fdt; layout; om; ok; on; oinit; oseed }

let bits x =
  Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 x))

let law_ordered c =
  let (F dt) = c.fdt in
  cover "a narrow float" (match c.fdt with F D.Float16 | F D.Bfloat16 -> true | _ -> false);
  cover "four windows" (c.layout <> Line_of 2 && c.ok >= 4);
  cover "an init" c.oinit;
  cover "one window holding products, from init" (c.ok = 1 && c.oinit);
  let st = Random.State.make [| c.oseed |] in
  let draw s =
    Nx.cast dt (Nx.init Nx.float64 s (fun _ -> Random.State.float st 2. -. 1.))
  in
  let a = draw [| c.om; c.ok |] and b = draw [| c.ok; c.on |] in
  let init = if c.oinit then Some (draw [| c.om; c.on |]) else None in
  let pattern = Nx.Pattern.v "m k, k n -> m n | k" in
  let t = match c.layout with Line_of t -> t | Grid_ab | Grid_ba -> 4 in
  (* The product closed, and asked for its sum then closed, with [a] and
     [b] cut along [k] by [cut] and [init] whole on [on]. *)
  let run cut on =
    let a = Nx.place (cut 1) a and b = Nx.place (cut 0) b in
    let init = Option.map (Nx.place on) init in
    let s = Nx.contract ?init ~sum:true dt pattern a b in
    ( bits (Nx.contract ?init dt pattern a b),
      bits (Nx.place (Nx.Placement.closed (Nx.placement s)) s) )
  in
  let closed, summed =
    match c.layout with
    | Line_of 2 -> run (fun axis -> Nx.Placement.split ~axis Two.v) Two.on
    | Line_of _ -> run (fun axis -> Nx.Placement.split ~axis Mem.v) Mem.on
    | Grid_ab ->
        run (fun axis -> Nx.Placement.mesh grid [ (axis, [ "a"; "b" ]) ]) Mem.on
    | Grid_ba ->
        run (fun axis -> Nx.Placement.mesh grid [ (axis, [ "b"; "a" ]) ]) Mem.on
  in
  let wide = match c.fdt with F D.Float32 | F D.Float64 -> true | F _ -> false in
  if wide && not c.oinit then
    equal (array int64) ~msg:"its sum closed, the same bits" closed summed;
  (* One window holding every product computes as one device does. *)
  if c.ok = 1 then
    equal (array int64) ~msg:"one window holding products, one device's bits"
      (bits (Nx.contract ?init dt pattern a b))
      closed;
  (* Nx.contract's bound over cut summed names, u|r| + (1 + u) γ(β(m) + t - 1,
     u) Σ|a||b| in the accumulator, [m] the most products a window holds and [t]
     the windows holding products, the [u|r|] term with [init] alone, then half
     the dtype's spacing [u_out] relative, or [tiny] below its normals; the
     float64 reference errs by γ(K + 2) of float64. *)
  let u, u_out, tiny =
    match c.fdt with
    | F D.Float64 -> (0x1p-53, 0., 0.)
    | F D.Float32 -> (0x1p-24, 0., 0.)
    | F D.Bfloat16 -> (0x1p-24, 0x1p-8, 0x1p-134)
    | F _ -> (0x1p-24, 0x1p-11, 0x1p-25)
  in
  let host x = Nx.to_array (Nx.cast Nx.float64 x) in
  let ha = host a and hb = host b and hi = Option.map host init in
  let gamma n u = n *. u /. (1. -. (n *. u)) in
  let m = (c.ok + t - 1) / t in
  let k = Float.of_int (beta m + Int.min c.ok t - 1)
  and k64 = Float.of_int (c.ok + 2) in
  for i = 0 to c.om - 1 do
    for j = 0 to c.on - 1 do
      let o = (i * c.on) + j in
      let start = Option.fold ~none:0. ~some:(fun h -> h.(o)) hi in
      let sum = ref start and mag = ref 0. in
      for l = 0 to c.ok - 1 do
        let p = ha.((i * c.ok) + l) *. hb.((l * c.on) + j) in
        sum := !sum +. p;
        mag := !mag +. Float.abs p
      done;
      let joins = if c.oinit then u *. Float.abs !sum else 0. in
      let g =
        joins +. ((1. +. u) *. gamma k u *. !mag)
        +. (gamma k64 0x1p-53 *. (!mag +. Float.abs start))
      in
      let bound = g +. (u_out *. (Float.abs !sum +. g)) +. tiny in
      let got = Int64.float_of_bits closed.(o) in
      at_most float_exact
        ~msg:(Printf.sprintf "(%d, %d): %h against %h" i j got !sum)
        ~than:bound (Nx_support.gap got !sum)
    done
  done

(* A product's sum over a grid is named by the cut's mesh axes in the cut's
   order, whichever order the mesh lists them in: a value placed at that
   partial placement adds to it, and the sum closes to the closed product
   plus the value. The values are small integers, so every sum is exact. *)
let named_sums =
  let ba = Nx.Mesh.v Mem.v [ ("b", 2); ("a", 2) ] in
  [
    ("cut over b then a, mesh a then b", (grid, [ "b"; "a" ]));
    ("cut over a then b, mesh a then b", (grid, [ "a"; "b" ]));
    ("cut over a then b, mesh b then a", (ba, [ "a"; "b" ]));
    ("cut over b then a, mesh b then a", (ba, [ "b"; "a" ]));
  ]

let law_named_sum (_, (mesh, order)) =
  let a = Nx.init Nx.float32 [| 2; 8 |] (fun i -> Float.of_int (i.(0) + i.(1)))
  and b = Nx.init Nx.float32 [| 8; 3 |] (fun i -> Float.of_int (i.(0) - i.(1)))
  and z = Nx.init Nx.float32 [| 2; 3 |] (fun i -> Float.of_int (i.(0) * i.(1))) in
  let cut axis = Nx.Placement.mesh mesh [ (axis, order) ] in
  let s = Nx.matmul ~sum:true (Nx.place (cut 1) a) (Nx.place (cut 0) b) in
  let named = Nx.Placement.partial mesh order [] in
  equal (Nx_support.placement ()) ~msg:"named" named (Nx.placement s);
  let y = Nx.add s (Nx.place named z) in
  same_product ~msg:"closed, plus the value" (Nx.add (Nx.matmul a b) z) y

(* A large init joins a cut sum after every product: four products of
   [0x1.8p-26] over four devices add [1.5 · 2^-24] to an init of [1], which
   rounds to [1 + 2^-23]; added to the first device's partial, the others'
   would each be lost against it. *)
let test_init_joins_last () =
  let a = Nx.full Nx.float32 [| 1; 4 |] 0x1.8p-13
  and b = Nx.full Nx.float32 [| 4; 1 |] 0x1p-13 in
  let init = Nx.place Mem.on (Nx.full Nx.float32 [| 1; 1 |] 1.) in
  let a = Nx.place (Nx.Placement.split ~axis:1 Mem.v) a
  and b = Nx.place (Nx.Placement.split ~axis:0 Mem.v) b in
  let y = Nx.contract ~init Nx.float32 (Nx.Pattern.v "m k, k n -> m n | k") a b in
  equal (array float_exact) [| 1. +. 0x1p-23 |]
    (Nx.to_array (Nx.place Nx.Host.on y))

(* Refusals *)

let x () = Nx.place (sum_at Line) (Nx.ones Nx.float32 [| 4 |])
let i () = Nx.place (sum_at Line) (Nx.ones Nx.int32 [| 4 |])
let whole () = Nx.place Mem.on (Nx.ones Nx.float32 [| 4 |])
let names = String.concat "; " (List.map Rig.name (Nx.Devices.to_list Mem.v))

let test_exp_refused () =
  raises
    (Invalid_argument
       ("Nx.exp: operand 0 lies at partial [" ^ names
      ^ "], a sum the operation does not keep; Nx.place (Placement.on s) x"))
    (fun () -> Nx.exp (x ()))

(* Operations not linear in a sum, each named with the operand it refuses. *)
let not_linear =
  [
    ("a sum times a sum", "Nx.mul", fun () -> ignore (Nx.mul (x ()) (x ())));
    ("a value over a sum", "Nx.div", fun () -> ignore (Nx.div (whole ()) (x ())));
    ("a maximum", "Nx.maximum", fun () -> ignore (Nx.maximum (x ()) (whole ())));
    ("a square root", "Nx.sqrt", fun () -> ignore (Nx.sqrt (x ())));
    ("a cumulative sum", "Nx.cumsum", fun () -> ignore (Nx.cumsum (x ())));
    ("a maximum along an axis", "Nx.max", fun () -> ignore (Nx.max (x ())));
    ( "a cast to an integer",
      "Nx.cast",
      fun () -> ignore (Nx.cast Nx.int32 (x ())) );
    ("an integer mean", "Nx.mean", fun () -> ignore (Nx.mean (i ())));
    ( "an integer division",
      "Nx.div",
      fun () -> ignore (Nx.div (i ()) (Nx.full_like (i ()) 2l)) );
    ( "sums over other groups",
      "Nx.add",
      fun () ->
        ignore
          (Nx.add (x ())
             (Nx.place (sum_at Over_b) (Nx.ones Nx.float32 [| 4 |]))) );
    ( "a sum along a cut axis",
      "Nx.sum",
      fun () ->
        ignore
          (Nx.sum (Nx.place (sum_at (Over_a 0)) (Nx.ones Nx.float32 [| 4 |])))
    );
  ]

(* An operand that is no sum, beside a sum, is named whole. *)
let test_beside_sum_refused () =
  let c =
    Nx.place (Nx.Placement.split ~axis:0 Mem.v) (Nx.ones Nx.float32 [| 4; 2 |])
  in
  let pm = Nx.place (sum_at Line) (Nx.ones Nx.float32 [| 3; 4 |]) in
  raises_match
    (fun e ->
      Exn.invalid_arg ~sub:"Nx.matmul: operand 1 lies at split" e
      && Exn.invalid_arg ~sub:"Nx.place (Placement.on s) y" e)
    (fun () -> Nx.matmul pm c)

let routing =
  group "routing"
    [
      (* 26 operations drawn alike: each misses a run of 600 cases with
         probability (25/26)^600 = 6e-11. *)
      prop ~count:600
        "a linear operation keeps a sum, and closing after equals closing first"
        (Gen.with_pp pp_case case) law_kept;
      prop ~count:400
        "closing after equals closing first but where a term overflows, a \
         zero's sign or an infinity's scaling decides"
        (Gen.with_pp pp_specials specials_case)
        law_specials;
      test
        "a value made at a sum holds its elements on each group's first device"
        test_creations;
      test "exp of a sum raises naming the place that closes it"
        test_exp_refused;
      cases "an operation that is not linear in a sum raises"
        ~id:(fun (id, _, _) -> id)
        not_linear
        (fun (_, by, f) ->
          raises_match (Exn.invalid_arg ~sub:(by ^ ": operand ")) f);
      test "an operand that is no sum beside a sum is named whole"
        test_beside_sum_refused;
    ]

let contractions =
  group "contractions"
    [
      prop ~count:300
        "a contraction over a cut summed axis asked for its sum closes to the \
         contraction of its closed operands"
        (Gen.with_pp pp_contraction contraction)
        law_contraction;
      cases "a product over a cut summed axis closes unless asked for its sum"
        ~id:fst closes law_closes;
      test "a sum reduce-scattered is the product's rows" test_reduce_scattered;
      test "einsum's sum over one operand keeps a sum" test_einsum_keeps;
      cases "every product takes ~sum and leaves a sum" ~id:fst products
        law_takes_sum;
      prop
        "a product whose summed axes no device cuts is the same asked for its \
         sum"
        (Gen.with_pp pp_uncut uncut_case)
        law_uncut_sum;
      cases "a product's sum is named by its cut's mesh axes in order" ~id:fst
        named_sums law_named_sum;
      test "a large init joins a cut sum after its products"
        test_init_joins_last;
      test "a bfloat16 sum rounds each term and each addition"
        test_rounded_terms;
      prop
        "a bfloat16 product over a cut summed axis, closed or asked for as a \
         sum, is within its bound"
        (Gen.with_pp pp_narrow narrow)
        law_narrow;
      prop ~count:300 ~examples:ordered_examples
        "a product asked for its sum at the accumulator's dtype closes to the \
         product's bits, within the contraction's bound"
        (Gen.with_pp pp_ordered ordered)
        law_ordered;
    ]

let () = exit (run "nx sums" [ routing; contractions ])
