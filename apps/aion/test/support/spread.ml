(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let alpha = 1e-3

let tau ~n count =
  let n = float_of_int n in
  sqrt (n *. (((2. *. float_of_int count /. alpha) ** (2. /. n)) -. 1.))

type check = {
  count : int;
  tau : float;
  worst : float;
  failures : (int * float * float * float) list;
}

let same_length fn a b x =
  let m = Array.length x in
  if Array.length a <> m || Array.length b <> m then
    invalid_arg ("Spread." ^ fn ^ ": arrays of different lengths")

(* [ok i] decides value [i]; [ratio i] is its distance over its bound. *)
let check ~n count ~ok ~ratio ~value ~reference ~bound =
  let tau = tau ~n count in
  let worst = ref 0. and failures = ref [] in
  for i = count - 1 downto 0 do
    worst := Float.max !worst (ratio tau i);
    if not (ok tau i) then
      failures := (i, value i, reference i, tau *. bound i) :: !failures
  done;
  { count; tau; worst = !worst; failures = !failures }

let ratio d b = if d = 0. then 0. else if b = 0. then infinity else d /. b

let values ~n ~reference ~spread x =
  same_length "values" reference spread x;
  let d i = Float.abs (x.(i) -. reference.(i)) in
  check ~n (Array.length x)
    ~ok:(fun tau i -> d i <= tau *. spread.(i))
    ~ratio:(fun tau i -> ratio (d i) (tau *. spread.(i)))
    ~value:(fun i -> x.(i))
    ~reference:(fun i -> reference.(i))
    ~bound:(fun i -> spread.(i))

let signs ~n ~reference ~spread x =
  same_length "signs" reference spread x;
  let differ i = x.(i) > 0. <> (reference.(i) > 0.) in
  let margin i = Float.abs reference.(i) in
  check ~n (Array.length x)
    ~ok:(fun tau i -> (not (differ i)) || margin i <= tau *. spread.(i))
    ~ratio:(fun tau i ->
      if differ i then ratio (margin i) (tau *. spread.(i)) else 0.)
    ~value:(fun i -> x.(i))
    ~reference:(fun i -> reference.(i))
    ~bound:(fun i -> spread.(i))

let levels ~n ~edges ~reference ~spread k =
  same_length "levels" reference spread (Array.map float_of_int k);
  let token y =
    let c = ref 0 in
    Array.iter (fun e -> if e < y then incr c) edges;
    !c
  in
  (* The edge between tokens [a] and [a + 1] is [edges.(a)]. *)
  let margin i =
    let a = token reference.(i) and b = k.(i) in
    if a = b then 0.
    else if abs (a - b) > 1 then infinity
    else Float.abs (reference.(i) -. edges.(min a b))
  in
  check ~n (Array.length k)
    ~ok:(fun tau i -> margin i <= tau *. spread.(i))
    ~ratio:(fun tau i -> ratio (margin i) (tau *. spread.(i)))
    ~value:(fun i -> float_of_int k.(i))
    ~reference:(fun i -> reference.(i))
    ~bound:(fun i -> spread.(i))

let pp name ppf c =
  Format.fprintf ppf "%s: %d values, tau %.2f, worst %.3g, %d failing" name
    c.count c.tau c.worst (List.length c.failures);
  List.iteri
    (fun j (i, v, r, b) ->
      if j < 5 then
        Format.fprintf ppf "@\n  %d: %.9g, reference %.9g, tau s %.3g" i v r b)
    c.failures

let holds name c = if c.failures <> [] then Windtrap.failf "%a" (pp name) c
