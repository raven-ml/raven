(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* talon under Rune.jit, Rune.vmap and Rune.grad. Under a trace, a step whose
   sizes follow from the sizes of its input computes what it computes eagerly,
   and one whose sizes depend on values raises at the trace. Under a derivative
   every step runs. *)

open Talon
open Windtrap

let f64 = Nx.float64

(* Frames

   Every case is a table over five arrays of one length:
   - "a" is the floats [x], null where [valid] is false; "k" is [x], a key that
     holds NaN, [-0.] and [0.]; "b" is [-x]. The floats are small dyadics and
     the special values, so a sum is exact in any order.
   - "i" is the [int32]s [is], their extremes included, "t" the booleans
     [flags] and "d" the dates of the days [ds], each null where [valid] is
     false at the row before, the first row reading the last. *)

let floats = [ -0.; 0.; 1.; -1.; 2.5; 3.; Float.nan; infinity; neg_infinity ]
let ints = [ Int32.min_int; -1l; 0l; 1l; 7l; Int32.max_int ]
let days = [ Int32.min_int; -1l; 0l; 1l; 20_000l; Int32.max_int ]
let lengths = [ 0; 1; 2; 3; 7 ]

type args = {
  x : (float, Nx.float64_elt, Nx.host) Nx.t;
  valid : Nx.host Nx.bool_t;
  is : (int32, Nx.int32_elt, Nx.host) Nx.t;
  flags : Nx.host Nx.bool_t;
  ds : (int32, Nx.int32_elt, Nx.host) Nx.t;
}

let args =
  Nx.Ptree.v (fun c a ->
      let open Nx.Ptree.Walk in
      let x = field c "x" tensor a.x in
      let valid = field c "valid" tensor a.valid in
      let is = field c "is" tensor a.is in
      let flags = field c "flags" tensor a.flags in
      let ds = field c "ds" tensor a.ds in
      { x; valid; is; flags; ds })

let signature () = Nx.Ptree.(args @-> returns tensor)

(* [rolled v] is [v] one row down, its last row first. *)
let rolled v =
  let n = Nx.dim 0 v in
  if n = 0 then v
  else
    Nx.concatenate ~axis:0
      [ Nx.slice [ Nx.R (n - 1, n) ] v; Nx.slice [ Nx.R (0, n - 1) ] v ]

let table a =
  let bits = Nx.cast Nx.bit a.valid in
  let rolled = Nx.cast Nx.bit (rolled a.valid) in
  let date =
    Column.of_layout (Type.Any Type.date)
      (Fixed { validity = Some rolled; values = Nx.P a.ds })
  in
  Talon.v
    [
      ("a", Column.of_tensor ~validity:bits a.x);
      ("k", Column.of_tensor a.x);
      ("b", Column.of_tensor (Nx.neg a.x));
      ("i", Column.of_tensor ~validity:rolled a.is);
      ("t", Column.of_tensor ~validity:rolled a.flags);
      ("d", Result.get_ok date);
    ]

type frame = {
  fx : float array;
  fvalid : bool array;
  fis : int32 array;
  fflags : bool array;
  fds : int32 array;
}

let pp_frame ppf f =
  let row ppf j =
    if f.fvalid.(j) then Format.fprintf ppf "%h" f.fx.(j)
    else Format.pp_print_string ppf "null";
    Format.fprintf ppf " %ld %b %ld" f.fis.(j) f.fflags.(j) f.fds.(j)
  in
  let sep ppf () = Format.pp_print_string ppf "; " in
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list ~pp_sep:sep row)
    (List.init (Array.length f.fx) Fun.id)

let frame_of n =
  Gen.(
    let row g = array ~size:(constant n) g in
    let+ fx = row (of_list floats)
    and+ fvalid = row (frequency [ (3, constant true); (1, constant false) ])
    and+ fis = row (of_list ints)
    and+ fflags = row bool
    and+ fds = row (of_list days) in
    { fx; fvalid; fis; fflags; fds })

let frame = Gen.with_pp pp_frame (Gen.bind (Gen.of_list lengths) frame_of)

(* Two frames of one length, the lanes of a map. *)
let lanes =
  Gen.with_pp
    (Format.pp_print_list pp_frame)
    Gen.(
      let* n = of_list lengths in
      let+ f = frame_of n and+ g = frame_of n in
      [ f; g ])

let arrays f =
  let v dt a = Nx.create dt [| Array.length a |] a in
  {
    x = v f64 f.fx;
    valid = v Nx.bool f.fvalid;
    is = v Nx.int32 f.fis;
    flags = v Nx.bool f.fflags;
    ds = v Nx.int32 f.fds;
  }

(* [stack args] is the lanes [args] of a map. *)
let stack args =
  let st get = Nx.stack ~axis:0 (List.map get args) in
  {
    x = st (fun a -> a.x);
    valid = st (fun a -> a.valid);
    is = st (fun a -> a.is);
    flags = st (fun a -> a.flags);
    ds = st (fun a -> a.ds);
  }

let cover_frame f =
  let n = Array.length f.fx in
  cover "no row" (n = 0);
  cover "one row" (n = 1);
  cover "a null" (Array.exists not f.fvalid);
  cover "a NaN" (Array.exists Float.is_nan f.fx);
  cover "an int extreme"
    (Array.exists (fun i -> i = Int32.min_int || i = Int32.max_int) f.fis)

(* Results

   A case's result is a [(rows, 2 * columns)] matrix: for each column its
   values as float64, [0.] under a null, then [1.] where it is null. Two
   results agree when they have one shape and each element has the same bits,
   every NaN being equal. *)

let run_ok q =
  match Query.run q with
  | Ok t -> t
  | Error e -> failwith (Format.asprintf "%a" Error.pp e)

let read t name =
  match Column.layout (Talon.column t name) with
  | Fixed { validity; values = Nx.P x } ->
      let n = Nx.dim 0 x in
      let held =
        match validity with
        | None -> Nx.ones Nx.bool [| n |]
        | Some v -> Nx.cast Nx.bool v
      in
      let x = Nx.cast f64 x in
      [ Nx.where held x (Nx.zeros_like x); Nx.cast f64 (Nx.logical_not held) ]
  | _ -> invalid_arg "read: a column of one element per row"

let matrix names t =
  match names with
  | [] -> Nx.zeros f64 [| Talon.rows t; 0 |]
  | names -> Nx.stack ~axis:1 (List.concat_map (read t) names)

let bits m =
  let bits v =
    if Float.is_nan v then Int64.min_int else Int64.bits_of_float v
  in
  (Nx.shape m, Array.map bits (Nx.to_array m))

let result_t =
  Testable.make ~pp:Nx.pp ~equal:(fun a b -> bits a = bits b)

(* [agrees name f] states that [f], compiled and mapped, computes what it
   computes eagerly. [f] is bound under [jit] once, so each length traces once.
*)
let agrees name f =
  let compiled = Rune.jit (signature ()) f in
  let eager fs = Nx.stack ~axis:0 (List.map (fun fr -> f (arrays fr)) fs) in
  group name
    [
      prop "jit" frame (fun fr ->
          cover_frame fr;
          let a = arrays fr in
          equal result_t (f a) (compiled a));
      prop "vmap" lanes (fun fs ->
          let stacked = stack (List.map arrays fs) in
          let mapped = Rune.vmap (signature ()) f stacked in
          equal result_t (eager fs) mapped);
    ]

(* [verb names build] runs [build] over the case's table and reads [names]. *)
let verb names build a =
  matrix names (run_ok (build (Query.of_table (table a))))

let a = Col.float "a"
let b = Col.float "b"
let all = [ "a"; "k"; "b"; "i"; "t"; "d" ]

(* [shorter q] is "r", [q]'s "b" past its first row: one row fewer. *)
let shorter q =
  Query.select Expr.[ "r" := b ] (Query.slice ~offset:1 ~length:max_int q)

(* Steps whose sizes follow from their input's *)

let static =
  let open Expr in
  [
    agrees "select"
      (verb [ "c"; "d" ]
         (Query.select [ "c" := a +. b; "d" := nx { f = Nx.abs } a ]));
    agrees "derive"
      (verb [ "c"; "e" ]
         (Query.derive
            [ "c" := if_ (a > float 1.) a b; "e" := coalesce [ a; b ] ]));
    agrees "sort on floats"
      (verb all (Query.sort [ Order.asc "a"; Order.desc "b" ]));
    agrees "sort on ints, booleans and dates"
      (verb all
         (Query.sort [ Order.desc "t"; Order.asc "d"; Order.desc "i" ]));
    agrees "sort nulls first"
      (verb all (Query.sort [ Order.nulls_first (Order.desc "i") ]));
    agrees "slice" (verb all (Query.slice ~offset:1 ~length:3));
    agrees "slice from the end" (verb all (Query.slice ~offset:(-2) ~length:3));
    agrees "append" (verb all (fun q -> Query.append q q));
    agrees "aggregate"
      (verb [ "s"; "c"; "m"; "lo"; "hi"; "f"; "l"; "med" ]
         (Query.aggregate ~by:[]
            [
              "s" := sum a;
              "c" := count a;
              "m" := mean a;
              "lo" := min a;
              "hi" := max a;
              "f" := first a;
              "l" := last a;
              "med" := median a;
            ]));
    agrees "n_unique"
      (verb [ "k"; "i"; "t"; "d" ]
         (Query.aggregate ~by:[]
            [
              "k" := n_unique (Col.float "k");
              "i" := n_unique (Col.int "i");
              "t" := n_unique (Col.bool "t");
              "d" := n_unique (Col.date "d");
            ]));
    agrees "frames"
      (verb [ "c"; "r"; "f" ]
         (Query.derive
            [
              "c" := shift 1 a;
              "r" := rank (Col.int "i");
              "f" := over ~order:[ Order.desc "d" ] (first a);
            ]));
    agrees "join by position"
      (verb [ "a"; "r" ] (fun q ->
           Query.join ~kind:Left ~on:Join.position (shorter q) q));
    agrees "join by position, counted"
      (verb [ "a"; "r" ] (fun q ->
           Query.join ~each_right:One ~on:Join.position (shorter q) q));
    agrees "join all"
      (verb [ "a"; "r" ] (fun q ->
           Query.join ~on:Join.all (Query.select [ "r" := b ] q) q));
    agrees "head and tail"
      (verb all (fun q -> Query.append (Kit.tail 2 q) (Kit.head 2 q)));
    agrees "top_k on floats" (verb all (Kit.top_k 2 [ Order.desc "a" ]));
    agrees "top_k on ints and booleans"
      (verb all (Kit.top_k 3 [ Order.desc "t"; Order.asc "i" ]));
    agrees "top_k on dates" (verb all (Kit.top_k 2 [ Order.desc "d" ]));
    agrees "take" (fun a ->
        let n = Nx.dim 0 a.x in
        let last = Nx.full Nx.int64 [| n |] (Int64.of_int (Int.pred n)) in
        let reversed = Nx.sub last (Nx.arange Nx.int64 0 n 1) in
        matrix all (Talon.take reversed (table a)));
  ]

(* Steps whose sizes depend on values *)

let fixed =
  let v dt xs = Nx.create dt [| Array.length xs |] xs in
  {
    x = v f64 [| 1.; 2.; 1. |];
    valid = v Nx.bool [| true; false; true |];
    is = v Nx.int32 [| 1l; 2l; 1l |];
    flags = v Nx.bool [| true; true; false |];
    ds = v Nx.int32 [| 0l; 1l; 0l |];
  }

(* [refuses name f] states that [f] runs eagerly and raises at the trace under
   [jit] and [vmap], naming the transformation and the read. *)
let refuses ?(read = "") name f =
  let traced by =
    Exn.invalid_arg ~sub:(read ^ "the value is traced by " ^ by)
  in
  test name (fun () ->
      ignore (f fixed);
      raises_match (traced "Rune.jit") (fun () ->
          Rune.jit (signature ()) f fixed);
      raises_match (traced "Rune.vmap") (fun () ->
          Rune.vmap (signature ()) f (stack [ fixed; fixed ])))

let dynamic =
  let open Expr in
  let keys = Query.select [ keep (Sel.names [ "k" ]) ] in
  let join kind q = Query.join ~kind ~on:(Join.keys [ "k" ]) (keys q) q in
  [
    refuses "filter" (verb [ "a" ] (Query.filter (a > float 1.)));
    refuses "aggregate by a key"
      (verb [ "s" ] (Query.aggregate ~by:[ "i" ] [ "s" := sum a ]));
    refuses "over a key"
      (verb [ "c" ] (Query.derive [ "c" := over ~by:[ "k" ] (sum a) ]));
    refuses "inner join on a key" (verb [ "a" ] (join Join.Inner));
    refuses "left join on a key" (verb [ "a" ] (join Join.Left));
    refuses "semi join on a key" (verb [ "a" ] (join Join.Semi));
    refuses "distinct" (verb [ "k" ] (fun q -> Kit.distinct (keys q)));
    refuses "an OCaml function"
      (verb [ "c" ]
         (Query.derive [ "c" := store Type.float64 (const Float.abs $ a) ]));
    refuses "a cast that may fail"
      (verb [ "c" ] (Query.derive [ "c" := cast Type.int64 a ]));
    refuses ~read:"Nx.to_array: " "values" (fun args ->
        match Query.values (Expr.option a) (Query.of_table (table args)) with
        | Ok _ -> args.x
        | Error _ -> args.x);
    refuses "equal" (fun args ->
        if Talon.equal (table args) (table args) then args.x
        else Nx.neg args.x);
    refuses "null_count" (fun args ->
        let c = Talon.column (table args) "a" in
        Nx.add_s args.x (Float.of_int (Column.null_count c)));
  ]

(* Failures that depend on values

   Under [jit] and [vmap], a failure that a value decides raises with the
   message it has eagerly, under [jit] when the call returns. *)

let alike expected f =
  let message f =
    match f () with
    | _ -> "no failure"
    | exception (Invalid_argument m | Failure m) -> m
  in
  contains ~sub:expected (message (fun () -> f fixed));
  contains ~sub:expected (message (fun () -> Rune.jit (signature ()) f fixed));
  contains ~sub:expected
    (message (fun () -> Rune.vmap (signature ()) f (stack [ fixed; fixed ])))

let checks =
  [
    test "Talon.to_tensor of a null" (fun () ->
        alike {|Talon.to_tensor: "a" has a null|} (fun args ->
            Talon.to_tensor f64 [ "a" ] (table args)));
    test "Column.to_tensor of a null" (fun () ->
        alike "Column.to_tensor: row 1 is null" (fun args ->
            Column.to_tensor f64 (Talon.column (table args) "a")));
    test "Talon.take past the rows" (fun () ->
        alike "Talon.take: index 3 of a table of 3 rows" (fun args ->
            let n = Nx.dim 0 args.x in
            let i = Nx.add_s (Nx.arange Nx.int64 0 n 1) 1L in
            Talon.to_tensor f64 [ "b" ] (Talon.take i (table args))));
    test "a count of a join by position" (fun () ->
        alike "the left row matches 0 rows, not one" (fun args ->
            let q = Query.of_table (table args) in
            let joined = Query.join ~each_left:One ~on:Join.position in
            matrix [ "a" ] (run_ok (joined (shorter q) q))));
  ]

(* Derivatives

   Under [grad] the floats keep their elements, so every step runs. The
   gradient of the sum of "a" squared, [0.] under a null, is [2 x] where "a"
   holds a value and [0.] under a null, through any step that keeps the rows,
   and only on the rows a filter keeps. *)

let finite = [ -0.; 0.; 1.; -1.; 2.5; 3. ]

let pp_case ppf (x, valid) =
  let row ppf (x, v) =
    if v then Format.fprintf ppf "%h" x else Format.fprintf ppf "%h null" x
  in
  Format.pp_print_list row ppf
    (List.combine (Array.to_list x) (Array.to_list valid))

let gradient_case =
  Gen.with_pp pp_case
  @@ Gen.(
    let* n = of_list lengths in
    let+ x = array ~size:(constant n) (of_list finite)
    and+ valid = array ~size:(constant n) bool in
    (x, valid))

let squares x valid q =
  let zeros = Nx.zeros Nx.int32 [| Nx.dim 0 x |] in
  let t = table { x; valid; is = zeros; flags = valid; ds = zeros } in
  let c = Expr.("c" := coalesce [ a *. a; float 0. ]) in
  let r = run_ok (Query.derive [ c ] (q (Query.of_table t))) in
  Nx.sum (Talon.to_tensor f64 [ "c" ] r)

(* The sign of a zero derivative is not part of the gradient. *)
let floats_t = array (float 1e-300)

let derivatives =
  let gradient q (x, valid) =
    let x = Nx.create f64 [| Array.length x |] x in
    let valid = Nx.create Nx.bool [| Array.length valid |] valid in
    Nx.to_array (Rune.grad' (fun x -> squares x valid q) x)
  in
  let twice x held = if held then 2. *. x else 0. in
  [
    prop "grad through a derive" gradient_case (fun ((x, valid) as c) ->
        equal floats_t (Array.map2 twice x valid) (gradient Fun.id c));
    prop "grad through a sort" gradient_case (fun ((x, valid) as c) ->
        let g = gradient (Query.sort [ Order.desc "b" ]) c in
        equal floats_t (Array.map2 twice x valid) g);
    prop "grad through a filter" gradient_case (fun ((x, valid) as c) ->
        let kept x held = if held && x > 0.5 then 2. *. x else 0. in
        let g = gradient (Query.filter Expr.(a > float 0.5)) c in
        equal floats_t (Array.map2 kept x valid) g);
    test "grad refuses values and display" (fun () ->
        let traced = Exn.invalid_arg ~sub:"the value is traced by Rune.grad" in
        let reads f =
          let g x =
            f (table { fixed with x });
            Nx.sum x
          in
          raises_match traced (fun () -> Rune.grad' g fixed.x)
        in
        reads (fun t -> ignore (Query.values a (Query.of_table t)));
        reads (fun t -> ignore (Format.asprintf "%a" Talon.pp t)));
    test "grad of a null raises eager's message" (fun () ->
        let f x = Talon.to_tensor f64 [ "a" ] (table { fixed with x }) in
        raises_match
          (Exn.invalid_arg ~sub:{|Talon.to_tensor: "a" has a null|})
          (fun () -> Rune.grad' (fun x -> Nx.sum (f x)) fixed.x));
  ]

let () =
  exit
    (run "Jit"
       [
         group "static sizes" static;
         group "sizes from values" dynamic;
         group "checks" checks;
         group "derivatives" derivatives;
       ])
