(* The law of launches: a schedule of random programs of views of buffers,
   each interpreted element by element, leaves in memory what its graph
   states, whichever order each launch runs its members in, and no two
   members of a launch write one element.

   A program is a list of writes of float32 buffers, each a store through a
   view of a buffer of a value read through views of buffers (the graph's
   states, so a later write does not change an earlier value): runs,
   strided columns, reversed runs, runs from a bound variable, bits read
   through a bitcast, stores of one write marked disjoint where they are
   made so, and stores of one write that may overlap, landing in the order
   listed.

   Its oracle (spec) applies the graph's writes one by one: each write
   computes its stores from the states named, then lands them. *)

open Windtrap
open Tolk

let dt = Dtype.Float32
let int_ i = Ops.Int i
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Programs *)

type view =
  | Whole
  | Run of int * int
  | Rows of int * int * int * int
  | Cols of int * int * int * int
  | Rev of int * int
  | Bound of int * int
  | Bit of int * int
  | Bytes of int * int

let pp_view ppf = function
  | Whole -> Format.fprintf ppf "whole"
  | Run (a, b) -> Format.fprintf ppf "run[%d,%d)" a b
  | Rows (h, w, a, b) -> Format.fprintf ppf "rows(%dx%d)[%d,%d)" h w a b
  | Cols (h, w, a, b) -> Format.fprintf ppf "cols(%dx%d)[%d,%d)" h w a b
  | Rev (a, b) -> Format.fprintf ppf "rev[%d,%d)" a b
  | Bound (a, b) -> Format.fprintf ppf "bound[%d,%d)" a b
  | Bit (a, b) -> Format.fprintf ppf "bit[%d,%d)" a b
  | Bytes (a, b) -> Format.fprintf ppf "bytes[%d,%d)" a b

let dims_of n = function
  | Whole -> [ n ]
  | Run (a, b) | Rev (a, b) | Bound (a, b) | Bit (a, b) | Bytes (a, b) ->
      [ b - a ]
  | Rows (_, w, a, b) -> [ b - a; w ]
  | Cols (h, _, a, b) -> [ h; b - a ]

let numel n v = List.fold_left ( * ) 1 (dims_of n v)

(* The view [v] of the state [st] of a buffer of [n] elements. *)
let apply_view st n v =
  match v with
  | Whole -> st
  | Run (a, b) -> Shape.shrink st [ Some (int_ a, int_ b) ]
  | Rev (a, b) -> Shape.flip (Shape.shrink st [ Some (int_ a, int_ b) ]) [ 0 ]
  | Bound (a, b) ->
      let start =
        Shape.bind
          (Ops.variable ("start" ^ string_of_int (b - a)) (`Int Bigint.zero)
             (`Int (Bigint.of_int n)))
          (`Int (Bigint.of_int a))
      in
      let len = Ops.int (b - a) in
      Shape.shrink st [ Some (Ops.Sym start, Ops.Sym Ops.O.(start + len)) ]
  | Rows (h, w, a, b) ->
      Shape.shrink
        (Shape.reshape st [ int_ h; int_ w ])
        [ Some (int_ a, int_ b); None ]
  | Cols (h, w, a, b) ->
      Shape.shrink
        (Shape.reshape st [ int_ h; int_ w ])
        [ None; Some (int_ a, int_ b) ]
  | Bit (a, b) ->
      Ops.bitcast
        (Ops.bitcast (Shape.shrink st [ Some (int_ a, int_ b) ]) Dtype.Int32)
        Dtype.Float32
  | Bytes (a, b) ->
      Ops.bitcast
        (Shape.shrink (Ops.bitcast st Dtype.Uint8)
           [ Some (int_ (a * 4), int_ (b * 4)) ])
        Dtype.Float32

let indices n v =
  match v with
  | Whole -> Array.init n Fun.id
  | Run (a, b) | Bound (a, b) | Bit (a, b) | Bytes (a, b) ->
      Array.init (b - a) (fun i -> a + i)
  | Rev (a, b) -> Array.init (b - a) (fun i -> b - 1 - i)
  | Rows (_, w, a, b) ->
      Array.init ((b - a) * w) (fun k -> ((a + (k / w)) * w) + (k mod w))
  | Cols (h, w, a, b) ->
      Array.init (h * (b - a)) (fun k ->
          ((k / (b - a)) * w) + a + (k mod (b - a)))

type read = { buf : int; older : int; rv : view }
type value = { reads : read list; ops : (int * float) list; const : bool }
type store = { dst : int; view : view; v : value }

type step =
  | Assign of store
  | Assemble of int * store list  (* marked disjoint *)
  | Crossed of store list * store list  (* two buffers, marked disjoint *)
  | Overlap of int * store list  (* unmarked, may overlap *)

type program = { sizes : int array; steps : step list }

let pp_read ppf r =
  Format.fprintf ppf "b%d%s:%a" r.buf
    (if r.older > 0 then Printf.sprintf "@-%d" r.older else "")
    pp_view r.rv

let pp_store ppf s =
  Format.fprintf ppf "b%d:%a <- %a" s.dst pp_view s.view
    (fun ppf v ->
      if v.const then Format.fprintf ppf "const"
      else
        Format.pp_print_list
          ~pp_sep:(fun p () -> Format.fprintf p " + ")
          pp_read ppf v.reads)
    s.v

let pp_step ppf =
  let stores =
    Format.pp_print_list ~pp_sep:(fun p () -> Format.fprintf p "; ") pp_store
  in
  function
  | Assign s -> Format.fprintf ppf "assign %a" pp_store s
  | Assemble (b, ss) -> Format.fprintf ppf "assemble b%d {%a}" b stores ss
  | Overlap (b, ss) -> Format.fprintf ppf "overlap b%d {%a}" b stores ss
  | Crossed (a, b) -> Format.fprintf ppf "crossed {%a}/{%a}" stores a stores b

let pp_program ppf p =
  Format.fprintf ppf "sizes [%s]@.  %a"
    (String.concat ";" (Array.to_list (Array.map string_of_int p.sizes)))
    (Format.pp_print_list ~pp_sep:(fun p () -> Format.fprintf p "@.  ") pp_step)
    p.steps

(* Generators: a seed's random choices *)

let pick r l = List.nth l (Random.State.int r (List.length l))
let int_in r lo hi = lo + Random.State.int r (hi - lo + 1)

let factorings n =
  List.filter_map
    (fun h -> if n mod h = 0 then Some (h, n / h) else None)
    (List.init n (fun i -> i + 1))

let gen_view r n =
  let run () =
    let a = int_in r 0 (n - 1) in
    Run (a, int_in r (a + 1) n)
  in
  match Random.State.int r 8 with
  | 0 -> Whole
  | 1 | 2 -> run ()
  | 3 ->
      let a = int_in r 0 (n - 1) in
      Rev (a, int_in r (a + 1) n)
  | 4 ->
      let a = int_in r 0 (n - 1) in
      let b = int_in r (a + 1) n in
      if Random.State.bool r then Bit (a, b) else Bytes (a, b)
  | k -> (
      match List.filter (fun (h, w) -> h > 1 && w > 1) (factorings n) with
      | [] -> run ()
      | fs ->
          let h, w = pick r fs in
          if k = 5 then
            let a = int_in r 0 (h - 1) in
            Rows (h, w, a, int_in r (a + 1) h)
          else
            let a = int_in r 0 (w - 1) in
            Cols (h, w, a, int_in r (a + 1) w))

(* A view of a buffer of [n] elements holding [m] of them, if one exists. *)
let view_of_numel r n m =
  if m > n then None
  else if m = n && Random.State.int r 4 = 0 then Some Whole
  else
    let a = int_in r 0 (n - m) in
    Some
      (match Random.State.int r 5 with
      | 0 -> Rev (a, a + m)
      | 1 -> Bit (a, a + m)
      | 2 -> Bytes (a, a + m)
      | _ -> Run (a, a + m))

(* The reads of a store of [m] elements: any buffer but [excluding], the
   targets of a write marked disjoint, whose promise stops its stores reading
   the elements another writes, through a view of the buffer's latest state
   or one up to two writes older. *)
let gen_value r sizes ~excluding m =
  if Random.State.int r 7 = 0 then { reads = []; ops = []; const = true }
  else
    let usable =
      List.filter
        (fun b -> not (List.mem b excluding))
        (List.init (Array.length sizes) Fun.id)
    in
    let reads =
      List.filter_map
        (fun _ ->
          if usable = [] then None
          else
            let b = pick r usable in
            Option.map
              (fun rv -> { buf = b; older = int_in r 0 2; rv })
              (view_of_numel r sizes.(b) m))
        (List.init (int_in r 1 2) Fun.id)
    in
    if reads = [] then { reads = []; ops = []; const = true }
    else
      {
        reads;
        ops =
          List.init (int_in r 0 2) (fun _ ->
              (Random.State.int r 3, Float.of_int (int_in r 2 7)));
        const = false;
      }

let gen_store r sizes b =
  let view = gen_view r sizes.(b) in
  let v = gen_value r sizes ~excluding:[] (numel sizes.(b) view) in
  { dst = b; view; v }

let gen_assembly r sizes b =
  let n = sizes.(b) in
  let pairs l =
    let rec go = function a :: (c :: _ as t) -> (a, c) :: go t | _ -> [] in
    go l
  in
  let cuts n k =
    List.sort_uniq compare
      (0 :: n :: List.init (k - 1) (fun _ -> int_in r 1 (n - 1)))
  in
  let fs = List.filter (fun (h, w) -> h > 1 && w > 1) (factorings n) in
  let views =
    if n < 2 then []
    else if fs = [] || Random.State.int r 3 = 0 then
      let bound = Random.State.int r 3 = 0 in
      List.map
        (fun (a, c) -> if bound then Bound (a, c) else Run (a, c))
        (pairs (cuts n (int_in r 2 4)))
    else
      let h, w = pick r fs in
      if Random.State.bool r then
        List.map (fun (a, c) -> Rows (h, w, a, c)) (pairs (cuts h (int_in r 2 4)))
      else
        List.map (fun (a, c) -> Cols (h, w, a, c)) (pairs (cuts w (int_in r 2 4)))
  in
  if List.compare_length_with views 2 < 0 then Assign (gen_store r sizes b)
  else
    Assemble
      ( b,
        List.map
          (fun view ->
            let v = gen_value r sizes ~excluding:[ b ] (numel n view) in
            { dst = b; view; v })
          views )

(* Two columns of two equal buffers, stored crosswise from shared values. *)
let gen_crossed r sizes ~first_target =
  let buffers = List.init (Array.length sizes) Fun.id in
  let targets = List.filter (fun b -> b >= first_target) buffers in
  let same =
    List.concat_map
      (fun i ->
        List.filter_map
          (fun j ->
            if i < j && sizes.(i) = sizes.(j) && sizes.(i) mod 2 = 0 then
              Some (i, j)
            else None)
          targets)
      targets
  in
  if same = [] then None
  else
    let i, j = pick r same in
    let n = sizes.(i) in
    let h = n / 2 in
    let view c = Cols (h, 2, c, c + 1) in
    let shared () = gen_value r sizes ~excluding:[ i; j ] h in
    let v0 = shared () and v1 = shared () in
    Some
      (Crossed
         ( [
             { dst = i; view = view 0; v = v0 };
             { dst = i; view = view 1; v = v1 };
           ],
           if Random.State.bool r then
             [
               { dst = j; view = view 0; v = v1 };
               { dst = j; view = view 1; v = v0 };
             ]
           else
             [
               { dst = j; view = view 0; v = v0 };
               { dst = j; view = view 1; v = v1 };
             ] ))

let gen_overlap r sizes b =
  let n = sizes.(b) in
  let fs = List.filter (fun (h, w) -> h > 1 && w > 1) (factorings n) in
  let view () =
    if fs = [] || Random.State.int r 3 = 0 then
      let a = int_in r 0 (n - 1) in
      Run (a, int_in r (a + 1) n)
    else
      let h, w = pick r fs in
      if Random.State.bool r then
        let a = int_in r 0 (h - 1) in
        Rows (h, w, a, int_in r (a + 1) h)
      else
        let a = int_in r 0 (w - 1) in
        Cols (h, w, a, int_in r (a + 1) w)
  in
  Overlap
    ( b,
      List.init (int_in r 2 3) (fun _ ->
          let view = view () in
          let v = gen_value r sizes ~excluding:[] (numel n view) in
          { dst = b; view; v }) )

let gen_program r =
  let n_bufs = int_in r 3 6 in
  let sizes = Array.init n_bufs (fun _ -> pick r [ 1; 4; 6; 8; 12; 16 ]) in
  let first_target = n_bufs / 2 in
  (* Each buffer is written once and whole first, so that any later read of
     it holds what the program wrote. *)
  let first =
    List.init n_bufs (fun b ->
        Assign
          {
            dst = b;
            view = Whole;
            v = { reads = []; ops = []; const = true };
          })
  in
  let steps =
    List.init (int_in r 2 8) (fun _ ->
        let b = int_in r first_target (n_bufs - 1) in
        match Random.State.int r 8 with
        | 0 | 1 | 2 -> Assign (gen_store r sizes b)
        | 3 | 4 -> gen_assembly r sizes b
        | 5 -> gen_overlap r sizes b
        | _ -> (
            match gen_crossed r sizes ~first_target with
            | Some s -> s
            | None -> gen_assembly r sizes b))
  in
  { sizes; steps = first @ steps }

(* The value of a store's elements, of its view's shape. *)

let build ?slots p =
  let device = "CPU" in
  (* Storage takes a slot from the counter: a slot given by hand could be one
     a schedule's own storage takes, and hash-consing would make them one.
     [slots] are those of an earlier build, which are the counter's. *)
  let slot i = Option.map (fun s -> s.(i)) slots in
  let base =
    Array.init (Array.length p.sizes) (fun i ->
        Ops.new_buffer ?slot:(slot i) device p.sizes.(i) dt)
  in
  let states = Array.map (fun b -> [ b ]) base in
  let latest b = List.hd states.(b) in
  (* The state a read names: [older] writes before the latest, or the oldest
     there is. *)
  let state_at (r : read) =
    let st = states.(r.buf) in
    List.nth st (min r.older (List.length st - 1))
  in
  let value s dims =
    let broadcast c =
      Shape.expand
        (Shape.reshape (Ops.cast c dt) (List.map (fun _ -> int_ 1) dims))
        (List.map int_ dims)
    in
    if s.v.const then broadcast (Ops.float 3.5)
    else
      let read (r : read) =
        Shape.reshape
          (apply_view (state_at r) p.sizes.(r.buf) r.rv)
          (List.map int_ dims)
      in
      let x =
        match List.map read s.v.reads with
        | [] -> assert false
        | r :: rs -> List.fold_left Ops.add r rs
      in
      List.fold_left
        (fun x (k, c) ->
          let c = broadcast (Ops.float c) in
          match k with
          | 0 -> Ops.add x c
          | 1 -> Ops.mul x c
          | _ -> Ops.maximum x c)
        x s.v.ops
  in
  let store_node s =
    Ops.store
      (apply_view (latest s.dst) p.sizes.(s.dst) s.view)
      (Ops.cast (value s (dims_of p.sizes.(s.dst) s.view)) dt)
  in
  let write after b stores = states.(b) <- after (latest b) stores :: states.(b) in
  let written = Hashtbl.create 8 in
  let write ?(after = Ops.after) b stores =
    Hashtbl.replace written b ();
    write after b stores
  in
  List.iter
    (function
      | Assign s -> write s.dst [ store_node s ]
      | Overlap (b, ss) -> write b (List.map store_node ss)
      | Assemble (b, ss) ->
          write ~after:Ops.after_disjoint b (List.map store_node ss)
      | Crossed (a, b) ->
          let bi = (List.hd a).dst and bj = (List.hd b).dst in
          let sa = List.map store_node a and sb = List.map store_node b in
          write ~after:Ops.after_disjoint bi sa;
          write ~after:Ops.after_disjoint bj sb)
    p.steps;
  (* Listed by buffer: the listing is part of the program, so it may not
     depend on node ids, which depend on the nodes that outlived earlier
     programs. *)
  let bufs = Hashtbl.fold (fun b () acc -> b :: acc) written [] in
  (Ops.sink (List.map latest (List.sort Int.compare bufs)), base)

(* The spec: each write's values computed from the state before it. *)

let initial s j = f32 (float_of_int (((j * 7) + (s * 13) + 3) mod 11) /. 4.)

let spec p =
  let memory =
    Array.init (Array.length p.sizes) (fun s ->
        Array.init p.sizes.(s) (initial s))
  in
  (* Each buffer's states, the latest first. *)
  let states = Array.map (fun m -> [ Array.copy m ]) memory in
  let read (r : read) =
    let st = states.(r.buf) in
    let m = List.nth st (min r.older (List.length st - 1)) in
    Array.map (fun i -> m.(i)) (indices p.sizes.(r.buf) r.rv)
  in
  let value s =
    let n = numel p.sizes.(s.dst) s.view in
    if s.v.const then Array.make n (f32 3.5)
    else
      let sum =
        match List.map read s.v.reads with
        | [] -> assert false
        | r :: rs ->
            List.fold_left
              (fun a b -> Array.mapi (fun i x -> f32 (x +. b.(i))) a)
              r rs
      in
      Array.map
        (fun x ->
          List.fold_left
            (fun x (k, c) ->
              match k with
              | 0 -> f32 (x +. c)
              | 1 -> f32 (x *. c)
              | _ -> Float.max x c)
            x s.v.ops)
        sum
  in
  let apply ss =
    let vs = List.map (fun s -> (s, value s)) ss in
    List.iter
      (fun (s, v) ->
        Array.iteri
          (fun k i -> memory.(s.dst).(i) <- v.(k))
          (indices p.sizes.(s.dst) s.view))
      vs;
    List.iter
      (fun b -> states.(b) <- Array.copy memory.(b) :: states.(b))
      (List.sort_uniq Int.compare (List.map (fun s -> s.dst) ss))
  in
  List.iter
    (function
      | Assign s -> apply [ s ]
      | Assemble (_, ss) | Overlap (_, ss) -> apply ss
      | Crossed (a, b) -> apply (a @ b))
    p.steps;
  memory

(* Interpreting a schedule: memory keyed by storage node, each launch run
   forwards and backwards from the same memory. *)

type place = { node : Ops.t; off : int; len : int }

let sint_value ~vars = function
  | Ops.Int i -> i
  | Sym s -> (
      match Interpreter.eval ~vars s with
      | `Int z -> Bigint.to_int z
      | `Float x -> int_of_float x
      | _ -> failwith "a view of a non-integer offset")

let rec place_of ~vars u =
  match (Ops.op u, Ops.arg u) with
  | Op.After, _ | Op.Bitcast, _ -> place_of ~vars (Ops.nth u 0)
  | Op.Shrink, _ ->
      Option.map
        (fun pl ->
          match Shape.marg u with
          | Shrink [ (o, n) ] ->
              {
                pl with
                off = pl.off + sint_value ~vars o;
                len = sint_value ~vars n;
              }
          | _ -> failwith "a view of rank above 1 as a call's argument")
        (place_of ~vars (Ops.nth u 0))
  | (Param | Buffer | Alloc), Param p when p.addrspace <> Dtype.Alu ->
      Some { node = u; off = 0; len = Shape.max_numel u }
  | _ -> None

let store_through_two = "cannot evaluate a store through more than one index"

let is_kernel_call u = Ops.op u = Call && Ops.op (Ops.nth u 0) = Sink
let untouched = `Float Float.nan

(* A value by its bits, so that NaNs and zeros compare by them. *)
let bits : Dtype.value -> string = function
  | `Float x -> Printf.sprintf "%Lx" (Int64.bits_of_float x)
  | `Bool b -> string_of_bool b
  | v -> Bigint.to_string (Dtype.Value.to_z v)

let run_linear ~base ~vars linear =
  let memory = Ops.Tbl.create 16 in
  let values_of pl =
    match Ops.Tbl.find_opt memory pl.node with
    | Some v -> v
    | None ->
        let size = Shape.max_numel pl.node in
        let v =
          let of_slot =
            List.find_index (fun b -> b == pl.node) (Array.to_list base)
          in
          match of_slot with
          | Some s ->
              Array.init size (fun j -> `Float (initial s j) : int -> Dtype.value)
          | None -> Array.make size untouched
        in
        Ops.Tbl.replace memory pl.node v;
        v
  in
  let writes_of call =
    if not (is_kernel_call call) then failwith "a call of no kernel";
    let args = List.tl (Ops.src call) in
    let buffers, params =
      List.fold_left
        (fun (bs, ps) (k, a) ->
          match place_of ~vars a with
          | Some pl -> ((k, Array.sub (values_of pl) pl.off pl.len) :: bs, ps)
          | None -> (
              match Ops.arg a with
              | Param { bound = Some v; _ } -> (bs, (k, v) :: ps)
              | Param { name = Some n; _ } when List.mem_assoc n vars ->
                  (bs, (k, List.assoc n vars) :: ps)
              | _ -> (
                  match Shape.ssimplify a with
                  | Int n -> (bs, (k, Dtype.Value.of_int n) :: ps)
                  | Sym _ -> failwith "an argument of no value")))
        ([], [])
        (List.mapi (fun k a -> (k, a)) args)
    in
    List.map
      (fun (k, i, v) ->
        let a = List.nth args k in
        match place_of ~vars a with
        | Some pl -> (pl, i, v)
        | None -> failwith "a write through no storage")
      (Interpreter.writes ~vars ~params ~buffers (Ops.nth call 0))
  in
  let apply (pl, i, v) = (values_of pl).(pl.off + i) <- v in
  let snapshot () =
    Ops.Tbl.fold (fun k v acc -> (k, Array.copy v) :: acc) memory []
  in
  let restore saved =
    let live = Ops.Tbl.fold (fun k _ acc -> k :: acc) memory [] in
    List.iter
      (fun k -> if not (List.mem_assq k saved) then Ops.Tbl.remove memory k)
      live;
    List.iter (fun (k, v) -> Ops.Tbl.replace memory k (Array.copy v)) saved
  in
  let launches = ref 0 in
  let run e =
    if Ops.op e = Call && Shape.is_launch (Ops.nth e 0) then begin
      incr launches;
      let members = Shape.launch_members e in
      let per = List.map writes_of members in
      let seen = Hashtbl.create 64 in
      List.iteri
        (fun m ws ->
          List.iter
            (fun (pl, i, _) ->
              let k = (Ops.hash pl.node, pl.off + i) in
              (match Hashtbl.find_opt seen k with
              | Some m' when m' <> m ->
                  failf "two members of a launch write element %d" (pl.off + i)
              | _ -> ());
              Hashtbl.replace seen k m)
            ws)
        per;
      let saved = snapshot () in
      let run_in order =
        restore saved;
        List.iter
          (fun m -> List.iter apply (writes_of m))
          order;
        Ops.Tbl.fold
          (fun k v acc -> (Ops.hash k, Array.map bits v) :: acc)
          memory []
      in
      let forwards = run_in members in
      let backwards = run_in (List.rev members) in
      equal bool ~msg:"members of a launch depend on their order" true
        (List.sort compare forwards = List.sort compare backwards)
    end
    else List.iter apply (writes_of e)
  in
  List.iter run (Ops.src linear);
  (memory, !launches)

(* The law *)

let order_independent seed =
  let r = Random.State.make [| seed |] in
  let p = gen_program r in
  let big, base = build p in
  cover "an assembly" (List.exists (function Assemble _ -> true | _ -> false) p.steps);
  cover "a crossed write"
    (List.exists (function Crossed _ -> true | _ -> false) p.steps);
  (* Rangeify refuses some graphs of views at verification: the schedule makes
     no statement about them. *)
  let linear, vars =
    match Schedule.create_linear_with_vars ~capturing:true big with
    | r -> r
    | exception Invalid_argument m
      when String.starts_with ~prefix:"UOp verification failed" m ->
        reject ()
  in
  let vars = List.map (fun (n, v) -> (n, `Int (Bigint.of_int v))) vars in
  (* A store through a bytes bitcast writes through two indices, which the
     interpreter does not read. *)
  let memory, launches =
    match run_linear ~base ~vars linear with
    | r -> r
    | exception Invalid_argument m when m = store_through_two -> reject ()
  in
  cover "a launch" (launches > 0);
  cover "an unmarked overlap"
    (List.exists (function Overlap _ -> true | _ -> false) p.steps);
  let expected = spec p in
  let msg = Format.asprintf "seed %d: %a" seed pp_program p in
  Array.iteri
    (fun s b ->
      let got =
        match Ops.Tbl.find_opt memory b with
        | Some v -> Array.to_list (Array.map bits v)
        | None ->
            Array.to_list
              (Array.map
                 (fun x -> bits (`Float x))
                 (Array.init p.sizes.(s) (initial s)))
      in
      equal (list string) ~msg
        (Array.to_list
           (Array.map (fun x -> bits (`Float x)) expected.(s)))
        got)
    base

let count_launches linear =
  List.length
    (List.filter
       (fun e -> Ops.op e = Call && Shape.is_launch (Ops.nth e 0))
       (Ops.src linear))

(* Process history: nodes are interned weakly, so a node that outlived an
   earlier build of a program keeps its id and one collected is made anew,
   younger. Node ids order nodes by that history alone, and no schedule may
   read it. *)

(* A schedule printed with the storage it made numbered by first use, above
   [slots], the program's own: a made slot is a process-wide counter's. The
   program's keep theirs, so that a swap of two of its calls shows. *)
let schedule_text ~slots (linear, vars) =
  let made u =
    match Ops.op u with
    | Alloc -> true
    | Buffer -> not (Array.mem (Ops.param_arg_of u).slot slots)
    | _ -> false
  in
  let storage = List.filter made (Ops.toposort ~calls:Enter linear) in
  let above = 1 + Array.fold_left max (-1) slots in
  let renumber i u =
    let p = Ops.param_arg_of u in
    (u, Ops.replace u ~arg:(Param { p with slot = above + i }))
  in
  let renumbered =
    Ops.substitute ~calls:Enter ~pass:Once linear (List.mapi renumber storage)
  in
  Format.asprintf "%a@.%s" Ops.pp renumbered
    (String.concat ","
       (List.map (fun (n, v) -> Printf.sprintf "%s=%d" n v) vars))

(* Whether each pair of [nodes] is ordered as [Ops.compare] orders them. *)
let id_order nodes =
  List.concat_map (fun a -> List.map (Ops.compare a) nodes) nodes

let uncached f = Setting.context [ B (Setting.scache, 0) ] f

let same_program_same_schedule seed =
  let r = Random.State.make [| seed |] in
  let p = gen_program r in
  let schedule big =
    match Schedule.create_linear_with_vars big with
    | r -> r
    | exception Invalid_argument m
      when String.starts_with ~prefix:"UOp verification failed" m ->
        reject ()
  in
  uncached @@ fun () ->
  (* In a function of its own, so that only [kept] outlives it. *)
  let first () =
    let big, base = build p in
    let slots = Array.map (fun b -> (Ops.param_arg_of b).slot) base in
    let kept = List.filter (fun _ -> Random.State.bool r) (Ops.src big) in
    (schedule_text ~slots (schedule big), slots, kept, id_order (Ops.src big))
  in
  let text, slots, kept, order = first () in
  Gc.full_major ();
  let big, _ = build ~slots p in
  cover "a node older in one build, younger in the other"
    (id_order (Ops.src big) <> order);
  let second = schedule big in
  cover "a launch" (count_launches (fst second) > 0);
  let msg = Format.asprintf "seed %d: %a" seed pp_program p in
  equal string ~msg text (schedule_text ~slots second);
  ignore (Sys.opaque_identity kept)

(* A store of one element and a store of many into other storage. The one
   element joins a launch of the many, and the many open none of its own: so
   the listing decides, and the order the nodes were built in does not. *)
let listing_decides () =
  let launches ~built_first ~listed_first =
    uncached @@ fun () ->
    let store n =
      let b = Ops.new_buffer "CPU" n dt in
      let one = Shape.reshape (Ops.cast (Ops.float 3.5) dt) [ int_ 1 ] in
      Ops.after b [ Ops.store b (Shape.expand one [ int_ n ]) ]
    in
    let many, single =
      if built_first = `Many then
        let m = store 4 in
        (m, store 1)
      else
        let s = store 1 in
        (store 4, s)
    in
    let listing =
      if listed_first = `Many then [ many; single ] else [ single; many ]
    in
    let linear, _ =
      Schedule.create_linear_with_vars ~capturing:true (Ops.sink listing)
    in
    count_launches linear
  in
  let launches ~listed_first =
    let a = launches ~built_first:`Many ~listed_first in
    Gc.full_major ();
    equal int ~msg:"built in the other order" a
      (launches ~built_first:`One ~listed_first);
    a
  in
  equal int ~msg:"the many first" 1 (launches ~listed_first:`Many);
  equal int ~msg:"the one first" 0 (launches ~listed_first:`One)

(* The mark: an assembly of columns, which no byte range separates, runs as
   one launch only because it is marked; unmarked, it runs apart. *)

let cat ~marked =
  let device = "CPU" in
  let out = Ops.new_buffer ~slot:0 device 16 dt in
  let columns = Shape.reshape out [ int_ 4; int_ 4 ] in
  let piece c slot =
    let b = Ops.new_buffer ~slot device 4 dt in
    Ops.store
      (Shape.shrink columns [ None; Some (int_ c, int_ (c + 1)) ])
      (Shape.reshape b [ int_ 4; int_ 1 ])
  in
  let stores = List.init 4 (fun c -> piece c (c + 1)) in
  Ops.sink [ (if marked then Ops.after_disjoint else Ops.after) out stores ]

let marked_columns_launch () =
  let launches ~marked =
    count_launches
      (fst (Schedule.create_linear_with_vars ~capturing:true (cat ~marked)))
  in
  equal int ~msg:"marked" 1 (launches ~marked:true);
  equal int ~msg:"unmarked" 0 (launches ~marked:false)

(* Two calls of functions compiled on their own write one buffer in turn,
   each through a write marked disjoint: the promise covers the stores of one
   write, so the kernels of the second call join no launch of the first's. *)
let calls_in_turn () =
  let device = "CPU" in
  let x = Ops.new_buffer ~slot:0 device 4 dt in
  let filled c =
    let p = Shape.param ~shape:[ int_ 4 ] ~device 0 dt in
    let at a b = Shape.shrink p [ Some (int_ a, int_ b) ] in
    let value n =
      Shape.expand
        (Shape.reshape (Ops.cast (Ops.float c) dt) [ int_ 1 ])
        [ int_ n ]
    in
    Ops.sink
      [
        Ops.after_disjoint p
          [ Ops.store (at 0 2) (value 2); Ops.store (at 2 4) (value 2) ];
      ]
  in
  let first = Ops.call ~precompile:true (filled 1.) [ x ] in
  let x1 = Ops.after x [ first ] in
  let second = Ops.call ~precompile:true (filled 2.) [ x1 ] in
  let big = Ops.sink [ Ops.after x1 [ second ] ] in
  let linear, _ = Schedule.create_linear_with_vars ~capturing:true big in
  equal int ~msg:"a launch a call" 2 (count_launches linear)

(* A write of no stores states nothing of the stores before it. *)
let no_stores_no_mark () =
  let x = Ops.new_buffer ~slot:0 "CPU" 4 dt in
  let written = Ops.after x [ Ops.store x (Ops.float 1.) ] in
  equal bool false (Ops.is_disjoint (Ops.after_disjoint written []))

(* The mark's tag reaches the launches through a schedule read back from the
   disk: a child made it, and another reads it. *)

let launches_of_cat () =
  print_endline
    ("launches "
    ^ string_of_int
        (count_launches
           (fst
              (Schedule.create_linear_with_vars ~capturing:true
                 (cat ~marked:true)))))

let launches_read db =
  match Disk_cache.child ~env:[ ("DEBUG", "3") ] ~cachedb:db "launches" with
  | Error err -> failf "the child failed: %s" err
  | Ok out -> (
      let lines = String.split_on_char '\n' out in
      let line = List.find (String.starts_with ~prefix:"launches ") lines in
      let hit =
        List.exists
          (fun l ->
            String.starts_with ~prefix:"scheduled " l
            && List.mem "hit" (String.split_on_char ' ' l))
          lines
      in
      ( hit,
        int_of_string
          (String.sub line 9 (String.length line - 9)) ))

let disk_keeps_mark () =
  let db = Disk_cache.fresh () in
  equal (pair bool int) ~msg:"made" (false, 1) (launches_read db);
  equal (pair bool int) ~msg:"read back" (true, 1) (launches_read db)

let () =
  Disk_cache.play [ ("launches", launches_of_cat) ];
  exit @@ run "Tolk.Launch_law"
    [
      group "launches"
        [
          prop ~count:1500
            "a schedule's launches write what its graph states, in either \
             order of their members"
            (Gen.int_range 1 1_000_000) order_independent;
          prop
            "a schedule does not depend on which nodes outlived an earlier \
             build"
            (Gen.int_range 1 1_000_000) same_program_same_schedule;
          test "the listing places a one-element store beside a store of many"
            listing_decides;
          test "a write marked disjoint runs its columns as one launch"
            marked_columns_launch;
          test "a schedule read back keeps a marked write's launch"
            disk_keeps_mark;
          test "marked writes of two calls run in turn" calls_in_turn;
          test "a marked write of no stores marks nothing" no_stores_no_mark;
        ];
    ]
