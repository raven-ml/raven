(* Tests of the launches of a schedule: the stores of one state run as one
   launch whose members are their kernels (Shape.launch), and write what each
   store's kernel alone writes, bit for bit, on the host and each GPU present.

   A program assembles pieces into the rows of one buffer, each piece a store
   into a run of rows of a value computed from buffers of its own: an
   elementwise value, a sum along a last axis, or a product of a matrix and a
   vector. Its schedule runs once as made and once with each launch run as
   its members, one call each; both write the same bits. *)

open Windtrap
open Tolk
module Engine = Tolk_engine

(* Devices *)

let device name = if name = "CPU" then Rig.host else Gpus.gpu name

(* A program of a type [name]'s device lacks makes no statement there. *)
let reject_unsupported name dt =
  if not (List.mem dt (Renderer.supported_dtypes (Engine.renderer (device name))))
  then reject ()

(* Programs *)

type kind = Elementwise | Sum | Matvec | Matmul

type piece = {
  kind : kind;
  rows : int;  (* the rows of the buffer it covers *)
  k : int;  (* the length of the axis it sums along *)
}

type program = {
  dtype : Dtype.t;
  width : int;  (* the elements of a row *)
  pieces : piece list;
  fill : bool;  (* whether a store of every row comes first *)
  later : bool;  (* whether a store over the first row comes after *)
  bound : bool;  (* whether each piece's first row is a bound variable *)
}

let pp_kind ppf k =
  Format.pp_print_string ppf
    (match k with
    | Elementwise -> "elementwise"
    | Sum -> "sum"
    | Matvec -> "matvec"
    | Matmul -> "matmul")

let pp_program ppf p =
  Format.fprintf ppf "%a %d wide%s%s%s: %a" Dtype.pp p.dtype p.width
    (if p.fill then ", filled" else "")
    (if p.later then ", then a row" else "")
    (if p.bound then ", bound starts" else "")
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       (fun ppf q ->
         Format.fprintf ppf "%a %d rows k %d" pp_kind q.kind q.rows q.k))
    p.pieces

let is_float dt = Dtype.is_float dt

(* A value of [dt] from a small integer: every type holds it exactly. *)
let element dt j : Dtype.value =
  let x = ((j * 7) + 3) mod 11 - 5 in
  if is_float dt then `Float (float_of_int x /. 4.)
  else if Dtype.is_unsigned dt then `Int (Bigint.of_int (x + 5))
  else `Int (Bigint.of_int x)

(* The value a piece stores, of shape [rows; width], from new buffers of
   [d]. *)
let value d dt width slot q =
  let buf n =
    incr slot;
    Ops.new_buffer ~slot:!slot d n dt
  in
  let r = q.rows in
  let shaped u s = Shape.reshape u (List.map (fun n -> Ops.Int n) s) in
  let full r w k = [ Ops.Int r; Int w; Int k ] in
  match q.kind with
  | Elementwise ->
      let x = shaped (buf (r * width)) [ r; width ] in
      if is_float dt then Ops.O.(Ops.exp2 (x * Ops.float 0.25) + Ops.float 1.)
      else Ops.O.((x * Ops.int 3) + Ops.int 1)
  | Sum ->
      let x = shaped (buf (r * width * q.k)) [ r; width; q.k ] in
      shaped (Shape.rop x Add [ 2 ]) [ r; width ]
  | Matvec ->
      let a = shaped (buf (r * width * q.k)) [ r; width; q.k ] in
      let v = shaped (buf q.k) [ 1; 1; q.k ] in
      let v = Shape.expand v (full r width q.k) in
      shaped (Shape.rop Ops.O.(a * v) Add [ 2 ]) [ r; width ]
  | Matmul ->
      let a = shaped (buf (r * q.k)) [ r; 1; q.k ] in
      let b = shaped (buf (width * q.k)) [ 1; width; q.k ] in
      let a = Shape.expand a (full r width q.k)
      and b = Shape.expand b (full r width q.k) in
      shaped (Shape.rop Ops.O.(a * b) Add [ 2 ]) [ r; width ]

let total p = List.fold_left (fun n q -> n + q.rows) 0 p.pieces

(* The program's sink, its output in slot 0. *)
(* [x] of [dt] broadcast to [shape]. *)
let broadcast x dt shape =
  let ones = List.map (fun _ -> Ops.Int 1) shape in
  Shape.expand
    (Shape.reshape (Ops.cast x dt) ones)
    (List.map (fun n -> Ops.Int n) shape)

let sink d p =
  let filled x shape = broadcast x p.dtype shape in
  let n = total p in
  let out = Ops.new_buffer ~slot:0 d (n * p.width) p.dtype in
  let slot = ref 0 in
  let state =
    if not p.fill then out
    else
      let zero = if is_float p.dtype then Ops.float 0.5 else Ops.int 2 in
      Ops.after out [ Ops.store out (filled zero [ n * p.width ]) ]
  in
  let rows u = Shape.reshape u [ Int n; Int p.width ] in
  let region u o r =
    if not p.bound then
      Shape.shrink (rows u) [ Some (Int o, Int (o + r)); None ]
    else
      let s =
        Ops.variable ("start" ^ string_of_int r) (`Int Bigint.zero)
          (`Int (Bigint.of_int n))
      in
      let start = Shape.bind s (`Int (Bigint.of_int o)) in
      Shape.shrink (rows u)
        [ Some (Ops.Sym start, Ops.Sym Ops.O.(start + Ops.int r)); None ]
  in
  let _, stores =
    List.fold_left
      (fun (o, stores) q ->
        let v = Ops.cast (value d p.dtype p.width slot q) p.dtype in
        (o + q.rows, Ops.store (region state o q.rows) v :: stores))
      (0, []) p.pieces
  in
  let state = Ops.after_disjoint state (List.rev stores) in
  let state =
    if not p.later then state
    else
      let one = if is_float p.dtype then Ops.float (-1.5) else Ops.int 7 in
      let first = Shape.shrink (rows state) [ Some (Int 0, Int 1); None ] in
      Ops.after state [ Ops.store first (filled one [ 1; p.width ]) ]
  in
  Ops.sink [ state ]

(* Running *)

let launches linear =
  List.filter
    (fun e -> Ops.op e = Call && Shape.is_launch (Ops.nth e 0))
    (Ops.src linear)

(* [linear] with each launch run as its members, one call each. *)
let apart linear =
  Ops.replace linear
    ~src:
      (List.concat_map
         (fun e ->
           if Ops.op e = Call && Shape.is_launch (Ops.nth e 0) then
             Shape.launch_members e
           else [ e ])
         (Ops.src linear))

let storage big =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | Buffer, Param p -> Some (n, p)
      | _ -> None)
    (Ops.toposort ~calls:Skip big)

(* The bits of a value, so that NaNs and zeros compare by their bits. *)
let bits : Dtype.value -> string = function
  | `Float x -> Printf.sprintf "%Lx" (Int64.bits_of_float x)
  | `Bool b -> string_of_bool b
  | v -> Bigint.to_string (Dtype.Value.to_z v)

(* What [linear], the schedule of [big] on [name], leaves in its buffer of
   slot 0. *)
let compiled name linear =
  let d = device name in
  Realize.compile_linear ~targets:(fun _ -> Engine.target d) linear

let written name big (linear, vars) =
  let d = device name in
  let devices _ = d in
  let compiled = compiled name linear in
  let bufs =
    List.map
      (fun (n, (p : Ops.param_arg)) ->
        let size = Shape.max_numel n in
        (n, p, Run.buffer d p.dtype (Array.init size (element p.dtype))))
      (storage big)
  in
  let bound = List.map (fun (n, _, b) -> (n, b)) bufs in
  let s = Engine.link ~devices ~bound compiled in
  Engine.run ~vars s [||];
  let out, p, b =
    List.find (fun (_, (p : Ops.param_arg), _) -> p.slot = 0) bufs
  in
  let n = Shape.max_numel out in
  Array.to_list (Array.map bits (Array.sub (Run.values p.dtype b) 0 n))

(* Laws *)

let alike p =
  match p.pieces with
  | q :: (_ :: _ as rest) -> List.for_all (fun q' -> q' = q) rest
  | _ -> false

let reduces q = q.kind <> Elementwise

let writes_as_apart name p =
  reject_unsupported name p.dtype;
  cover "one piece" (List.length p.pieces = 1);
  cover "alike pieces" (alike p);
  cover "unlike pieces" (List.length p.pieces > 1 && not (alike p));
  cover "a reducing piece beside one that is not"
    (List.exists reduces p.pieces
    && List.exists (fun q -> not (reduces q)) p.pieces);
  cover "reducing pieces of two kernels"
    (List.length (List.sort_uniq compare (List.filter reduces p.pieces)) > 1);
  cover "a fill" p.fill;
  cover "a later state" p.later;
  cover "bound starts" p.bound;
  cover "a float" (is_float p.dtype);
  cover "an integer" (not (is_float p.dtype));
  let big = sink name p in
  let linear, vars = Schedule.create_linear_with_vars big in
  let ls = launches linear in
  (* Pieces of one kernel that reduces nothing run as one launch. *)
  let msg = Format.asprintf "launches of %a" pp_program p in
  if alike p && not (reduces (List.hd p.pieces)) then
    equal int ~msg 1 (List.length ls);
  if List.length p.pieces > 1 && not (List.exists reduces p.pieces) then
    at_least int ~msg ~than:1 (List.length ls);
  (* A member that reduces shares a launch of its target only with members
     of its kernel. *)
  let ren = Engine.renderer (device name) in
  List.iter
    (fun l ->
      let bodies = List.map (fun m -> Ops.nth m 0) (Shape.launch_members l) in
      let reducing b =
        List.exists (fun u -> Ops.op u = Reduce) (Ops.toposort ~calls:Skip b)
      in
      if List.exists reducing bodies then
        equal int ~msg:"kernels of a launch of a reducing member" 1
          (List.length (List.sort_uniq Ops.compare bodies)))
    (List.filter
       (fun e -> Ops.op e = Call && Shape.is_launch (Ops.nth e 0))
       (List.concat_map (Realize.split_launch ren) ls));
  equal (list string) ~msg:(Format.asprintf "%a" pp_program p)
    (written name big (apart linear, vars))
    (written name big (linear, vars))

(* The bits a program of elementwise integer pieces leaves in its buffer: each
   piece's rows hold [3 x + 1] of its buffer [x], whose element [j] is
   [element dt j]. *)
let assembled p =
  List.concat_map
    (fun q ->
      List.init (q.rows * p.width) (fun j ->
          match element p.dtype j with
          | `Int x -> string_of_int ((3 * Bigint.to_int x) + 1)
          | _ -> invalid_arg "assembled: an integer program"))
    p.pieces

(* However many stores of one state a program makes, its buffer holds what
   each stored. *)
let writes_pieces name p =
  reject_unsupported name p.dtype;
  cover "past a hundred pieces" (List.length p.pieces > 100);
  let big = sink name p in
  equal (list string) ~msg:(Format.asprintf "%a" pp_program p) (assembled p)
    (written name big (Schedule.create_linear_with_vars big))

(* Generators *)

let dtypes =
  Dtype.
    [
      Float32; Float16; Bfloat16; Float64; Int8; Int16; Int32; Int64; Uint8;
      Uint16; Uint32; Uint64;
    ]

let gen_piece =
  Gen.(
    map
      (fun (kind, rows, k) -> { kind; rows; k })
      (triple
         (frequency
            [
              (3, constant Elementwise);
              (1, constant Sum);
              (1, constant Matvec);
              (1, constant Matmul);
            ])
         (int_range 1 4)
         (frequency [ (1, constant 1); (2, constant 16); (1, constant 33) ])))

let gen_program =
  Gen.(
    bind (int_range 0 (List.length dtypes - 1)) (fun i ->
        bind
          (frequency
             [ (1, constant 1); (3, int_range 2 8); (1, int_range 9 32) ])
          (fun n ->
            bind bool (fun same ->
                bind gen_piece (fun first ->
                    bind (list ~size:(constant (n - 1)) gen_piece) (fun rest ->
                        map
                          (fun ((width, fill), (later, bound)) ->
                            {
                              dtype = List.nth dtypes i;
                              width;
                              pieces =
                                (if same then List.init n (fun _ -> first)
                                 else first :: rest);
                              fill;
                              later;
                              bound;
                            })
                          (pair
                             (pair
                                (frequency
                                   [ (1, constant 1); (2, constant 64) ])
                                bool)
                             (pair bool bool))))))))

(* Up to 160 elementwise pieces of [int32] rows. *)
let gen_many =
  Gen.(
    map
      (fun ((n, rows), (width, fill)) ->
        {
          dtype = Int32;
          width;
          pieces = List.init n (fun _ -> { kind = Elementwise; rows; k = 1 });
          fill;
          later = false;
          bound = false;
        })
      (pair
         (pair
            (frequency [ (1, int_range 1 8); (3, int_range 101 160) ])
            (int_range 1 2))
         (pair
            (one_of [ constant 1; constant 9; constant 64 ])
            bool)))

(* A program of 101 one-row pieces, the fewest past a hundred. *)
let past_a_hundred =
  {
    dtype = Int32;
    width = 9;
    pieces = List.init 101 (fun _ -> { kind = Elementwise; rows = 1; k = 1 });
    fill = true;
    later = false;
    bound = false;
  }

(* Cases *)

(* The sources of the programs of the launches of [big]'s schedule on
   [name]. *)
let launch_sources name big =
  let linear, _ = Schedule.create_linear_with_vars big in
  List.filter_map
    (fun e ->
      match Ops.src e with
      | prg :: _ when Ops.op e = Call && Ops.op prg = Program -> (
          let sink = Ops.nth prg 0 in
          let source =
            List.find_opt (fun u -> Ops.op u = Source) (Ops.src prg)
          in
          match (Ops.arg sink, source) with
          | Kernel { name = "launch"; _ }, Some src -> (
              match Ops.arg src with String t -> Some t | _ -> None)
          | _ -> None)
      | _ -> None)
    (Ops.src (compiled name linear))

let occurrences sub s =
  let n = String.length sub in
  let rec go i k =
    if i + n > String.length s then k
    else if String.sub s i n = sub then go (i + n) (k + 1)
    else go (i + 1) k
  in
  go 0 0

let sixteen kind dtype =
  {
    dtype;
    width = 16;
    pieces = List.init 16 (fun _ -> { kind; rows = 16; k = 16 });
    fill = false;
    later = false;
    bound = false;
  }

let one_body name () =
  match launch_sources name (sink name (sixteen Elementwise Float32)) with
  | [ src ] ->
      equal int ~msg:"bodies" 1 (occurrences "void member(" src);
      equal int ~msg:"calls" 16 (occurrences "member(" src - 1)
  | srcs -> failf "%d launches" (List.length srcs)

let out_of_line name attribute () =
  match launch_sources name (sink name (sixteen Elementwise Float32)) with
  | [ src ] -> equal int ~msg:"out of line" 1 (occurrences attribute src)
  | srcs -> failf "%d launches" (List.length srcs)

(* [p]'s launch on [name] writes what its members apart write, and its one
   launch's source. *)
let as_apart name p =
  let big = sink name p in
  let linear, vars = Schedule.create_linear_with_vars big in
  equal int ~msg:"launches" 1 (List.length (launches linear));
  equal (list string)
    (written name big (apart linear, vars))
    (written name big (linear, vars));
  match launch_sources name big with
  | [ src ] -> src
  | srcs -> failf "%d launches" (List.length srcs)

(* Sixteen products of [16; 16] half matrices, on tensor cores on CUDA. *)
let products name () =
  let src = as_apart name (sixteen Matmul Float16) in
  if name = "CUDA" then
    at_least int ~msg:"matrix instructions" ~than:1
      (occurrences "mma.sync" src)

(* Sixteen products of [16; 16] half matrices on [name], whose launch's
   members hold no workgroup memory: each product sums through it, so each
   runs on its own, and writes what it writes apart. *)
let products_apart name () =
  let big = sink name (sixteen Matmul Float16) in
  let linear, vars = Schedule.create_linear_with_vars big in
  equal int ~msg:"launches" 0 (List.length (launch_sources name big));
  equal (list string)
    (written name big (apart linear, vars))
    (written name big (linear, vars))

let pieces dtype rows =
  {
    dtype;
    width = 64;
    pieces = List.map (fun r -> { kind = Elementwise; rows = r; k = 1 }) rows;
    fill = false;
    later = false;
    bound = false;
  }

(* On a GPU, members of fewer threads than the launch run gated by the
   thread index. *)
let fewer_threads () =
  let src = as_apart "CUDA" (pieces Float32 [ 1; 64; 1; 64 ]) in
  at_least int ~msg:"thread gates" ~than:1 (occurrences "lidx0<" src)

(* On the host, a launch whose cost the engine splits into blocks, whose
   bounds fall inside members of unequal iterations. *)
let blocks () =
  let p = pieces Float32 [ 100; 300; 37 ] in
  let big = sink "CPU" p in
  let linear, _ = Schedule.create_linear_with_vars big in
  let ops =
    List.concat_map
      (fun e ->
        match Ops.src e with
        | prg :: _ when Ops.op prg = Program -> (
            match Ops.arg (Ops.nth prg 0) with
            | Kernel { name = "launch"; estimates = Some { ops = Int n; _ }; _ }
              ->
                [ n ]
            | _ -> [])
        | _ -> [])
      (Ops.src (compiled "CPU" linear))
  in
  (match ops with
  | [ n ] -> at_least int ~msg:"operations, two blocks'" ~than:(1 lsl 19) n
  | ns -> failf "%d launches" (List.length ns));
  ignore (as_apart "CPU" p)

(* Stores of values each kernel makes first, read by the stores: pieces of
   two kernels each. *)
let made_first () =
  let d = "CPU" in
  let out = Ops.new_buffer ~slot:0 d 256 Float32 in
  let rows = Shape.reshape out [ Int 16; Int 16 ] in
  let store i =
    let x = Ops.new_buffer ~slot:(i + 1) d 16 Float32 in
    let value = Ops.contiguous (Ops.exp2 (Shape.reshape x [ Int 1; Int 16 ])) in
    Ops.store (Shape.shrink rows [ Some (Int i, Int (i + 1)); None ]) value
  in
  Ops.sink [ Ops.after_disjoint out (List.init 16 store) ]

(* The most storage made by the schedule that is live at once: each takes
   the entries from the first that takes it to the last. *)
let peak big linear =
  let given = List.map fst (storage big) in
  let entries = Array.of_list (Ops.src linear) in
  let first = Ops.Tbl.create 16 and last = Ops.Tbl.create 16 in
  Array.iteri
    (fun i e ->
      List.iter
        (fun a ->
          let b = Ops.storage_base a in
          if (not (List.memq b given)) && Ops.addrspace a <> Some Dtype.Alu
          then begin
            if not (Ops.Tbl.mem first b) then Ops.Tbl.replace first b i;
            Ops.Tbl.replace last b i
          end)
        (Ops.src_without_body e))
    entries;
  let live i =
    Ops.Tbl.fold
      (fun b f n -> if f <= i && i <= Ops.Tbl.find last b then n + 1 else n)
      first 0
  in
  List.fold_left max 0 (List.init (Array.length entries) live)

(* A launch never makes storage live longer than the schedule's order does:
   each value is made and freed in turn, one at a time, however the stores
   could run together. *)
let storage_order_holds () =
  let big = made_first () in
  let linear = fst (Schedule.create_linear_with_vars ~capturing:true big) in
  equal int ~msg:"values live at once" 1 (peak big linear)

(* The launch of [sixteen kind Float32]'s program on the host, split for a
   renderer with [f]'s limits, as lengths of the launches it becomes. *)
let split ?launch_members ?launch_args ?launch_locals kind =
  let big = sink "CPU" (sixteen kind Float32) in
  let linear, _ = Schedule.create_linear_with_vars big in
  match launches linear with
  | [ e ] ->
      let target = Engine.target Rig.host in
      let ren = Renderer.v ?launch_members ?launch_args ?launch_locals target in
      List.map
        (fun l ->
          if Ops.op l = Call && Shape.is_launch (Ops.nth l 0) then
            List.length (Shape.launch_members l)
          else 1)
        (Realize.split_launch ren e)
  | ls -> failf "%d launches" (List.length ls)

let fits_target () =
  let same = List.init 8 (fun _ -> 2) in
  equal (list int) ~msg:"members" same
    (split ~launch_members:2 Elementwise);
  (* A member passes its region of the output and its source. *)
  equal (list int) ~msg:"arguments" same
    (split ~launch_args:5 Elementwise);
  equal (list int) ~msg:"one reducing kernel holds a launch" [ 16 ]
    (split Matmul);
  equal (list int) ~msg:"a member that reduces, no workgroup memory"
    (List.init 16 (fun _ -> 1))
    (split ~launch_locals:false Matmul);
  equal (list int) ~msg:"buffers" same
    (Setting.context
       [ Setting.B (Setting.max_kernel_buffers, 5) ]
       (fun () -> split Elementwise))

(* On CUDA, a state of more members than its renderer's launch holds runs
   as several launches, alike ones one program, writing what its members
   apart write. *)
let past_the_cap () =
  let d = device "CUDA" in
  let cap = (Engine.renderer d).launch_members in
  let n = cap + (cap / 2) in
  let p = { (pieces Float32 (List.init n (fun _ -> 1))) with width = 64 } in
  let big = sink "CUDA" p in
  let linear, vars = Schedule.create_linear_with_vars big in
  let ran =
    List.filter
      (fun e ->
        match Ops.src e with
        | prg :: _ when Ops.op e = Call && Ops.op prg = Program -> (
            match Ops.arg (Ops.nth prg 0) with
            | Kernel { name = "launch"; _ } -> true
            | _ -> false)
        | _ -> false)
      (Ops.src (compiled "CUDA" linear))
  in
  equal int ~msg:"launches" 2 (List.length ran);
  equal (list string)
    (written "CUDA" big (apart linear, vars))
    (written "CUDA" big (linear, vars))

(* States read as they were

   A kernel of one write, or a kernel beside a write, that reads a state reads
   it as it was: run on the host, each program leaves the values the states
   it names hold. *)

let part lo hi u = Shape.shrink u [ Some (Int lo, Int hi) ]
let flip u = Shape.flip u [ 0 ]
let fl = Ops.float
let f32 ~n slot = Ops.new_buffer ~slot "CPU" n Float32

(* [j] at element [j] of the buffer of slot 0, [10 j] at slot 1. *)
let start (p : Ops.param_arg) j =
  `Float (float_of_int (j * if p.slot = 0 then 1 else 10))

(* Every element of [big]'s buffers after its schedule runs on the host, in
   slot order. *)
let finals big =
  let d = device "CPU" in
  let linear, vars =
    Setting.context [ B (Setting.scache, 0) ] (fun () ->
        Schedule.create_linear_with_vars big)
  in
  let compiled = compiled "CPU" linear in
  let bufs =
    List.map
      (fun (n, (p : Ops.param_arg)) ->
        (p, Run.buffer d p.dtype (Array.init (Shape.max_numel n) (start p))))
      (storage big)
  in
  let bound =
    List.map2 (fun (n, _) (_, b) -> (n, b)) (storage big) bufs
  in
  Engine.run ~vars (Engine.link ~devices:(fun _ -> d) ~bound compiled) [||];
  let bufs =
    List.sort (fun ((p : Ops.param_arg), _) (q, _) -> compare p.slot q.slot) bufs
  in
  List.concat_map
    (fun ((p : Ops.param_arg), b) -> Array.to_list (Run.values p.dtype b))
    bufs

let close = Testable.float_rel ~rel:1e-6 ~abs:0.

let holds big expected =
  equal (list close) expected
    (List.map
       (function `Float x -> x | _ -> failf "a float")
       (finals (Ops.sink big)))

(* [x] holding 0 to 11, [10, 12) reversed from [8, 10) as it was, and [8, 10)
   overwritten by the write's other store. *)
let sibling_stores x =
  [
    Ops.store (part 10 12 x) (flip (part 8 10 x));
    Ops.store (part 8 10 x) (fl 5.);
  ]

let sibling_read order () =
  let x = f32 ~n:12 0 in
  holds
    [ Ops.after x (order (sibling_stores x)) ]
    [ 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7.; 5.; 5.; 9.; 8. ]

(* The stores of [sibling_read]'s kind under a later write of [x], which alone
   the sink names. *)
let chained_read order () =
  let x = f32 ~n:12 0 in
  let x1 =
    Ops.after x
      (order
         [
           Ops.store (part 0 2 x) (fl 5.);
           Ops.store (part 2 4 x) Ops.O.(part 0 2 x + fl 1.);
         ])
  in
  holds
    [ Ops.after x1 [ Ops.store (part 8 10 x1) (fl 9.) ] ]
    [ 5.; 5.; 1.; 2.; 4.; 5.; 6.; 7.; 9.; 9.; 10.; 11. ]

(* As [chained_read], the later write over [x] itself, its destination a run
   of the first write's state: only that destination reaches the first
   write. *)
let read_under_destination () =
  let x = f32 ~n:12 0 in
  let x1 =
    Ops.after x
      [
        Ops.store (part 0 2 x) (fl 5.);
        Ops.store (part 2 4 x) Ops.O.(part 0 2 x + fl 1.);
      ]
  in
  holds
    [ Ops.after x [ Ops.store (part 8 10 x1) (fl 9.) ] ]
    [ 5.; 5.; 1.; 2.; 4.; 5.; 6.; 7.; 9.; 9.; 10.; 11. ]

(* A join over the view a store writes reads the state under it: [x] as it
   was, once [y] is stored. *)
let read_join_of_view order () =
  let x = f32 ~n:8 0 and y = f32 ~n:8 1 and z = f32 ~n:5 2 in
  let v = part 1 6 x in
  let x' = Ops.after x [ Ops.store v (fl 7.) ] in
  let y' = Ops.after y [ Ops.store y (fl 2.) ] in
  let z' = Ops.after z [ Ops.store z (Ops.after v [ y' ]) ] in
  holds
    (order [ x'; y'; z' ])
    ([ 0.; 7.; 7.; 7.; 7.; 7.; 6.; 7. ]
    @ List.init 8 (fun _ -> 2.)
    @ [ 1.; 2.; 3.; 4.; 5. ])

(* Two stores in a chain, the second into a run of the buffer as it was, that
   run read elsewhere as it was: the second writes over the first's state. *)
let store_over_state_read () =
  let x = f32 ~n:12 0 and y = f32 ~n:2 1 in
  let a = Ops.after x [ Ops.store (part 0 2 x) (fl 1.) ] in
  let b = Ops.after a [ Ops.store (part 8 10 x) (fl 2.) ] in
  let y' = Ops.after y [ Ops.store y (part 8 10 x) ] in
  holds [ b; y' ]
    [ 1.; 1.; 2.; 3.; 4.; 5.; 6.; 7.; 2.; 2.; 10.; 11.; 8.; 9. ]

(* An After over a view names that view's state, and a reader of the view as
   it was runs before the write. *)
let read_view_after_over order () =
  let x = f32 ~n:8 0 and y = f32 ~n:5 1 in
  let v = part 1 6 x in
  let x' = Ops.after v [ Ops.store v (fl 7.) ] in
  let y' = Ops.after y [ Ops.store y v ] in
  holds (order [ x'; y' ])
    [ 0.; 7.; 7.; 7.; 7.; 7.; 6.; 7.; 1.; 2.; 3.; 4.; 5. ]

(* A store into a run and a reader of it that share a computed value, which
   could run in one kernel over one loop: the reader reads the run as it was. *)
let read_sharing_value order () =
  let x = f32 ~n:8 0 and w = f32 ~n:5 1 and y = f32 ~n:5 2 in
  let v = part 1 6 x in
  let twice = Ops.O.(w * fl 2.) in
  let x' = Ops.after x [ Ops.store v twice ] in
  let y' = Ops.after y [ Ops.store y Ops.O.(v + twice) ] in
  (* [w] is 0 10 20 30 40, doubled 0 20 40 60 80. *)
  holds
    (order [ x'; y' ])
    ([ 0.; 0.; 20.; 40.; 60.; 80.; 6.; 7. ]
    @ [ 0.; 10.; 20.; 30.; 40. ]
    @ [ 1.; 22.; 43.; 64.; 85. ])

(* A store of a run reversed, then a store over that run, the state between
   read: the reversal stages what it overwrites. *)
let reversed_then_stored () =
  let x = f32 ~n:8 0 and z = f32 ~n:5 1 in
  let v = part 1 6 x in
  let a = Ops.after x [ Ops.store v (flip v) ] in
  let b = Ops.after a [ Ops.store v (fl 5.) ] in
  let z' = Ops.after z [ Ops.store z (part 1 6 a) ] in
  holds [ b; z' ]
    [ 0.; 5.; 5.; 5.; 5.; 5.; 6.; 7.; 5.; 4.; 3.; 2.; 1. ]

(* A reader of a run that also waits for the store into it reads a copy of
   the run as it was. *)
let read_view_waiting () =
  let x = f32 ~n:8 0 and y = f32 ~n:5 1 in
  let v = part 1 6 x in
  let x' = Ops.after x [ Ops.store v (fl 7.) ] in
  let y' = Ops.after y [ Ops.store y Ops.O.(v + part 1 6 x') ] in
  let copies = Schedule.copies () in
  holds [ x'; y' ]
    [ 0.; 7.; 7.; 7.; 7.; 7.; 6.; 7.; 8.; 9.; 10.; 11.; 12. ];
  at_least int ~msg:"copies" ~than:(copies + 1) (Schedule.copies ())

(* Each store reads only its own elements, as the state under the write was.
   Marked disjoint, the stores are one write over that state, which neither
   overwrites before the other reads it. *)
let own_elements () =
  let x = f32 ~n:4 0 in
  let bump lo hi = Ops.store (part lo hi x) Ops.O.(part lo hi x + fl 1.) in
  let copies = Schedule.copies () in
  holds [ Ops.after_disjoint x [ bump 0 2; bump 2 4 ] ] [ 1.; 2.; 3.; 4. ];
  equal int ~msg:"copies" copies (Schedule.copies ())

(* Each store reads the elements the other writes. *)
let crossed_reads () =
  let x = f32 ~n:4 0 in
  holds
    [
      Ops.after x
        [
          Ops.store (part 0 2 x) (part 2 4 x);
          Ops.store (part 2 4 x) (part 0 2 x);
        ];
    ]
    [ 2.; 3.; 0.; 1. ]

(* [x] holding 0 to 7 written through a run of it, and that run of [x] as it
   was stored into [y]. *)
let view_read rd order () =
  let x = f32 ~n:8 0 and y = f32 ~n:8 1 in
  let x' = Ops.after x [ Ops.store (part 1 6 x) (fl 7.) ] in
  let lo, hi = rd in
  let y' = Ops.after y [ Ops.store (part 0 (hi - lo) y) (part lo hi x) ] in
  let tail = List.init (8 - (hi - lo)) (fun j -> float_of_int ((hi - lo + j) * 10)) in
  holds
    (order [ x'; y' ])
    ([ 0.; 7.; 7.; 7.; 7.; 7.; 6.; 7. ]
    @ List.init (hi - lo) (fun j -> float_of_int (lo + j))
    @ tail)

(* A store through a later state of the buffer, and a reader of the first of
   its runs as it was. *)
let chained_view order () =
  let x = f32 ~n:8 0 and y = f32 ~n:5 1 in
  let a = Ops.after x [ Ops.store (part 1 6 x) (fl 7.) ] in
  let b = Ops.after a [ Ops.store (part 3 8 x) (fl 8.) ] in
  let y' = Ops.after y [ Ops.store y (part 1 6 x) ] in
  holds (order [ b; y' ])
    ([ 0.; 7.; 7.; 8.; 8.; 8.; 8.; 8. ] @ [ 1.; 2.; 3.; 4.; 5. ])

(* A store through a later state sharing a value with a reader of the first
   state, through a run the store does not name. *)
let chained_share order () =
  let x = f32 ~n:8 0 and y = f32 ~n:4 1 and w = f32 ~n:4 2 in
  let twice = Ops.O.(w * fl 2.) in
  let a = Ops.after x [ Ops.store (part 0 4 x) (fl 7.) ] in
  let b = Ops.after a [ Ops.store (part 2 6 x) twice ] in
  let y' = Ops.after y [ Ops.store y Ops.O.(part 0 4 x + twice) ] in
  (* [w] is 0 10 20 30, doubled 0 20 40 60. *)
  holds (order [ b; y' ])
    ([ 7.; 7.; 0.; 20.; 40.; 60.; 6.; 7. ]
    @ [ 0.; 21.; 42.; 63. ]
    @ [ 0.; 10.; 20.; 30. ])

(* A loop's call stores through one run of the buffer, listed after a write
   of it, and reads the first state of another: a copy made before that
   write, once, not each trip. *)
let loop_reads_older_run () =
  let x = f32 ~n:16 0 in
  let vw = part 0 4 x and vr = part 8 12 x in
  let pre = Ops.after x [ Ops.store vw (fl 1.); Ops.store vr (fl 2.) ] in
  let r = Ops.range ~axis_type:Weak (Ops.Int 4) [ 0 ] in
  let a0 = Shape.param ~shape:[ Ops.Int 4 ] 0 Float32
  and a1 = Shape.param ~shape:[ Ops.Int 4 ] 1 Float32 in
  let body =
    Ops.sink ~kernel:(Ops.kernel_info ())
      [
        Ops.end_
          (Ops.store (Ops.index a0 [ r ]) Ops.O.(Ops.index a1 [ r ] + fl 1.))
          [ r ];
      ]
  in
  let trip = Ops.range ~axis_type:Loop (Ops.Int 3) [ 100 ] in
  let loop = Ops.end_ (Ops.call ~precompile:true body [ vw; vr ]) [ trip ] in
  holds
    [ Ops.after pre [ loop ] ]
    [
      9.; 10.; 11.; 12.; 4.; 5.; 6.; 7.; 2.; 2.; 2.; 2.; 12.; 13.; 14.; 15.;
    ]

(* Two writes of [x] over the whole of it, beside one through a view: no
   storage holds two successors of a state, whichever is listed first. *)
let two_whole order () =
  let x = f32 ~n:8 0 in
  let v = part 2 6 x in
  let w1 = Ops.after x [ Ops.store x (fl 1.) ] in
  let w2 = Ops.after x [ Ops.store x (fl 3.) ] in
  let u = Ops.after v [ Ops.store v (fl 2.) ] in
  raises_match (Exn.invalid_arg ~sub:"two writes over one state") (fun () ->
      ignore (finals (Ops.sink (order [ w1; w2; u ]))))

(* Listed order *)

(* Two overlapping stores of [x], listed in [order]. *)
let listed_constants order expected () =
  let x = f32 ~n:8 0 in
  holds
    [
      Ops.after x
        (order
           [ Ops.store (part 0 5 x) (fl 1.); Ops.store (part 3 8 x) (fl 2.) ]);
    ]
    expected

(* [p]'s elements 0 to 5 doubled, then 2 to 7 tripled, each store reading [p]
   as it was. *)
let own_overlap p =
  Ops.after p
    [
      Ops.store (part 0 6 p) Ops.O.(part 0 6 p * fl 2.);
      Ops.store (part 2 8 p) Ops.O.(part 2 8 p * fl 3.);
    ]

let tripled = [ 0.; 2.; 6.; 9.; 12.; 15.; 18.; 21. ]

let listed_overlap () = holds [ own_overlap (f32 ~n:8 0) ] tripled

(* [p]'s run 1 to 3 as it was, stored after a store into the run 0 to 2, which
   it overlaps. *)
let listed_reads_earlier () =
  let x = f32 ~n:8 0 in
  holds
    [
      Ops.after x
        [
          Ops.store (part 0 2 x) (fl 5.);
          Ops.store (part 1 3 x) Ops.O.(part 0 2 x + fl 1.);
        ];
    ]
    [ 5.; 1.; 2.; 3.; 4.; 5.; 6.; 7. ]

(* A store listed twice lands at its first place: listed again, it is a store
   the state already holds. *)
let listed_twice () =
  let x = f32 ~n:8 0 in
  let a = Ops.store (part 0 6 x) (fl 1.) in
  let b = Ops.store (part 2 8 x) (fl 2.) in
  holds [ Ops.after x [ a; b; a ] ] [ 1.; 1.; 2.; 2.; 2.; 2.; 2.; 2. ]

(* A body that lists overlapping stores of its parameter. *)
let listing_body ?device () =
  let p = Shape.param ~shape:[ Ops.Int 8 ] ?device 0 Float32 in
  Ops.sink [ own_overlap p ]

(* The same listing in a function's body, and in a body a loop runs. Each trip
   doubles elements 0 and 1 and triples 2 to 7 of what the trip before left,
   each store reading the elements as the trip began. *)
let listed_in_call ?device () =
  let x = f32 ~n:8 0 in
  let call = Ops.call ~precompile:true (listing_body ?device ()) [ x ] in
  holds [ Ops.after x [ call ] ] tripled

(* The same listing in a body inlined at its call. *)
let listed_inlined () =
  let x = f32 ~n:8 0 in
  let call = Ops.call (listing_body ~device:"CPU" ()) [ x ] in
  holds [ Ops.after x [ call ] ] tripled

(* [p] doubled by a function of it. *)
let doubling () =
  let p = Shape.param ~shape:[ Ops.Int 8 ] ~device:"CPU" 0 Float32 in
  Ops.sink [ Ops.store p Ops.O.(p * fl 2.) ]

(* The call loads through the parameter it stores through: its load and store
   are one, and it doubles what the store before it left. *)
let call_after_store () =
  let x = f32 ~n:8 0 in
  holds
    [
      Ops.after x
        [
          Ops.store (part 0 4 x) (fl 9.);
          Ops.call ~precompile:true (doubling ()) [ x ];
        ];
    ]
    [ 18.; 18.; 18.; 18.; 8.; 10.; 12.; 14. ]

(* An After over the whole of [x] and one over a view of it are two writes
   over one state, in either listing: which lands last would be the listing's
   choice. *)
let whole_and_view order () =
  let x = f32 ~n:8 0 in
  let v = part 2 6 x in
  let w = Ops.after x [ Ops.store x (fl 1.) ] in
  let u = Ops.after v [ Ops.store v (fl 2.) ] in
  raises_match (Exn.invalid_arg ~sub:"two writes over one state") (fun () ->
      ignore (finals (Ops.sink (order [ w; u ]))))

let listed_in_loop () =
  let x = f32 ~n:8 0 in
  let trip = Ops.range ~axis_type:Loop (Ops.Int 3) [ 100 ] in
  let call =
    Ops.call ~precompile:true (listing_body ~device:"CPU" ()) [ x ]
  in
  holds
    [ Ops.after x [ Ops.end_ call [ trip ] ] ]
    [ 0.; 8.; 54.; 81.; 108.; 135.; 162.; 189. ]

let as_it_was =
  group "states read as they were"
    [
      test "a store reads what its write's other store writes as it was"
        (sibling_read Fun.id);
      test "a store reads what its write's other store writes, listed last"
        (sibling_read List.rev);
      test "a store under a later write reads its write's other store as it was"
        (chained_read Fun.id);
      test "a store under a later write reads it, listed last"
        (chained_read List.rev);
      test "a store only a later destination reaches reads it as it was"
        read_under_destination;
      test "a join of the view a store writes reads the state under it"
        (read_join_of_view Fun.id);
      test "a join of the view a store writes, listed last"
        (read_join_of_view List.rev);
      test "a store over a state, into a run of the buffer read elsewhere"
        store_over_state_read;
      test "a reader of a view an After is over reads it as it was"
        (read_view_after_over Fun.id);
      test "a reader of a view an After is over, listed last"
        (read_view_after_over List.rev);
      test "a reader sharing a value with the store into its run reads it as \
            it was"
        (read_sharing_value Fun.id);
      test "a reader sharing a value with the store, listed last"
        (read_sharing_value List.rev);
      test "a reversal of a run, then a store over it, leaves the state between"
        reversed_then_stored;
      test "a reader of a run that waits for the store into it reads a copy"
        read_view_waiting;
      test
        "stores of a disjoint write reading only the elements they write take \
         no copy"
        own_elements;
      test "stores of one write reading each other's elements read them as \
            they were"
        crossed_reads;
      test "a reader through the view a write stores into reads it as it was"
        (view_read (1, 6) Fun.id);
      test "a reader through the view, listed first" (view_read (1, 6) List.rev);
      test "a reader through another view reads it as it was"
        (view_read (2, 5) Fun.id);
      test "a chain of stores through views of one buffer, its first run read"
        (chained_view Fun.id);
      test "a chain of stores through views, the reader listed first"
        (chained_view List.rev);
      test "a chained store and a reader of the first state sharing a value"
        (chained_share Fun.id);
      test "a chained store and a sharing reader, the reader listed first"
        (chained_share List.rev);
      test "a loop reading an older run of its storage reads a copy of the run"
        loop_reads_older_run;
    ]

let listed =
  group "an After lists its stores in the order they land"
    [
      test "the later store wins" (listed_constants Fun.id
        [ 1.; 1.; 1.; 2.; 2.; 2.; 2.; 2. ]);
      test "listed the other way, the other wins" (listed_constants List.rev
        [ 1.; 1.; 1.; 1.; 1.; 2.; 2.; 2. ]);
      test "each store reads its own elements as they were" listed_overlap;
      test "a later store reads what an earlier one wrote as it was"
        listed_reads_earlier;
      test "a store listed twice lands at its first place" listed_twice;
      test "the listing holds in a function's body, parameters of a device"
        (listed_in_call ~device:"CPU");
      test "the listing holds in a function's body, parameters of none"
        (listed_in_call ?device:None);
      test "the listing holds in a body inlined at its call" listed_inlined;
      test "the listing holds in a body a loop runs" listed_in_loop;
      test "a call loading through what it stores doubles what was listed before"
        call_after_store;
      test "an After of a state and one of a view of it raise, whole first"
        (whole_and_view Fun.id);
      test "an After of a state and one of a view of it raise, view first"
        (whole_and_view List.rev);
      test "two writes over the whole of a state raise, whole first"
        (two_whole Fun.id);
      test "two writes over the whole of a state raise, view first"
        (two_whole List.rev);
      test "two writes over the whole of a state raise, view between"
        (two_whole (function
          | [ a; b; c ] -> [ a; c; b ]
          | l -> l));
    ]

(* The launches on each GPU present: a GPU's tests skip without one. *)
let on_a_gpu kind =
  group kind
    [
      test
        "sixteen products run as one launch, on tensor cores on CUDA, or \
         apart where a launch's members hold no workgroup memory"
        (fun () ->
          if (Engine.renderer (device kind)).launch_locals then products kind ()
          else products_apart kind ());
      prop ~count:30
        "the stores of one state run as one launch and write what their \
         kernels apart write"
        gen_program (writes_as_apart kind);
      (* Each case compiles a program of its own, whose pipelines a GPU
         builds anew, so the host draws the breadth. A GPU takes one draw
         beside a fixed case past a hundred pieces, which reaches the cover
         on every seed. *)
      prop ~count:1 ~examples:[ past_a_hundred ]
        "the stores of one state write each piece's values" gen_many
        (writes_pieces kind);
    ]

let () =
  exit @@ run "Tolk.Launch"
    [
      as_it_was;
      listed;
      group "cases"
        [
          test "on the host, alike members compile one body" (one_body "CPU");
          test "on CUDA, alike members compile one body" (one_body "CUDA");
          test "on the host, a launch's bodies are out of line"
            (out_of_line "CPU" "noinline");
          test "on CUDA, a launch's bodies are out of line"
            (out_of_line "CUDA" "__noinline__");
          test "a launch never makes storage live longer" storage_order_holds;
          test "a launch is split by what its target can run" fits_target;
          test "on the host, a launch's blocks split its members" blocks;
          test "on CUDA, members of fewer threads run gated" fewer_threads;
          test "on CUDA, a state past a launch's members runs as several"
            past_the_cap;
          test "on the host, sixteen products run as one launch"
            (products "CPU");
        ];
      group "launches"
        [
          prop ~count:60
            "on the host, the stores of one state run as one launch and write \
             what their kernels apart write"
            gen_program (writes_as_apart "CPU");
          prop ~count:12
            "on the host, the stores of one state write each piece's values"
            gen_many (writes_pieces "CPU");
        ];
      group "GPUs" (List.map on_a_gpu Gpus.kinds);
    ]
