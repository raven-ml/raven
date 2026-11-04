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

let gpus = [ ("CUDA", lazy (Rig.open_ (Rig_cuda.gpu 0))) ]

let device name =
  if name = "CPU" then Rig.host
  else
    match Lazy.force (List.assoc name gpus) with
    | Ok d -> d
    | Error why -> skip ~reason:why ()

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
  let state = Ops.after state (List.rev stores) in
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
  if alike p && not (reduces (List.hd p.pieces)) then
    equal int ~msg:"launches" 1 (List.length ls);
  if List.length p.pieces > 1 && not (List.exists reduces p.pieces) then
    at_least int ~msg:"launches" ~than:1 (List.length ls);
  (* A member that reduces shares its launch only with members of its
     kernel. *)
  List.iter
    (fun l ->
      let bodies = List.map (fun m -> Ops.nth m 0) (Shape.launch_members l) in
      let reducing b =
        List.exists (fun u -> Ops.op u = Reduce) (Ops.toposort ~calls:Skip b)
      in
      if List.exists reducing bodies then
        equal int ~msg:"kernels of a launch of a reducing member" 1
          (List.length (List.sort_uniq Ops.compare bodies)))
    ls;
  equal (list string) ~msg:(Format.asprintf "%a" pp_program p)
    (written name big (apart linear, vars))
    (written name big (linear, vars))

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

(* A schedule ordered by its storage makes and frees each piece's storage in
   turn, which a launch would hold at once. *)
let by_storage () =
  let big = sink "CPU" (sixteen Elementwise Float32) in
  let linear, _ = Schedule.create_linear_with_vars ~by_storage:true big in
  equal int ~msg:"launches" 0 (List.length (launches linear));
  equal int ~msg:"calls" 16 (List.length (Ops.src linear))

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

let () =
  exit @@ run "Tolk.Launch"
    [
      group "cases"
        [
          test "on the host, alike members compile one body" (one_body "CPU");
          test "on CUDA, alike members compile one body" (one_body "CUDA");
          test "on the host, a launch's bodies are out of line"
            (out_of_line "CPU" "noinline");
          test "on CUDA, a launch's bodies are out of line"
            (out_of_line "CUDA" "__noinline__");
          test "a schedule ordered by its storage makes no launch" by_storage;
          test "on the host, a launch's blocks split its members" blocks;
          test "on CUDA, members of fewer threads run gated" fewer_threads;
          test "on CUDA, a state past a launch's members runs as several"
            past_the_cap;
          test "on the host, sixteen products run as one launch"
            (products "CPU");
          test "on CUDA, sixteen products on tensor cores run as one launch"
            (products "CUDA");
        ];
      group "launches"
        [
          prop ~count:60
            "on the host, the stores of one state run as one launch and write \
             what their kernels apart write"
            gen_program (writes_as_apart "CPU");
          prop ~count:30
            "on CUDA, the stores of one state run as one launch and write what \
             their kernels apart write"
            gen_program (writes_as_apart "CUDA");
        ];
    ]
