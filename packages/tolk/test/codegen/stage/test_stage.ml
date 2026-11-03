open Windtrap
open Tolk

let is o u = Op.equal (Ops.op u) o

(* Products

   [c = Σ_r float32 a.(i, r) * float32 b.(r, j)] in float32, [batch] times,
   for [a] of [m] by [k] and [b] of [k] by [n], of 16-bit floats: as rune
   lowers a contraction of narrow floats. [b] is stored as [k] rows of [n]
   ([K; N]) or as [n] rows of [k] ([N; K]). A Toeplitz [a] reads row [i] from
   element [i] on, [a.(i, r) = t.(i + r)]: its threads' and lanes' terms
   overlap, which staging refuses. *)

type layout = Kn | Nk

type product = {
  dt : Dtype.t;
  batch : int;
  m : int;
  k : int;
  n : int;
  layout : layout;
  toeplitz : bool;
}

let a_size p = p.batch * if p.toeplitz then p.m + p.k else p.m * p.k

let kernel ?opts_to_apply p =
  let c = Shape.param ~shape:[ Ops.Int (p.batch * p.m * p.n) ] 0 Float32
  and a = Shape.param ~shape:[ Ops.Int (a_size p) ] 1 p.dt
  and b = Shape.param ~shape:[ Ops.Int (p.batch * p.k * p.n) ] 2 p.dt in
  let rows = p.m + p.k and mk = p.m * p.k and kn = p.k * p.n
  and mn = p.m * p.n in
  let open Ops.O in
  let z = Ops.range (Int p.batch) [ 0 ]
  and i = Ops.range (Int p.m) [ 1 ]
  and j = Ops.range (Int p.n) [ 2 ]
  and r = Ops.range (Int p.k) [ 3 ] in
  let at x idx = Ops.cast (Ops.index x [ idx ]) Float32 in
  let ai =
    if p.toeplitz then (z * int rows) + i + r
    else (z * int mk) + (i * int p.k) + r
  and bi =
    match p.layout with
    | Kn -> (z * int kn) + (r * int p.n) + j
    | Nk -> (z * int kn) + (j * int p.k) + r
  in
  let sum = Ops.reduce (at a ai * at b bi) Add [ r ] in
  let store =
    Ops.store (Ops.index c [ (z * int mn) + (i * int p.n) + j ]) sum
  in
  Ops.sink ~kernel:(Ops.kernel_info ?opts_to_apply ())
    [ Ops.end_ store [ z; i; j ] ]

(* The optimisations the hand-coded rules choose for [p] on [r], staged or
   not, a core padding axes it does not divide. *)
let chosen ~staged p r =
  Setting.context
    Setting.[ B (stage, staged); B (tc_opt, 2) ]
    (fun () ->
      let s = Postrange.Scheduler.v (kernel p) r in
      Postrange.Scheduler.convert_loop_to_global s;
      Postrange.Scheduler.applied_opts (Heuristic.hand_coded_optimizations s))

(* Floats of magnitudes from [2^-8] to [2^8], whose products' sums round. *)
let values seed size =
  let st = Random.State.make [| seed |] in
  Array.init size (fun _ ->
      let f = Float.of_int (Random.State.int st 2048 - 1024) /. 1024. in
      `Float (Float.ldexp f (Random.State.int st 17 - 8)))

(* Renderers *)

let cuda = lazy (Rig.open_ (Rig_cuda.gpu 0))

(* CUDA's first GPU and its renderer, skipping without them. *)
let on_cuda () =
  match Lazy.force cuda with
  | Error why -> skip ~reason:why ()
  | Ok d -> (
      match Tolk_engine.renderer d with
      | r -> (d, r)
      | exception Failure why -> skip ~reason:why ())

(* [r] without shared memory, named apart: it stages nothing. *)
let unstaged (r : Renderer.t) =
  Renderer.v ~name:(r.name ^ " without shared memory") ~suffix:r.suffix
    ~supports_float4:r.supports_float4 ~registers:r.registers
    ~vector_alu:r.vector_alu ~has_local:r.has_local ~has_shared:false
    ~global_max:r.global_max ~local_max:r.local_max
    ?global_prod_max:r.global_prod_max ~shared_max:r.shared_max
    ~async_copy:r.async_copy ~tensor_cores:r.tensor_cores
    ~per_byte_first:r.per_byte_first ~extra_matcher:r.extra_matcher
    ~packed_args:r.packed_args ~code_for_op:r.code_for_op ~native:r.native
    ~render:r.render ~compiler:r.compiler r.target

(* CUDA's renderer of [r]'s target without asynchronous copies, which stages
   in one stage. *)
let one_stage (r : Renderer.t) =
  Renderer.with_compiler r.compiler (Cstyle.cuda_sync r.target)

let target device renderer arch =
  { Helpers.Target.device; renderer; arch; interface = ""; indices = "" }

(* CUDA's renderer of an sm_89, which compiles nothing, skipping where it
   cannot be made. *)
let sm89 ?(sync = false) () =
  let t = target "CUDA" "CUDA" "sm_89" in
  match if sync then Cstyle.cuda_sync t else Cstyle.cuda t with
  | r -> Renderer.with_compiler (Renderer.Compiler.v Fun.id) r
  | exception Failure why -> skip ~reason:why ()

(* Programs *)

let says s m =
  let n = String.length s in
  let rec at i =
    i + n <= String.length m && (String.sub m i n = s || at (i + 1))
  in
  at 0

let compiled r k =
  match Codegen.to_program k r with
  | prg -> prg
  | exception Renderer.Compiler.Compile_error m
    when says "cannot open source file" m ->
      skip ~reason:("a header is missing, set CUDA_PATH: " ^ m) ()

let source prg =
  match Ops.arg (Ops.nth prg 2) with
  | String src -> src
  | _ -> failwith "a program's third source is its source"

let count s text =
  let n = String.length s in
  let rec go i acc =
    if i + n > String.length text then acc
    else if String.sub text i n = s then go (i + n) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

(* The buffers in shared memory of a linearized program. *)
let locals lin =
  List.filter
    (fun u -> (is Alloc u || is Buffer u) && Ops.addrspace u = Some Dtype.Local)
    (Ops.toposort ~calls:Skip lin)

(* Draws *)

let dims = Gen.of_list [ 16; 32; 48; 64; 96; 128; 256 ]

let products =
  let open Gen in
  let+ dt = of_list [ Dtype.Float16; Dtype.Bfloat16 ]
  and+ batch = frequency [ (3, constant 1); (1, constant 2) ]
  and+ m = dims
  and+ n = dims
  and+ k = of_list [ 32; 48; 64; 160; 256 ]
  and+ padded = frequency [ (4, constant false); (1, constant true) ]
  and+ layout = of_list [ Kn; Nk ]
  and+ toeplitz = frequency [ (5, constant false); (1, constant true) ] in
  let m, k = if padded then (m - 4, k - 8) else (m, k) in
  { dt; batch; m; k; n; layout; toeplitz }

let pp_product ppf p =
  Format.fprintf ppf "%a%s [%d; %d] by %s [%d; %d], %d times" Dtype.pp p.dt
    (if p.toeplitz then " toeplitz" else "")
    p.m p.k
    (match p.layout with Kn -> "[K; N]" | Nk -> "[N; K]")
    p.k p.n p.batch

(* The bits law *)

(* A kernel staged computes the bits of the kernel of the same optimisations
   unstaged: staging moves operands and never rounds them. *)
let same_bits ((p, staged_tile, depth, seed) as _case) =
  let d, r = on_cuda () in
  let staged = if depth = 2 then r else one_stage r in
  let opts = chosen ~staged:staged_tile p r in
  let k = kernel ~opts_to_apply:opts p in
  let inputs =
    [ (1, values seed (a_size p)); (2, values (seed + 1) (p.batch * p.k * p.n)) ]
  in
  let run r = List.assoc 0 (Run.on d (compiled r k) inputs) in
  let prg = compiled staged k in
  let src = source prg in
  let cored = says "mma.sync" src in
  let operands = List.length (locals (Ops.nth prg 1)) in
  let shared = operands > 0 in
  cover "both operands staged" (cored && operands = 2);
  cover "an operand refused, the other staged" (cored && operands = 1);
  cover "an axis the core pads"
    (cored && shared && (p.m mod 16 <> 0 || p.k mod 16 <> 0));
  cover "two stages, cp.async" (shared && says "cp.async" src);
  cover "one stage" (shared && depth = 1);
  cover "the staged tile" (shared && staged_tile);
  cover "the unstaged tile, staged" (shared && not staged_tile);
  cover "batched" (shared && p.batch > 1);
  cover "an [N; K] operand" (shared && p.layout = Nk);
  equal (array Dtypes.value) (run (unstaged r)) (run staged)

let law =
  prop ~count:24
    ~examples:
      [
        ( { dt = Bfloat16; batch = 1; m = 128; k = 64; n = 128; layout = Kn;
            toeplitz = false },
          true, 2, 1 );
        ( { dt = Float16; batch = 2; m = 60; k = 56; n = 32; layout = Nk;
            toeplitz = false },
          true, 1, 2 );
        ( { dt = Bfloat16; batch = 1; m = 64; k = 64; n = 64; layout = Kn;
            toeplitz = true },
          false, 2, 3 );
        ( { dt = Float16; batch = 1; m = 96; k = 160; n = 64; layout = Kn;
            toeplitz = false },
          false, 2, 4 );
      ]
    "a staged kernel computes the bits of the same optimisations unstaged"
    (Gen.with_pp
       (fun ppf (p, staged, depth, seed) ->
         Format.fprintf ppf "%a, %s tile, %d stages, seed %d" pp_product p
           (if staged then "staged" else "unstaged")
           depth seed)
       Gen.(
         quad products bool (of_list [ 1; 2 ]) (int_range 0 1_000_000)))
    same_bits

(* The barriers of a staged kernel

   Every thread runs the same instructions, so two accesses of one element of
   shared memory by different threads are ordered only by a barrier between
   them in the instructions. Over two trips of each loop, every store into
   shared memory and every load of an element it may store, in either order,
   are separated by a barrier: a load after the store reads what it stored,
   and a store after the load does not overwrite what the load reads. An
   asynchronous copy completes only at a barrier that waits for it. *)

type access = { at : int; buf : Ops.t; write : bool; elements : int list }

(* The value of the integer or boolean node [u] where each hardware index and
   range is bound by [env]. *)
let rec eval env u =
  match Ops.op u with
  | Op.Const -> Ops.value u
  | Op.Special | Op.Range -> `Int (Bigint.of_int (List.assq u env))
  | Op.Cast | Op.After -> eval env (Ops.nth u 0)
  | o when Op.Set.mem o Op.Set.alu ->
      Ops.exec_alu o (Ops.dtype u) (List.map (eval env) (Ops.src u))
  | o -> failf "an index computes %s" (Op.name o)

let to_int = function `Int z -> Bigint.to_int z | _ -> failwith "an integer"
let holds = function `Bool b -> b | _ -> failwith "a boolean"

(* The values of the threads of a workgroup: each binding of the local
   indices of [instrs], and every workgroup index at [0]. *)
let threads instrs =
  let specials = List.filter (is Special) instrs in
  let size u = Dtype.Value.to_int (Ops.vmax u) + 1 in
  List.fold_left
    (fun envs u ->
      match Ops.arg u with
      | String n when n.[0] = 'l' ->
          List.concat_map
            (fun env -> List.init (size u) (fun v -> (u, v) :: env))
            envs
      | _ -> List.map (fun env -> (u, 0) :: env) envs)
    [ [] ] specials

(* The accesses of shared memory and the barriers ([None]) of the instructions
   [instrs], each loop walked for its first two trips: the elements each
   access reads or writes over the threads that reach it. *)
let walk instrs =
  let all = threads instrs in
  let instrs = Array.of_list instrs in
  let access i trips conds x write =
    let at = Ops.nth x 0 in
    if Ops.addrspace at <> Some Dtype.Local then []
    else
      let width = Shape.max_numel at in
      let elements =
        List.concat_map
          (fun env ->
            let env = trips @ env in
            if List.for_all (fun c -> holds (eval env c)) conds then
              let i = to_int (eval env (Ops.nth at 1)) in
              List.init width (fun e -> i + e)
            else [])
          all
      in
      [ Some { at = i; buf = Ops.buf_uop at; write; elements } ]
  in
  let closes r e = is End e && List.memq r (List.tl (Ops.src e)) in
  let rec from trips conds i stop =
    if i >= stop then []
    else
      let u = instrs.(i) in
      match Ops.op u with
      | Op.Range ->
          let close = ref (i + 1) in
          while not (closes u instrs.(!close)) do incr close done;
          let n = Dtype.Value.to_int (Ops.vmax u) + 1 in
          List.concat
            (List.init (min 2 n) (fun t ->
                 from ((u, t) :: trips) conds (i + 1) !close))
          @ from trips conds (!close + 1) stop
      | Op.If -> from trips (Ops.nth u 0 :: conds) (i + 1) stop
      | Op.Endif -> from trips (List.tl conds) (i + 1) stop
      | Op.Barrier -> None :: from trips conds (i + 1) stop
      | Op.Store -> access i trips conds u true @ from trips conds (i + 1) stop
      | Op.Load -> access i trips conds u false @ from trips conds (i + 1) stop
      | _ -> from trips conds (i + 1) stop
  in
  from [] [] 0 (Array.length instrs)

(* The first store and load of one element with no barrier between them. *)
let unordered events =
  let rec go seen = function
    | [] -> None
    | None :: rest -> go [] rest
    | Some a :: rest -> (
        let shared b = List.exists (fun e -> List.mem e a.elements) b.elements in
        let clash b = b.buf == a.buf && b.write <> a.write && shared b in
        match List.find_opt clash seen with
        | Some b -> Some (b, a)
        | None -> go (a :: seen) rest)
  in
  go [] events

let pp_access ppf a =
  let lo = List.fold_left min max_int a.elements
  and hi = List.fold_left max min_int a.elements in
  Format.fprintf ppf "%s of elements %d to %d, instruction %d"
    (if a.write then "a store" else "a load")
    lo hi a.at

let wait_all = {|asm volatile("cp.async.wait_all;" ::: "memory"); __syncthreads();|}

(* The renderers the law draws: CUDA's in two stages and in one, and HIP's
   for RDNA3, whose A operand broadcasts a lane bit, and RDNA4, both in one
   stage of copies through registers. *)
type target = Cuda of int | Hip of string

let renderer_of = function
  | Cuda depth -> sm89 ~sync:(depth = 1) ()
  | Hip arch -> (
      match Cstyle.hip (target "AMD" "HIP" arch) with
      | r -> Renderer.with_compiler (Renderer.Compiler.v Fun.id) r
      | exception Failure why -> skip ~reason:why ())

let ordered (p, t) =
  let r = renderer_of t in
  let depth = match t with Cuda d -> d | Hip _ -> 1 in
  let k = kernel ~opts_to_apply:(chosen ~staged:true p r) p in
  let prg = Codegen.to_program k r in
  let instrs = Ops.src (Ops.nth prg 1) in
  let src = source prg in
  let shared = locals (Ops.nth prg 1) <> [] in
  cover "two stages" (shared && depth = 2);
  cover "one stage" (shared && depth = 1);
  cover "HIP" (shared && match t with Hip _ -> true | Cuda _ -> false);
  cover "a loop" (shared && List.exists (is Range) instrs);
  (match unordered (walk instrs) with
  | None -> ()
  | Some (a, b) ->
      failf "%a then %a with no barrier between them" pp_access a pp_access b);
  if says "tg_cp_async(" src then
    equal ~msg:"barriers that wait for the copies" int
      (count "__syncthreads();" src)
      (count wait_all src)

let barriers =
  prop ~count:40
    ~examples:
      [
        ( { dt = Bfloat16; batch = 1; m = 128; k = 64; n = 128; layout = Kn;
            toeplitz = false },
          Cuda 2 );
        ( { dt = Float16; batch = 1; m = 64; k = 256; n = 64; layout = Nk;
            toeplitz = false },
          Cuda 1 );
        ( { dt = Float16; batch = 1; m = 64; k = 128; n = 64; layout = Kn;
            toeplitz = false },
          Hip "gfx1100" );
        ( { dt = Bfloat16; batch = 1; m = 128; k = 64; n = 128; layout = Nk;
            toeplitz = false },
          Hip "gfx1201" );
      ]
    "a staged kernel separates each store into shared memory and each load of \
     its elements by a barrier, over two trips"
    (Gen.with_pp
       (fun ppf (p, t) ->
         Format.fprintf ppf "%a, %s" pp_product p
           (match t with
           | Cuda d -> Printf.sprintf "CUDA, %d stages" d
           | Hip arch -> "HIP " ^ arch))
       (Gen.pair products
          (Gen.of_list [ Cuda 1; Cuda 2; Hip "gfx1100"; Hip "gfx1201" ])))
    ordered

(* Races

   The race checker of CUDA's toolkit, [compute-sanitizer], run on this suite's
   staged kernels, reports any two accesses of one element of shared memory by
   different threads with no barrier between them, as hazards, whether or not
   the values they move happen to agree. Its default report waits for a CUDA
   synchronisation, which rig's waits never call, so the suite asks for each
   hazard. *)

(* Integers from -2 to 2: a product's sums are exact. *)
let small seed size =
  let st = Random.State.make [| seed |] in
  Array.init size (fun _ -> `Float (Float.of_int (Random.State.int st 5 - 2)))

let exact p a b =
  Array.init (p.m * p.n) (fun e ->
      let i = e / p.n and j = e mod p.n in
      let s = ref 0. in
      for q = 0 to p.k - 1 do
        let v = function `Float f -> f | _ -> 0. in
        s := !s +. (v a.((i * p.k) + q) *. v b.((q * p.n) + j))
      done;
      `Float !s)

let cube =
  { dt = Bfloat16; batch = 1; m = 128; k = 128; n = 128; layout = Kn;
    toeplitz = false }

(* A product of 128 cubed in [depth] stages is the exact product. *)
let run_staged depth () =
  let d, r = on_cuda () in
  let r = if depth = 2 then r else one_stage r in
  let k = kernel ~opts_to_apply:(chosen ~staged:true cube r) cube in
  let prg = compiled r k in
  equal ~msg:"operands staged" int 2 (List.length (locals (Ops.nth prg 1)));
  let a = small 1 (cube.m * cube.k) and b = small 2 (cube.k * cube.n) in
  equal (array Dtypes.value) (exact cube a b)
    (List.assoc 0 (Run.on d prg [ (1, a); (2, b) ]))

let checked = "kernels the race checker runs"
let in_two = "a product of 128 cubed in two stages of cp.async is exact"
let in_one = "a product of 128 cubed in one stage is exact"

let checked_kernels =
  group checked [ test in_two (run_staged 2); test in_one (run_staged 1) ]

(* [compute-sanitizer] attaches to the process it starts, so each kernel runs
   in this suite's own process ([--in-process]). *)
let racecheck kernel () =
  ignore (on_cuda ());
  let tool =
    match Sys.getenv_opt "CUDA_PATH" with
    | Some root -> Filename.concat root "bin/compute-sanitizer"
    | None -> skip ~reason:"CUDA_PATH names no toolkit" ()
  in
  if not (Sys.file_exists tool) then skip ~reason:(tool ^ " is missing") ();
  let logs = Filename.temp_dir "racecheck" "" in
  let args =
    [| tool; "--tool"; "racecheck"; "--racecheck-report"; "hazard";
       "--error-exitcode"; "1"; Sys.executable_name; "-o"; logs;
       "--in-process"; "-f"; checked ^ " › " ^ kernel |]
  in
  let out = Unix.open_process_args_in tool args in
  let text = In_channel.input_all out in
  match Unix.close_process_in out with
  | Unix.WEXITED 0 -> ()
  | _ -> failf "the race checker reports:\n%s" text

let races =
  group "races"
    [
      test "the race checker finds no hazard in a kernel staged in two stages"
        (racecheck in_two);
      test "the race checker finds no hazard in a kernel staged in one stage"
        (racecheck in_one);
    ]

(* The layout *)

(* The 16-byte slot of a 128-byte line that element [e] of 2 bytes falls on. *)
let slot e = e * 2 / 16 mod 8

let at ~cols col row =
  Shape.sym_infer (Sym (Stage.layout ~itemsize:2 ~cols (Ops.int col) (Ops.int row))) []

let distinct l = List.length (List.sort_uniq Int.compare l)

let spread chunks =
  let cols = 8 * chunks in
  for col = 0 to chunks - 1 do
    for row0 = 0 to 7 do
      let rows =
        List.init 8 (fun r -> slot (at ~cols (8 * col) ((8 * row0) + r)))
      in
      equal
        ~msg:
          (Printf.sprintf "slots of rows %d to %d at chunk %d" (8 * row0)
             ((8 * row0) + 7) col)
        int 8 (distinct rows)
    done
  done;
  (* A copy's 8 threads of a phase write 8 consecutive chunks, from a
     multiple of 8. *)
  for group = 0 to (chunks / 2) - 1 do
    let first = 8 * group in
    let chunk f = slot (at ~cols (8 * (f mod chunks)) (f / chunks)) in
    equal
      ~msg:(Printf.sprintf "slots of chunks %d to %d" first (first + 7))
      int 8
      (distinct (List.init 8 (fun f -> chunk (first + f))))
  done

let layout =
  group "the layout"
    [
      cases "eight rows at a chunk, and eight chunks in a row's order, fall \
             on eight slots of a line"
        ~id:(Printf.sprintf "%d chunks a row") (List.init 64 succ) spread;
      cases "rows of 128 and 256 bytes are CuTe's swizzles"
        ~id:(fun (cols, col, row, _) -> Printf.sprintf "%d columns, row %d, column %d" cols row col)
        [
          (64, 0, 0, 0); (64, 0, 1, 72); (64, 8, 1, 64); (64, 17, 3, 201);
          (128, 0, 1, 136); (128, 40, 2, 312); (128, 64, 9, 1224);
        ]
        (fun (cols, col, row, offset) -> equal int offset (at ~cols col row));
    ]

(* Stages

   Two stages where the target copies asynchronously and both fit the shared
   memory beside what other cores take, one where only one fits or copies
   wait, none where one does not fit, a tile holds less than a copy of 16
   bytes, the target has no shared memory, or staging is off. *)

let cuda_target = Result.get_ok (Helpers.Target.of_string "CUDA:CUDA:sm_89")

let target_with ~async ~shared =
  Renderer.v ~has_shared:shared ~async_copy:async ~shared_max:49152
    cuda_target

let stages =
  cases "a core takes as many stages as its tiles fit"
    ~id:(fun (name, _, _, _, _, _) -> name)
    [
      ("two that fit", true, true, 0, [ 8192; 8192 ], 2);
      ("one where two do not fit", true, true, 0, [ 16384; 16384 ], 1);
      ("one where copies wait", false, true, 0, [ 8192; 8192 ], 1);
      ("none past shared memory", true, true, 0, [ 32768; 32768 ], 0);
      ("none beside other cores' tiles", true, true, 40000, [ 8192; 8192 ], 0);
      ("one beside other cores' tiles", true, true, 20000, [ 8192; 8192 ], 1);
      ("none for a tile under a copy", true, true, 0, [ 8; 8192 ], 0);
      ("none without shared memory", true, false, 0, [ 8192; 8192 ], 0);
    ]
    (fun (_, async, shared, used, tiles, expected) ->
      equal int expected
        (Stage.depth ~used (target_with ~async ~shared) tiles))

let stage_off () =
  Setting.context
    [ Setting.B (Setting.stage, false) ]
    (fun () ->
      equal int 0
        (Stage.depth (target_with ~async:true ~shared:true) [ 8192; 8192 ]))

(* The staged tile

   A staged core's workgroup is 2 by 2 warps, 128 threads, of 64 by 64
   outputs, each halved, M then N, to 32 by 32 while the kernel keeps fewer
   than 16384 / stages threads; K is unrolled to 32. *)

let tile (dt, m, k, n, sync) =
  let r = sm89 ~sync () in
  let p = { dt; batch = 1; m; k; n; layout = Kn; toeplitz = false } in
  let s = Postrange.Scheduler.v (kernel p) r in
  Postrange.Scheduler.convert_loop_to_global s;
  let o =
    Setting.context
      [ Setting.B (Setting.stage, true) ]
      (fun () -> Heuristic.hand_coded_optimizations s)
  in
  let module K = Postrange.Scheduler in
  let size i = match List.nth (K.full_shape o) i with Int n -> n | Sym _ -> 0 in
  let product l = List.fold_left (fun n i -> n * size i) 1 l in
  ( product (K.axes_of o [ Global ]),
    product (K.axes_of o [ Warp; Local ]),
    product (K.axes_of ~reduce:true o [ Weak; Loop ]) )

let staged_tiles =
  cases "a staged core's workgroups, threads and trips"
    ~id:(fun ((_, m, k, n, sync), _) ->
      Printf.sprintf "%d by %d over %d, %s" m n k
        (if sync then "one stage" else "two stages"))
    [
      ((Dtype.Bfloat16, 2048, 2048, 2048, false), (256, 128, 64));
      ((Dtype.Bfloat16, 512, 2880, 4096, false), (128, 128, 90));
      ((Dtype.Bfloat16, 512, 2880, 5760, false), (180, 128, 90));
      ((Dtype.Bfloat16, 512, 2880, 512, false), (64, 128, 90));
      ((Dtype.Bfloat16, 512, 2880, 4096, true), (128, 128, 90));
      ((Dtype.Float16, 256, 2880, 256, true), (16, 128, 90));
    ]
    (fun (case, expected) ->
      equal (triple int int int) expected (tile case))

(* Sources *)

let rendered p file =
  let r = sm89 () in
  let k = kernel ~opts_to_apply:(chosen ~staged:true p r) p in
  expect_file ~file (source (Codegen.to_program k r))

let sources =
  group "sources"
    [
      test "a bfloat16 product of 2048 cubed" (fun () ->
          rendered
            { dt = Bfloat16; batch = 1; m = 2048; k = 2048; n = 2048;
              layout = Kn; toeplitz = false }
            "packages/tolk/test/codegen/stage/bf16_2048.cu");
      test "a float16 product whose core pads M and K" (fun () ->
          rendered
            { dt = Float16; batch = 1; m = 60; k = 56; n = 64; layout = Kn;
              toeplitz = false }
            "packages/tolk/test/codegen/stage/f16_padded.cu");
    ]

let () =
  exit
    (Windtrap.run "Tolk.Stage"
       [
         group "values" [ law ];
         group "barriers" [ barriers ];
         checked_kernels;
         races;
         layout;
         group "stages" [ stages; test "none when staging is off" stage_off ];
         staged_tiles;
         sources;
       ])
