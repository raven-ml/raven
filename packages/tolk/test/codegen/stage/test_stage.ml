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
  let z = Ops.range ~axis_type:Weak (Int p.batch) [ 0 ]
  and i = Ops.range ~axis_type:Weak (Int p.m) [ 1 ]
  and j = Ops.range ~axis_type:Weak (Int p.n) [ 2 ]
  and r = Ops.range ~axis_type:Weak (Int p.k) [ 3 ] in
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

(* [r] rebuilt with [has_shared] and [registers], named [name]. *)
let rebuilt ~name ?(has_shared = true) ?registers (r : Renderer.t) =
  let registers = Option.value registers ~default:r.registers in
  Renderer.v ~name ~suffix:r.suffix ~supports_float4:r.supports_float4
    ~half8:r.half8 ~registers ~vector_lanes:r.vector_lanes
    ~has_local:r.has_local ~has_shared ~global_max:r.global_max
    ~local_max:r.local_max ?global_prod_max:r.global_prod_max
    ~shared_max:r.shared_max ~async_copy:r.async_copy
    ~tensor_cores:r.tensor_cores
    ~per_byte_first:r.per_byte_first ~extra_matcher:r.extra_matcher
    ~packed_args:r.packed_args ~code_for_op:r.code_for_op ~native:r.native
    ~render:r.render ~compiler:r.compiler r.target

(* [r] without shared memory: it stages nothing. *)
let unstaged (r : Renderer.t) =
  rebuilt ~name:(r.name ^ " without shared memory") ~has_shared:false r

(* CUDA's renderer of [r]'s target without asynchronous copies, which stages
   in one stage. *)
let one_stage (r : Renderer.t) =
  Renderer.with_compiler r.compiler (Cstyle.cuda_sync r.target)

let target device renderer arch =
  { Helpers.Target.device; renderer; arch; interface = ""; indices = "" }

(* CUDA's renderer of an sm_89, which compiles nothing. *)
let sm89 ?(sync = false) () =
  let t = target "CUDA" "CUDA" "sm_89" in
  if sync then Cstyle.cuda_sync t else Cstyle.without_compiler t

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
    [
      (1, values seed (a_size p));
      (2, values (seed + 1) (p.batch * p.k * p.n));
    ]
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
        let shared b =
          List.exists (fun e -> List.mem e a.elements) b.elements
        in
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

let wait_all =
  {|asm volatile("cp.async.wait_all;" ::: "memory"); __syncthreads();|}

(* The renderers the law draws: CUDA's in two stages and in one, and HIP's
   for RDNA3, whose A operand broadcasts a lane bit, and RDNA4, both in one
   stage of copies through registers. *)
type target = Cuda of int | Hip of string

let renderer_of = function
  | Cuda depth -> sm89 ~sync:(depth = 1) ()
  | Hip arch -> Cstyle.without_compiler (target "AMD" "HIP" arch)

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

(* A core under another sum

   [c[i, j] = Σ_h Σ_r a[i, r] · b'[h, r, j]], [b'] the bfloat16 of
   [b[h, r, j] · s[h]] computed in the kernel: [a]'s tile reads [i] and [r]
   alone, [b'] has no tile, and the core's loop [r] runs inside the sum over
   [h], in two stages. A first tile copied once, outside [h], would leave slot
   [0] holding a chunk of [r]'s last trips at [h]'s second trip. Integers from -2 to 2 times
   scales of [1] or [2]: every sum is exact. *)
let outer = (64, 4, 64, 64)

let outer_kernel ?opts_to_apply () =
  let m, h, k, n = outer in
  let c = Shape.param ~shape:[ Ops.Int (m * n) ] 0 Float32
  and a = Shape.param ~shape:[ Ops.Int (m * k) ] 1 Dtype.Bfloat16
  and b = Shape.param ~shape:[ Ops.Int (h * k * n) ] 2 Dtype.Bfloat16
  and s = Shape.param ~shape:[ Ops.Int h ] 3 Float32 in
  let kn = k * n in
  let open Ops.O in
  let i = Ops.range ~axis_type:Weak (Int m) [ 0 ]
  and j = Ops.range ~axis_type:Weak (Int n) [ 1 ]
  and g = Ops.range ~axis_type:Weak (Int h) [ 2 ]
  and r = Ops.range ~axis_type:Weak (Int k) [ 3 ] in
  let at x idx = Ops.cast (Ops.index x [ idx ]) Float32 in
  let scaled =
    Ops.cast
      (Ops.cast
         (at b ((g * int kn) + (r * int n) + j) * Ops.index s [ g ])
         Dtype.Bfloat16)
      Float32
  in
  let sum = Ops.reduce (at a ((i * int k) + r) * scaled) Add [ g; r ] in
  let store = Ops.store (Ops.index c [ (i * int n) + j ]) sum in
  Ops.sink ~kernel:(Ops.kernel_info ?opts_to_apply ())
    [ Ops.end_ store [ i; j ] ]

let outer_exact a b sc =
  let m, h, k, n = outer in
  let v = function `Float f -> f | _ -> 0. in
  Array.init (m * n) (fun e ->
      let i = e / n and j = e mod n in
      let t = ref 0. in
      for g = 0 to h - 1 do
        for q = 0 to k - 1 do
          t :=
            !t
            +. v a.((i * k) + q)
               *. (v b.((g * k * n) + (q * n) + j) *. v sc.(g))
        done
      done;
      `Float !t)

let core_under_a_sum () =
  let d, r = on_cuda () in
  let opts =
    Setting.context
      Setting.[ B (stage, true) ]
      (fun () ->
        let s = Postrange.Scheduler.v (outer_kernel ()) r in
        Postrange.Scheduler.convert_loop_to_global s;
        Postrange.Scheduler.applied_opts
          (Heuristic.hand_coded_optimizations s))
  in
  let prg = compiled r (outer_kernel ~opts_to_apply:opts ()) in
  let src = source prg in
  equal ~msg:"a core" bool true (says "mma.sync" src);
  equal ~msg:"a's tile staged" int 1 (List.length (locals (Ops.nth prg 1)));
  (* Two stages: the first trip's copy and the next trip's, beside the
     definition of the copy. *)
  equal ~msg:"two stages" int 3 (count "tg_cp_async(" src);
  let m, h, k, n = outer in
  let a = small 1 (m * k) and b = small 2 (h * k * n) in
  let sc = Array.init h (fun g -> `Float (Float.of_int (1 + (g mod 2)))) in
  equal (array Dtypes.value) (outer_exact a b sc)
    (List.assoc 0 (Run.on d prg [ (1, a); (2, b); (3, sc) ]))

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
  group checked
    [
      test in_two (run_staged 2);
      test in_one (run_staged 1);
    ]

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

(* Serial ranges around a core

   [c.(o, i, j) = Σ_r a.(i, r) * b.(o, r + j)] in float32, or its sum over [o]
   as well, for [a] of [m] by [k] and [b] Toeplitz in each of [outer] blocks,
   of bfloat16. [b] reads [o] and its tile overlaps, so it stays unstaged; [a]
   reads no [o] and stages, its tile stepping through the core's loop [l]. The
   core runs inside [o], and [l]'s loop inside or around [o]'s:

   {v
     for o:            serial: a loop, a bound in memory, a sum outside l
       for l:          the core's loop, copying a's tiles
         c += a[l] * b[o, l]
   v}

   [a]'s first tile is copied before [l]'s loop at each trip of [o] around it:
   copied once, outside [o], its slot would hold a later tile of [l] at [o]'s
   second trip. Inside [l], [o] reads the tile of [l]'s trip. *)

type around =
  | Loop_range  (** [o] a {!Ops.Axis_type.Loop}. *)
  | Bound_in_memory  (** [o] weak, of a bound read from memory. *)
  | Sum_outside  (** [o] closed by the core's sum, outside [l]. *)
  | Sum_inside  (** [o] closed by the core's sum, inside [l]. *)

let pp_around ppf a =
  Format.pp_print_string ppf
    (match a with
    | Loop_range -> "a loop"
    | Bound_in_memory -> "a weak range bounded by memory"
    | Sum_outside -> "a range the core's sum closes outside its loop"
    | Sum_inside -> "a range the core's sum closes inside its loop")

let summed = function
  | Sum_outside | Sum_inside -> true
  | Loop_range | Bound_in_memory -> false

type nest = { around : around; outer : int; m : int; k : int; n : int }

(* The sum's ranges nest in the order of their identities, the first
   outermost: [o] takes [0] to run outside [r]'s [3], and [4] inside. *)
let nested ?opts_to_apply p =
  let c =
    Shape.param
      ~shape:[ Ops.Int ((if summed p.around then 1 else p.outer) * p.m * p.n) ]
      0 Float32
  and a = Shape.param ~shape:[ Ops.Int (p.m * p.k) ] 1 Bfloat16
  and b = Shape.param ~shape:[ Ops.Int (p.outer * (p.k + p.n)) ] 2 Bfloat16
  and bound = Shape.param ~shape:[ Ops.Int 1 ] 3 Int32 in
  let rows = p.k + p.n and mn = p.m * p.n in
  let open Ops.O in
  let o =
    match p.around with
    | Loop_range -> Ops.range ~axis_type:Loop (Int p.outer) [ 0 ]
    | Bound_in_memory ->
        let n = Ops.load (Ops.index bound [ int 0 ]) [] in
        Ops.range ~axis_type:Weak (Sym (Ops.cast n Dtype.Weak_int)) [ 0 ]
    | Sum_outside -> Ops.range ~axis_type:Weak (Int p.outer) [ 0 ]
    | Sum_inside -> Ops.range ~axis_type:Weak (Int p.outer) [ 4 ]
  and i = Ops.range ~axis_type:Weak (Int p.m) [ 1 ]
  and j = Ops.range ~axis_type:Weak (Int p.n) [ 2 ]
  and r = Ops.range ~axis_type:Weak (Int p.k) [ 3 ] in
  let at x idx = Ops.cast (Ops.index x [ idx ]) Float32 in
  let prod = at a ((i * int p.k) + r) * at b ((o * int rows) + r + j) in
  let store =
    if summed p.around then
      Ops.end_
        (Ops.store
           (Ops.index c [ (i * int p.n) + j ])
           (Ops.reduce prod Add [ o; r ]))
        [ i; j ]
    else
      Ops.end_
        (Ops.store
           (Ops.index c [ (o * int mn) + (i * int p.n) + j ])
           (Ops.reduce prod Add [ r ]))
        [ o; i; j ]
  in
  Ops.sink ~kernel:(Ops.kernel_info ?opts_to_apply ()) [ store ]

let nested_exact p a b =
  let v = function `Float f -> f | _ -> 0. in
  let one o i j =
    let s = ref 0. in
    for q = 0 to p.k - 1 do
      s := !s +. (v a.((i * p.k) + q) *. v b.((o * (p.k + p.n)) + q + j))
    done;
    !s
  in
  let mn = p.m * p.n in
  if summed p.around then
    Array.init mn (fun e ->
        let s = ref 0. in
        for o = 0 to p.outer - 1 do
          s := !s +. one o (e / p.n) (e mod p.n)
        done;
        `Float !s)
  else
    Array.init (p.outer * mn) (fun e ->
        `Float (one (e / mn) (e mod mn / p.n) (e mod p.n)))

(* The core over [j], [i] and [r]: the choices pair [a]'s [i] with each of
   [b]'s ranges, [o] then [j], and each range of the sum, the last varying
   fastest. [o] of one trip would leave the kernel, so [outer] is at least
   [2]. *)
let nested_opts around =
  let axis =
    match around with
    | Loop_range | Bound_in_memory -> 1
    | Sum_outside -> 2
    | Sum_inside -> 3
  in
  [ Opt.Tc { axis; tc_select = -1; tc_opt = 2; use_tc = 1 } ]

(* The core's loop runs [k / 16] trips, and [a]'s tile holds the core's 16
   rows by 16 terms. *)
let core_trips = 16
let core_tile = 16 * 16

let nested_value (p, depth, seed) =
  let d, r = on_cuda () in
  let r = if depth = 2 then r else one_stage r in
  let prg = compiled r (nested ~opts_to_apply:(nested_opts p.around) p) in
  let src = source prg and lin = Ops.nth prg 1 in
  equal ~msg:"a core" bool true (says "mma.sync" src);
  (match locals lin with
  | [ buf ] ->
      equal ~msg:"the staged slots" int (depth * core_tile)
        (Shape.max_numel buf)
  | bufs -> failf "%d staged operands, not 1" (List.length bufs));
  (match unordered (walk (Ops.src lin)) with
  | None -> ()
  | Some (a, b) ->
      failf "%a then %a with no barrier between them" pp_access a pp_access b);
  let later = p.k / core_trips > 2 in
  cover "one stage" (depth = 1);
  List.iter
    (fun a ->
      cover
        (Format.asprintf "two stages under %a" pp_around a)
        (depth = 2 && later && p.around = a))
    [ Loop_range; Bound_in_memory; Sum_outside; Sum_inside ];
  let a = small seed (p.m * p.k)
  and b = small (seed + 1) (p.outer * (p.k + p.n)) in
  let bound = [| `Int (Bigint.of_int p.outer) |] in
  equal (array Dtypes.value) (nested_exact p a b)
    (List.assoc 0 (Run.on d prg [ (1, a); (2, b); (3, bound) ]))

let nests =
  let open Gen in
  let+ around = of_list [ Loop_range; Bound_in_memory; Sum_outside; Sum_inside ]
  and+ outer = int_range 2 4
  and+ m = of_list [ 16; 32 ]
  and+ k = of_list [ 32; 48; 64; 128 ]
  and+ n = of_list [ 16; 32 ] in
  { around; outer; m; k; n }

let pp_nest ppf p =
  Format.fprintf ppf "[%d; %d] by [%d; %d] under %a of %d trips" p.m p.k p.k
    p.n pp_around p.around p.outer

let serial =
  let example around = ({ around; outer = 3; m = 16; k = 64; n = 16 }, 2, 1) in
  prop ~count:16
    ~examples:
      (List.map example [ Loop_range; Bound_in_memory; Sum_outside; Sum_inside ]
      @ [ ({ around = Loop_range; outer = 2; m = 32; k = 48; n = 16 }, 1, 2) ])
    "a core staged under a serial range is exact, its copies ordered by \
     barriers"
    (Gen.with_pp
       (fun ppf (p, depth, seed) ->
         Format.fprintf ppf "%a, %d stages, seed %d" pp_nest p depth seed)
       Gen.(triple nests (of_list [ 1; 2 ]) (int_range 0 1_000_000)))
    nested_value

(* The layout *)

(* The 16-byte slot of a 128-byte line that element [e] of 2 bytes falls on. *)
let slot e = e * 2 / 16 mod 8

let at ~cols col row =
  let offset = Stage.layout ~itemsize:2 ~cols (Ops.int col) (Ops.int row) in
  Shape.sym_infer (Sym offset) []

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
        ~id:(fun (cols, col, row, _) ->
          Printf.sprintf "%d columns, row %d, column %d" cols row col)
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


(* Decoded panels on the host

   [c = Σ_r a.(i, r) * (float32 codes.(j, r) * scales.(j, r / 32))] in float32,
   for [a] of [m] by [k], and [n] rows of [k] byte codes, each block of 32
   scaled by its float32 scale: a product whose right operand is decoded from
   integer codes, as a quantised matrix's is. The codes are [Uint8], or [codes]
   when given. *)

type decoded = { dm : int; dk : int; dn : int }

let decoded_kernel ?(codes = Dtype.Uint8) ?opts_to_apply p =
  let c = Shape.param ~shape:[ Ops.Int (p.dm * p.dn) ] 0 Float32
  and a = Shape.param ~shape:[ Ops.Int (p.dm * p.dk) ] 1 Float32
  and codes = Shape.param ~shape:[ Ops.Int (p.dn * p.dk) ] 2 codes
  and scales = Shape.param ~shape:[ Ops.Int (p.dn * p.dk / 32) ] 3 Float32 in
  let blocks = p.dk / 32 in
  let open Ops.O in
  let i = Ops.range ~axis_type:Weak (Int p.dm) [ 0 ]
  and j = Ops.range ~axis_type:Weak (Int p.dn) [ 1 ]
  and r = Ops.range ~axis_type:Weak (Int p.dk) [ 2 ] in
  let value =
    Ops.cast (Ops.index codes [ (j * int p.dk) + r ]) Float32
    * Ops.index scales [ (j * int blocks) + (r // int 32) ]
  in
  let sum = Ops.reduce (Ops.index a [ (i * int p.dk) + r ] * value) Add [ r ] in
  let store = Ops.store (Ops.index c [ (i * int p.dn) + j ]) sum in
  Ops.sink
    ~kernel:(Ops.kernel_info ?opts_to_apply ())
    [ Ops.end_ store [ i; j ] ]

(* Operands over [r], each with whether it is decoded: codes are integers
   loaded from integer storage, through integer arithmetic. *)
let decoded_operands =
  let r = Ops.range ~axis_type:Weak (Ops.Int 64) [ 0 ] in
  let param slot dt = Shape.param ~shape:[ Ops.Int 64 ] slot dt in
  let read slot dt = Ops.index (param slot dt) [ r ] in
  let f32 u = Ops.cast u Float32 and u32 n = Ops.int ~dtype:Uint32 n in
  let h = f32 (read 0 Bfloat16) in
  (* ReLU as nx computes it on a bfloat16: its bits through an integer, the
     sign selected. *)
  let sign = Ops.where (Ops.lt (Ops.float 0.) h) (u32 0xffffffff) (u32 0) in
  let relu =
    Ops.bitcast (Ops.bitwise_and (Ops.bitcast h Uint32) sign) Float32
  in
  let codes = read 1 Uint8 in
  let scale = Ops.bitcast (Ops.shl (Ops.cast codes Uint32) (u32 23)) Float32 in
  [
    ("a byte code", f32 codes, true);
    ("an int8 code by a scale", Ops.mul (f32 (read 2 Int8)) (read 3 Float32),
      true);
    ("an E8M0 scale's bits", scale, true);
    ("a widened bfloat16", h, false);
    ("ReLU of a bfloat16", relu, false);
    ("a range's position", f32 r, false);
    ("a gather at loaded positions",
      Ops.index (param 4 Float32) [ read 5 Int32 ], false);
  ]

let decoded_kinds =
  cases "a decoded operand converts integer codes it loads"
    ~id:(fun (name, _, _) -> name)
    decoded_operands
    (fun (_, o, decoded) -> equal bool decoded (Stage.decoded o))

let host_target = Tolk_engine.target Rig.host
let host = lazy (Cstyle.clang host_target)

(* The host's renderer with a register file of [count] registers of [bytes]
   bytes, which compiles nothing. *)
let host_file (count, bytes) =
  let r = Lazy.force host in
  Renderer.with_compiler
    (Renderer.Compiler.v Fun.id)
    (rebuilt ~name:r.name ~has_shared:r.has_shared
       ~registers:(fun () -> Some { Renderer.count; bytes })
       r)

(* The optimisations the hand-coded rules choose for [p] on [r]. *)
let decoded_opts ?codes p r =
  let s = Postrange.Scheduler.v (decoded_kernel ?codes p) r in
  Postrange.Scheduler.convert_loop_to_global s;
  Postrange.Scheduler.applied_opts (Heuristic.hand_coded_optimizations s)

let staged_program ?codes ~staged p r =
  Setting.context
    Setting.[ B (stage, staged) ]
    (fun () ->
      let k =
        decoded_kernel ?codes ~opts_to_apply:(decoded_opts ?codes p r) p
      in
      Codegen.linearize k r)

(* Whether [u] loads codes: bytes of global memory. *)
let code_load u =
  is Load u
  && Dtype.mem (Ops.dtype u) Dtype.[ Uint8; Int8 ]
  && Ops.addrspace (Ops.nth u 0) = Some Dtype.Global

(* The times the instructions of [prg] decode each code: at each load of a code,
   the product of the trips of the loops around it whose ranges its address does
   not read. *)
let decodes prg =
  let instrs = Ops.src (Ops.nth prg 1) in
  (* A loop the launch splits into blocks runs all its iterations. *)
  let split =
    match Ops.arg (Ops.nth prg 0) with
    | Kernel { split = Some { iterations = Int n; _ }; _ } -> Some n
    | _ -> None
  in
  let constant u =
    List.for_all (fun u -> is Const u || is Cast u) (Ops.toposort ~calls:Skip u)
  in
  let trips r =
    match split with
    | Some n when not (constant (Ops.nth r 0)) -> n
    | _ -> Dtype.Value.to_int (Ops.vmax r) + 1
  in
  (* The ranges an address reads: a range's own sources order its loop after
     others' and are not read. *)
  let reads x =
    let seen = Ops.Tbl.create 64 and acc = ref [] in
    let rec walk u =
      if not (Ops.Tbl.mem seen u) then begin
        Ops.Tbl.add seen u ();
        if is Range u then acc := u :: !acc else List.iter walk (Ops.src u)
      end
    in
    walk x;
    !acc
  in
  let rec go open_ most = function
    | [] -> most
    | u :: rest when is Range u -> go (u :: open_) most rest
    | u :: rest when is End u ->
        go
          (List.filter (fun r -> not (List.memq r (List.tl (Ops.src u)))) open_)
          most rest
    | u :: rest when code_load u ->
        let n =
          List.fold_left
            (fun n r ->
              if List.memq r (reads (Ops.nth u 0)) then n else n * trips r)
            1 open_
        in
        go open_ (max most n) rest
    | _ :: rest -> go open_ most rest
  in
  go [] 0 instrs

(* The rows of a tile: the largest divisor of [m] up to 16 that leaves half the
   registers to the rest, one register a row. *)
let rec tile_rows m at_most =
  if m mod at_most = 0 then at_most else tile_rows m (at_most - 1)

(* A panel holds the values one trip of the columns' loop reads: the 4 columns
   the tile upcasts by [k] values, of 4 bytes. *)
let panel_fits p = 4 * p.dk * 4 <= 64 * 1024

let expected_decodes p (count, _) =
  let trips = p.dm / tile_rows p.dm (min 16 (count / 2)) in
  if trips >= 2 && panel_fits p then 1 else trips

let decoded_cases =
  let shapes =
    List.concat_map
      (fun dm ->
        List.map (fun dk -> { dm; dk; dn = 64 }) [ 32; 1024; 4096; 8192 ])
      [ 1; 2; 8; 10; 16; 17; 24; 64; 128 ]
  in
  let rows =
    List.concat_map
      (fun file -> List.map (fun p -> (p, file)) shapes)
      [ (16, 32); (32, 16) ]
  in
  cases
    "on the host a decoded operand read by several row tiles is decoded once"
    ~id:(fun (p, (c, b)) ->
      Printf.sprintf "%d rows by %d of %d on %dx%d" p.dm p.dk p.dn c b)
    rows
    (fun (p, file) ->
      let prg = staged_program ~staged:true p (host_file file) in
      equal int (expected_decodes p file) (decodes prg))

let decoded_stage_off =
  test "staging off decodes once a row tile" (fun () ->
      let p = { dm = 64; dk = 1024; dn = 64 } in
      equal int 8
        (decodes (staged_program ~staged:false p (host_file (16, 32)))))

(* Floats of 8 significant bits times powers of two, codes of any byte, and
   scales of powers of two: the sums round. *)
let decoded_inputs seed p =
  let st = Random.State.make [| seed |] in
  let float () =
    `Float
      (Float.ldexp
         (Float.of_int (Random.State.int st 256 - 128))
         (Random.State.int st 9 - 8))
  in
  [
    (1, Array.init (p.dm * p.dk) (fun _ -> float ()));
    ( 2,
      Array.init (p.dn * p.dk) (fun _ ->
          `Int (Bigint.of_int (Random.State.int st 256))) );
    ( 3,
      Array.init
        (p.dn * p.dk / 32)
        (fun _ -> `Float (Float.ldexp 1. (Random.State.int st 9 - 4))) );
  ]

let decoded_bits (p, seed) =
  let r = Lazy.force host in
  let run staged =
    let prg = staged_program ~staged p r in
    let prg = Codegen.compile prg r in
    List.assoc 0 (Run.on_host prg (decoded_inputs seed p))
  in
  let staged = decodes (staged_program ~staged:true p r) in
  let unstaged = decodes (staged_program ~staged:false p r) in
  cover "staged" (staged < unstaged);
  cover "one row tile" (unstaged = 1);
  cover "past the panel" (unstaged > 1 && staged = unstaged);
  equal (array Dtypes.value) (run false) (run true)

(* The value of the address [u] at the first trip of every loop, a block's
   loop of a split launch included: a range, or a load of a block's bounds, is
   [0]. *)
let rec at_first u =
  match Ops.op u with
  | Op.Const -> Ops.value u
  | Op.Range | Op.Load -> `Int Bigint.zero
  | Op.Cast | Op.After -> at_first (Ops.nth u 0)
  | o when Op.Set.mem o Op.Set.alu ->
      Ops.exec_alu o (Ops.dtype u) (List.map at_first (Ops.src u))
  | o -> failf "an address computes %s" (Op.name o)

(* The code loads of [prg], each as the first byte it reads at the first trip
   of every loop and the bytes it reads. *)
let code_reads prg =
  List.filter_map
    (fun u ->
      if code_load u then
        let at = Ops.nth u 0 in
        Some (to_int (at_first (Ops.nth at 1)), Shape.max_numel at)
      else None)
    (Ops.src (Ops.nth prg 1))

(* The fewest loads of registers of [bytes] that read the bytes of [reads]: a
   load for each register's width of each run of consecutive bytes. *)
let fewest_loads bytes reads =
  let rec go loads = function
    | [] -> loads
    | (start, width) :: rest ->
        let rec extend stop = function
          | (s, w) :: rest when s = stop -> extend (s + w) rest
          | rest -> (stop, rest)
        in
        let stop, rest = extend (start + width) rest in
        go (loads + ((stop - start + bytes - 1) / bytes)) rest
  in
  go 0 (List.sort compare reads)

let decoded_loads =
  let rows =
    List.concat_map
      (fun file ->
        List.concat_map
          (fun codes ->
            List.concat_map
              (fun dm ->
                List.map
                  (fun dk -> ({ dm; dk; dn = 64 }, codes, file))
                  [ 32; 1024; 4096 ])
              [ 1; 2; 10 ])
          Dtype.[ Uint8; Int8 ])
      [ (16, 16); (16, 32); (32, 64) ]
  in
  cases "on the host a decoded product loads its codes a register at a time"
    ~id:(fun (p, codes, (c, b)) ->
      Format.asprintf "%d rows by %d of %d %a on %dx%d" p.dm p.dk p.dn Dtype.pp
        codes c b)
    rows
    (fun (p, codes, ((_, bytes) as file)) ->
      let reads =
        code_reads (staged_program ~codes ~staged:true p (host_file file))
      in
      equal
        ~msg:
          (String.concat " "
             (List.map (fun (s, w) -> Printf.sprintf "%d+%d" s w) reads))
        int (fewest_loads bytes reads) (List.length reads))

(* Products whose sums are exact in any order: [a] of integers from -4 to 4,
   codes of any value of their type and scales of 1, 2 or 4, so that every
   partial sum of [k] terms up to 4096 is an integer below 2^24. *)
let exact_inputs seed codes p =
  let st = Random.State.make [| seed |] in
  let lo, hi = if codes = Dtype.Int8 then (-128, 127) else (0, 255) in
  let a = Array.init (p.dm * p.dk) (fun _ -> Random.State.int st 9 - 4)
  and c =
    Array.init (p.dn * p.dk) (fun _ -> lo + Random.State.int st (hi - lo + 1))
  and s =
    Array.init (p.dn * p.dk / 32) (fun _ -> 1 lsl Random.State.int st 3)
  in
  (a, c, s)

let decoded_value ((p, codes), seed) =
  let a, c, s = exact_inputs seed codes p in
  let expected =
    Array.init (p.dm * p.dn) (fun e ->
        let i = e / p.dn and j = e mod p.dn in
        let sum = ref 0 in
        for r = 0 to p.dk - 1 do
          let scale = s.((j * p.dk / 32) + (r / 32)) in
          sum := !sum + (a.((i * p.dk) + r) * c.((j * p.dk) + r) * scale)
        done;
        `Float (Float.of_int !sum))
  in
  let r = Lazy.force host in
  let prg = Codegen.compile (staged_program ~codes ~staged:true p r) r in
  let float x = `Float (Float.of_int x) in
  let int x = `Int (Bigint.of_int x) in
  let got =
    List.assoc 0
      (Run.on_host prg
         [
           (1, Array.map float a);
           (2, Array.map int c);
           (3, Array.map float s);
         ])
  in
  equal (array Dtypes.value) expected got

let decoded_values =
  let gen =
    let open Gen in
    let+ dm = of_list [ 1; 2; 3; 10 ]
    and+ dk = of_list [ 32; 96; 1024; 4096 ]
    and+ dn = of_list [ 4; 8; 64 ]
    and+ codes = of_list Dtype.[ Uint8; Int8 ]
    and+ seed = int_range 0 1_000_000 in
    (({ dm; dk; dn }, codes), seed)
  in
  prop ~count:12
    ~examples:
      [
        (({ dm = 1; dk = 1024; dn = 64 }, Dtype.Uint8), 1);
        (({ dm = 1; dk = 4096; dn = 8 }, Dtype.Int8), 2);
        (({ dm = 10; dk = 32; dn = 64 }, Dtype.Uint8), 3);
      ]
    "on the host a decoded product computes the product of the decoded codes"
    (Gen.with_pp
       (fun ppf ((p, codes), seed) ->
         Format.fprintf ppf "%d rows by [%d; %d] of %a, seed %d" p.dm p.dn p.dk
           Dtype.pp codes seed)
       gen)
    decoded_value

let decoded_law =
  let gen =
    let open Gen in
    let+ dm = of_list [ 1; 3; 8; 10; 16; 24; 64 ]
    and+ dk =
      frequency [ (4, of_list [ 32; 96; 256; 1024 ]); (1, constant 8192) ]
    and+ dn = of_list [ 4; 8; 12; 32 ]
    and+ seed = int_range 0 1_000_000 in
    ({ dm; dk; dn }, seed)
  in
  prop ~count:16
    ~examples:
      [
        ({ dm = 64; dk = 256; dn = 8 }, 1);
        ({ dm = 10; dk = 1024; dn = 4 }, 2);
        ({ dm = 1; dk = 64; dn = 4 }, 3);
        ({ dm = 64; dk = 8192; dn = 8 }, 4);
      ]
    "on the host a staged decoded product computes the bits of the same \
     optimisations unstaged"
    (Gen.with_pp
       (fun ppf (p, seed) ->
         Format.fprintf ppf "%d rows by [%d; %d], seed %d" p.dm p.dn p.dk seed)
       gen)
    decoded_bits

let () =
  exit
    (Windtrap.run "Tolk.Stage"
       [
         group "values" [ law ];
         group "barriers" [ barriers ];
         checked_kernels;
         races;
         group "serial ranges" [ serial ];
         layout;
         group "stages" [ stages; test "none when staging is off" stage_off ];
         test
           "a core under another sum whose tile does not read it is the exact \
            product"
           core_under_a_sum;
         staged_tiles;
         sources;
         group "decoded panels"
           [
             decoded_kinds;
             decoded_law;
             decoded_cases;
             decoded_stage_off;
             decoded_loads;
             decoded_values;
           ];
       ])
