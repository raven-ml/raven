open Windtrap
open Tolk

let is o u = Op.equal (Ops.op u) o

(* Products

   [c = Σ_r float32 a.(i, r) * float32 b.(r, j)] in float32, [batch] times, for
   [a] of [m] by [k] and [b] of [k] by [n], of 16-bit floats: as rune lowers a
   contraction of narrow floats. [b] is stored as [k] rows of [n] ([K; N]) or as
   [n] rows of [k] ([N; K]). A Toeplitz [a] reads row [i] from element [i] on,
   [a.(i, r) = t.(i + r)]: its threads' and lanes' terms overlap, which staging
   refuses. An affine [a] is computed, as a norm's output is, from float32 [x]
   of [m] by [k] and float32 [γ] and [β] of [k]: [a.(i, r)] is the 16-bit float
   of [x.(i, r) · γ.(r) + β.(r)], whose [γ] and [β] the sum's padded trips read
   through their own gates, [β] nonzero. *)

type layout = Kn | Nk
type form = Plain | Toeplitz | Affine

type product = {
  dt : Dtype.t;
  batch : int;
  m : int;
  k : int;
  n : int;
  layout : layout;
  form : form;
}

let a_size p = p.batch * if p.form = Toeplitz then p.m + p.k else p.m * p.k

let kernel ?opts_to_apply p =
  let c = Shape.param ~shape:[ Ops.Int (p.batch * p.m * p.n) ] 0 Float32
  and a =
    Shape.param
      ~shape:[ Ops.Int (a_size p) ]
      1
      (if p.form = Affine then Float32 else p.dt)
  and b = Shape.param ~shape:[ Ops.Int (p.batch * p.k * p.n) ] 2 p.dt
  and gamma = Shape.param ~shape:[ Ops.Int p.k ] 3 Float32
  and beta = Shape.param ~shape:[ Ops.Int p.k ] 4 Float32 in
  let rows = p.m + p.k
  and mk = p.m * p.k
  and kn = p.k * p.n
  and mn = p.m * p.n in
  let open Ops.O in
  let z = Ops.range ~axis_type:Weak (Int p.batch) [ 0 ]
  and i = Ops.range ~axis_type:Weak (Int p.m) [ 1 ]
  and j = Ops.range ~axis_type:Weak (Int p.n) [ 2 ]
  and r = Ops.range ~axis_type:Weak (Int p.k) [ 3 ] in
  let at x idx = Ops.cast (Ops.index x [ idx ]) Float32 in
  let ai =
    if p.form = Toeplitz then (z * int rows) + i + r
    else (z * int mk) + (i * int p.k) + r
  and bi =
    match p.layout with
    | Kn -> (z * int kn) + (r * int p.n) + j
    | Nk -> (z * int kn) + (j * int p.k) + r
  in
  let av =
    match p.form with
    | Affine ->
        let affine =
          (Ops.index a [ ai ] * Ops.index gamma [ r ]) + Ops.index beta [ r ]
        in
        Ops.cast (Ops.cast affine p.dt) Float32
    | Plain | Toeplitz -> at a ai
  in
  let sum = Ops.reduce (av * at b bi) Add [ r ] in
  let store =
    Ops.store (Ops.index c [ (z * int mn) + (i * int p.n) + j ]) sum
  in
  Ops.sink
    ~kernel:(Ops.kernel_info ?opts_to_apply ())
    [ Ops.end_ store [ z; i; j ] ]

(* The optimisations the hand-coded rules choose for [k] on [r], staged or not,
   a core padding axes it does not divide. *)
let optimised ?(staged = true) k r =
  Setting.context
    Setting.[ B (stage, staged); B (tc_opt, 2) ]
    (fun () ->
      let s = Postrange.Scheduler.v k r in
      Postrange.Scheduler.convert_loop_to_global s;
      Postrange.Scheduler.applied_opts (Heuristic.hand_coded_optimizations s))

let chosen ~staged p r = optimised ~staged (kernel p) r

(* Floats of magnitudes from [2^-8] to [2^8], whose products' sums round. *)
let values seed size =
  let st = Random.State.make [| seed |] in
  Array.init size (fun _ ->
      let f = Float.of_int (Random.State.int st 2048 - 1024) /. 1024. in
      `Float (Float.ldexp f (Random.State.int st 17 - 8)))

(* Integers from -2 to 2: a product's sums are exact. *)
let small seed size =
  let st = Random.State.make [| seed |] in
  Array.init size (fun _ -> `Float (Float.of_int (Random.State.int st 5 - 2)))

let float_of = function
  | `Float f -> f
  | `Int z -> Bigint.to_float z
  | _ -> invalid_arg "a number"

(* Renderers *)

(* CUDA's first GPU and its renderer, skipping without them. *)
let on_cuda () =
  let d = Gpus.gpu "CUDA" in
  (d, Run.renderer d)

(* [r] rebuilt with [has_shared], [registers], [units] and [in_flight], named
   [name]. *)
let rebuilt ~name ?(has_shared = true) ?registers ?units ?in_flight
    (r : Renderer.t) =
  let registers = Option.value registers ~default:r.registers in
  let units = Option.value units ~default:r.units
  and in_flight = Option.value in_flight ~default:r.in_flight in
  Renderer.v ~name ~suffix:r.suffix ~supports_float4:r.supports_float4
    ~half8:r.half8 ~registers ~vector_lanes:r.vector_lanes
    ~has_local:r.has_local ~has_shared ~global_max:r.global_max
    ~local_max:r.local_max ?global_prod_max:r.global_prod_max
    ~shared_max:r.shared_max ~async_copy:r.async_copy
    ~tensor_cores:r.tensor_cores ~per_byte_first:r.per_byte_first ~units
    ~in_flight ~extra_matcher:r.extra_matcher ~packed_args:r.packed_args
    ~code_for_op:r.code_for_op ~native:r.native ~render:r.render
    ~compiler:r.compiler r.target

(* [r] without shared memory: it stages nothing. *)
let unstaged (r : Renderer.t) =
  rebuilt ~name:(r.name ^ " without shared memory") ~has_shared:false r

(* CUDA's renderer of [r]'s target without asynchronous copies, which stages in
   one stage. *)
let one_stage (r : Renderer.t) =
  Renderer.with_compiler r.compiler (Cstyle.cuda_sync r.target)

let target device renderer arch =
  { Helpers.Target.device; renderer; arch; interface = ""; indices = "" }

(* CUDA's renderer of an sm_89, which compiles nothing; [sync] without
   asynchronous copies. *)
let sm89 ?(sync = false) () =
  let t = target "CUDA" "CUDA" "sm_89" in
  if sync then Cstyle.cuda_sync t else Cstyle.without_compiler t

(* The workgroups of the kernel [k] staged on sm_89, the threads of one and its
   trips of the core's loops, as the hand-coded rules choose them. *)
let planned ?(sync = false) k =
  let r = sm89 ~sync () in
  let s = Postrange.Scheduler.v k r in
  Postrange.Scheduler.convert_loop_to_global s;
  let o =
    Setting.context
      [ Setting.B (Setting.stage, true) ]
      (fun () -> Heuristic.hand_coded_optimizations s)
  in
  let module K = Postrange.Scheduler in
  let size i =
    match List.nth (K.full_shape o) i with Int n -> n | Sym _ -> 0
  in
  let product l = List.fold_left (fun n i -> n * size i) 1 l in
  ( product (K.axes_of o [ Global ]),
    product (K.axes_of o [ Warp; Local ]),
    product (K.axes_of ~reduce:true o [ Weak; Loop ]) )

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

let instrs prg = Ops.src (Ops.nth prg 1)

(* The tensor cores a program runs. *)
let cores prg = List.length (List.filter (is Wmma) (instrs prg))
let a_core = satisfies ~claim:"a tensor core" Testable.int (fun n -> n > 0)

(* The buffers in shared memory of a linearized program. *)
let locals lin =
  List.filter
    (fun u -> (is Alloc u || is Buffer u) && Ops.addrspace u = Some Dtype.Local)
    (Ops.toposort ~calls:Skip lin)

(* The elements of each buffer in shared memory of [prg], least first: a tile's,
   or its two stages'. *)
let buffers prg =
  List.sort Int.compare (List.map Shape.max_numel (locals (Ops.nth prg 1)))

(* The shared buffers of [k] lowered for [r]'s target in one stage. *)
let one_stage_buffers k (r : Renderer.t) =
  buffers (Codegen.linearize k (Cstyle.cuda_sync r.target))

let slot b = match Ops.arg b with Param p -> Some p.slot | _ -> None

(* A copy into shared memory: a store of a load of global memory, the slot of
   the buffer it reads, the elements of the shared buffer it writes and the
   bytes it moves. *)
type copy = { from : int; into : int; bytes : int }

let copies prg =
  List.filter_map
    (fun u ->
      if not (is Store u) then None
      else
        let into = Ops.nth u 0 and v = Ops.nth u 1 in
        if Ops.addrspace into <> Some Dtype.Local || not (is Load v) then None
        else
          let at = Ops.nth v 0 in
          if Ops.addrspace at <> Some Dtype.Global then None
          else
            Option.map
              (fun from ->
                {
                  from;
                  into = Shape.max_numel (Ops.buf_uop into);
                  bytes = Shape.max_numel into * Dtype.itemsize (Ops.dtype into);
                })
              (slot (Ops.buf_uop at)))
    (instrs prg)

(* The slots of the global buffers [prg] copies into shared memory. *)
let copied prg =
  List.sort_uniq Int.compare (List.map (fun c -> c.from) (copies prg))

(* The elements of the shared buffer [prg] copies slot [s] into. *)
let tile_of prg s =
  List.sort_uniq Int.compare
    (List.filter_map
       (fun c -> if c.from = s then Some c.into else None)
       (copies prg))

(* Draws *)

let dims = Gen.of_list [ 16; 32; 48; 64; 96; 128; 256 ]

let products =
  let open Gen in
  let+ dt = of_list [ Dtype.Float16; Dtype.Bfloat16 ]
  and+ batch = frequency [ (3, constant 1); (1, constant 2) ]
  and+ m = dims
  and+ n = dims
  and+ k = of_list [ 16; 32; 48; 64; 160; 256 ]
  and+ padded =
    frequency
      [
        (3, constant `None); (1, constant `M); (1, constant `K); (1, constant `N);
      ]
  and+ layout = of_list [ Kn; Nk ]
  and+ form =
    frequency
      [ (4, constant Plain); (1, constant Toeplitz); (1, constant Affine) ]
  in
  (* One axis padded, within the heuristic's 6/5 of the work: M and K to the
     core's 16, at most 32 / 28 times, or N by 8 columns, which the core's 8
     divides and the staged tile pads to its 64. *)
  let m, k, n =
    match padded with
    | `M when m >= 32 -> (m - 4, k, n)
    | `K when k >= 48 -> (m, k - 8, n)
    | `N when n >= 32 -> (m, k, n - 8)
    | _ -> (m, k, n)
  in
  { dt; batch; m; k; n; layout; form }

let pp_product ppf p =
  Format.fprintf ppf "%a%s [%d; %d] by %s [%d; %d], %d times" Dtype.pp p.dt
    (match p.form with
    | Plain -> ""
    | Toeplitz -> " toeplitz"
    | Affine -> " affine")
    p.m p.k
    (match p.layout with Kn -> "[K; N]" | Nk -> "[N; K]")
    p.k p.n p.batch

let opt = Testable.make ~pp:Opt.pp ~equal:Opt.equal

(* Staging *)

(* A kernel staged computes the bits of the kernel of the same optimisations
   unstaged: staging moves operands and never rounds them. *)
let same_bits (p, staged_tile, depth, seed) =
  let d, r = on_cuda () in
  let staged = if depth = 2 then r else one_stage r in
  let opts = chosen ~staged:staged_tile p r in
  let k = kernel ~opts_to_apply:opts p in
  let nonzero = function `Float 0. -> `Float 1. | v -> v in
  let inputs =
    [
      (1, values seed (a_size p)); (2, values (seed + 1) (p.batch * p.k * p.n));
    ]
    @
    if p.form = Affine then
      [
        (3, values (seed + 2) p.k);
        (4, Array.map nonzero (values (seed + 3) p.k));
      ]
    else []
  in
  let run r = List.assoc 0 (Run.on d (compiled r k) inputs) in
  let prg = compiled staged k in
  let cored = cores prg > 0 in
  let operands = List.length (locals (Ops.nth prg 1)) in
  let shared = operands > 0 in
  let two =
    depth = 2 && shared
    && List.map (( * ) 2) (one_stage_buffers k r) = buffers prg
  in
  cover "both operands staged" (cored && operands = 2);
  cover "an operand refused, the other staged" (cored && operands = 1);
  cover "an axis the core pads"
    (cored && shared && (p.m mod 16 <> 0 || p.k mod 16 <> 0));
  (* The staged tile pads N by 8 tiles of the core's 8 columns, M by 4 of its 16
     rows. *)
  let padded_by n =
    cored && shared
    && List.exists
         (function Opt.Padto { amount; _ } -> amount = n | _ -> false)
         opts
  in
  cover "N padded to the staged tile" (padded_by 8);
  cover "M padded to the staged tile" (padded_by 4);
  cover "two stages" two;
  cover "one stage" (shared && depth = 1);
  cover "the staged tile" (shared && staged_tile);
  cover "the unstaged tile, staged" (shared && not staged_tile);
  cover "batched" (shared && p.batch > 1);
  cover "an [N; K] operand" (shared && p.layout = Nk);
  cover "one core's K" (shared && p.k = 16);
  cover "an affine operand staged over a padded K, β nonzero"
    (cored && operands >= 2 && p.form = Affine && p.k mod 16 <> 0);
  equal (array Dtypes.value) (run (unstaged r)) (run staged)

let bits_law =
  (* The examples reach every cover; the draws add new shapes each run. *)
  prop ~count:8
    ~examples:
      [
        ( {
            dt = Bfloat16;
            batch = 1;
            m = 128;
            k = 64;
            n = 128;
            layout = Kn;
            form = Plain;
          },
          true,
          2,
          1 );
        ( {
            dt = Float16;
            batch = 2;
            m = 62;
            k = 56;
            n = 32;
            layout = Nk;
            form = Plain;
          },
          true,
          1,
          2 );
        ( {
            dt = Bfloat16;
            batch = 1;
            m = 128;
            k = 64;
            n = 120;
            layout = Kn;
            form = Plain;
          },
          true,
          2,
          5 );
        ( {
            dt = Bfloat16;
            batch = 1;
            m = 64;
            k = 64;
            n = 64;
            layout = Kn;
            form = Toeplitz;
          },
          false,
          2,
          3 );
        ( {
            dt = Float16;
            batch = 1;
            m = 96;
            k = 160;
            n = 64;
            layout = Kn;
            form = Plain;
          },
          false,
          2,
          4 );
        ( {
            dt = Bfloat16;
            batch = 1;
            m = 128;
            k = 56;
            n = 128;
            layout = Nk;
            form = Affine;
          },
          true,
          2,
          6 );
        ( {
            dt = Bfloat16;
            batch = 1;
            m = 44;
            k = 64;
            n = 120;
            layout = Kn;
            form = Plain;
          },
          true,
          2,
          7 );
        ( {
            dt = Float16;
            batch = 1;
            m = 64;
            k = 16;
            n = 64;
            layout = Nk;
            form = Plain;
          },
          true,
          2,
          8 );
      ]
    "a staged kernel computes the bits of the same optimisations unstaged"
    (Gen.with_pp
       (fun ppf (p, staged, depth, seed) ->
         Format.fprintf ppf "%a, %s tile, %d stages, seed %d" pp_product p
           (if staged then "staged" else "unstaged")
           depth seed)
       Gen.(quad products bool (of_list [ 1; 2 ]) (int_range 0 1_000_000)))
    same_bits

(* Gathered rows

   A routed product: each slot [σ] of [blocks] blocks of 64 slots reads the row
   of [a] its map entry names and the expert matrix of its block's owner, [c.(σ,
   j) = Σ_r a.(map.(σ) / topk, r) · w.(owner.(σ / 64), r, j)], in float32, the
   products of bfloat16 values. [map] holds each slot's route position, [tokens
   · topk] for a pad slot, which reads [a] as [0]: a pad's outputs are [+0].
   Positions are each expert's in order, in blocks of 64 from its first; each
   token's [topk] positions draw their experts at random, so a block may read a
   row twice. [w] is bfloat16, of [K; N] matrices, or MXFP4: [N; K] matrices of
   4-bit codes, two a byte, the low nibble first, each block of 32 scaled by a
   power of two of 8 bits, decoded into bfloat16 in the kernel. The positions,
   the map's rows, the clamp and the gate are rune's for
   [Nx.matmul_segments]. *)

type weights = Float_weights | Mxfp4

type routed = {
  weights : weights;
  tokens : int;
  topk : int;
  experts : int;
  block : int;
  rk : int;
  rn : int;
  rseed : int;
}

(* E2M1's 16 values, by code. *)
let e2m1 =
  [|
    0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6.; -0.; -0.5; -1.; -1.5; -2.; -3.; -4.; -6.;
  |]

(* The float32 value of the E2M1 code [c], an integer, by integer arithmetic as
   nx.quant decodes it: a sign bit, two bits of exponent [e] and one of mantissa
   [m], the magnitude [m / 2] for [e = 0] and [(2 + m) 2^(e - 2)] otherwise, as
   quarters. *)
let decode_e2m1 c =
  let open Ops.O in
  let m = c land int 1 and e = (c lsr int 1) land int 3 in
  let quarters = Ops.where (Ops.eq e (int 0)) (m * int 2) ((m + int 2) lsl e) in
  let v = Ops.cast quarters Float32 * Ops.float ~dtype:Float32 0.25 in
  Ops.where (Ops.eq (c lsr int 3) (int 1)) (Ops.neg v) v

(* Each slot's route position, its count of real ones, and each block's owner:
   the blocks a routing of [n] positions over [g] experts takes at most, [m + (n
   - m) / 64] for [m = min n g], as rune bounds them. *)
let routing p =
  let st = Random.State.make [| p.rseed |] in
  let positions = p.tokens * p.topk in
  let ids = Array.init positions (fun _ -> Random.State.int st p.experts) in
  let m = min positions p.experts in
  let blocks = m + ((positions - m) / p.block) in
  let map = Array.make (blocks * p.block) positions in
  let owner = Array.make blocks 0 in
  let next = ref 0 in
  for e = 0 to p.experts - 1 do
    let rank = ref 0 in
    Array.iteri
      (fun pos id ->
        if id = e then begin
          if !rank mod p.block = 0 then begin
            owner.(!next) <- e;
            incr next
          end;
          map.(((!next - 1) * p.block) + (!rank mod p.block)) <- pos;
          incr rank
        end)
      ids
  done;
  (map, owner)

let routed_kernel ?opts_to_apply p ~blocks =
  let k = p.rk and n = p.rn and positions = p.tokens * p.topk in
  let slots = blocks * p.block and last = positions - 1 in
  let kn = k * n and halves = k / 2 and scaled = k / 32 in
  let c = Shape.param ~shape:[ Ops.Int (slots * n) ] 0 Float32
  and a = Shape.param ~shape:[ Ops.Int (p.tokens * k) ] 1 Bfloat16
  and map = Shape.param ~shape:[ Ops.Int slots ] 2 Int64
  and owner = Shape.param ~shape:[ Ops.Int blocks ] 3 Int32
  and w = Shape.param ~shape:[ Ops.Int (p.experts * kn) ] 4 Bfloat16
  and codes = Shape.param ~shape:[ Ops.Int (p.experts * n * halves) ] 4 Uint8
  and scales =
    Shape.param ~shape:[ Ops.Int (p.experts * n * scaled) ] 5 Uint8
  in
  let open Ops.O in
  let b = Ops.range ~axis_type:Weak (Int blocks) [ 0 ]
  and i = Ops.range ~axis_type:Weak (Int p.block) [ 1 ]
  and j = Ops.range ~axis_type:Weak (Int n) [ 2 ]
  and r = Ops.range ~axis_type:Weak (Int k) [ 3 ] in
  let pos = Ops.index map [ (b * int p.block) + i ] in
  let inside =
    Ops.lt (Ops.bitcast pos Uint64) (Ops.int ~dtype:Uint64 positions)
  in
  let at =
    Ops.minimum
      (Ops.maximum pos (Ops.int ~dtype:Int64 0))
      (Ops.int ~dtype:Int64 last)
  in
  let row = Ops.cast at Weak_int // int p.topk in
  let av =
    Ops.where inside
      (Ops.index a [ (row * int k) + r ])
      (Ops.float ~dtype:Bfloat16 0.)
  in
  let e = Ops.cast (Ops.index owner [ b ]) Weak_int in
  let bv =
    match p.weights with
    | Float_weights -> Ops.index w [ (e * int kn) + (r * int n) + j ]
    | Mxfp4 ->
        let wrow = (e * int n) + j in
        let code =
          Ops.cast
            (Ops.index codes [ (wrow * int halves) + (r // int 2) ])
            Weak_int
        in
        let nibble = (code lsr (r % int 2 * int 4)) land int 15 in
        let bits =
          Ops.cast
            (Ops.index scales [ (wrow * int scaled) + (r // int 32) ])
            Uint32
        in
        let scale = Ops.bitcast (bits lsl Ops.int ~dtype:Uint32 23) Float32 in
        Ops.cast (decode_e2m1 nibble * scale) Bfloat16
  in
  let sum = Ops.reduce (Ops.cast av Float32 * Ops.cast bv Float32) Add [ r ] in
  let out = (((b * int p.block) + i) * int n) + j in
  let store = Ops.store (Ops.index c [ out ]) sum in
  Ops.sink
    ~kernel:(Ops.kernel_info ?opts_to_apply ())
    [ Ops.end_ store [ b; i; j ] ]

(* The kernel's inputs: [a] and float weights of [values], or with [small],
   integers from -2 to 2, codes of any byte and scales from [2^-4] to [2^4]
   (bytes 123 to 131). *)
let routed_inputs ?(small = false) p =
  let map, owner = routing p in
  let int x = `Int (Bigint.of_int x) in
  let draw seed size =
    if small then
      let st = Random.State.make [| seed |] in
      Array.init size (fun _ ->
          `Float (Float.of_int (Random.State.int st 5 - 2)))
    else values seed size
  in
  let st = Random.State.make [| p.rseed + 2 |] in
  let weights =
    match p.weights with
    | Float_weights -> [ (4, draw (p.rseed + 1) (p.experts * p.rk * p.rn)) ]
    | Mxfp4 ->
        let bytes n lo hi =
          Array.init n (fun _ -> int (lo + Random.State.int st (hi - lo + 1)))
        in
        [
          (4, bytes (p.experts * p.rn * p.rk / 2) 0 255);
          (5, bytes (p.experts * p.rn * p.rk / 32) 123 131);
        ]
  in
  ( map,
    owner,
    [
      (1, draw p.rseed (p.tokens * p.rk));
      (2, Array.map int map);
      (3, Array.map int owner);
    ]
    @ weights )

(* The routed product's kernel, with the optimisations [opts] if given, else
   those the compiler chooses. *)
let routed_program ?opts p =
  let map, _ = routing p in
  routed_kernel ?opts_to_apply:opts p ~blocks:(Array.length map / p.block)

(* The optimisations the compiler applied to [prg]. *)
let applied prg =
  match Ops.arg (Ops.nth prg 0) with
  | Kernel { applied_opts; _ } -> applied_opts
  | _ -> failwith "a program's first source is its kernel"

let pp_routed ppf p =
  Format.fprintf ppf
    "%s, %d tokens, top %d of %d experts in blocks of %d, [%d; %d], seed %d"
    (match p.weights with Float_weights -> "bfloat16" | Mxfp4 -> "MXFP4")
    p.tokens p.topk p.experts p.block p.rk p.rn p.rseed

(* A dense decoded product

   [c.(t, j) = Σ_r x.(t, r) · w.(j, r)] in float32, the products of bfloat16
   values, for [x] of [tokens] rows of [k] and [w] MXFP4 of [n] rows of [k],
   decoded into bfloat16 in the kernel: codes [n; k / 2], two a byte, the low
   nibble first, [offset] bytes into their buffer, and a power of two of 8 bits
   each block of 32, scales [n; k / 32], as [Nx_quant.apply] reads a packed
   matrix. The codes stage by copies of 16 bytes and the scales, which the
   core's loop moves by a byte a trip, are held whole; codes at an offset that
   no copy of 16 bytes keeps leave the operand unstaged, its scales with it. *)

type dense = { dtokens : int; dk : int; dn : int; offset : int }

let dense_kernel ?opts_to_apply p =
  let k = p.dk and n = p.dn and t = p.dtokens in
  let halves = k / 2 and scaled = k / 32 in
  let c = Shape.param ~shape:[ Ops.Int (t * n) ] 0 Float32
  and x = Shape.param ~shape:[ Ops.Int (t * k) ] 1 Bfloat16
  and codes = Shape.param ~shape:[ Ops.Int (p.offset + (n * halves)) ] 2 Uint8
  and scales = Shape.param ~shape:[ Ops.Int (n * scaled) ] 3 Uint8 in
  let open Ops.O in
  let i = Ops.range ~axis_type:Weak (Int t) [ 0 ]
  and j = Ops.range ~axis_type:Weak (Int n) [ 1 ]
  and r = Ops.range ~axis_type:Weak (Int k) [ 2 ] in
  let code =
    Ops.cast
      (Ops.index codes [ int p.offset + (j * int halves) + (r // int 2) ])
      Weak_int
  in
  let nibble = (code lsr (r % int 2 * int 4)) land int 15 in
  let bits =
    Ops.cast (Ops.index scales [ (j * int scaled) + (r // int 32) ]) Uint32
  in
  let scale = Ops.bitcast (bits lsl Ops.int ~dtype:Uint32 23) Float32 in
  let w = Ops.cast (decode_e2m1 nibble * scale) Bfloat16 in
  let xv = Ops.index x [ (i * int k) + r ] in
  let sum = Ops.reduce (Ops.cast xv Float32 * Ops.cast w Float32) Add [ r ] in
  let store = Ops.store (Ops.index c [ (i * int n) + j ]) sum in
  Ops.sink
    ~kernel:(Ops.kernel_info ?opts_to_apply ())
    [ Ops.end_ store [ i; j ] ]

(* [x] of [values], or with [small] of integers from -2 to 2, codes of any byte
   and scales from [2^-4] to [2^4] (bytes 123 to 131). *)
let dense_inputs ?(small = false) p =
  let st = Random.State.make [| p.dtokens |] in
  let int x = `Int (Bigint.of_int x) in
  let bytes n lo hi =
    Array.init n (fun _ -> int (lo + Random.State.int st (hi - lo + 1)))
  in
  let x =
    if small then
      Array.init (p.dtokens * p.dk) (fun _ ->
          `Float (Float.of_int (Random.State.int st 5 - 2)))
    else values (p.dtokens + 1) (p.dtokens * p.dk)
  in
  [
    (1, x);
    (2, bytes (p.offset + (p.dn * p.dk / 2)) 0 255);
    (3, bytes (p.dn * p.dk / 32) 123 131);
  ]

let pp_dense ppf p =
  Format.fprintf ppf "%d tokens, [%d; %d], codes at %d" p.dtokens p.dn p.dk
    p.offset

(* The float32 value of MXFP4's weight [q] of row [row]: its nibble's E2M1 value
   scaled by [2^(s - 127)]. *)
let mxfp4_weight ~codes ~scales ~k row q =
  let byte = int_of_float (float_of codes.((row * k / 2) + (q / 2))) in
  let nibble = (byte lsr (q mod 2 * 4)) land 15 in
  let s = int_of_float (float_of scales.((row * k / 32) + (q / 32))) in
  Float.ldexp e2m1.(nibble) (s - 127)

(* A computed operand

   An attention block's output projection as rune lowers it: [c[t, j] = Σ_r x[t,
   r] · w[j, r]] in float32, over [r] of [heads · 64], whose operand [x[t, r]]
   is the bfloat16 of [o[h, t, d] · kept[h, t]] at [h = r / 64] and [d = r mod
   64], for [o] of bfloat16 laid out [heads; tokens; 64] and [kept] of float32
   laid out [heads; tokens]. [w] is bfloat16 [N; K], or Q8_0: each block of 32
   values of a row in 34 bytes, a float16 scale then 32 int8 codes, the product
   summed a block at a time, each block's sum times its scale. The loads [x]
   computes on are staged where they step through one loop with the other
   operand's, beside a decoded [w] too. *)

type weights_of = Bf16_rows | Q8_0_rows
type computed = { weights_of : weights_of; tokens : int; head : int }

let heads = 8
let outputs = 256
let qk = 32
let q8_bytes = 2 + qk

let computed_kernel ?opts_to_apply p =
  let head = p.head in
  let k = heads * head and t = p.tokens and n = outputs in
  let blocks = k / qk and plane = t * head and per_head = head / qk in
  let row = blocks * q8_bytes in
  let c = Shape.param ~shape:[ Ops.Int (t * n) ] 0 Float32
  and o = Shape.param ~shape:[ Ops.Int (heads * t * head) ] 1 Dtype.Bfloat16
  and kept = Shape.param ~shape:[ Ops.Int (heads * t) ] 2 Float32
  and w = Shape.param ~shape:[ Ops.Int (n * k) ] 3 Dtype.Bfloat16
  and q8 = Shape.param ~shape:[ Ops.Int (n * blocks * q8_bytes) ] 3 Uint8 in
  let open Ops.O in
  let x tok h d =
    let v =
      Ops.cast (Ops.index o [ (h * int plane) + (tok * int head) + d ]) Float32
    in
    Ops.cast
      (Ops.cast (v * Ops.index kept [ (h * int t) + tok ]) Bfloat16)
      Float32
  in
  let sum, tok, j =
    match p.weights_of with
    | Bf16_rows ->
        let r = Ops.range ~axis_type:Weak (Int k) [ 0 ]
        and tok = Ops.range ~axis_type:Weak (Int t) [ 1 ]
        and j = Ops.range ~axis_type:Weak (Int n) [ 2 ] in
        let wv = Ops.cast (Ops.index w [ (j * int k) + r ]) Float32 in
        let sum = Ops.reduce (x tok (r // int head) (r % int head) * wv) in
        (sum Add [ r ], tok, j)
    | Q8_0_rows ->
        let e = Ops.range ~axis_type:Weak (Int qk) [ 0 ]
        and b = Ops.range ~axis_type:Weak (Int blocks) [ 1 ]
        and tok = Ops.range ~axis_type:Weak (Int t) [ 2 ]
        and j = Ops.range ~axis_type:Weak (Int n) [ 3 ] in
        let block = (j * int row) + (b * int q8_bytes) in
        let code =
          Ops.cast
            (Ops.bitcast (Ops.index q8 [ block + int 2 + e ]) Int8)
            Float32
        in
        let byte i = Ops.cast (Ops.index q8 [ block + int i ]) Uint16 in
        let scale =
          Ops.cast
            (Ops.bitcast
               (byte 0 + (byte 1 lsl Ops.int ~dtype:Uint16 8))
               Float16)
            Float32
        in
        let d = (b % int per_head * int qk) + e in
        let inner = Ops.reduce (x tok (b // int per_head) d * code) Add [ e ] in
        (Ops.reduce (inner * scale) Add [ b ], tok, j)
  in
  let store = Ops.store (Ops.index c [ (tok * int n) + j ]) sum in
  Ops.sink
    ~kernel:(Ops.kernel_info ?opts_to_apply ())
    [ Ops.end_ store [ tok; j ] ]

(* [o] and [kept] of [values]; bfloat16 weights of [values], or Q8_0 blocks of
   any codes, each scale a float16 from 2^-5 to 2^2 (high bytes 0x28 to
   0x43). *)
let computed_inputs p =
  let k = heads * p.head and t = p.tokens in
  let int x = `Int (Bigint.of_int x) in
  let st = Random.State.make [| p.tokens + 3 |] in
  let weights =
    match p.weights_of with
    | Bf16_rows -> values (p.tokens + 2) (outputs * k)
    | Q8_0_rows ->
        Array.init
          (outputs * k / qk * q8_bytes)
          (fun i ->
            match i mod q8_bytes with
            | 1 -> int (0x28 + Random.State.int st 0x1c)
            | _ -> int (Random.State.int st 256))
  in
  [
    (1, values p.tokens (heads * t * p.head));
    (2, values (p.tokens + 1) (heads * t));
    (3, weights);
  ]

(* Sums over heads

   [c[i, j] = Σ_g Σ_r a[g, i, r] · b[g, r, j]] over [g] of [h] blocks of [r] of
   [k], of bfloat16 values, summed in float32: both tiles step through [g] and
   [r]. *)
let heads_kernel ?opts_to_apply ?bound (m, h, k, n) =
  let c = Shape.param ~shape:[ Ops.Int (m * n) ] 0 Float32
  and a = Shape.param ~shape:[ Ops.Int (h * m * k) ] 1 Dtype.Bfloat16
  and b = Shape.param ~shape:[ Ops.Int (h * k * n) ] 2 Dtype.Bfloat16
  and count = Shape.param ~shape:[ Ops.Int 1 ] 3 Int32 in
  let mk = m * k and kn = k * n in
  let open Ops.O in
  let i = Ops.range ~axis_type:Weak (Int m) [ 0 ]
  and j = Ops.range ~axis_type:Weak (Int n) [ 1 ]
  and g =
    match bound with
    | None -> Ops.range ~axis_type:Weak (Int h) [ 2 ]
    | Some () ->
        let n = Ops.load (Ops.index count [ int 0 ]) [] in
        Ops.range ~axis_type:Weak (Sym (Ops.cast n Dtype.Weak_int)) [ 2 ]
  and r = Ops.range ~axis_type:Weak (Int k) [ 3 ] in
  let at x idx = Ops.cast (Ops.index x [ idx ]) Float32 in
  let sum =
    Ops.reduce
      (at a ((g * int mk) + (i * int k) + r)
      * at b ((g * int kn) + (r * int n) + j))
      Add [ g; r ]
  in
  let store = Ops.store (Ops.index c [ (i * int n) + j ]) sum in
  Ops.sink
    ~kernel:(Ops.kernel_info ?opts_to_apply ())
    [ Ops.end_ store [ i; j ] ]

let heads_exact (m, h, k, n) a b =
  Array.init (m * n) (fun e ->
      let i = e / n and j = e mod n in
      let t = ref 0. in
      for g = 0 to h - 1 do
        for q = 0 to k - 1 do
          t :=
            !t
            +. float_of a.((g * m * k) + (i * k) + q)
               *. float_of b.((g * k * n) + (q * n) + j)
        done
      done;
      `Float !t)

(* [c.(i, j) = Σ_(g, r) a.(g, i, r) * b.(j, g k + r)] in float32: a sum over [h]
   heads of [k] dimensions, [a] laid out heads first, as attention's output
   reaches its projection, and [b] stored [N; h k]. *)
let merged_kernel ?opts_to_apply (m, h, k, n) =
  let c = Shape.param ~shape:[ Ops.Int (m * n) ] 0 Float32
  and a = Shape.param ~shape:[ Ops.Int (h * m * k) ] 1 Float32
  and b = Shape.param ~shape:[ Ops.Int (n * h * k) ] 2 Float32 in
  let open Ops.O in
  let i = Ops.range ~axis_type:Weak (Int m) [ 0 ]
  and j = Ops.range ~axis_type:Weak (Int n) [ 1 ]
  and g = Ops.range ~axis_type:Weak (Int h) [ 2 ]
  and r = Ops.range ~axis_type:Weak (Int k) [ 3 ] in
  let sum =
    Ops.reduce
      (Ops.index a [ (g * int Stdlib.(m * k)) + (i * int k) + r ]
      * Ops.index b [ (j * int Stdlib.(h * k)) + (g * int k) + r ])
      Add [ g; r ]
  in
  let store = Ops.store (Ops.index c [ (i * int n) + j ]) sum in
  Ops.sink
    ~kernel:(Ops.kernel_info ?opts_to_apply ())
    [ Ops.end_ store [ i; j ] ]

(* Serial ranges around a core

   [c.(o, i, j) = Σ_r a.(i, r) * b.(o, r + j)] in float32, or its sum over [o]
   as well, for [a] of [m] by [k] and [b] Toeplitz in each of [outer] blocks, of
   bfloat16. [b] reads [o] and its tile overlaps, so it stays unstaged; [a]
   reads no [o] and stages, its tile stepping through the core's loop [l]. The
   core runs inside [o], and [l]'s loop inside or around [o]'s:

   {v for o: serial: a loop, a bound in memory, a sum outside l for l: the
   core's loop, copying a's tiles c += a[l] * b[o, l] v}

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

(* The sum's ranges nest in the order of their identities, the first outermost:
   [o] takes [0] to run outside [r]'s [3], and [4] inside. *)
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
  let one o i j =
    let s = ref 0. in
    for q = 0 to p.k - 1 do
      s :=
        !s
        +. (float_of a.((i * p.k) + q) *. float_of b.((o * (p.k + p.n)) + q + j))
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

(* The core over [j], [i] and [r]: the choices pair [a]'s [i] with each of [b]'s
   ranges, [o] then [j], and each range of the sum, the last varying fastest.
   [o] of one trip would leave the kernel, so [outer] is at least [2]. *)
let nested_opts around =
  let axis =
    match around with
    | Loop_range | Bound_in_memory -> 1
    | Sum_outside -> 2
    | Sum_inside -> 3
  in
  [ Opt.Tc { axis; tc_select = -1; tc_opt = 2; use_tc = 1 } ]

let pp_nest ppf p =
  Format.fprintf ppf "[%d; %d] by [%d; %d] under %a of %d trips" p.m p.k p.k p.n
    pp_around p.around p.outer

(* The barriers of a staged kernel

   Every thread runs the same instructions, so two accesses of one element of
   shared memory by different threads are ordered only by a barrier between them
   in the instructions. Over two trips of each loop, every store into shared
   memory and every load of an element it may store, in either order, are
   separated by a barrier: a load after the store reads what it stored, and a
   store after the load does not overwrite what the load reads. *)

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

(* The values of the threads of a workgroup: each binding of the local indices
   of [instrs], and every workgroup index at [0]. *)
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
   [instrs], each loop walked for its first two trips: the elements each access
   reads or writes over the threads that reach it. *)
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
          while not (closes u instrs.(!close)) do
            incr close
          done;
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

(* The staged kernels the barrier law draws, each lowered for sm_89 in two
   stages or in one, or for HIP's RDNA3, whose A operand broadcasts a lane bit,
   and RDNA4, both in one stage of copies through registers. *)
type target = Cuda of int | Hip of string

type staged_kernel =
  | Product of product * target
  | Gathered of routed
  | Decoded of dense
  | Computed of computed
  | Nested of nest * int
  | Registers of product
  | Merged

let renderer_of = function
  | Cuda depth -> sm89 ~sync:(depth = 1) ()
  | Hip arch -> Cstyle.without_compiler (target "AMD" "HIP" arch)

let pp_staged ppf = function
  | Product (p, t) ->
      Format.fprintf ppf "%a, %s" pp_product p
        (match t with
        | Cuda d -> Printf.sprintf "CUDA, %d stages" d
        | Hip arch -> "HIP " ^ arch)
  | Gathered p -> pp_routed ppf p
  | Decoded p -> pp_dense ppf p
  | Computed p ->
      Format.fprintf ppf "%s computed operand, %d tokens, heads of %d"
        (match p.weights_of with
        | Bf16_rows -> "bfloat16"
        | Q8_0_rows -> "Q8_0")
        p.tokens p.head
  | Nested (p, depth) -> Format.fprintf ppf "%a, %d stages" pp_nest p depth
  | Registers p -> Format.fprintf ppf "in registers, %a" pp_product p
  | Merged -> Format.pp_print_string ppf "256 by 256 over 4 heads of 110"

let merged = (256, 4, 110, 256)

let lowered_staged = function
  | Product (p, t) ->
      let r = renderer_of t in
      Codegen.to_program (kernel ~opts_to_apply:(chosen ~staged:true p r) p) r
  | Gathered p -> Codegen.to_program (routed_program p) (sm89 ())
  | Decoded p -> Codegen.to_program (dense_kernel p) (sm89 ())
  | Computed p -> Codegen.to_program (computed_kernel p) (sm89 ())
  | Nested (p, depth) ->
      Codegen.to_program
        (nested ~opts_to_apply:(nested_opts p.around) p)
        (sm89 ~sync:(depth = 1) ())
  | Registers p ->
      let r = sm89 () in
      Codegen.to_program (kernel ~opts_to_apply:(chosen ~staged:true p r) p) r
  | Merged ->
      let r = sm89 () in
      Codegen.to_program
        (merged_kernel
           ~opts_to_apply:(optimised (merged_kernel merged) r)
           merged)
        r

let ordered k =
  let prg = lowered_staged k in
  let instrs = instrs prg in
  let staged = locals (Ops.nth prg 1) <> [] in
  let family f = cover f staged in
  (match k with
  | Product (_, Cuda 2) -> family "a product in two stages"
  | Product (_, Cuda _) -> family "a product in one stage"
  | Product (_, Hip _) -> family "a product on HIP"
  | Gathered _ -> family "gathered rows"
  | Decoded _ -> family "a decoded operand"
  | Computed _ -> family "a computed operand"
  | Nested (p, _) ->
      family (Format.asprintf "a core under %a" pp_around p.around)
  | Registers _ -> family "a product in registers"
  | Merged -> family "a product in registers over heads");
  cover "a loop" (staged && List.exists (is Range) instrs);
  match unordered (walk instrs) with
  | None -> ()
  | Some (a, b) ->
      failf "%a then %a with no barrier between them" pp_access a pp_access b

let staged_kernels =
  let open Gen in
  let gathered =
    let+ weights = of_list [ Float_weights; Mxfp4 ]
    and+ tokens = of_list [ 1; 12; 64 ]
    and+ experts = of_list [ 4; 8 ]
    and+ block = of_list [ 8; 64 ]
    and+ rseed = int_range 0 1_000_000 in
    Gathered
      { weights; tokens; topk = 4; experts; block; rk = 160; rn = 64; rseed }
  and decoded =
    let+ dtokens = of_list [ 12; 64 ]
    and+ dk = of_list [ 256; 512 ]
    and+ offset = of_list [ 0; 8 ] in
    Decoded { dtokens; dk; dn = 128; offset }
  and computed =
    let+ weights_of = of_list [ Bf16_rows; Q8_0_rows ]
    and+ tokens = of_list [ 12; 64 ]
    and+ head = of_list [ 32; 64 ] in
    Computed { weights_of; tokens; head }
  and nests =
    let+ around =
      of_list [ Loop_range; Bound_in_memory; Sum_outside; Sum_inside ]
    and+ outer = int_range 2 4
    and+ m = of_list [ 16; 32 ]
    and+ k = of_list [ 32; 48; 64; 128 ]
    and+ n = of_list [ 16; 32 ]
    and+ depth = of_list [ 1; 2 ] in
    Nested ({ around; outer; m; k; n }, depth)
  and products =
    let+ p = products
    and+ t = of_list [ Cuda 1; Cuda 2; Hip "gfx1100"; Hip "gfx1201" ] in
    Product (p, t)
  in
  frequency
    [ (8, products); (2, gathered); (2, decoded); (1, computed); (3, nests) ]

let barriers =
  let product p t = Product (p, t)
  and plain dt m k n layout =
    { dt; batch = 1; m; k; n; layout; form = Plain }
  in
  let nest around depth =
    Nested ({ around; outer = 3; m = 16; k = 64; n = 16 }, depth)
  in
  (* The examples reach every cover; the draws add new shapes each run. *)
  prop ~count:8
    ~examples:
      [
        product (plain Bfloat16 128 64 128 Kn) (Cuda 2);
        product (plain Float16 64 256 64 Nk) (Cuda 1);
        product (plain Float16 64 128 64 Kn) (Hip "gfx1100");
        product (plain Bfloat16 128 64 128 Nk) (Hip "gfx1201");
        (* N one tile of RDNA4's core, 16 columns: no pad of N *)
        product (plain Float16 28 248 16 Kn) (Hip "gfx1201");
        Gathered
          {
            weights = Float_weights;
            tokens = 64;
            topk = 4;
            experts = 8;
            block = 64;
            rk = 160;
            rn = 64;
            rseed = 1;
          };
        Gathered
          {
            weights = Mxfp4;
            tokens = 128;
            topk = 4;
            experts = 8;
            block = 64;
            rk = 320;
            rn = 128;
            rseed = 2;
          };
        Decoded { dtokens = 64; dk = 512; dn = 128; offset = 0 };
        Computed { weights_of = Bf16_rows; tokens = 64; head = 32 };
        nest Loop_range 2;
        nest Bound_in_memory 2;
        nest Sum_outside 2;
        nest Sum_inside 2;
        Nested ({ around = Loop_range; outer = 2; m = 32; k = 48; n = 16 }, 1);
        Registers
          {
            dt = Float32;
            batch = 1;
            m = 256;
            k = 160;
            n = 256;
            layout = Nk;
            form = Plain;
          };
        Merged;
      ]
    "a staged kernel separates each store into shared memory and each load of \
     its elements by a barrier, over two trips"
    (Gen.with_pp pp_staged staged_kernels)
    ordered

(* A core under a serial range runs its first copy at each trip of the range
   (ae27c3a50a: copied once outside the range, a core under a loop of 3 trips
   gave 7720 of its 12288 outputs wrong). Integers from -2 to 2: every sum is
   exact. *)

(* The core's loop runs [k / 16] trips, and [a]'s tile holds the core's 16 rows
   by 16 terms. *)
let core_trips = 16
let core_tile = 16 * 16

let nested_value (p, depth, seed) =
  let d, r = on_cuda () in
  let r = if depth = 2 then r else one_stage r in
  let prg = compiled r (nested ~opts_to_apply:(nested_opts p.around) p) in
  a_core (cores prg);
  equal ~msg:"a's tile staged" (list int) [ depth * core_tile ] (buffers prg);
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

let serial =
  let example around = ({ around; outer = 3; m = 16; k = 64; n = 16 }, 2, 1) in
  prop ~count:16
    ~examples:
      (List.map example [ Loop_range; Bound_in_memory; Sum_outside; Sum_inside ]
      @ [ ({ around = Loop_range; outer = 2; m = 32; k = 48; n = 16 }, 1, 2) ])
    "a core staged under a serial range computes the exact product"
    (Gen.with_pp
       (fun ppf (p, depth, seed) ->
         Format.fprintf ppf "%a, %d stages, seed %d" pp_nest p depth seed)
       Gen.(triple nests (of_list [ 1; 2 ]) (int_range 0 1_000_000)))
    nested_value

(* A nest of loops

   [heads_kernel]'s product over [g] of [h] blocks, [r] of 512: a block's terms
   pass one running sum's, so its trips cannot take a block whole, and both
   tiles step through [g] and [r]'s trips as one nest, in two stages. The kernel
   computes the bits of the same optimisations unstaged. *)
let nest = (64, 4, 512, 64)

let[@inline never] [@local never] over_nested_loops () =
  let d, r = on_cuda () in
  let module K = Postrange.Scheduler in
  let o =
    Setting.context
      Setting.[ B (stage, true) ]
      (fun () ->
        let s = K.v (heads_kernel nest) r in
        K.convert_loop_to_global s;
        Heuristic.hand_coded_optimizations s)
  in
  (* The sum's loops of more than one trip: [g]'s and the core's. *)
  let trips i =
    match List.nth (K.full_shape o) i with Int n -> n | Sym _ -> 0
  in
  let loops =
    List.filter (fun i -> trips i > 1) (K.axes_of ~reduce:true o [ Weak; Loop ])
  in
  equal ~msg:"loops of more than one trip" int 2 (List.length loops);
  let k = heads_kernel ~opts_to_apply:(K.applied_opts o) nest in
  let prg = compiled r k in
  a_core (cores prg);
  equal ~msg:"both tiles staged" (list int) [ 1; 2 ] (copied prg);
  (* Two stages: each tile's buffer holds twice what one stage's does. *)
  equal ~msg:"two stages" (list int)
    (List.map (( * ) 2) (one_stage_buffers k r))
    (buffers prg);
  let m, h, kk, n = nest in
  let inputs = [ (1, values 1 (h * m * kk)); (2, values 2 (h * kk * n)) ] in
  let unstaged = List.assoc 0 (Run.on d (compiled (unstaged r) k) inputs) in
  equal (array Dtypes.value) unstaged (List.assoc 0 (Run.on d prg inputs))

(* [heads_kernel] over 8 blocks of 32: the staged tile unrolls [K] to 32 values
   a trip, so the core's own range runs one trip, and both tiles step through
   [g] alone, in two stages. Integers from -2 to 2: every sum is exact. *)
let whole = (64, 8, 32, 64)

let core_taken_whole () =
  let d, r = on_cuda () in
  let k =
    heads_kernel ~opts_to_apply:(optimised (heads_kernel whole) r) whole
  in
  let prg = compiled r k in
  a_core (cores prg);
  equal ~msg:"both tiles staged" (list int) [ 1; 2 ] (copied prg);
  equal ~msg:"two stages" (list int)
    (List.map (( * ) 2) (one_stage_buffers k r))
    (buffers prg);
  let m, h, kk, n = whole in
  let a = small 1 (h * m * kk) and b = small 2 (h * kk * n) in
  equal (array Dtypes.value) (heads_exact whole a b)
    (List.assoc 0 (Run.on d prg [ (1, a); (2, b) ]))

(* The same nest with the count of blocks read from memory: its extent is not
   constant, so the tiles copy in one stage. *)
let symbolic_nest () =
  let r = sm89 () in
  let opts = optimised (heads_kernel nest) r in
  let k = heads_kernel ~opts_to_apply:opts ~bound:() nest in
  let prg = Codegen.to_program k r in
  a_core (cores prg);
  equal ~msg:"both tiles staged" (list int) [ 1; 2 ] (copied prg);
  equal (list int) (one_stage_buffers k r) (buffers prg)

(* A product of 128 cubed *)

let cube =
  {
    dt = Bfloat16;
    batch = 1;
    m = 128;
    k = 128;
    n = 128;
    layout = Kn;
    form = Plain;
  }

let exact (p : product) a b =
  Array.init (p.m * p.n) (fun e ->
      let i = e / p.n and j = e mod p.n in
      let s = ref 0. in
      for q = 0 to p.k - 1 do
        s := !s +. (float_of a.((i * p.k) + q) *. float_of b.((q * p.n) + j))
      done;
      `Float !s)

let cube_program r =
  compiled r (kernel ~opts_to_apply:(chosen ~staged:true cube r) cube)

let cube_a = lazy (small 1 (cube.m * cube.k))
let cube_b = lazy (small 2 (cube.k * cube.n))

let[@inline never] [@local never] in_two_stages () =
  let d, r = on_cuda () in
  let prg = cube_program r in
  equal ~msg:"both operands staged" (list int) [ 1; 2 ] (copied prg);
  let a = Lazy.force cube_a and b = Lazy.force cube_b in
  equal (array Dtypes.value) (exact cube a b)
    (List.assoc 0 (Run.on d prg [ (1, a); (2, b) ]))

let[@inline never] [@local never] in_one_stage () =
  let d, r = on_cuda () in
  let prg = cube_program (one_stage r) in
  equal ~msg:"both operands staged" (list int) [ 1; 2 ] (copied prg);
  let a = Lazy.force cube_a and b = Lazy.force cube_b in
  equal (array Dtypes.value) (exact cube a b)
    (List.assoc 0 (Run.on d prg [ (1, a); (2, b) ]))

(* CUDA's source

   A copy of 16 or 8 bytes into shared memory renders as an asynchronous copy of
   that many bytes, its gate as the copy's zero fill, and a barrier waits for
   the copies before it. *)

(* The calls of [f] in [src], its definition left out. *)
let calls f src = count (f ^ "(") src - 1

(* The copies of [bytes] bytes of [prg], those gated, and all. *)
let async_copies ~bytes prg =
  List.length (List.filter (fun c -> c.bytes = bytes) (copies prg))

let gated_copies prg =
  List.length
    (List.filter
       (fun u ->
         is Store u
         && Ops.addrspace (Ops.nth u 0) = Some Dtype.Local
         && is Load (Ops.nth u 1)
         && List.length (Ops.src (Ops.nth u 1)) = 3)
       (instrs prg))

let cuda_source =
  let cube_on_sm89 () =
    let r = sm89 () in
    Codegen.to_program
      (kernel ~opts_to_apply:(chosen ~staged:true cube r) cube)
      r
  in
  group "CUDA's source"
    [
      cases "a copy of 16 bytes is one cp.async of 16 bytes" ~id:fst
        [
          ("of bfloat16 operands", cube_on_sm89);
          ( "of MXFP4's codes and scales",
            fun () ->
              Codegen.to_program
                (dense_kernel { dtokens = 64; dk = 512; dn = 128; offset = 0 })
                (sm89 ()) );
        ]
        (fun (_, prg) ->
          let prg = prg () in
          satisfies ~claim:"some copies" int
            (fun n -> n > 0)
            (async_copies ~bytes:16 prg);
          equal int
            (async_copies ~bytes:16 prg)
            (calls "tg_cp_async" (source prg)));
      test "a copy of 8 bytes is one cp.async of 8 bytes" (fun () ->
          let r = sm89 () in
          let prg =
            Codegen.to_program
              (merged_kernel
                 ~opts_to_apply:(optimised (merged_kernel merged) r)
                 merged)
              r
          in
          satisfies ~claim:"some copies" int
            (fun n -> n > 0)
            (async_copies ~bytes:8 prg);
          equal int
            (async_copies ~bytes:8 prg)
            (calls "tg_cp_async8" (source prg)));
      test "a gated copy fills zeros where its gate fails" (fun () ->
          let p =
            {
              weights = Float_weights;
              tokens = 128;
              topk = 4;
              experts = 8;
              block = 64;
              rk = 160;
              rn = 128;
              rseed = 7;
            }
          in
          let prg = Codegen.to_program (routed_program p) (sm89 ()) in
          let src = source prg in
          let gated_calls =
            List.length
              (List.filter
                 (fun l -> says "tg_cp_async(" l && says ", !(" l)
                 (String.split_on_char '\n' src))
          in
          satisfies ~claim:"some gated copies" int
            (fun n -> n > 0)
            (gated_copies prg);
          equal int (gated_copies prg) gated_calls);
      test "a barrier waits for the copies before it" (fun () ->
          let src = source (cube_on_sm89 ()) in
          let wait =
            {|asm volatile("cp.async.wait_all;" ::: "memory"); __syncthreads();|}
          in
          satisfies ~claim:"some barriers" int
            (fun n -> n > 0)
            (count "__syncthreads();" src);
          equal int (count "__syncthreads();" src) (count wait src));
    ]

(* The race checker

   compute-sanitizer's racecheck, run once over the kernels tagged [race],
   reports any two accesses of one element of shared memory by different threads
   with no barrier between them, as hazards, whether or not the values they move
   happen to agree. It tracks the process it starts and those it starts in turn,
   as windtrap's worker. Each such kernel runs from a function of its own, named
   for what it stages, and the sanitizer's record of each launch and of each
   hazard holds the host backtrace that names the function. *)

let race = "race"

(* A run of the race checker: its exit status, its output, the summary of each
   log, the function each launch was made from, and for each hazard the function
   of its own backtrace, else of the launch before it, since the sanitizer
   reports a kernel's hazards when the kernel ends. *)
type race_run = {
  status : Unix.process_status;
  output : string;
  summaries : string list;
  launches : string list;
  hazards : (string * string option) list;
}

(* The records of a sanitizer log: each starts at a line whose text after the
   prefix does not start with a space. *)
let records log =
  let prefix = "========= " in
  let starts l =
    String.starts_with ~prefix l
    && String.length l > String.length prefix
    && l.[String.length prefix] <> ' '
  in
  let close cur acc =
    match cur with [] -> acc | c -> String.concat "\n" (List.rev c) :: acc
  in
  let rec go cur acc = function
    | [] -> List.rev (close cur acc)
    | l :: rest when starts l -> go [ l ] (close cur acc) rest
    | l :: rest -> go (if cur = [] then [] else l :: cur) acc rest
  in
  go [] [] (String.split_on_char '\n' log)

(* The index of [sub] in [s], if any. *)
let find sub s =
  let n = String.length sub in
  let rec at i =
    match String.index_from_opt s i sub.[0] with
    | Some j when j + n <= String.length s ->
        if String.sub s j n = sub then Some j else at (j + 1)
    | _ -> None
  in
  at 0

(* The function of this suite that a record's host backtrace names first,
   without the number the compiler appends to its name. *)
let launcher r =
  let prefix = "Test_stage." in
  match find prefix r with
  | None -> None
  | Some i -> (
      let i = i + String.length prefix in
      let j = String.index_from r i ' ' in
      let name = String.sub r i (j - i) in
      match String.rindex_opt name '_' with
      | Some k -> Some (String.sub name 0 k)
      | None -> Some name)

let race_run =
  fixture (fun () ->
      ignore (on_cuda ());
      let tool =
        match Sys.getenv_opt "CUDA_PATH" with
        | Some root ->
            Filename.concat (Filename.concat root "bin") "compute-sanitizer"
        | None -> skip ~reason:"CUDA_PATH names no toolkit" ()
      in
      if not (Sys.file_exists tool) then skip ~reason:(tool ^ " is missing") ();
      let dir = temp_dir ~prefix:"racecheck" () in
      let output = Filename.concat dir "output" in
      let args =
        [|
          tool;
          "--tool";
          "racecheck";
          "--racecheck-report";
          "hazard";
          "--target-processes";
          "all";
          "--dump-kernel-launches";
          (* At the default limit of 100 records the summary claim still counts
             the hazards past it. *)
          "--log-file";
          Filename.concat dir "%p.log";
          Sys.executable_name;
          "--tag";
          race;
        |]
      in
      (* The run's own reports would overwrite this run's. *)
      let reports v =
        List.exists
          (fun prefix -> String.starts_with ~prefix v)
          [ "WINDTRAP_JUNIT="; "WINDTRAP_JSON="; "WINDTRAP_OUTPUT=" ]
      in
      let env =
        Array.of_list
          (List.filter
             (fun v -> not (reports v))
             (Array.to_list (Unix.environment ())))
      in
      let fd = Unix.openfile output [ O_WRONLY; O_CREAT; O_TRUNC ] 0o600 in
      let pid =
        Fun.protect
          ~finally:(fun () -> Unix.close fd)
          (fun () -> Unix.create_process_env tool args env Unix.stdin fd fd)
      in
      let _, status = Unix.waitpid [] pid in
      let read f = In_channel.with_open_bin f In_channel.input_all in
      let logs =
        List.filter
          (String.ends_with ~suffix:".log")
          (Array.to_list (Sys.readdir dir))
      in
      let logs =
        List.map (fun f -> records (read (Filename.concat dir f))) logs
      in
      let summary r = find "RACECHECK SUMMARY" r <> None in
      let hazard r =
        let header =
          match String.index_opt r '\n' with
          | Some i -> String.sub r 0 i
          | None -> r
        in
        find "hazard" header <> None && not (summary r)
      in
      let rec attribute launch acc = function
        | [] -> acc
        | r :: rest when hazard r ->
            let owner =
              match launcher r with Some _ as f -> f | None -> launch
            in
            attribute launch ((r, owner) :: acc) rest
        | r :: rest when summary r -> attribute launch acc rest
        | r :: rest -> (
            match launcher r with
            | Some _ as f -> attribute f acc rest
            | None -> attribute launch acc rest)
      in
      let records = List.concat logs in
      {
        status;
        output = read output;
        summaries = List.filter summary records;
        launches =
          List.filter_map
            (fun r -> if hazard r || summary r then None else launcher r)
            records;
        hazards = List.concat_map (fun l -> List.rev (attribute None [] l)) logs;
      })

let race_free (what, f) =
  test ("the race checker finds no hazard in " ^ what) (fun () ->
      let run = race_run () in
      satisfies ~claim:"launches the race checker checked" int
        (fun n -> n > 0)
        (List.length (List.filter (String.equal f) run.launches));
      match List.filter (fun (_, owner) -> owner = Some f) run.hazards with
      | [] -> ()
      | (first, _) :: _ as found ->
          failf "%d hazards, the first:\n%s" (List.length found) first)

(* Each log's summary counts its errors and warnings, the hazards of every
   kernel, whatever function they are put down to. *)
let race_summary () =
  let run = race_run () in
  satisfies ~claim:"a summary" int (fun n -> n > 0) (List.length run.summaries);
  List.iter
    (satisfies ~claim:"no error or warning" string (fun r ->
         find "(0 errors, 0 warnings)" r <> None))
    run.summaries

let races =
  group "races"
    (test "the kernels the race checker runs pass under it" (fun () ->
         match (race_run ()).status with
         | Unix.WEXITED 0 -> ()
         | _ -> failf "the run under the race checker:\n%s" (race_run ()).output)
    :: test "the race checker reports no error or warning" race_summary
    :: List.map race_free
         [
           ("a kernel staged in two stages", "in_two_stages");
           ("a kernel staged in one stage", "in_one_stage");
           ("a kernel staged over a nest of loops", "over_nested_loops");
           ("a kernel of gathered rows", "of_gathered_rows");
           ( "a kernel of gathered rows whose scales are held whole",
             "of_gathered_mxfp4" );
           ("a kernel of a computed operand", "of_computed_operand");
           ("a kernel whose scales are held whole", "of_held_scales");
           ( "products in registers staged in trips of several steps",
             "in_registers" );
           ("a product summed over heads in copies of 8 bytes", "in_copies_of_8");
         ])

(* Depth *)

type depth_case = {
  async : bool;
  local : bool;
  has_shared : bool;
  on : bool;
  used : int;
  tiles : int list;
}

let shared_max = 49152

(* [stage.mli]'s depth: none where staging is off, the target has no local
   threads or shared memory, there are no tiles or one holds less than a copy of
   16 bytes; else two stages where copies are asynchronous and two fit beside
   [used], one where one does, none otherwise. *)
let expected_depth c =
  let stage = List.fold_left ( + ) 0 c.tiles in
  if
    (not c.on) || (not c.local) || (not c.has_shared) || c.tiles = []
    || List.exists (fun t -> t < 16) c.tiles
  then 0
  else if c.async && c.used + (2 * stage) <= shared_max then 2
  else if c.used + stage <= shared_max then 1
  else 0

let depth_of c =
  let r =
    Renderer.v ~has_local:c.local ~has_shared:c.has_shared ~async_copy:c.async
      ~shared_max
      (target "CUDA" "CUDA" "sm_89")
  in
  Setting.context
    [ Setting.B (Setting.stage, c.on) ]
    (fun () -> Stage.depth ~used:c.used r c.tiles)

let pp_depth ppf c =
  Format.fprintf ppf "tiles [%s] beside %d bytes%s%s%s%s"
    (String.concat "; " (List.map string_of_int c.tiles))
    c.used
    (if c.async then ", asynchronous" else "")
    (if c.local then "" else ", no local threads")
    (if c.has_shared then "" else ", no shared memory")
    (if c.on then "" else ", staging off")

let depth_law =
  let usual =
    Gen.frequency [ (5, Gen.constant true); (1, Gen.constant false) ]
  in
  let gen =
    let open Gen in
    let+ async = bool
    and+ local = usual
    and+ has_shared = usual
    and+ on = usual
    and+ used = of_list [ 0; 1; 8192; 24576; 49152 ]
    and+ tiles =
      list ~size:(int_range 0 3)
        (of_list [ 8; 15; 16; 17; 8192; 12288; 16384; 24575; 24576; 24577 ])
    in
    { async; local; has_shared; on; used; tiles }
  in
  let case ?(async = true) ?(local = true) ?(has_shared = true) ?(on = true)
      ?(used = 0) tiles =
    { async; local; has_shared; on; used; tiles }
  in
  prop ~count:200
    ~examples:
      [
        case [ 8192; 8192 ];
        (* Two stages fill shared memory to its last byte, and one byte over
           leaves one. *)
        case [ 12288; 12288 ];
        case ~used:1 [ 12288; 12288 ];
        case ~used:24576 [ 12288 ];
        case ~used:24576 [ 12288; 12288 ];
        case ~used:24577 [ 12288; 12288 ];
        case ~async:false [ 8192; 8192 ];
        case [ 16; 16 ];
        case [ 15; 8192 ];
        case [];
        case ~local:false [ 8192 ];
        case ~has_shared:false [ 8192 ];
        case ~on:false [ 8192 ];
      ]
    "a core takes as many stages as its tiles fit" (Gen.with_pp pp_depth gen)
    (fun c ->
      let d = expected_depth c in
      let valid =
        c.on && c.local && c.has_shared && c.tiles <> []
        && List.for_all (fun t -> t >= 16) c.tiles
      in
      let stage = List.fold_left ( + ) 0 c.tiles in
      cover "two stages" (d = 2);
      cover "two stages filling shared memory"
        (d = 2 && c.used + (2 * stage) = shared_max);
      cover "one stage where two do not fit" (d = 1 && c.async);
      cover "one stage where copies wait" (d = 1 && not c.async);
      cover "one stage filling shared memory"
        (d = 1 && c.used + stage = shared_max);
      cover "none past shared memory" (valid && d = 0);
      cover "none for a tile under a copy"
        (List.exists (fun t -> t < 16) c.tiles);
      cover "none for a tile of one copy staged" (d > 0 && List.mem 16 c.tiles);
      cover "none without tiles" (c.tiles = []);
      cover "none without local threads" (not c.local);
      cover "none without shared memory" (not c.has_shared);
      cover "none with staging off" (not c.on);
      equal int d (depth_of c))

(* Gathered rows *)

let gathered_bits (p : routed) =
  let d, r = on_cuda () in
  let prg = compiled r (routed_program p) in
  let map, _, inputs = routed_inputs p in
  let positions = p.tokens * p.topk in
  let run prg = List.assoc 0 (Run.on d prg inputs) in
  let cored = cores prg > 0 in
  let staged = List.length (locals (Ops.nth prg 1)) in
  let blocks = Array.length map / p.block in
  let block b = Array.sub map (b * p.block) p.block in
  let pads b = Array.exists (( = ) positions) (block b) in
  let reals b = Array.exists (( <> ) positions) (block b) in
  let twice b =
    let rows =
      List.filter_map
        (fun x -> if x = positions then None else Some (x / p.topk))
        (Array.to_list (block b))
    in
    List.length (List.sort_uniq Int.compare rows) < List.length rows
  in
  let some f = List.exists f (List.init blocks Fun.id) in
  let copied = copied prg in
  let rows = List.mem 1 copied
  and weights = List.mem 4 copied
  and scales = List.mem 5 copied in
  cover "a core" cored;
  cover "MXFP4 weights on a core" (cored && p.weights = Mxfp4);
  cover "MXFP4 codes staged, scales held"
    (cored && p.weights = Mxfp4 && weights && scales);
  cover "MXFP4 weights unstaged, their scales' tile no run of whole copies"
    (cored && p.weights = Mxfp4 && not weights);
  cover "float weights on a core" (cored && p.weights = Float_weights);
  cover "a block of real and pad rows" (some (fun b -> pads b && reals b));
  cover "a block of pad rows alone" (some (fun b -> not (reals b)));
  cover "a row read twice in a block" (some twice);
  List.iter
    (fun t -> cover (Printf.sprintf "%d tokens" t) (p.tokens = t))
    [ 1; 12; 64; 128; 512 ];
  (* Whether several warps read the rows decides which: [of_gathered_rows] and
     [alone] pin each. *)
  cover "gathered rows staged" (cored && rows);
  cover "gathered rows left unstaged" (cored && not rows);
  (* A decoded operand is staged only where each of its loads is copied or held,
     and its scales are held only beside its copied codes. The held scales, at
     most [128 · 320 / 32] bytes, always fit beside the stages, so MXFP4's codes
     are copied exactly where its scales are held: neither where a workgroup's
     rows of scales are no run of whole 16-byte copies, as 8 rows of 5 bytes
     over [160] values. *)
  if cored then begin
    if p.weights = Mxfp4 then
      satisfies ~msg:"codes copied exactly where scales are held"
        ~claim:"both or neither" (pair bool bool)
        (fun (w, s) -> w = s)
        (weights, scales);
    equal ~msg:"each staged buffer copies a buffer" int staged
      (List.length copied)
  end;
  let staged = run prg in
  let opts = applied prg in
  equal (array Dtypes.value)
    (run (compiled (unstaged r) (routed_program ~opts p)))
    staged;
  Array.iteri
    (fun s x ->
      if x = positions then
        for j = 0 to p.rn - 1 do
          equal
            ~msg:(Printf.sprintf "slot %d, a pad, column %d" s j)
            Dtypes.value (`Float 0.)
            staged.((s * p.rn) + j)
        done)
    map

let routed_law =
  let gen =
    let open Gen in
    let+ weights = of_list [ Float_weights; Mxfp4 ]
    and+ tokens = of_list [ 1; 12; 64; 128; 512 ]
    and+ topk = of_list [ 1; 2; 4 ]
    and+ experts = of_list [ 4; 8; 32 ]
    and+ rk = of_list [ 64; 160; 320 ]
    and+ rn = of_list [ 64; 128 ]
    and+ block = of_list [ 8; 64 ]
    and+ rseed = int_range 0 1_000_000 in
    { weights; tokens; topk; experts; block; rk; rn; rseed }
  in
  (* rune's blocks: 8 rows below 16 positions an expert, 64 from there. *)
  let example weights (tokens, experts) =
    let block = if tokens * 4 < 16 * experts then 8 else 64 in
    {
      weights;
      tokens;
      topk = 4;
      experts;
      block;
      rk = 320;
      rn = 128;
      rseed = tokens;
    }
  in
  prop ~count:8
    ~examples:
      (List.concat_map
         (fun w ->
           List.map (example w)
             [ (1, 8); (12, 8); (64, 32); (128, 8); (512, 8) ])
         [ Float_weights; Mxfp4 ]
      @ [
          {
            weights = Mxfp4;
            tokens = 1;
            topk = 1;
            experts = 4;
            block = 64;
            rk = 160;
            rn = 64;
            rseed = 0;
          };
          (* A core over one gathered row, which it leaves unstaged: no other
             example does, so every seed reaches the cover. *)
          {
            weights = Float_weights;
            tokens = 1;
            topk = 1;
            experts = 4;
            block = 64;
            rk = 160;
            rn = 64;
            rseed = 0;
          };
        ])
    "a staged product of gathered rows computes the bits of the same \
     optimisations unstaged, a pad row's outputs +0"
    (Gen.with_pp pp_routed gen)
    gathered_bits

(* A routed product of float weights over 128 tokens, of integers from -2 to 2:
   its sums are exact. *)
let exact_routed =
  {
    weights = Float_weights;
    tokens = 128;
    topk = 4;
    experts = 8;
    block = 64;
    rk = 160;
    rn = 128;
    rseed = 7;
  }

(* The exact outputs of a routed product whose weight [o j q] is the weight of
   expert [o] at column [j] and term [q]. *)
let routed_exact (p : routed) map owner a weight =
  let positions = p.tokens * p.topk in
  Array.init
    (Array.length map * p.rn)
    (fun e ->
      let s = e / p.rn and j = e mod p.rn in
      let o = owner.(s / p.block) in
      let t = ref 0. in
      if map.(s) <> positions then
        for q = 0 to p.rk - 1 do
          t := !t +. (float_of a.((map.(s) / p.topk * p.rk) + q) *. weight o j q)
        done;
      `Float !t)

let[@inline never] [@local never] of_gathered_rows () =
  let d, r = on_cuda () in
  let p = exact_routed in
  let prg = compiled r (routed_program p) in
  equal ~msg:"rows and weights staged" (list int) [ 1; 4 ] (copied prg);
  let map, owner, inputs = routed_inputs ~small:true p in
  let a = List.assoc 1 inputs and w = List.assoc 4 inputs in
  let weight o j q = float_of w.((o * p.rk * p.rn) + (q * p.rn) + j) in
  equal (array Dtypes.value)
    (routed_exact p map owner a weight)
    (List.assoc 0 (Run.on d prg inputs))

(* The same routed product with MXFP4 weights, of codes of any byte scaled by
   [2^-4] to [2^4]: its sums are exact. Its codes are staged beside the rows and
   its scales held whole, one stage of each trip's copies. *)
let exact_mxfp4 = { exact_routed with weights = Mxfp4; rk = 320 }

let[@inline never] [@local never] of_gathered_mxfp4 () =
  let d, r = on_cuda () in
  let p = exact_mxfp4 in
  let prg = compiled r (routed_program p) in
  equal ~msg:"rows, codes and scales staged" (list int) [ 1; 4; 5 ] (copied prg);
  let map, owner, inputs = routed_inputs ~small:true p in
  let a = List.assoc 1 inputs
  and codes = List.assoc 4 inputs
  and scales = List.assoc 5 inputs in
  let weight o j q = mxfp4_weight ~codes ~scales ~k:p.rk ((o * p.rn) + j) q in
  equal (array Dtypes.value)
    (routed_exact p map owner a weight)
    (List.assoc 0 (Run.on d prg inputs))

(* A core reading gathered rows copies its tiles in one stage: each holds what
   it holds on a target whose copies wait. *)
let gathered_one_stage () =
  let prg r = Codegen.to_program (routed_program exact_routed) r in
  let two = prg (sm89 ()) and one = prg (sm89 ~sync:true ()) in
  equal ~msg:"rows and weights staged" (list int) [ 1; 4 ] (copied two);
  equal ~msg:"the rows' tile" (list int) (tile_of one 1) (tile_of two 1);
  equal ~msg:"the weights' tile" (list int) (tile_of one 4) (tile_of two 4)

(* Gathered rows that each warp reads alone: blocks of 8 rows, the core's two
   warps on two blocks, as gpt-oss's gate and up projection with bfloat16
   experts splits its rows over 64 tokens. Staged, they would save no load and
   add a wait a trip. The weights, gathered at each warp's block's expert, are
   rows each warp reads alone too: nothing stages. *)
let alone () =
  let r = sm89 () in
  let p =
    {
      weights = Float_weights;
      tokens = 64;
      topk = 4;
      experts = 32;
      block = 8;
      rk = 160;
      rn = 128;
      rseed = 64;
    }
  in
  let opts =
    Opt.
      [
        Tc { axis = 1; tc_select = -1; tc_opt = 0; use_tc = 1 };
        Split { axis = 0; amount = 2; target = Local; top = false };
      ]
  in
  let prg = Codegen.to_program (routed_program ~opts p) r in
  let threads =
    match Ops.arg prg with
    | Program { local_size; _ } ->
        List.fold_left
          (fun n -> function Ops.Int k -> n * k | Sym _ -> n)
          1 local_size
    | _ -> 0
  in
  a_core (cores prg);
  equal ~msg:"two warps" int 64 threads;
  equal ~msg:"the tiles staged" (list int) [] (copied prg)

(* A routed MXFP4 product over [K = 320] at 128 tokens on sm_89 copies its rows
   and codes and steps its core through 5 trips of 64, which its sums take in
   blocks of 4, the second padded: its trips past the fifth read no global
   memory. At the last trip of the kernel's loops, the eighth, each index of
   global memory that reads a loop is invalid (c0790355e4: the padded trips'
   copies read past the codes). *)
let padded_trips () =
  let p =
    {
      weights = Mxfp4;
      tokens = 128;
      topk = 4;
      experts = 8;
      block = 64;
      rk = 320;
      rn = 128;
      rseed = 0;
    }
  in
  let sink = Codegen.blocked_sums (routed_program p) (sm89 ()) in
  let nodes = Ops.toposort ~calls:Skip sink in
  let loop u =
    is Range u && List.mem (Ops.axis_type u) Ops.Axis_type.[ Weak; Loop ]
  in
  let loops = List.filter loop nodes in
  let last r = (r, Shape.const_like r (`Int (Dtype.Value.to_z (Ops.vmax r)))) in
  let reads_loops u =
    is Index u
    && Ops.addrspace (Ops.nth u 0) = Some Dtype.Global
    && List.exists (fun r -> Ops.Nodes.mem r (Ops.ranges (Ops.nth u 1))) loops
  in
  let reads = List.filter reads_loops nodes in
  let slots us =
    List.sort_uniq Int.compare
      (List.filter_map (fun u -> slot (Ops.nth u 0)) us)
  in
  let copied =
    List.filter
      (fun u ->
        is Store u && Ops.addrspace (Ops.nth (Ops.nth u 0) 0) = Some Dtype.Local)
      nodes
    |> List.concat_map (fun u ->
        List.filter reads_loops (Ops.toposort ~calls:Skip (Ops.nth u 1)))
  in
  equal ~msg:"the loops' trips" (list int) [ 2; 4 ]
    (List.sort Int.compare
       (List.map (fun r -> Dtype.Value.to_int (Ops.vmax r) + 1) loops));
  equal ~msg:"buffers read in the loops" (list int) [ 1; 4 ] (slots reads);
  equal ~msg:"buffers copied in the loops" (list int) [ 1; 4 ] (slots copied);
  (* A vector index's gate selects its lanes' indices or their invalid lanes. *)
  let at_last u =
    let i = Ops.nth u 1 in
    (not (is Where i))
    || Shape.resolve
         (Ops.substitute ~calls:Skip ~pass:Once (Ops.nth i 0)
            (List.map last loops))
  in
  equal ~msg:"indices read at the last trip" (list Uops.uop) []
    (List.filter at_last reads)

(* Computed operands *)

(* The buffers a core of [p] copies: [o]'s tile beside [w]'s, or beside a
   decoded [w]'s none. [kept], read once a head, is copied where a trip takes a
   head whole, the loops of every tile then stepping through the heads alone, as
   each core's trip here does on CUDA's renderer. *)
let copied_of p =
  match p.weights_of with Bf16_rows -> [ 1; 2; 3 ] | Q8_0_rows -> [ 1 ]

let computed_bits p =
  let d, r = on_cuda () in
  let prg = compiled r (computed_kernel p) in
  let inputs = computed_inputs p in
  let run prg = List.assoc 0 (Run.on d prg inputs) in
  if p.tokens >= 64 then a_core (cores prg);
  if cores prg > 0 then
    equal ~msg:"the operands' loads staged" (list int) (copied_of p)
      (copied prg);
  let opts = applied prg in
  equal (array Dtypes.value)
    (run (compiled (unstaged r) (computed_kernel ~opts_to_apply:opts p)))
    (run prg)

let computed_law =
  cases
    "a staged computed operand computes the bits of the same optimisations \
     unstaged"
    ~id:(fun p ->
      Printf.sprintf "%s, %d tokens, heads of %d"
        (match p.weights_of with
        | Bf16_rows -> "bfloat16"
        | Q8_0_rows -> "Q8_0")
        p.tokens p.head)
    (List.concat_map
       (fun weights_of ->
         List.map
           (fun tokens -> { weights_of; tokens; head = 64 })
           [ 12; 64; 128; 512 ])
       [ Bf16_rows; Q8_0_rows ]
    @ List.map
        (fun tokens -> { weights_of = Bf16_rows; tokens; head = 32 })
        [ 64; 512 ])
    computed_bits

(* The output projection of a computed operand over 64 tokens, in heads of 32
   whose loads all step through the heads, of integers from -2 to 2 scaled by 1
   or 2: its sums are exact. Each tile is copied in two stages. *)
let[@inline never] [@local never] of_computed_operand () =
  let d, r = on_cuda () in
  let p = { weights_of = Bf16_rows; tokens = 64; head = 32 } in
  let prg = compiled r (computed_kernel p) in
  let k = computed_kernel ~opts_to_apply:(applied prg) p in
  equal ~msg:"o, kept and w staged" (list int) [ 1; 2; 3 ] (copied prg);
  equal ~msg:"two stages" (list int)
    (List.map (( * ) 2) (one_stage_buffers k r))
    (buffers prg);
  let head = p.head in
  let kk = heads * head and t = p.tokens in
  let o = small 1 (heads * t * head) and w = small 2 (outputs * kk) in
  let kept =
    Array.init (heads * t) (fun i -> `Float (Float.of_int (1 + (i mod 2))))
  in
  let expected =
    Array.init (t * outputs) (fun e ->
        let tok = e / outputs and j = e mod outputs in
        let s = ref 0. in
        for q = 0 to kk - 1 do
          let h = q / head and dd = q mod head in
          s :=
            !s
            +. float_of o.((h * t * head) + (tok * head) + dd)
               *. float_of kept.((h * t) + tok)
               *. float_of w.((j * kk) + q)
        done;
        `Float !s)
  in
  equal (array Dtypes.value) expected
    (List.assoc 0 (Run.on d prg [ (1, o); (2, kept); (3, w) ]))

(* A core under another sum

   [c[i, j] = Σ_h Σ_r a[i, r] · b'[h, r, j]], [b'] the bfloat16 of [b[h, r, j] ·
   s[h]] computed in the kernel: [a]'s tile reads [i] and [r] alone, and the
   core's loop [r] runs inside the sum over [h], in two stages. [b]'s tile steps
   through [h] as well as [r], another loop than [a]'s, so [b'] stays unstaged.
   Integers from -2 to 2 times scales of [1] or [2]: every sum is exact. *)
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
  Ops.sink
    ~kernel:(Ops.kernel_info ?opts_to_apply ())
    [ Ops.end_ store [ i; j ] ]

let core_under_a_sum () =
  let d, r = on_cuda () in
  let k = outer_kernel ~opts_to_apply:(optimised (outer_kernel ()) r) () in
  let prg = compiled r k in
  a_core (cores prg);
  equal ~msg:"a's tile staged" (list int) [ 1 ] (copied prg);
  equal ~msg:"two stages" (list int)
    (List.map (( * ) 2) (one_stage_buffers k r))
    (buffers prg);
  let m, h, kk, n = outer in
  let a = small 1 (m * kk) and b = small 2 (h * kk * n) in
  let sc = Array.init h (fun g -> `Float (Float.of_int (1 + (g mod 2)))) in
  let expected =
    Array.init (m * n) (fun e ->
        let i = e / n and j = e mod n in
        let t = ref 0. in
        for g = 0 to h - 1 do
          for q = 0 to kk - 1 do
            t :=
              !t
              +. float_of a.((i * kk) + q)
                 *. (float_of b.((g * kk * n) + (q * n) + j) *. float_of sc.(g))
          done
        done;
        `Float !t)
  in
  equal (array Dtypes.value) expected
    (List.assoc 0 (Run.on d prg [ (1, a); (2, b); (3, sc) ]))

(* Decoded operands *)

let dense_bits p =
  let d, r = on_cuda () in
  let prg = compiled r (dense_kernel p) in
  let run prg = List.assoc 0 (Run.on d prg (dense_inputs p)) in
  let cored = cores prg > 0 in
  if p.dtokens >= 64 then a_core (cores prg);
  let copied = copied prg in
  if cored then
    equal
      ~msg:
        "x, and the codes and scales where a copy of 16 bytes keeps the codes"
      (list int)
      (if p.offset mod 16 = 0 then [ 1; 2; 3 ] else [ 1 ])
      copied;
  if List.mem 2 copied then
    equal ~msg:"the codes' copies" (list int) [ 16 ]
      (List.sort_uniq Int.compare
         (List.filter_map
            (fun c -> if c.from = 2 then Some c.bytes else None)
            (copies prg)));
  cover "a core" cored;
  cover "codes staged, scales held" (cored && List.mem 3 copied);
  cover "codes at an offset unstaged" (cored && p.offset mod 16 <> 0);
  cover "codes over the loop within a line" (cored && p.dk / 2 <= 128);
  let opts = applied prg in
  equal (array Dtypes.value)
    (run (compiled (unstaged r) (dense_kernel ~opts_to_apply:opts p)))
    (run prg)

let dense_law =
  let gen =
    let open Gen in
    let+ dtokens = of_list [ 1; 12; 64; 512 ]
    and+ dk = of_list [ 256; 512; 2880 ]
    and+ dn = of_list [ 128; 256 ]
    and+ offset = of_list [ 0; 8; 16 ] in
    { dtokens; dk; dn; offset }
  in
  prop ~count:8
    ~examples:
      (List.map
         (fun dtokens -> { dtokens; dk = 512; dn = 128; offset = 0 })
         [ 1; 12; 64; 512 ]
      @ [
          { dtokens = 64; dk = 512; dn = 128; offset = 8 };
          { dtokens = 64; dk = 256; dn = 128; offset = 0 };
        ])
    "a staged dense MXFP4 product computes the bits of the same optimisations \
     unstaged"
    (Gen.with_pp pp_dense gen) dense_bits

(* A dense MXFP4 product of 64 tokens, of integers from -2 to 2 times codes
   scaled by [2^-4] to [2^4]: its sums are exact. Its scales are held whole
   beside the staged codes and rows. *)
let[@inline never] [@local never] of_held_scales () =
  let d, r = on_cuda () in
  let p = { dtokens = 64; dk = 512; dn = 128; offset = 0 } in
  let prg = compiled r (dense_kernel p) in
  equal ~msg:"x, codes and scales staged" (list int) [ 1; 2; 3 ] (copied prg);
  let inputs = dense_inputs ~small:true p in
  let x = List.assoc 1 inputs
  and codes = List.assoc 2 inputs
  and scales = List.assoc 3 inputs in
  let expected =
    Array.init (p.dtokens * p.dn) (fun e ->
        let t = e / p.dn and j = e mod p.dn in
        let s = ref 0. in
        for q = 0 to p.dk - 1 do
          s :=
            !s
            +. float_of x.((t * p.dk) + q)
               *. mxfp4_weight ~codes ~scales ~k:p.dk j q
        done;
        `Float !s)
  in
  equal (array Dtypes.value) expected (List.assoc 0 (Run.on d prg inputs))

(* On a target that copies in one stage the decoded operand stays unstaged, its
   codes and scales loads, and [x] alone is copied. *)
let decoded_without_async () =
  let p = { dtokens = 64; dk = 512; dn = 128; offset = 0 } in
  let prg = Codegen.to_program (dense_kernel p) (sm89 ~sync:true ()) in
  a_core (cores prg);
  equal (list int) [ 1 ] (copied prg)

(* Products in registers

   Float32 products whose operands both lie along the sum, [b] stored [N; K],
   take the staged product's tile: 512 by 256 over 440, blocked, its trips of 20
   values a loop of 5 steps of 4, and 256 by 256 over 160 and an affine operand
   over 64, short of the units, at 32 values and 8 steps; and over 110, which 16
   bytes do not divide, in copies of 8 bytes. Each copies both operands' loads
   into shared memory and computes the bits of the same optimisations unstaged,
   whose loop runs one step a trip. *)
let products_in_registers =
  List.map
    (fun (m, k, n, form) ->
      { dt = Float32; batch = 1; m; k; n; layout = Nk; form })
    [
      (512, 440, 256, Plain);
      (256, 160, 256, Plain);
      (128, 64, 128, Affine);
      (256, 110, 256, Plain);
    ]

let[@inline never] [@local never] in_registers p =
  let d, r = on_cuda () in
  let k = kernel ~opts_to_apply:(chosen ~staged:true p r) p in
  let prg = compiled r k in
  satisfies ~claim:"both operands copied" (list int)
    (fun l -> List.mem 1 l && List.mem 2 l)
    (copied prg);
  let inputs =
    [ (1, values 1 (a_size p)); (2, values 2 (p.k * p.n)) ]
    @ if p.form = Affine then [ (3, values 3 p.k); (4, values 4 p.k) ] else []
  in
  let unstaged = List.assoc 0 (Run.on d (compiled (unstaged r) k) inputs) in
  equal (array Dtypes.value) unstaged (List.assoc 0 (Run.on d prg inputs))

(* A sum over 4 heads of 110 float32 dimensions: a row of [a] starts every 440
   bytes, so its tiles copy in runs of 8 bytes, and the sum's loops, the heads'
   and the dimensions' trips, are a nest that two stages step through. Short of
   the units, the tile is 64 rows by 32 columns, and a trip takes 22 values, 11
   steps of 2 of the 110 in 5 trips. *)
let[@inline never] [@local never] in_copies_of_8 () =
  let d, r = on_cuda () in
  let k =
    merged_kernel ~opts_to_apply:(optimised (merged_kernel merged) r) merged
  in
  let prg = compiled r k in
  equal ~msg:"copies of 8 bytes" (list int) [ 8 ]
    (List.sort_uniq Int.compare (List.map (fun c -> c.bytes) (copies prg)));
  equal ~msg:"two stages" (list int) [ 2 * 32 * 22; 2 * 64 * 22 ] (buffers prg);
  let m, h, kk, n = merged in
  let inputs = [ (1, values 1 (h * m * kk)); (2, values 2 (n * h * kk)) ] in
  let unstaged = List.assoc 0 (Run.on d (compiled (unstaged r) k) inputs) in
  equal (array Dtypes.value) unstaged (List.assoc 0 (Run.on d prg inputs))

(* SwiGLU's two projections of one input, [c.(i, j) = (Σ_r x.(i, r) · w.(j, r))
   · (Σ_r x.(i, r) · v.(j, r))] in float32: the first product copies [x]'s tile
   and the second reads it. *)
let swiglu ?opts_to_apply (m, n, k) =
  let c = Shape.param ~shape:[ Ops.Int (m * n) ] 0 Float32
  and x = Shape.param ~shape:[ Ops.Int (m * k) ] 1 Float32
  and w = Shape.param ~shape:[ Ops.Int (n * k) ] 2 Float32
  and v = Shape.param ~shape:[ Ops.Int (n * k) ] 3 Float32 in
  let open Ops.O in
  let i = Ops.range ~axis_type:Weak (Int m) [ 1 ]
  and j = Ops.range ~axis_type:Weak (Int n) [ 2 ]
  and r = Ops.range ~axis_type:Weak (Int k) [ 0 ] in
  let sum y =
    Ops.reduce
      (Ops.index x [ (i * int k) + r ] * Ops.index y [ (j * int k) + r ])
      Add [ r ]
  in
  let store = Ops.store (Ops.index c [ (i * int n) + j ]) (sum w * sum v) in
  Ops.sink
    ~kernel:(Ops.kernel_info ?opts_to_apply ())
    [ Ops.end_ store [ i; j ] ]

let shared_input () =
  let r = sm89 () in
  let sizes = (1024, 512, 256) in
  let k = swiglu ~opts_to_apply:(optimised (swiglu sizes) r) sizes in
  let prg = Codegen.to_program k r in
  equal ~msg:"the buffers copied" (list int) [ 1; 2; 3 ] (copied prg);
  equal ~msg:"x's tile copied once" int 3 (List.length (locals (Ops.nth prg 1)))

(* Product tiles *)

type tile_case = { async : bool; itemsize : int; tk : int; short : bool }

(* [stage.mli]'s tile: on a target that copies in two stages, 16 threads along M
   by 8 along N, 8 by 8 outputs a thread or 4 by 4 where [short], and a run of
   16 bytes where those values divide K at least twice, else of 8 bytes where
   they divide K and K holds at least 256 bytes. *)
let expected_tile c =
  let run bytes =
    if bytes mod c.itemsize <> 0 then None
    else
      let run = bytes / c.itemsize in
      let enough =
        if bytes = 16 then c.tk / run >= 2 else c.tk * c.itemsize >= 256
      in
      if c.tk mod run = 0 && enough then Some run else None
  in
  if not c.async then None
  else
    match (run 16, run 8) with
    | Some run, _ | None, Some run ->
        Some
          { Stage.threads = (16, 8); outputs = (if c.short then 4 else 8); run }
    | None, None -> None

let product_tile_t =
  Testable.make
    ~pp:(fun ppf { Stage.threads = tm, tn; outputs; run } ->
      Format.fprintf ppf "%d by %d threads, %d outputs, runs of %d" tm tn
        outputs run)
    ~equal:( = )

let tile_law =
  let gen =
    let open Gen in
    let+ async = bool
    and+ itemsize = of_list [ 1; 2; 4; 8; 16 ]
    and+ tk = of_list [ 1; 2; 4; 8; 16; 24; 32; 36; 64; 110; 128; 440 ]
    and+ short = bool in
    { async; itemsize; tk; short }
  in
  let case ?(async = true) ?(short = false) itemsize tk =
    { async; itemsize; tk; short }
  in
  prop ~count:200
    ~examples:
      [
        case 4 8;
        case 4 4;
        case 4 110;
        case 4 36;
        case 2 24;
        case ~short:true 2 64;
        case ~async:false 4 64;
      ]
    "a staged product's tile copies runs of 16 bytes, or of 8 where they do \
     not divide K"
    (Gen.with_pp
       (fun ppf c ->
         Format.fprintf ppf "%d values of %d bytes%s%s" c.tk c.itemsize
           (if c.async then "" else ", copies that wait")
           (if c.short then ", short" else ""))
       gen)
    (fun c ->
      let expected = expected_tile c in
      let run_bytes =
        Option.map (fun (t : Stage.product_tile) -> t.run * c.itemsize) expected
      in
      cover "none where copies wait" (not c.async);
      cover "none where K is short or undivided" (c.async && expected = None);
      cover "runs of 16 bytes" (run_bytes = Some 16);
      cover "runs of 8 bytes" (run_bytes = Some 8);
      cover "short" (c.short && expected <> None);
      equal (option product_tile_t) expected
        (Stage.product_tile
           (sm89 ~sync:(not c.async) ())
           ~itemsize:c.itemsize ~k:c.tk ~short:c.short))

(* Steps

   On sm_89, which copies asynchronously, a trip of a dense MXFP4 product takes
   64 values, so that each row of codes a trip is a sector of 32 bytes: two
   stages of [rows · 32] bytes of codes for a workgroup's [rows] rows, beside
   two of [x]'s 64 values of each of its tokens, and the scales held whole,
   [rows · k / 32]. Over [5760; 2880] a workgroup takes 16 rows and 64 tokens at
   64 tokens, 160 rows and 64 tokens at 512; over [128; 256], 8 rows and 16
   tokens, whose codes, 128 bytes a row over the whole loop, are copied each
   trip all the same, beside the held scales. The shared buffers' elements: *)

let decoded_trips =
  cases "a dense MXFP4 product's staged rows cover a sector a trip"
    ~id:(fun (p, _) -> Format.asprintf "%a" pp_dense p)
    [
      ( { dtokens = 64; dk = 2880; dn = 5760; offset = 0 },
        [ 2 * 16 * 32; 16 * 90; 2 * 64 * 64 ] );
      ( { dtokens = 512; dk = 2880; dn = 5760; offset = 0 },
        [ 2 * 64 * 64; 2 * 160 * 32; 160 * 90 ] );
      ( { dtokens = 64; dk = 256; dn = 128; offset = 0 },
        [ 8 * 8; 2 * 8 * 32; 2 * 16 * 64 ] );
    ]
    (fun (p, expected) ->
      equal (list int) expected
        (buffers (Codegen.to_program (dense_kernel p) (sm89 ()))))

(* The staged tile

   A staged core's workgroup is 2 by 2 warps, 128 threads, of 64 by 64 outputs,
   each halved, M then N, to 32 by 32 while the kernel keeps fewer than 16384 /
   stages threads; K is unrolled to 32. In two stages on sm_89, whose renderer
   states its units and their bytes in flight, the busiest unit's copies a trip
   reach those bytes where shared memory and the sum's blocks allow: 512 by 512
   over 2880 keeps one workgroup of 64 by 64 a unit and takes 96 values a trip,
   30 trips. M pads as N does: 110 by 440 takes the tile of 440 by 110. 16 rows
   by 440 are one tile of M, whose pad has no axis, so the core stays unstaged
   with N's 55 tiles unpadded. *)

let staged_tiles =
  cases "a staged core's workgroups, threads and trips"
    ~id:(fun ((_, m, k, n, sync), _) ->
      Printf.sprintf "%d by %d over %d, %s" m n k
        (if sync then "one stage" else "two stages"))
    [
      ((Dtype.Bfloat16, 2048, 2048, 2048, false), (256, 128, 64));
      ((Dtype.Bfloat16, 512, 2880, 4096, false), (128, 128, 90));
      ((Dtype.Bfloat16, 512, 2880, 5760, false), (180, 128, 90));
      ((Dtype.Bfloat16, 512, 2880, 512, false), (64, 128, 30));
      ((Dtype.Bfloat16, 512, 2880, 4096, true), (128, 128, 90));
      ((Dtype.Float16, 256, 2880, 256, true), (16, 128, 90));
      ((Dtype.Bfloat16, 110, 2880, 440, false), (56, 128, 15));
      ((Dtype.Bfloat16, 440, 2880, 110, false), (56, 128, 15));
      ((Dtype.Bfloat16, 16, 64, 440, false), (55, 32, 4));
    ]
    (fun ((dt, m, k, n, sync), expected) ->
      equal (triple int int int) expected
        (planned ~sync
           (kernel { dt; batch = 1; m; k; n; layout = Kn; form = Plain })))

(* Rows already in their slots, as a routed layer's second product reads the
   first's outputs: slot [σ] of [blocks] blocks of [block] rows reads row [σ] of
   [a] and the bfloat16 [K; N] matrix of its block's expert, [c.(σ, j) = Σ_r
   a.(σ, r) · w.(owner.(σ / block), r, j)]. *)
let slotted_kernel ~block ~blocks ~experts ~k ~n =
  let slots = blocks * block and kn = k * n in
  let c = Shape.param ~shape:[ Ops.Int (slots * n) ] 0 Float32
  and a = Shape.param ~shape:[ Ops.Int (slots * k) ] 1 Bfloat16
  and owner = Shape.param ~shape:[ Ops.Int blocks ] 2 Int32
  and w = Shape.param ~shape:[ Ops.Int (experts * kn) ] 3 Bfloat16 in
  let open Ops.O in
  let s = Ops.range ~axis_type:Weak (Int slots) [ 0 ]
  and j = Ops.range ~axis_type:Weak (Int n) [ 1 ]
  and r = Ops.range ~axis_type:Weak (Int k) [ 2 ] in
  let e = Ops.cast (Ops.index owner [ s // int block ]) Weak_int in
  let av = Ops.index a [ (s * int k) + r ]
  and bv = Ops.index w [ (e * int kn) + (r * int n) + j ] in
  let sum = Ops.reduce (Ops.cast av Float32 * Ops.cast bv Float32) Add [ r ] in
  let store = Ops.store (Ops.index c [ (s * int n) + j ]) sum in
  Ops.sink ~kernel:(Ops.kernel_info ()) [ Ops.end_ store [ s; j ] ]

(* A block's rows read one matrix, so a staged warp holds as many of the core's
   16-row tiles as a block has: 3 of 48 rows, 4 of 64, and one workgroup loads a
   block's tile of the matrix. A block takes 20 workgroups of 2 warps along its
   2560 outputs, and the core 90 trips of 32 of its 2880 values. 49 blocks of 48
   rows are 147 tiles, which no warp's 4 tiles divide: rows in runs take no pad,
   which would leave a warp one tile. *)
let slotted_tiles =
  cases "a staged warp holds a block's rows of one matrix"
    ~id:(fun (blocks, block) -> Printf.sprintf "%d blocks of %d" blocks block)
    [ (50, 48); (50, 64); (49, 48) ]
    (fun (blocks, block) ->
      equal (triple int int int)
        (blocks * 20, 64, 90)
        (planned (slotted_kernel ~block ~blocks ~experts:8 ~k:2880 ~n:2560)))

(* The splits the hand-coded rules choose for a routed product of float weights
   on CUDA, staged or not. A core reading gathered rows copies in one stage and
   keeps the unstaged splits. *)
let routed_tiles =
  cases "a core reading gathered rows keeps the unstaged splits"
    ~id:(Printf.sprintf "%d tokens") [ 12; 64; 128; 512 ] (fun tokens ->
      let p =
        {
          weights = Float_weights;
          tokens;
          topk = 4;
          experts = 8;
          block = 64;
          rk = 160;
          rn = 128;
          rseed = tokens;
        }
      in
      let r = sm89 () in
      let opts staged = optimised ~staged (routed_program p) r in
      equal (list opt) (opts false) (opts true))

(* A staged core's plans under another sum, on renderers that state units and
   their bytes in flight or not, compute the same bits: blocks of [r]'s terms of
   64, 320 and 512, which the core takes whole or in trips; 352 and 416, whose
   trips the sum's running sums pad, so that a nest's next trip passes the inner
   loop's bound; and 3, which leaves the core [g] for its [K], over 16 and 32
   blocks, with [r] inside it. Where the two renderers lower one program, its
   instructions equal, it runs once. *)
let heads_opts r sizes = optimised (heads_kernel sizes) r

let lower_and_compile r k =
  let lin = Codegen.linearize k r in
  match Codegen.compile lin r with
  | prg -> prg
  | exception Renderer.Compiler.Compile_error m
    when says "cannot open source file" m ->
      skip ~reason:("a header is missing, set CUDA_PATH: " ^ m) ()

let heads_bits ((sizes, units, in_flight), seed) =
  let d, r = on_cuda () in
  let name =
    Printf.sprintf "%s of %d units, %d in flight" r.name units in_flight
  in
  let stated = rebuilt ~name ~units ~in_flight r
  and unstated =
    rebuilt ~name:(r.name ^ " of no units") ~units:0 ~in_flight:0 r
  in
  let _, h, k, _ = sizes in
  let m, _, _, n = sizes in
  let kernel r = heads_kernel ~opts_to_apply:(heads_opts r sizes) sizes in
  let k1 = kernel stated and k0 = kernel unstated in
  let lin1 = Codegen.linearize k1 stated
  and lin0 = Codegen.linearize k0 unstated in
  let same = Ops.equal (Ops.nth lin1 1) (Ops.nth lin0 1) in
  let staged = locals (Ops.nth lin1 1) <> [] in
  cover "a plan the units move, staged" ((not same) && staged);
  cover "a plan they leave" same;
  cover "a padded block, staged" (staged && (k = 352 || k = 416));
  cover "the core's K from the blocks, staged" (staged && k = 3);
  if not same then begin
    let inputs =
      [ (1, values seed (h * m * k)); (2, values (seed + 1) (h * k * n)) ]
    in
    let run r k = List.assoc 0 (Run.on d (lower_and_compile r k) inputs) in
    equal (array Dtypes.value) (run unstated k0) (run stated k1)
  end

let heads_law =
  let open Gen in
  let sizes =
    let* k = of_list [ 3; 64; 320; 352; 416; 512 ] in
    let+ h = if k = 3 then of_list [ 16; 32 ] else of_list [ 2; 4; 8 ]
    and+ m = of_list [ 32; 64 ]
    and+ n = of_list [ 32; 64 ] in
    (m, h, k, n)
  in
  (* The examples reach every cover; the draws add new shapes each run. *)
  prop ~count:8
    ~examples:
      [
        (((64, 4, 512, 64), 100, 18304), 1);
        (((64, 8, 320, 64), 24, 32768), 2);
        (((64, 32, 3, 64), 100, 18304), 3);
        (((64, 4, 352, 64), 100, 18304), 4);
        (((64, 2, 416, 64), 24, 32768), 5);
        (* One unit of 4 KiB in flight keeps the plan of no units. *)
        (((64, 4, 64, 64), 1, 4096), 6);
        (* 16 blocks of 320, more blocks than the draws give a K of 64 to
           512. *)
        (((64, 16, 320, 64), 100, 18304), 7);
      ]
    "a staged core's plans under another sum at any units and bytes in flight \
     compute the bits of the plan that reads neither"
    (Gen.with_pp
       (fun ppf (((m, h, k, n), u, f), seed) ->
         Format.fprintf ppf
           "%d by %d over %d blocks of %d, %d units, %d bytes in flight, seed \
            %d"
           m n h k u f seed)
       (pair
          (triple sizes
             (of_list [ 1; 24; 100; 142 ])
             (of_list [ 4096; 18304; 32768 ]))
          (int_range 0 1_000_000)))
    heads_bits

(* A staged core's tile and its trips follow the renderer's compute units and
   the bytes a unit keeps in flight, and never the order of a sum: a key
   projection of 64 tokens by 512 over 2880 takes one plan on CUDA's renderer,
   another on the same renderer stating a part of 24 units, and a third where
   neither is stated, and the three compute the same bits. *)
let spread_bits () =
  let d, r = on_cuda () in
  let p =
    {
      dt = Bfloat16;
      batch = 1;
      m = 64;
      k = 2880;
      n = 512;
      layout = Nk;
      form = Plain;
    }
  in
  let fewer = rebuilt ~name:(r.name ^ " of 24 units") ~units:24 r
  and unstated =
    rebuilt ~name:(r.name ^ " of no units") ~units:0 ~in_flight:0 r
  in
  let opts r = chosen ~staged:true p r in
  not_equal ~msg:"24 units" (list opt) (opts r) (opts fewer);
  not_equal ~msg:"no units" (list opt) (opts r) (opts unstated);
  let inputs =
    [ (1, values 1 (a_size p)); (2, values 2 (p.batch * p.k * p.n)) ]
  in
  let run r =
    let prg = compiled r (kernel ~opts_to_apply:(opts r) p) in
    List.assoc 0 (Run.on d prg inputs)
  in
  let bits = run r in
  equal ~msg:"24 units" (array Dtypes.value) bits (run fewer);
  equal ~msg:"no units" (array Dtypes.value) bits (run unstated)

(* Layout *)

(* The offset in a tile of rows of [chunks] copies of 16 bytes of the element at
   [col] in [row], of [itemsize] bytes. *)
let offset ~itemsize ~chunks col row =
  let cols = chunks * 16 / itemsize in
  let at = Stage.layout ~itemsize ~cols (Ops.int col) (Ops.int row) in
  Shape.sym_infer (Sym at) []

(* The 16-byte slot of a 128-byte line that an offset of [itemsize]-byte
   elements falls on. *)
let slot_of ~itemsize e = e * itemsize / 16 mod 8
let distinct l = List.length (List.sort_uniq Int.compare l)
let rec pow2_factor n = if n mod 2 = 0 then 2 * pow2_factor (n / 2) else 1

type swizzle = { itemsize : int; chunks : int; row : int; chunk : int }

let swizzles =
  let open Gen in
  let+ itemsize = of_list [ 1; 2; 4; 8; 16 ]
  and+ chunks = int_range 1 64
  and+ row = int_range 0 63
  and+ chunk = int_range 0 63 in
  { itemsize; chunks; row; chunk = chunk mod chunks }

let pp_swizzle ppf s =
  Format.fprintf ppf "rows of %d chunks of %d-byte items, row %d, chunk %d"
    s.chunks s.itemsize s.row s.chunk

(* A row's chunks are permuted by a factor of their count, 8 at most. *)
let cover_permutations s =
  let g = min 8 (pow2_factor s.chunks) in
  List.iter
    (fun w -> cover (Printf.sprintf "%d chunks permuted" w) (g = w))
    [ 1; 2; 4; 8 ]

let swizzle_examples =
  List.map
    (fun chunks -> { itemsize = 2; chunks; row = 3; chunk = 1 })
    [ 3; 6; 12; 24 ]

let layout_laws =
  [
    prop ~count:100 ~examples:swizzle_examples
      "eight consecutive rows at one chunk fall on eight slots of a line"
      (Gen.with_pp pp_swizzle swizzles) (fun s ->
        cover_permutations s;
        let v = 16 / s.itemsize in
        equal int 8
          (distinct
             (List.init 8 (fun r ->
                  slot_of ~itemsize:s.itemsize
                    (offset ~itemsize:s.itemsize ~chunks:s.chunks (s.chunk * v)
                       (s.row + r))))));
    prop ~count:100 ~examples:swizzle_examples
      "eight consecutive chunks of a tile from a multiple of eight, in row \
       order, fall on eight slots of a line" (Gen.with_pp pp_swizzle swizzles)
      (fun s ->
        cover_permutations s;
        let v = 16 / s.itemsize in
        let first = ((s.row * s.chunks) + s.chunk) / 8 * 8 in
        let chunk f =
          slot_of ~itemsize:s.itemsize
            (offset ~itemsize:s.itemsize ~chunks:s.chunks
               (v * (f mod s.chunks))
               (f / s.chunks))
        in
        equal int 8 (distinct (List.init 8 (fun f -> chunk (first + f)))));
    prop ~count:100 ~examples:swizzle_examples
      "a row's elements take the row's offsets, each chunk's together"
      (Gen.with_pp pp_swizzle swizzles) (fun s ->
        cover_permutations s;
        let v = 16 / s.itemsize and cols = s.chunks * 16 / s.itemsize in
        let at col = offset ~itemsize:s.itemsize ~chunks:s.chunks col s.row in
        equal (list int)
          (List.init cols (fun c -> (s.row * cols) + c))
          (List.sort Int.compare (List.init cols at));
        equal ~msg:"a chunk's elements" (list int)
          (List.init v (fun e -> at (s.chunk * v) + e))
          (List.init v (fun e -> at ((s.chunk * v) + e))));
    cases
      "a row of no whole number of chunks, or of items over 16 bytes, is laid \
       out in order"
      ~id:(fun (itemsize, cols) ->
        Printf.sprintf "%d items of %d bytes" cols itemsize)
      [ (2, 12); (1, 24); (4, 5); (32, 4) ]
      (fun (itemsize, cols) ->
        let at (row, col) =
          Shape.sym_infer
            (Sym (Stage.layout ~itemsize ~cols (Ops.int col) (Ops.int row)))
            []
        in
        let elements =
          List.concat_map
            (fun row -> List.init cols (fun col -> (row, col)))
            (List.init 4 Fun.id)
        in
        equal (list int)
          (List.map (fun (row, col) -> (row * cols) + col) elements)
          (List.map at elements));
  ]

(* Decoded operands of the host *)

(* Operands over [r], each with whether it is decoded: codes are integers loaded
   from integer storage, through integer arithmetic. *)
let decoded_operands () =
  let r = Ops.range ~axis_type:Weak (Ops.Int 64) [ 0 ] in
  let param slot dt = Shape.param ~shape:[ Ops.Int 64 ] slot dt in
  let read slot dt = Ops.index (param slot dt) [ r ] in
  let f32 u = Ops.cast u Float32 and u32 n = Ops.int ~dtype:Uint32 n in
  let h = f32 (read 0 Bfloat16) in
  (* ReLU as nx computes it on a bfloat16: its bits through an integer, the sign
     selected. *)
  let sign = Ops.where (Ops.lt (Ops.float 0.) h) (u32 0xffffffff) (u32 0) in
  let relu =
    Ops.bitcast (Ops.bitwise_and (Ops.bitcast h Uint32) sign) Float32
  in
  let codes = read 1 Uint8 in
  let scale = Ops.bitcast (Ops.shl (Ops.cast codes Uint32) (u32 23)) Float32 in
  ( [
      ("a byte code", f32 codes);
      ("an int8 code by a scale", Ops.mul (f32 (read 2 Int8)) (read 3 Float32));
      ("an E8M0 scale's bits", scale);
    ],
    [
      ("a widened bfloat16", h);
      ("ReLU of a bfloat16", relu);
      ("a range's position", f32 r);
      ( "a gather at loaded positions",
        Ops.index (param 4 Float32) [ read 5 Int32 ] );
    ] )

let decoded_kinds =
  [
    cases "an operand that converts integer codes it loads is decoded" ~id:fst
      (fst (decoded_operands ()))
      (fun (_, o) -> satisfies ~claim:"decoded" Uops.uop Stage.decoded o);
    cases "an operand of no such codes is not decoded" ~id:fst
      (snd (decoded_operands ()))
      (fun (_, o) ->
        satisfies ~claim:"not decoded" Uops.uop
          (fun o -> not (Stage.decoded o))
          o);
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

let host_target () = Run.target Rig.host
let host = lazy (Cstyle.clang (host_target ()))

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
  let instrs = instrs prg in
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

(* The pad of the rows, the kernel's first axis, that the hand-coded rules
   choose for [p] on [r], if any: a multiple of the rows a tile then takes
   (D188). *)
let rows_pad p r =
  List.find_map
    (function Opt.Padto { axis = 0; amount } -> Some amount | _ -> None)
    (optimised (decoded_kernel p) r)

(* The output lanes of the tile the hand-coded rules choose for [p] on [r]: its
   rows by its columns. *)
let tile_lanes p r =
  let module S = Postrange.Scheduler in
  let s = S.v (decoded_kernel p) r in
  S.convert_loop_to_global s;
  let s = Heuristic.hand_coded_optimizations s in
  List.fold_left
    (fun n a ->
      match List.nth (S.full_shape s) a with Ops.Int k -> n * k | Sym _ -> n)
    1
    (S.axes_of ~reduce:false s [ Upcast ])

(* A panel holds the values one trip of the columns' loop reads: the columns the
   tile upcasts by [k] values, of 4 bytes. *)
let panel_fits p columns = columns * p.dk * 4 <= 64 * 1024

let expected_decodes p ((count, _) as file) =
  let rows, trips =
    match rows_pad p (host_file file) with
    | Some m -> (m, (p.dm + m - 1) / m)
    | None ->
        let rows = tile_rows p.dm (min 16 (count / 2)) in
        (rows, p.dm / rows)
  in
  let columns = tile_lanes p (host_file file) / rows in
  if trips >= 2 && panel_fits p columns then 1 else trips

let decoded_once =
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
  test "with staging off, each row tile decodes the codes again" (fun () ->
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

(* The decoded product [p] lowered on the host with its staged optimisations,
   staged or not, each lowered once. *)
let decoded_bits (p, seed) =
  let r = Lazy.force host in
  let opts =
    Setting.context Setting.[ B (stage, true) ] (fun () -> decoded_opts p r)
  in
  let lowered staged =
    Setting.context
      Setting.[ B (stage, staged) ]
      (fun () -> Codegen.linearize (decoded_kernel ~opts_to_apply:opts p) r)
  in
  let on = lowered true and off = lowered false in
  let staged = decodes on and unstaged = decodes off in
  cover "staged" (staged < unstaged);
  cover "one row tile" (unstaged = 1);
  cover "past the panel" (unstaged > 1 && staged = unstaged);
  let run prg =
    List.assoc 0 (Run.on_host (Codegen.compile prg r) (decoded_inputs seed p))
  in
  equal (array Dtypes.value) (run off) (run on)

let decoded_law =
  let gen =
    let open Gen in
    let+ dm = of_list [ 1; 3; 8; 10; 16; 24; 64 ]
    and+ dk = of_list [ 32; 96; 256; 1024 ]
    and+ dn = of_list [ 4; 8; 12; 32 ]
    and+ seed = int_range 0 1_000_000 in
    ({ dm; dk; dn }, seed)
  in
  (* The examples reach every cover; the draws add new shapes each run. *)
  prop ~count:8
    ~examples:
      [
        ({ dm = 64; dk = 256; dn = 8 }, 1);
        ({ dm = 10; dk = 1024; dn = 4 }, 2);
        ({ dm = 1; dk = 64; dn = 4 }, 3);
        (* Past the panel: 8 columns of 8192 values. *)
        ({ dm = 64; dk = 8192; dn = 8 }, 4);
      ]
    "on the host a staged decoded product computes the bits of the same \
     optimisations unstaged"
    (Gen.with_pp
       (fun ppf (p, seed) ->
         Format.fprintf ppf "%d rows by [%d; %d], seed %d" p.dm p.dn p.dk seed)
       gen)
    decoded_bits

(* The value of the address [u] at the first trip of every loop, a block's loop
   of a split launch included: a range, or a load of a block's bounds, is
   [0]. *)
let rec at_first u =
  match Ops.op u with
  | Op.Const -> Ops.value u
  | Op.Range | Op.Load -> `Int Bigint.zero
  | Op.Cast | Op.After -> at_first (Ops.nth u 0)
  | o when Op.Set.mem o Op.Set.alu ->
      Ops.exec_alu o (Ops.dtype u) (List.map at_first (Ops.src u))
  | o -> failf "an address computes %s" (Op.name o)

(* The code loads of [prg], each as the first byte it reads at the first trip of
   every loop and the bytes it reads. *)
let code_reads prg =
  List.filter_map
    (fun u ->
      if code_load u then
        let at = Ops.nth u 0 in
        Some (to_int (at_first (Ops.nth at 1)), Shape.max_numel at)
      else None)
    (instrs prg)

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
           (1, Array.map float a); (2, Array.map int c); (3, Array.map float s);
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
            {
              dt = Bfloat16;
              batch = 1;
              m = 2048;
              k = 2048;
              n = 2048;
              layout = Kn;
              form = Plain;
            }
            "packages/tolk/test/codegen/stage/bf16_2048.cu");
      test "a float16 product whose core pads M and K" (fun () ->
          rendered
            {
              dt = Float16;
              batch = 1;
              m = 62;
              k = 56;
              n = 64;
              layout = Kn;
              form = Plain;
            }
            "packages/tolk/test/codegen/stage/f16_padded.cu");
    ]

let () =
  exit
    (Windtrap.run "Tolk.Stage"
       [
         group "staging"
           [
             bits_law;
             barriers;
             serial;
             test
               "a nest of loops is staged in two stages and computes the \
                unstaged bits"
               over_nested_loops
             |> tag race;
             test
               "a core whose K its tile takes whole is staged in two stages \
                and computes the exact product"
               core_taken_whole;
             test "a nest of an extent read from memory is staged in one stage"
               symbolic_nest;
             tag race
             @@ test
                  "a product of 128 cubed staged in two stages is the exact \
                   product"
                  in_two_stages;
             tag race
             @@ test
                  "a product of 128 cubed staged in one stage is the exact \
                   product"
                  in_one_stage;
             cuda_source;
             races;
           ];
         group "depth" [ depth_law ];
         group "operands"
           [
             group "gathered rows"
               [
                 routed_law;
                 tag race
                 @@ test "a routed product of gathered rows is exact"
                      of_gathered_rows;
                 tag race
                 @@ test
                      "a routed MXFP4 product of gathered rows, its scales \
                       held whole, is exact"
                      of_gathered_mxfp4;
                 test
                   "a core reading gathered rows copies its tiles in one stage"
                   gathered_one_stage;
                 test "gathered rows each warp reads alone stay unstaged" alone;
                 test "a staged core's padded trips read no global memory"
                   padded_trips;
               ];
             group "computed"
               [
                 computed_law;
                 tag race
                 @@ test "a product of a computed operand is exact"
                      of_computed_operand;
                 test
                   "a computed operand whose loads step through another loop \
                    than the plain operand's stays unstaged, the plain one \
                    staged under the sum"
                   core_under_a_sum;
               ];
             group "decoded"
               [
                 dense_law;
                 tag race
                 @@ test
                      "a dense MXFP4 product, its scales held whole, is exact"
                      of_held_scales;
                 test "a decoded operand stays unstaged where copies wait"
                   decoded_without_async;
               ];
           ];
         group "stages_product"
           [
             tag race
             @@ cases
                  "products in registers in trips of several steps compute \
                   their unstaged twins' bits"
                  ~id:(Format.asprintf "%a" pp_product)
                  products_in_registers in_registers;
             tag race
             @@ test
                  "a product summed over heads of 110 copies runs of 8 bytes \
                   and computes its unstaged twin's bits"
                  in_copies_of_8;
             test "a load two products read is copied once" shared_input;
           ];
         group "product_tile" [ tile_law ];
         group "steps" [ decoded_trips ];
         group "tile"
           [
             staged_tiles;
             slotted_tiles;
             routed_tiles;
             heads_law;
             test
               "a staged core's plans at two counts of units compute the same \
                bits"
               spread_bits;
           ];
         group "layout" layout_laws;
         group "decoded" decoded_kinds;
         group "decodes"
           [
             decoded_law;
             decoded_once;
             decoded_stage_off;
             decoded_loads;
             decoded_values;
           ];
         sources;
       ])
