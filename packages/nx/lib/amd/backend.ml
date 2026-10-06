(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's kernels on AMD GPUs, from the code objects the library carries.

   An elementwise operation names its module's key, its operands and its result;
   the operands' views are coalesced; the module's contiguous form [c] runs when
   every view is C-contiguous after merging, and its strided form [s] otherwise;
   one launch on the device's compute queue. Reductions and matrix products take
   paths of their own (Reductions, Matrix products). A module's kernels are
   loaded on a device at their first use, and kept while the device is. A GPU
   that no carried target covers refuses every kernel, as do the operations,
   dtypes and layouts the kernels do not serve. *)

module View = Nx_array.View
module Program = Nx_device.Program
module Code_object = Nx_amd_code_object

let name = "nx.amd"
let refuse fmt = Printf.ksprintf (fun r -> raise (Nx_backend.Refused r)) fmt

(* The kernel ABI (kernels/src/common.h): a workgroup's threads, and the most
   axes and source operands of a strided form. *)
let threads = 256
let max_rank = 32
let max_operands = 4

(* Workgroups: enough for each thread to take an element, at most [waves] per
   compute unit, past which each thread walks several. *)
let waves = 8

(* Shipped targets *)

(* The carried target whose code objects GPU [gpu] runs, by their generic
   processor's members. A GPU of several dies takes AQL packets, which launches
   do not encode. *)
let target ~gpu ~aql =
  let targets =
    List.map
      (fun (t, co) ->
        match Code_object.of_string co with
        | Ok co -> (t, co)
        | Error e -> failwith (Printf.sprintf "nx.amd: %s: %s" t e))
      (Lazy.force Archive.targets)
  in
  if aql then Error (gpu ^ " takes AQL packets, on its several dies")
  else
    match List.find_opt (fun (_, co) -> Code_object.runs_on co gpu) targets with
    | Some (t, _) -> Ok t
    | None ->
        Error
          (Printf.sprintf "nx.amd ships kernels for %s; this GPU is %s"
             (String.concat ", " (List.map fst targets))
             gpu)

(* Devices *)

(* What a device needs to run kernels: its carried target, or why it has none,
   its properties, and the kernels loaded on it, by module and name. *)
type device = {
  target : (string, string) result;
  props : Nx_amd_device.props;
  programs : (string, Program.t) Hashtbl.t;
}

let devices : (Nx_device.t * device) list ref = ref []
let lock = Mutex.create ()

let device d =
  Mutex.protect lock @@ fun () ->
  match List.find_opt (fun (d', _) -> Nx_device.equal d d') !devices with
  | Some (_, s) -> s
  | None ->
      let a = Option.get (Nx_amd_device.of_device d) in
      let s =
        {
          target = target ~gpu:(Nx_device.arch d) ~aql:(Nx_amd_device.aql a);
          props = Nx_amd_device.props a;
          programs = Hashtbl.create 16;
        }
      in
      devices := (d, s) :: !devices;
      s

(* The kernel [name] of [key]'s module on [d]. Two domains that load one kernel
   at once both load it, and find the same load. *)
let program d s target key name =
  let path = target ^ "/" ^ key in
  let id = path ^ "/" ^ name in
  match Mutex.protect lock (fun () -> Hashtbl.find_opt s.programs id) with
  | Some p -> p
  | None ->
      let binary = Option.get (Archive.find path) in
      let p =
        match Program.load d ~binary ~name with
        | Ok p -> p
        | Error why -> failwith why
      in
      Mutex.protect lock (fun () -> Hashtbl.replace s.programs id p);
      p

(* Running a kernel *)

type operand = Operand : ('a, 'b) Nx_array.t -> operand

let address (Operand a) = Nativeint.to_int (Nx_device.Buffer.address a.buffer)
let itemsize (Operand a) = Nx_dtype.itemsize a.dtype
let view (Operand a) = a.view
let buffer (Operand a) = a.buffer
let at o v = address o + (View.offset v * itemsize o)
let cdiv a b = (a + b - 1) / b
let unit_stride v = View.ndim v = 0 || View.strides v = [| 1 |]

(* [dst]'s device, its state and carried target, whose archive holds [key]. *)
let locate key (Operand d) =
  let dev = Nx_device.Buffer.device d.buffer in
  let s = device dev in
  let target = match s.target with Ok t -> t | Error e -> refuse "%s" e in
  if not (Archive.mem (target ^ "/" ^ key)) then refuse "no kernel %s" key;
  (dev, s, target)

let units s = s.props.compute_units * s.props.xccs

(* Kernel parameters, as 64-bit words. *)
let args f =
  let b = Buffer.create 2048 in
  f (fun x -> Buffer.add_int64_le b (Int64.of_int x));
  Buffer.contents b

(* [a] padded with zeros to [max_rank] words. *)
let words i64 a =
  for i = 0 to max_rank - 1 do
    i64 (if i < Array.length a then a.(i) else 0)
  done

let dispatch program groups args : Nx_amd_device.dispatch =
  { program; groups = (groups, 1, 1); threads = (threads, 1, 1); args }

(* Runs [key]'s module writing [dst] from [srcs]. *)
let run key ~dst srcs =
  let (Operand d) = dst in
  let dev, s, target = locate key dst in
  let ops = dst :: srcs in
  let views = View.coalesce (List.map view ops) in
  let n = View.numel d.view and rank = View.ndim (List.hd views) in
  if rank > max_rank then
    refuse "operands of %d axes once merged; kernels take %d" rank max_rank;
  if List.length srcs > max_operands then
    refuse "%d operands; kernels take %d" (List.length srcs) max_operands;
  if n > 0 then begin
    let groups = Int.min (cdiv n threads) (waves * units s) in
    let d =
      if List.for_all unit_stride views then
        dispatch (program dev s target key "c") groups
        @@ args (fun i64 ->
            List.iter2 (fun o v -> i64 (at o v)) ops views;
            i64 n;
            i64 groups)
      else
        let dv = List.hd views and svs = Array.of_list (List.tl views) in
        dispatch (program dev s target key "s") groups
        @@ args (fun i64 ->
            i64 (at dst dv);
            List.iter (fun o -> i64 (address o)) srcs;
            i64 n;
            i64 rank;
            i64 groups;
            words i64 (View.shape dv);
            for k = 0 to max_operands - 1 do
              i64 (if k < Array.length svs then View.offset svs.(k) else 0)
            done;
            for k = 0 to max_operands - 1 do
              words i64
                (if k < Array.length svs then View.strides svs.(k) else [||])
            done)
    in
    Nx_amd_device.launch ~touches:(List.map buffer ops) [ d ]
  end

(* Reductions

   Each element of the result folds a row, the operand's elements over the
   reduced axes in C order. The rows' axes and the reduced axes coalesce apart.
   A row is folded by [lanes] threads: up to a workgroup's, enough to cover it,
   where its elements are adjacent, and otherwise as many as fill the device
   with the rows, so that neighbouring threads read neighbouring rows. Fewer
   rows than two per compute unit, of more than [split] elements, are folded in
   parts of at least [split] elements by a first pass, whose partials a second
   pass folds. *)

let split = 4096
let rec pow2 ?(p = 1) n = if p >= n then p else pow2 ~p:(2 * p) n

(* The axes of [v] that [keep] selects, as a view. *)
let axes_of v keep =
  let ix = List.filter keep (List.init (View.ndim v) Fun.id) in
  let pick a = Array.of_list (List.map (fun i -> a.(i)) ix) in
  View.create ~offset:(View.offset v)
    ~strides:(pick (View.strides v))
    (pick (View.shape v))

(* Runs the reduction module [key] writing [dst] from [x] folded over [axes]. *)
let fold key ~dst x ~axes =
  let dev, s, target = locate key dst in
  let reduced i = Array.mem i axes in
  let xo =
    List.nth
      (View.coalesce [ view dst; axes_of (view x) (Fun.negate reduced) ])
      1
  in
  let xr = List.hd (View.coalesce [ axes_of (view x) reduced ]) in
  let rows = View.numel (view dst) and len = View.numel xr in
  let rank = Int.max (View.ndim xo) (View.ndim xr) in
  if rank > max_rank then
    refuse "rows or reductions of %d axes once merged; kernels take %d" rank
      max_rank;
  if rows > 0 then begin
    let units = units s in
    let parts =
      if rows < 2 * units && len > split then
        Int.min (cdiv (waves * units) rows) (len / split)
      else 1
    in
    let chunk = cdiv len parts and items = rows * parts in
    let span = Int.min threads (pow2 chunk) in
    let lanes =
      let r = View.ndim xr in
      if r = 0 || (View.strides xr).(r - 1) = 1 then span
      else Int.min span (pow2 (cdiv (waves * units * threads) items))
    in
    let groups lanes n = Int.min (cdiv n (threads / lanes)) (waves * units) in
    let scratch =
      if parts = 1 then None
      else Some (Nx_device.Buffer.create dev Int64 (2 * items))
    in
    let pv, pi =
      match scratch with
      | None -> (0, 0)
      | Some b ->
          let a = Nativeint.to_int (Nx_device.Buffer.address b) in
          (a, a + (8 * items))
    in
    let g = groups lanes items in
    let first =
      if
        len = 0
        || (unit_stride xr && (View.ndim xo = 0 || View.strides xo = [| len |]))
      then
        dispatch (program dev s target key "c") g
        @@ args (fun i64 ->
            i64 (at dst (view dst));
            i64 (at x xo);
            i64 pv;
            i64 pi;
            List.iter i64 [ rows; len; lanes; parts; chunk; g ])
      else
        dispatch (program dev s target key "s") g
        @@ args (fun i64 ->
            i64 (at dst (view dst));
            i64 (address x);
            i64 pv;
            i64 pi;
            List.iter i64 [ rows; len; lanes; parts; chunk; g ];
            List.iter i64 [ View.ndim xo; View.ndim xr; View.offset xo ];
            words i64 (View.shape xo);
            words i64 (View.strides xo);
            words i64 (View.shape xr);
            words i64 (View.strides xr))
    in
    let second () =
      let lanes = Int.min threads (pow2 parts) in
      let g = groups lanes rows in
      dispatch (program dev s target key "f") g
      @@ args (fun i64 ->
          i64 (at dst (view dst));
          List.iter i64 [ pv; pi; rows; parts; lanes; g ])
    in
    let touches = buffer dst :: buffer x :: Option.to_list scratch in
    Nx_amd_device.launch ~touches
      (if parts = 1 then [ first ] else [ first; second () ])
  end

(* Matrix products

   A workgroup computes a tile of [tile] x [tile] outputs of one matrix of the
   batch: the grid's x covers the columns, its y the rows and its z the batch.
   The batch axes of the result and of both operands coalesce together, an
   operand's stride 0 where it broadcasts. *)

let tile = 64
let max_groups = 0xffff_ffff

(* Operand [v]'s batch axes under the result's batch shape [batch]: its strides,
   aligned to the last axes, 0 where it lacks an axis or holds one matrix along
   it. *)
let batch_view batch v =
  let r = Array.length batch and n = View.ndim v - 2 in
  let shape = View.shape v and strides = View.strides v in
  View.create
    ~strides:
      (Array.init r (fun i ->
           let a = i - (r - n) in
           if a < 0 || shape.(a) = 1 then 0 else strides.(a)))
    batch

(* Runs the product module [key] writing [dst] from [a] and [b]. *)
let product key ~dst a b =
  let dev, s, target = locate key dst in
  let shape = View.shape (view dst) in
  let r = Array.length shape - 2 in
  let m = shape.(r) and n = shape.(r + 1) and batch = Array.sub shape 0 r in
  let av = view a and bv = view b in
  let k = (View.shape av).(View.ndim av - 1) in
  let nb = Array.fold_left ( * ) 1 batch in
  let views =
    View.coalesce
      [ View.create batch; batch_view batch av; batch_view batch bv ]
  in
  let d = List.nth views 0
  and va = List.nth views 1
  and vb = List.nth views 2 in
  if View.ndim d > max_rank then
    refuse "batches of %d axes once merged; kernels take %d" (View.ndim d)
      max_rank;
  if nb > max_groups then refuse "%d matrices; kernels take %d" nb max_groups;
  if m * n * nb > 0 then begin
    (* The strides of [v]'s rows and columns. *)
    let rows v = (View.strides v).(View.ndim v - 2)
    and cols v = (View.strides v).(View.ndim v - 1) in
    let args =
      args (fun i64 ->
          i64 (at dst (view dst));
          i64 (address a);
          i64 (address b);
          List.iter i64 [ m; n; k; View.ndim d ];
          List.iter i64 [ View.offset av; rows av; cols av ];
          List.iter i64 [ View.offset bv; rows bv; cols bv ];
          words i64 (View.shape d);
          words i64 (View.strides va);
          words i64 (View.strides vb))
    in
    Nx_amd_device.launch
      ~touches:[ buffer dst; buffer a; buffer b ]
      [
        {
          program = program dev s target key "s";
          groups = (cdiv n tile, cdiv m tile, nb);
          threads = (threads, 1, 1);
          args;
        };
      ]
  end

(* The name of a dtype the kernels serve, as module keys spell it. *)
let served (type a b) (dt : (a, b) Nx_dtype.t) =
  match dt with
  | Complex64 | Complex128 -> refuse "no complex dtypes"
  | Int4 -> refuse "no int4"
  | UInt4 -> refuse "no uint4"
  | Bit -> refuse "no bit"
  | _ -> Nx_dtype.to_string dt

(* Kernels *)

(* The names of the kinds, as module keys spell them. *)
let unary_name : Nx_backend.unary -> string = function
  | Neg -> "neg"
  | Recip -> "recip"
  | Abs -> "abs"
  | Sqrt -> "sqrt"
  | Sign -> "sign"
  | Exp -> "exp"
  | Log -> "log"
  | Log1p -> "log1p"
  | Expm1 -> "expm1"
  | Sin -> "sin"
  | Cos -> "cos"
  | Tan -> "tan"
  | Asin -> "asin"
  | Acos -> "acos"
  | Atan -> "atan"
  | Sinh -> "sinh"
  | Cosh -> "cosh"
  | Tanh -> "tanh"
  | Trunc -> "trunc"
  | Ceil -> "ceil"
  | Floor -> "floor"
  | Round -> "round"
  | Erf -> "erf"

let binary_name : Nx_backend.binary -> string = function
  | Add -> "add"
  | Sub -> "sub"
  | Mul -> "mul"
  | Fdiv -> "fdiv"
  | Idiv -> "idiv"
  | Mod -> "mod"
  | Pow -> "pow"
  | Atan2 -> "atan2"
  | Maximum -> "maximum"
  | Minimum -> "minimum"
  | And -> "and"
  | Or -> "or"
  | Xor -> "xor"

let compare_name : Nx_backend.compare -> string = function
  | Equal -> "equal"
  | Not_equal -> "not_equal"
  | Less -> "less"
  | Less_equal -> "less_equal"

let reduce_name : Nx_backend.reduce -> string = function
  | Sum -> "sum"
  | Prod -> "prod"
  | Max -> "max"
  | Min -> "min"

let arg_reduce_name : Nx_backend.arg_reduce -> string = function
  | Argmax -> "argmax"
  | Argmin -> "argmin"

module Kernels : Nx_backend.S = struct
  let name = name
  let runs_on d = Option.is_some (Nx_amd_device.of_device d)
  let owns = runs_on
  let no what = refuse "no %s" what

  let contiguous (type a b) (x : (a, b) Nx_array.t) ~(dst : (a, b) Nx_array.t) =
    ignore (served x.dtype);
    run
      (Printf.sprintf "contiguous.%d" (Nx_dtype.itemsize x.dtype))
      ~dst:(Operand dst) [ Operand x ]

  let cast (type a b c d) (x : (a, b) Nx_array.t) ~(dst : (c, d) Nx_array.t) =
    let s = served x.dtype and d = served dst.dtype in
    if s = d then
      run
        (Printf.sprintf "contiguous.%d" (Nx_dtype.itemsize x.dtype))
        ~dst:(Operand dst) [ Operand x ]
    else run (Printf.sprintf "cast.%s.%s" s d) ~dst:(Operand dst) [ Operand x ]

  let unary (type a b) k (x : (a, b) Nx_array.t) ~(dst : (a, b) Nx_array.t) =
    let dt = served x.dtype in
    match (k : Nx_backend.unary) with
    | (Trunc | Ceil | Floor | Round) when not (Nx_dtype.is_float x.dtype) ->
        contiguous x ~dst
    | _ ->
        run
          (Printf.sprintf "unary.%s.%s" (unary_name k) dt)
          ~dst:(Operand dst) [ Operand x ]

  let binary (type a b) k (a : (a, b) Nx_array.t) b ~(dst : (a, b) Nx_array.t) =
    run
      (Printf.sprintf "binary.%s.%s" (binary_name k) (served a.dtype))
      ~dst:(Operand dst) [ Operand a; Operand b ]

  let compare (type a b) k (a : (a, b) Nx_array.t) b ~dst =
    run
      (Printf.sprintf "compare.%s.%s" (compare_name k) (served a.dtype))
      ~dst:(Operand dst) [ Operand a; Operand b ]

  let fma (type a b) (a : (a, b) Nx_array.t) b c ~(dst : (a, b) Nx_array.t) =
    run
      (Printf.sprintf "fma.%s" (served a.dtype))
      ~dst:(Operand dst)
      [ Operand a; Operand b; Operand c ]

  let where (type a b) cond (a : (a, b) Nx_array.t) b ~(dst : (a, b) Nx_array.t)
      =
    ignore (served a.dtype);
    run
      (Printf.sprintf "where.%d" (Nx_dtype.itemsize a.dtype))
      ~dst:(Operand dst)
      [ Operand cond; Operand a; Operand b ]

  let threefry _ _ ~dst:_ = no "threefry"

  let reduce (type a b) k ~axes (x : (a, b) Nx_array.t)
      ~(dst : (a, b) Nx_array.t) =
    fold
      (Printf.sprintf "reduce.%s.%s" (reduce_name k) (served x.dtype))
      ~dst:(Operand dst) (Operand x) ~axes

  let scan _ ~axis:_ _ ~dst:_ = no "scans"

  let arg_reduce (type a b) k ~axis (x : (a, b) Nx_array.t) ~dst =
    fold
      (Printf.sprintf "arg_reduce.%s.%s" (arg_reduce_name k) (served x.dtype))
      ~dst:(Operand dst) (Operand x) ~axes:[| axis |]

  let sort ~descending:_ ~axis:_ _ ~dst:_ = no "sort"
  let argsort ~descending:_ ~axis:_ _ ~dst:_ = no "argsort"
  let group _ ~dst:_ = no "group"
  let pad _ _ _ ~dst:_ = no "pad"
  let cat ~axis:_ _ ~dst:_ = no "cat"
  let gather ~axis:_ _ _ ~dst:_ = no "gather"

  let scatter ~mode:_ ~unique:_ ~axis:_ ~indices:_ ~updates:_ _ ~dst:_ =
    no "scatter"

  let update _ ~starts:_ _ ~dst:_ = no "update"

  let unfold ~kernel_size:_ ~stride:_ ~dilation:_ ~padding:_ _ ~dst:_ =
    no "unfold"

  let fold ~output_size:_ ~kernel_size:_ ~stride:_ ~dilation:_ ~padding:_ _
      ~dst:_ =
    no "fold"

  let matmul (type a b) (a : (a, b) Nx_array.t) b ~(dst : (a, b) Nx_array.t) =
    product
      (Printf.sprintf "matmul.%s" (served a.dtype))
      ~dst:(Operand dst) (Operand a) (Operand b)

  let fft ~inverse:_ ~axes:_ _ ~dst:_ = no "fft"
  let rfft ~axes:_ _ ~dst:_ = no "rfft"
  let irfft ~axes:_ ~s:_ _ ~dst:_ = no "irfft"
  let cholesky ~upper:_ _ ~dst:_ = no "cholesky"
  let qr ~reduced:_ _ ~q:_ ~r:_ = no "qr"
  let lu _ ~lu:_ ~pivots:_ ~perm:_ = no "lu"
  let svd _ ~u:_ ~s:_ ~vt:_ = no "svd"
  let eig _ ~values:_ ~vectors:_ = no "eig"
  let eigh _ ~values:_ ~vectors:_ = no "eigh"

  let solve_triangular ~upper:_ ~transpose:_ ~unit_diag:_ _ _ ~dst:_ =
    no "solve_triangular"
end

let backend = Nx_backend.v (module Kernels)
