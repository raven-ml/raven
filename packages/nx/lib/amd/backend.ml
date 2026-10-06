(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's kernels on AMD GPUs, from the code objects the library carries.

   An elementwise operation names its module's key, its operands and its result;
   the operands' views are coalesced; the module's contiguous form [c] runs when
   every view is C-contiguous after merging, and its strided form [s] otherwise;
   one launch on the device's compute queue. Reductions, scans, sorts, scatters,
   windows and matrix products take paths of their own. A module's kernels are
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

(* Modules *)

(* A module: its family and the names that select its instance, as module keys
   spell them, such as [binary], [add] and [float32]; "" where it has none.
   Kernels are looked up by module and name, and the key's string is built only
   when a module is first looked up on a device. *)
type modname = { family : string; kind : string; dtype : string }

let modname ?(kind = "") family dtype = { family; kind; dtype }

let key_of m =
  String.concat "."
    (List.filter (fun n -> n <> "") [ m.family; m.kind; m.dtype ])

(* The name of an element width, in bytes. *)
let width_name = function
  | 1 -> "1"
  | 2 -> "2"
  | 4 -> "4"
  | 8 -> "8"
  | w -> string_of_int w

(* Devices *)

(* What a device needs to run kernels: its carried target, or why it has none,
   its properties, whether its target carries each module looked up, and the
   kernels loaded on it, by module and name. *)
type device = {
  target : (string, string) result;
  props : Nx_amd_device.props;
  modules : (modname, bool) Hashtbl.t;
  programs : (modname * string, Program.t) Hashtbl.t;
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
          modules = Hashtbl.create 16;
          programs = Hashtbl.create 16;
        }
      in
      devices := (d, s) :: !devices;
      s

(* [find t k] under the lock, which [Hashtbl.find_opt] never raises from. *)
let find t k =
  Mutex.lock lock;
  let v = Hashtbl.find_opt t k in
  Mutex.unlock lock;
  v

let remember t k v = Mutex.protect lock (fun () -> Hashtbl.replace t k v)

(* The kernel [name] of module [m] on [d]. Two domains that load one kernel at
   once both load it, and find the same load. *)
let program d s target m name =
  match find s.programs (m, name) with
  | Some p -> p
  | None ->
      let binary = Option.get (Archive.find (target ^ "/" ^ key_of m)) in
      let p =
        match Program.load d ~binary ~name with
        | Ok p -> p
        | Error why -> failwith why
      in
      remember s.programs (m, name) p;
      p

(* The name of a dtype the kernels serve, as module keys spell it. *)
let served (type a b) (dt : (a, b) Nx_dtype.t) =
  match dt with
  | Complex64 | Complex128 -> refuse "no complex dtypes"
  | Int4 -> refuse "no int4"
  | UInt4 -> refuse "no uint4"
  | Bit -> refuse "no bit"
  | _ -> Nx_dtype.to_string dt

(* Running a kernel *)

type operand = Operand : ('a, 'b) Nx_array.t -> operand

let address (Operand a) = Nativeint.to_int (Nx_device.Buffer.address a.buffer)
let itemsize (Operand a) = Nx_dtype.itemsize a.dtype
let view (Operand a) = a.view
let buffer (Operand a) = a.buffer
let served_name (Operand a) = served a.dtype
let at o v = address o + (View.offset v * itemsize o)
let cdiv a b = (a + b - 1) / b
let unit_stride v = View.ndim v = 0 || View.strides v = [| 1 |]

(* [dst]'s device, its state and carried target, whose archive holds module
   [m]. *)
let locate m (Operand d) =
  let dev = Nx_device.Buffer.device d.buffer in
  let s = device dev in
  let target = match s.target with Ok t -> t | Error e -> refuse "%s" e in
  let carried =
    match find s.modules m with
    | Some c -> c
    | None ->
        let c = Archive.mem (target ^ "/" ^ key_of m) in
        remember s.modules m c;
        c
  in
  if not carried then refuse "no kernel %s" (key_of m);
  (dev, s, target)

let units s = s.props.compute_units * s.props.xccs

(* Kernel parameters, as 64-bit words, then [raw] whole, written in a domain's
   scratch and copied out at their length. *)
let scratch = Domain.DLS.new_key (fun () -> Bytes.create 4096)

let args ?raw f =
  let b = Domain.DLS.get scratch in
  let n = ref 0 in
  f (fun x ->
      Bytes.set_int64_le b !n (Int64.of_int x);
      n := !n + 8);
  Option.iter
    (fun w ->
      Bytes.set_int64_le b !n w;
      n := !n + 8)
    raw;
  Bytes.sub_string b 0 !n

(* [a] padded with zeros to [max_rank] words. *)
let words i64 a =
  for i = 0 to max_rank - 1 do
    i64 (if i < Array.length a then a.(i) else 0)
  done

let dispatch program groups args : Nx_amd_device.dispatch =
  { program; groups = (groups, 1, 1); threads = (threads, 1, 1); args }

(* The words of a strided form's [meta]: [n], the extents of [views]' shape,
   which they share, [groups], and each view's offset and strides. *)
let meta i64 ~n ~groups views =
  let vs = Array.of_list views in
  let shape = View.shape vs.(0) in
  i64 n;
  i64 (Array.length shape);
  i64 groups;
  words i64 shape;
  for k = 0 to max_operands - 1 do
    i64 (if k < Array.length vs then View.offset vs.(k) else 0)
  done;
  for k = 0 to max_operands - 1 do
    words i64 (if k < Array.length vs then View.strides vs.(k) else [||])
  done

let groups_of s n = Int.min (cdiv n threads) (waves * units s)

(* The run of [key]'s elementwise module writing [dst] from [srcs], its
   parameters followed by [extra], if [dst] has elements. *)
let elementwise ?(extra = []) key ~dst srcs =
  let (Operand d) = dst in
  let dev, s, target = locate key dst in
  let ops = dst :: srcs in
  let views = View.coalesce (List.map view ops) in
  let n = View.numel d.view and rank = View.ndim (List.hd views) in
  if rank > max_rank then
    refuse "operands of %d axes once merged; kernels take %d" rank max_rank;
  if List.length srcs > max_operands then
    refuse "%d operands; kernels take %d" (List.length srcs) max_operands;
  if n = 0 then None
  else
    let groups = groups_of s n in
    Option.some
    @@
    if List.for_all unit_stride views then
      dispatch (program dev s target key "c") groups
      @@ args (fun i64 ->
          List.iter2 (fun o v -> i64 (at o v)) ops views;
          i64 n;
          i64 groups;
          List.iter i64 extra)
    else
      dispatch (program dev s target key "s") groups
      @@ args (fun i64 ->
          i64 (at dst (List.hd views));
          List.iter (fun o -> i64 (address o)) srcs;
          meta i64 ~n ~groups (List.tl views);
          List.iter i64 extra)

(* Runs [key]'s module writing [dst] from [srcs]. *)
let run ?extra key ~dst srcs =
  Option.iter
    (fun d ->
      Nx_amd_device.launch ~touches:(List.map buffer (dst :: srcs)) [ d ])
    (elementwise ?extra key ~dst srcs)

(* Moves

   gather, pad and the window writes of cat and update move bytes: their modules
   are keyed by element width. *)

let width (Operand a) = Nx_dtype.itemsize a.dtype

(* The bits of [v] as an element of [dt], zero-extended to 64. *)
let bits (type a b) (dt : (a, b) Nx_dtype.t) (v : a) =
  let e = Nx_array.Elements.create dt 1 in
  Nx_array.Elements.fill dt e v;
  let bytes = Nx_device.Buffer.bigarray Bigarray.char e in
  let w = ref 0L in
  for i = Bigarray.Array1.dim bytes - 1 downto 0 do
    w :=
      Int64.logor (Int64.shift_left !w 8) (Int64.of_int (Char.code bytes.{i}))
  done;
  !w

(* The C-contiguous strides of [shape]. *)
let c_strides shape =
  let n = Array.length shape in
  let st = Array.make n 1 in
  for i = n - 2 downto 0 do
    st.(i) <- st.(i + 1) * shape.(i + 1)
  done;
  st

(* The run of [key]'s place module writing [x] into the window of [dst] at
   [offset] of [x]'s shape, moved by [corner]: the address of a vector of
   positions, its rank, offset and stride, and the destination strides they move
   along. *)
let place key ~dst x ~offset ~corner:(starts, rank, off, str, dstr) =
  let (Operand d) = dst in
  let dev, s, target = locate key dst in
  let window =
    View.create ~offset
      ~strides:(c_strides (View.shape d.view))
      (View.shape (view x))
  in
  let views = View.coalesce [ window; view x ] in
  let w = List.hd views and xv = List.nth views 1 in
  let n = View.numel xv in
  if View.ndim w > max_rank then
    refuse "operands of %d axes once merged; kernels take %d" (View.ndim w)
      max_rank;
  if n = 0 then None
  else
    let groups = groups_of s n in
    let corner i64 =
      List.iter i64 [ rank; off; str ];
      words i64 dstr
    in
    Option.some
    @@
    if List.for_all unit_stride views then
      dispatch (program dev s target key "c") groups
      @@ args (fun i64 ->
          i64 (at dst w);
          i64 (at x xv);
          i64 starts;
          i64 n;
          i64 groups;
          corner i64)
    else
      dispatch (program dev s target key "s") groups
      @@ args (fun i64 ->
          i64 (address dst);
          i64 (address x);
          i64 starts;
          meta i64 ~n ~groups views;
          corner i64)

let gather ~axis (indices : Nx_backend.index_array) x ~dst =
  let (Operand xa) = x in
  let v = xa.view in
  let strides =
    Array.mapi (fun d s -> if d = axis then 0 else s) (View.strides v)
  in
  (* [x] over the indices' shape, its axis left to the index. *)
  let along =
    {
      xa with
      view =
        View.create ~offset:(View.offset v) ~strides (View.shape indices.view);
    }
  in
  run
    (modname "gather" (width_name (width x)))
    ~dst
    [ Operand along; Operand indices ]
    ~extra:[ (View.strides v).(axis); (View.shape v).(axis) ]

let pad padding fill x ~dst =
  let (Operand d) = dst in
  let key = modname "pad" (width_name (width dst)) in
  let dev, s, target = locate key dst in
  let n = View.numel d.view and shape = View.shape d.view in
  let rank = Array.length shape in
  if rank > max_rank then refuse "%d axes; kernels take %d" rank max_rank;
  if n > 0 then begin
    let groups = groups_of s n in
    let lo = Array.map fst padding in
    let hi = Array.mapi (fun i l -> l + (View.shape (view x)).(i)) lo in
    let args =
      args ~raw:fill (fun i64 ->
          i64 (at dst d.view);
          i64 (address x);
          List.iter i64 [ n; rank; groups; View.offset (view x) ];
          words i64 shape;
          words i64 lo;
          words i64 hi;
          words i64 (View.strides (view x)))
    in
    Nx_amd_device.launch
      ~touches:[ buffer dst; buffer x ]
      [ dispatch (program dev s target key "s") groups args ]
  end

let no_corner = (0, 0, 0, 0, [||])

let cat ~axis xs ~dst =
  let (Operand d) = dst in
  let key = modname "place" (width_name (width dst)) in
  let step = (c_strides (View.shape d.view)).(axis) in
  let _, runs =
    List.fold_left
      (fun (pos, runs) x ->
        let offset = View.offset d.view + (pos * step) in
        let r = place key ~dst x ~offset ~corner:no_corner in
        (pos + (View.shape (view x)).(axis), Option.to_list r @ runs))
      (0, []) xs
  in
  if runs <> [] then
    Nx_amd_device.launch
      ~touches:(buffer dst :: List.map buffer xs)
      (List.rev runs)

let update x ~(starts : Nx_backend.index_array) v ~dst =
  let (Operand d) = dst in
  let shape = View.shape d.view in
  let copy =
    elementwise (modname "contiguous" (width_name (width dst))) ~dst [ x ]
  in
  let corner =
    ( at (Operand starts) starts.view,
      Array.length shape,
      0,
      View.stride 0 starts.view,
      c_strides shape )
  in
  let write =
    place
      (modname "place" (width_name (width dst)))
      ~dst v ~offset:(View.offset d.view) ~corner
  in
  match Option.to_list copy @ Option.to_list write with
  | [] -> ()
  | runs ->
      Nx_amd_device.launch
        ~touches:[ buffer dst; buffer x; buffer v; starts.buffer ]
        runs

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

(* The parts of a row of [len] elements among [rows] rows: one, or for fewer
   rows than two per compute unit, of more than [split] elements, enough parts
   of at least [split] elements to fill the device. *)
let parts_of s ~rows ~len =
  let units = units s in
  if rows < 2 * units && len > split then
    Int.min (cdiv (waves * units) rows) (len / split)
  else 1

(* Scratch for the partials of [items] runs, and the addresses of their
   accumulators and positions. *)
let partials dev ~parts ~items =
  if parts = 1 then (None, 0, 0)
  else
    let b = Nx_device.Buffer.create dev Int64 (2 * items) in
    let a = Nativeint.to_int (Nx_device.Buffer.address b) in
    (Some b, a, a + (8 * items))

(* The first pass of the reduction module [key] over [x]'s rows [xo] and reduced
   elements [xr]: each row's fold into [dst], or with [parts] above 1 each run's
   of [chunk] elements into the partials [pv] and [pi]. *)
let fold_pass dev s target key ~dst x ~xo ~xr ~rows ~len ~parts ~chunk ~pv ~pi =
  let units = units s in
  let items = rows * parts in
  let span = Int.min threads (pow2 chunk) in
  let lanes =
    let r = View.ndim xr in
    if r = 0 || (View.strides xr).(r - 1) = 1 then span
    else Int.min span (pow2 (cdiv (waves * units * threads) items))
  in
  let g = Int.min (cdiv items (threads / lanes)) (waves * units) in
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
    let parts = parts_of s ~rows ~len in
    let chunk = cdiv len parts in
    let scratch, pv, pi = partials dev ~parts ~items:(rows * parts) in
    let first =
      fold_pass dev s target key ~dst x ~xo ~xr ~rows ~len ~parts ~chunk ~pv ~pi
    in
    let second () =
      let lanes = Int.min threads (pow2 parts) in
      let g = Int.min (cdiv rows (threads / lanes)) (waves * units s) in
      dispatch (program dev s target key "f") g
      @@ args (fun i64 ->
          i64 (at dst (view dst));
          List.iter i64 [ pv; pi; rows; parts; lanes; g ])
    in
    let touches = buffer dst :: buffer x :: Option.to_list scratch in
    Nx_amd_device.launch ~touches
      (if parts = 1 then [ first ] else [ first; second () ])
  end

(* Scans

   Each row, the operand's elements along the axis, runs into the result's. The
   rows' axes of both coalesce together. A thread runs along each row where rows
   are short or many enough to fill the device; otherwise a workgroup runs along
   each row, or for few long rows along each of its parts, which start from the
   fold of the parts before them: reduce's first pass folds them. *)

(* Rows at most this long run on a thread each. *)
let short = 64

(* Runs the scan module [key] writing [dst] from [x] along [axis], with the
   reduction module [fold_key] of its kind and dtype. *)
let scan key ~fold_key ~dst x ~axis =
  let dev, s, target = locate key dst in
  let without v = axes_of v (fun i -> i <> axis) in
  let views = View.coalesce [ without (view dst); without (view x) ] in
  let dov = List.hd views and xov = List.nth views 1 in
  let rows = View.numel dov and len = (View.shape (view x)).(axis) in
  if View.ndim dov > max_rank then
    refuse "rows of %d axes once merged; kernels take %d" (View.ndim dov)
      max_rank;
  if rows > 0 && len > 0 then begin
    let units = units s in
    let thread = len <= short || rows >= waves * units * threads in
    let parts = if thread then 1 else parts_of s ~rows ~len in
    let chunk = cdiv len parts and items = rows * parts in
    let scratch, pv, pi = partials dev ~parts ~items in
    let g =
      if thread then Int.min (cdiv rows threads) (waves * units)
      else Int.min items (waves * units)
    in
    let run =
      dispatch (program dev s target key (if thread then "t" else "w")) g
      @@ args (fun i64 ->
          i64 (at dst dov);
          i64 (address x);
          i64 pv;
          List.iter i64 [ rows; len; parts; chunk; g; View.ndim dov ];
          i64 (View.offset xov);
          i64 (View.stride axis (view x));
          i64 (View.stride axis (view dst));
          words i64 (View.shape dov);
          words i64 (View.strides xov);
          words i64 (View.strides dov))
    in
    let totals () =
      let xo = List.nth (View.coalesce [ View.create (View.shape dov); xov ]) 1
      and xr = View.create ~strides:[| View.stride axis (view x) |] [| len |] in
      fold_pass dev s target fold_key ~dst x ~xo ~xr ~rows ~len ~parts ~chunk
        ~pv ~pi
    in
    Nx_amd_device.launch
      ~touches:(buffer dst :: buffer x :: Option.to_list scratch)
      (if parts = 1 then [ run ] else [ totals (); run ])
  end

(* Sorts

   Each row's elements become keys paired with their positions, in scratch
   padded to a power of two; a bitonic network sorts the padded rows, its steps
   of distance at most half a block in the workgroup's memory and the longer
   ones over all slots; the elements at the sorted positions, or the positions,
   are the result. Every step is a run of one launch. *)

let block = 2048

type sorted = Elements | Positions

let log2 n =
  let rec go k = if 1 lsl k >= n then k else go (k + 1) in
  go 0

(* Runs the sort module [key] writing [x]'s [sorted] rows along [axis] into
   [dst]. *)
let sort_rows key sorted ~descending ~axis x ~dst =
  let dev, s, target = locate key dst in
  let without v = axes_of v (fun i -> i <> axis) in
  let views = View.coalesce [ without (view dst); without (view x) ] in
  let dov = List.hd views and xov = List.nth views 1 in
  let rows = View.numel dov and len = (View.shape (view x)).(axis) in
  if View.ndim dov > max_rank then
    refuse "rows of %d axes once merged; kernels take %d" (View.ndim dov)
      max_rank;
  if rows > 0 && len > 0 then begin
    let units = units s in
    let plog = log2 len in
    let p = 1 lsl plog in
    let total = rows * p in
    let scratch = Nx_device.Buffer.create dev Int64 (2 * total) in
    let keys = Nativeint.to_int (Nx_device.Buffer.address scratch) in
    let pos = keys + (8 * total) in
    let flip =
      if not descending then 0
      else if itemsize x = 8 then -1
      else (1 lsl (8 * itemsize x)) - 1
    in
    let spread n = Int.min (cdiv n threads) (waves * units) in
    let meta i64 groups =
      List.iter i64 [ rows; len; plog; groups; View.ndim dov ];
      List.iter i64 [ View.offset xov; View.stride axis (view x) ];
      List.iter i64 [ View.stride axis (view dst); flip ];
      words i64 (View.shape dov);
      words i64 (View.strides xov);
      words i64 (View.strides dov)
    in
    let run name groups f =
      dispatch (program dev s target key name) groups (args f)
    in
    let pairs =
      let g = spread total in
      run "k" g (fun i64 ->
          i64 (address x);
          i64 keys;
          i64 pos;
          meta i64 g)
    in
    let local klo khi =
      let g = Int.min (cdiv total block) (waves * units) in
      run "l" g (fun i64 ->
          List.iter i64 [ keys; pos; total; plog; klo; khi; g ])
    in
    let step k j =
      let g = spread (total / 2) in
      run "g" g (fun i64 -> List.iter i64 [ keys; pos; total; plog; k; j; g ])
    in
    (* Stages past a block: their long steps over all slots, then the short ones
       in blocks. *)
    let rec stages k acc =
      if k > p then List.rev acc
      else
        let rec long j acc =
          if j < block then acc else long (j / 2) (step k j :: acc)
        in
        stages (2 * k) (local k k :: long (k / 2) acc)
    in
    let result =
      let g = spread (rows * len) in
      match sorted with
      | Elements ->
          run "v" g (fun i64 ->
              i64 (at dst dov);
              i64 (address x);
              i64 pos;
              meta i64 g)
      | Positions ->
          run "i" g (fun i64 ->
              i64 (at dst dov);
              i64 pos;
              meta i64 g)
    in
    let network =
      if p = 1 then [] else local 2 (Int.min p block) :: stages (2 * block) []
    in
    Nx_amd_device.launch
      ~touches:[ buffer dst; buffer x; scratch ]
      ((pairs :: network) @ [ result ])
  end

(* Scatters

   The destination starts as a copy of the operand; each position's scratch, two
   64-bit words, holds what its updates decide: the last update's index under
   [`Set], a sum and a mark under [`Add], the winning key and the first NaN
   update's index under [`Max] and [`Min]. Every pass is a run of one launch,
   which orders their atomics. *)

let scatter_rows ~mode ~unique ~axis ~(indices : Nx_backend.index_array)
    ~updates x ~dst =
  let (Operand d) = dst in
  let w = width_name (width dst) in
  let m =
    match (mode : Nx_backend.scatter) with
    | `Set -> modname "scatter_set" w
    | `Add -> modname "scatter_add" (served_name dst)
    | `Max -> modname "scatter_max" (served_name dst)
    | `Min -> modname "scatter_min" (served_name dst)
  in
  let dev, s, target = locate m dst in
  let shape = View.shape indices.view in
  let rank = Array.length shape in
  if rank > max_rank then refuse "%d axes; kernels take %d" rank max_rank;
  let n = View.numel indices.view and positions = View.numel d.view in
  let copy = elementwise (modname "contiguous" w) ~dst [ x ] in
  let runs =
    if n = 0 || positions = 0 then []
    else
      let units = units s in
      let spread k = Int.min (cdiv k threads) (waves * units) in
      let gu = spread n and gp = spread positions in
      let run name groups f =
        dispatch (program dev s target m name) groups (args f)
      in
      let meta i64 =
        List.iter i64 [ n; rank; gu; axis; (View.shape d.view).(axis) ];
        i64 (View.offset indices.view);
        i64 (View.offset (view updates));
        words i64 shape;
        words i64 (View.strides indices.view);
        words i64 (View.strides (view updates));
        words i64 (c_strides (View.shape d.view))
      in
      let dst_at = at dst d.view
      and idx = address (Operand indices)
      and up = address updates in
      let scratch () =
        let b = Nx_device.Buffer.create dev Int64 (2 * positions) in
        let a = Nativeint.to_int (Nx_device.Buffer.address b) in
        (b, a, a + (8 * positions))
      in
      match mode with
      | `Set when unique ->
          [
            ( None,
              run "u" gu (fun i64 ->
                  List.iter i64 [ dst_at; idx; up ];
                  meta i64) );
          ]
      | `Set ->
          let b, pa, _ = scratch () in
          [
            (Some b, run "c" gp (fun i64 -> List.iter i64 [ pa; positions; gp ]));
            ( None,
              run "w" gu (fun i64 ->
                  List.iter i64 [ pa; idx ];
                  meta i64) );
            ( None,
              run "v" gu (fun i64 ->
                  List.iter i64 [ dst_at; pa; idx; up ];
                  meta i64) );
          ]
      | `Add ->
          let b, pa, pb = scratch () in
          [
            ( Some b,
              run "i" gp (fun i64 ->
                  List.iter i64 [ dst_at; pa; pb; positions; gp ]) );
            ( None,
              run "a" gu (fun i64 ->
                  List.iter i64 [ pa; pb; idx; up ];
                  meta i64) );
            ( None,
              run "s" gp (fun i64 ->
                  List.iter i64 [ dst_at; pa; pb; positions; gp ]) );
          ]
      | `Max | `Min ->
          let b, pa, pb = scratch () in
          [
            ( Some b,
              run "i" gp (fun i64 ->
                  List.iter i64 [ dst_at; pa; pb; positions; gp ]) );
            ( None,
              run "k" gu (fun i64 ->
                  List.iter i64 [ pa; idx; up ];
                  meta i64) );
            ( None,
              run "n" gu (fun i64 ->
                  List.iter i64 [ dst_at; pa; pb; idx; up ];
                  meta i64) );
            ( None,
              run "s" gp (fun i64 ->
                  List.iter i64 [ dst_at; pa; pb; idx; up ];
                  meta i64;
                  List.iter i64 [ positions; gp ]) );
          ]
  in
  match Option.to_list copy @ List.map snd runs with
  | [] -> ()
  | ds ->
      Nx_amd_device.launch
        ~touches:
          (buffer dst :: buffer x :: buffer updates :: indices.buffer
         :: List.filter_map fst runs)
        ds

(* Windows

   Each result element is computed from its index: an unfold element reads the
   one element its tap covers, or 0 in the padding; a fold element sums the taps
   that cover it, in order. The operand's leading axes coalesce. *)

(* The windows along each axis of [extent], as an unfold counts them. *)
let windows ~kernel_size ~stride ~dilation ~padding extent =
  Array.mapi
    (fun d e ->
      let eff = (dilation.(d) * (kernel_size.(d) - 1)) + 1 in
      let padded = e + fst padding.(d) + snd padding.(d) in
      if padded < eff then 0 else ((padded - eff) / stride.(d)) + 1)
    extent

(* Runs [m]'s kernel [name] writing [dst] from [x], whose first [lead] axes are
   its leading ones, over [extent] with [xstr] the operand's spatial strides,
   and [kstep] and [lstep] the strides of a fold operand's taps and windows. *)
let window_run m name ~dst x ~lead ~kernel_size ~stride ~dilation ~padding
    ~extent ~xstr ~kstep ~lstep =
  let dev, s, target = locate m dst in
  let k = Array.length kernel_size in
  let lv = List.hd (View.coalesce [ axes_of (view x) (fun i -> i < lead) ]) in
  if View.ndim lv > max_rank || k > max_rank then
    refuse "windows of %d axes and %d leading axes once merged; kernels take %d"
      k (View.ndim lv) max_rank;
  let n = View.numel (view dst) in
  if n > 0 then begin
    let win = windows ~kernel_size ~stride ~dilation ~padding extent in
    let kprod = Array.fold_left ( * ) 1 kernel_size
    and l = Array.fold_left ( * ) 1 win in
    let groups = groups_of s n in
    Nx_amd_device.launch
      ~touches:[ buffer dst; buffer x ]
      [
        dispatch (program dev s target m name) groups
        @@ args (fun i64 ->
            i64 (at dst (view dst));
            i64 (address x);
            List.iter i64 [ n; groups; View.ndim lv; k; kprod; l ];
            List.iter i64 [ View.offset lv; kstep; lstep ];
            words i64 (View.shape lv);
            words i64 (View.strides lv);
            words i64 kernel_size;
            words i64 stride;
            words i64 dilation;
            words i64 (Array.map fst padding);
            words i64 extent;
            words i64 win;
            words i64 xstr);
      ]
  end

let unfold_windows ~kernel_size ~stride ~dilation ~padding x ~dst =
  let shape = View.shape (view x) and strides = View.strides (view x) in
  let r = Array.length shape and k = Array.length kernel_size in
  window_run
    (modname "unfold" (width_name (width x)))
    "u" ~dst x ~lead:(r - k) ~kernel_size ~stride ~dilation ~padding
    ~extent:(Array.sub shape (r - k) k)
    ~xstr:(Array.sub strides (r - k) k)
    ~kstep:0 ~lstep:0

let fold_windows ~output_size ~kernel_size ~stride ~dilation ~padding x ~dst =
  let strides = View.strides (view x) in
  let r = Array.length strides in
  window_run
    (modname "fold" (served_name x))
    "f" ~dst x ~lead:(r - 2) ~kernel_size ~stride ~dilation ~padding
    ~extent:output_size ~xstr:[||]
    ~kstep:strides.(r - 2)
    ~lstep:strides.(r - 1)

(* Threefry *)

(* Runs the threefry module hashing [counter]'s word pairs under [key]'s into
   [dst], pairs along their last axis. *)
let threefry key counter ~dst =
  let key_ = modname "threefry" "" in
  let dev, s, target = locate key_ dst in
  let shape = View.shape (view dst) in
  let last = Array.length shape - 1 in
  let pairs o = axes_of (view o) (fun i -> i < last) in
  let views =
    View.coalesce
      [ View.create (Array.sub shape 0 last); pairs key; pairs counter ]
  in
  let n = View.numel (List.hd views) in
  if View.ndim (List.hd views) > max_rank then
    refuse "operands of %d axes once merged; kernels take %d"
      (View.ndim (List.hd views))
      max_rank;
  if n > 0 then begin
    let groups = groups_of s n in
    Nx_amd_device.launch
      ~touches:[ buffer dst; buffer key; buffer counter ]
      [
        dispatch (program dev s target key_ "s") groups
        @@ args (fun i64 ->
            i64 (at dst (view dst));
            i64 (address key);
            i64 (address counter);
            meta i64 ~n ~groups (List.tl views);
            i64 (View.stride last (view key));
            i64 (View.stride last (view counter)));
      ]
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
      (modname "contiguous" (width_name (Nx_dtype.itemsize x.dtype)))
      ~dst:(Operand dst) [ Operand x ]

  let cast (type a b c d) (x : (a, b) Nx_array.t) ~(dst : (c, d) Nx_array.t) =
    let s = served x.dtype and d = served dst.dtype in
    if s = d then
      run
        (modname "contiguous" (width_name (Nx_dtype.itemsize x.dtype)))
        ~dst:(Operand dst) [ Operand x ]
    else run (modname "cast" ~kind:s d) ~dst:(Operand dst) [ Operand x ]

  let unary (type a b) k (x : (a, b) Nx_array.t) ~(dst : (a, b) Nx_array.t) =
    let dt = served x.dtype in
    match (k : Nx_backend.unary) with
    | (Trunc | Ceil | Floor | Round) when not (Nx_dtype.is_float x.dtype) ->
        contiguous x ~dst
    | _ ->
        run
          (modname "unary" ~kind:(unary_name k) dt)
          ~dst:(Operand dst) [ Operand x ]

  let binary (type a b) k (a : (a, b) Nx_array.t) b ~(dst : (a, b) Nx_array.t) =
    run
      (modname "binary" ~kind:(binary_name k) (served a.dtype))
      ~dst:(Operand dst) [ Operand a; Operand b ]

  let compare (type a b) k (a : (a, b) Nx_array.t) b ~dst =
    run
      (modname "compare" ~kind:(compare_name k) (served a.dtype))
      ~dst:(Operand dst) [ Operand a; Operand b ]

  let fma (type a b) (a : (a, b) Nx_array.t) b c ~(dst : (a, b) Nx_array.t) =
    run
      (modname "fma" (served a.dtype))
      ~dst:(Operand dst)
      [ Operand a; Operand b; Operand c ]

  let where (type a b) cond (a : (a, b) Nx_array.t) b ~(dst : (a, b) Nx_array.t)
      =
    ignore (served a.dtype);
    run
      (modname "where" (width_name (Nx_dtype.itemsize a.dtype)))
      ~dst:(Operand dst)
      [ Operand cond; Operand a; Operand b ]

  let threefry key counter ~dst =
    threefry (Operand key) (Operand counter) ~dst:(Operand dst)

  let reduce (type a b) k ~axes (x : (a, b) Nx_array.t)
      ~(dst : (a, b) Nx_array.t) =
    fold
      (modname "reduce" ~kind:(reduce_name k) (served x.dtype))
      ~dst:(Operand dst) (Operand x) ~axes

  let scan (type a b) k ~axis (x : (a, b) Nx_array.t) ~(dst : (a, b) Nx_array.t)
      =
    let kind = reduce_name k and dt = served x.dtype in
    scan (modname "scan" ~kind dt)
      ~fold_key:(modname "reduce" ~kind dt)
      ~dst:(Operand dst) (Operand x) ~axis

  let arg_reduce (type a b) k ~axis (x : (a, b) Nx_array.t) ~dst =
    fold
      (modname "arg_reduce" ~kind:(arg_reduce_name k) (served x.dtype))
      ~dst:(Operand dst) (Operand x) ~axes:[| axis |]

  let sort (type a b) ~descending ~axis (x : (a, b) Nx_array.t)
      ~(dst : (a, b) Nx_array.t) =
    sort_rows
      (modname "sort" (served x.dtype))
      Elements ~descending ~axis (Operand x) ~dst:(Operand dst)

  let argsort (type a b) ~descending ~axis (x : (a, b) Nx_array.t) ~dst =
    sort_rows
      (modname "sort" (served x.dtype))
      Positions ~descending ~axis (Operand x) ~dst:(Operand dst)

  let group _ ~dst:_ = no "group"

  let pad (type a b) padding (v : a) (x : (a, b) Nx_array.t)
      ~(dst : (a, b) Nx_array.t) =
    ignore (served x.dtype);
    pad padding (bits x.dtype v) (Operand x) ~dst:(Operand dst)

  let cat (type a b) ~axis (xs : (a, b) Nx_array.t list)
      ~(dst : (a, b) Nx_array.t) =
    ignore (served dst.dtype);
    cat ~axis (List.map (fun x -> Operand x) xs) ~dst:(Operand dst)

  let gather (type a b) ~axis indices (x : (a, b) Nx_array.t)
      ~(dst : (a, b) Nx_array.t) =
    ignore (served x.dtype);
    gather ~axis indices (Operand x) ~dst:(Operand dst)

  let scatter (type a b) ~mode ~unique ~axis ~indices
      ~(updates : (a, b) Nx_array.t) (x : (a, b) Nx_array.t)
      ~(dst : (a, b) Nx_array.t) =
    ignore (served x.dtype);
    scatter_rows ~mode ~unique ~axis ~indices ~updates:(Operand updates)
      (Operand x) ~dst:(Operand dst)

  let update (type a b) (x : (a, b) Nx_array.t) ~starts v
      ~(dst : (a, b) Nx_array.t) =
    ignore (served x.dtype);
    update (Operand x) ~starts (Operand v) ~dst:(Operand dst)

  let unfold (type a b) ~kernel_size ~stride ~dilation ~padding
      (x : (a, b) Nx_array.t) ~(dst : (a, b) Nx_array.t) =
    ignore (served x.dtype);
    unfold_windows ~kernel_size ~stride ~dilation ~padding (Operand x)
      ~dst:(Operand dst)

  let fold (type a b) ~output_size ~kernel_size ~stride ~dilation ~padding
      (x : (a, b) Nx_array.t) ~(dst : (a, b) Nx_array.t) =
    fold_windows ~output_size ~kernel_size ~stride ~dilation ~padding
      (Operand x) ~dst:(Operand dst)

  let matmul (type a b) (a : (a, b) Nx_array.t) b ~(dst : (a, b) Nx_array.t) =
    product
      (modname "matmul" (served a.dtype))
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
