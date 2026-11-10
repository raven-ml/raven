(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module S = Nx_cuda_support
module D = Nx_array.Dtype

let strf = Printf.sprintf

(* The first [n] differences [show i] names, of the [len] indices [same]
   rejects: [] when there is none. *)
let differences ?(n = 8) len same show =
  let rec go i acc k =
    if i = len || k = n then List.rev acc
    else if same i then go (i + 1) acc k
    else go (i + 1) (show i :: acc) (k + 1)
  in
  go 0 [] 0

let no_difference d = equal (list string) [] d

(* Floats from bytes *)

let f32 s i = Int32.float_of_bits (String.get_int32_le s (4 * i))
let f64 s i = Int64.float_of_bits (String.get_int64_le s (8 * i))
let bits32 x = Int32.bits_of_float x

let put32 xs =
  let b = Bytes.create (4 * List.length xs) in
  List.iteri (fun i x -> Bytes.set_int32_le b (4 * i) (bits32 x)) xs;
  Bytes.to_string b

let put64 xs =
  let b = Bytes.create (8 * List.length xs) in
  List.iteri
    (fun i x -> Bytes.set_int64_le b (8 * i) (Int64.bits_of_float x))
    xs;
  Bytes.to_string b

(* The kernels' run over buffers *)

let launch g name ~n ps =
  S.run g
    (S.record (S.harness g)
       [ S.launch name ~grid:((n + 255) / 256, 1, 1) ~block:256 ps ])

(* The device's codecs *)

(* Every code of [dt] decodes, and drawn doubles encode, to the bits nx.array's
   host codecs give. The draws: every edge below, then doubles of a random sign
   and an exponent in [-spread, spread]. *)
let codecs (type s) (dt : (float, s) D.t) ~code ~spread =
  let g = S.gpu () in
  let codes = 1 lsl D.bits dt and n = 1 lsl 20 in
  let x = S.buffer g (8 * n) in
  S.generate g x D.Float64 (Wide spread) ~seed:13;
  let ties =
    List.concat_map
      (fun k ->
        List.concat_map
          (fun j ->
            let t = Float.ldexp 1. k *. (1. +. Float.ldexp 1. (-j)) in
            [ t; Float.succ t; Float.pred t ])
          [ 2; 3; 4; 8; 11; 12 ])
      (List.init 301 (fun k -> k - 160))
  in
  let edges =
    [ 0.; infinity; nan; Float.max_float; 57344.; 61436.; 61440.; 61444. ]
    @ ties
    |> List.concat_map (fun x -> [ x; -.x ])
  in
  let xs =
    put64 edges ^ String.sub (S.read x) 0 (8 * (n - List.length edges))
  in
  S.write x xs;
  let dec = S.buffer g (8 * codes) and enc = S.buffer g (D.bits dt / 8 * n) in
  launch g "codecs" ~n [ A x; A dec; A enc; W n; D (D.code dt, 0) ];
  let host_dec =
    let cs = Nx_array.of_array code [| codes |] (Array.init codes Fun.id) in
    Nx_array.to_array (Option.get (Nx_array.bitcast dt cs))
  in
  let dec = S.read dec in
  no_difference
    (differences codes
       (fun c ->
         Int64.bits_of_float (f64 dec c) = Int64.bits_of_float host_dec.(c))
       (fun c -> strf "%#x decodes to %h, not %h" c (f64 dec c) host_dec.(c)));
  let host_enc =
    let v = Nx_array.of_array dt [| n |] (Array.init n (f64 xs)) in
    Nx_array.to_array (Option.get (Nx_array.bitcast code v))
  in
  let enc = S.read enc in
  let get i =
    if codes > 256 then String.get_uint16_le enc (2 * i) else Char.code enc.[i]
  in
  no_difference
    (differences n
       (fun i -> get i = host_enc.(i))
       (fun i ->
         strf "%h encodes to %#x, not %#x" (f64 xs i) (get i) host_enc.(i)))

let codecs_f16 () = codecs D.Float16 ~code:D.Uint16 ~spread:30
let codecs_bf16 () = codecs D.Bfloat16 ~code:D.Uint16 ~spread:140
let codecs_e4m3 () = codecs D.Float8_e4m3fn ~code:D.Uint8 ~spread:20
let codecs_e5m2 () = codecs D.Float8_e5m2 ~code:D.Uint8 ~spread:30

(* Contract *)

(* How a case draws its operands: floats of exponents in [-e, e] (integers over
   their range); those with the second half of each row of a negating its first,
   so that sums cancel; those with NaN, infinities, -0 and subnormals (integer
   extremes) at every seventh element; or float64 elements all of the
   largest finite magnitude, their signs drawn, so that every product
   overflows. *)
type draw = Spread of int | Cancel | Edges | Overflow

(* A contraction y[z, i, j] = init[z, i, j] + Σ_k a[z, i, k] · b[z, j, k] of
   [batch], [m], [n], [k], a and b of [dt] in the layout whose [`K] or [`Free]
   axis is contiguous, [pad] elements between batch elements, accumulated in
   [acc], y of [out]. *)
type case = {
  dt : any;
  acc : any;
  out : any;
  batch : int;
  m : int;
  n : int;
  k : int;
  la : [ `K | `Free ];
  lb : [ `K | `Free ];
  init : [ `None | `Bias | `Full ];
  draw : draw;
  pad : int;
}

and any = D.any

let code (D.Any dt) = D.code dt
let width (D.Any dt) = D.bits dt / 8

let case_name c =
  let name (D.Any dt) = D.name dt in
  let l = function `K -> "k" | `Free -> "f" in
  let draw =
    match c.draw with
    | Spread e -> strf "spread %d" e
    | Cancel -> "cancel"
    | Edges -> "edges"
    | Overflow -> "overflow"
  in
  strf "%s->%s acc %s %dx%dx%dx%d %s%s%s %s pad %d" (name c.dt) (name c.out)
    (name c.acc) c.batch c.m c.n c.k (l c.la) (l c.lb)
    (match c.init with `None -> "" | `Bias -> " bias" | `Full -> " init")
    draw c.pad

(* An operand of [rows] x [k] per batch element holding [values] (their bytes
   with k contiguous), laid out with its k or its rows contiguous, [pad]
   elements between batch elements, with [skip] elements before its first. *)
let operand g dt values ~batch ~rows ~k ~layout ~skip ~pad =
  let w = width dt in
  let n = batch * ((rows * k) + pad) in
  let buffer = S.buffer g (w * (skip + Int.max 1 n)) in
  let strides =
    match layout with
    | `K -> [| (rows * k) + pad; k; 1 |]
    | `Free -> [| (rows * k) + pad; 1; rows |]
  in
  let laid = Bytes.make (w * n) '\000' in
  (* With k contiguous a batch element's rows are one run in both: one blit
     each. With rows contiguous, element by element. *)
  for z = 0 to batch - 1 do
    match layout with
    | `K ->
        let src = z * rows * k and dst = z * strides.(0) in
        Bytes.blit_string values (w * src) laid (w * dst) (w * rows * k)
    | `Free ->
        for r = 0 to rows - 1 do
          for q = 0 to k - 1 do
            let src = (z * rows * k) + (r * k) + q in
            let dst =
              (z * strides.(0)) + (r * strides.(1)) + (q * strides.(2))
            in
            Bytes.blit_string values (w * src) laid (w * dst) w
          done
        done
  done;
  if n > 0 then
    S.write
      (Rig.Buffer.view buffer ~first:(w * skip) ~length:(w * n))
      (Bytes.to_string laid);
  {
    S.buffer;
    dtype = code dt;
    shape = [| batch; rows; k |];
    strides;
    first = w * skip;
  }

let view (o : S.operand) =
  let all = S.read o.buffer in
  {
    Nx_gpu_ref.bytes = String.sub all o.first (String.length all - o.first);
    dtype = o.dtype;
    strides = o.strides;
  }

(* The codes of a float dtype's NaN, infinities, -0 and two subnormals, and an
   integer dtype's extremes and -1, little-endian. float8 e4m3fn has no
   infinity: its largest finite values stand in. *)
let edges (D.Any d) =
  let codes =
    match d with
    | D.Float64 ->
        [
          0x7ff8000000000000;
          0x7ff0000000000000;
          -0x10000000000000;
          min_int;
          1;
          0x8000000000000;
        ]
    | D.Float32 ->
        [ 0x7fc00000; 0x7f800000; 0xff800000; 0x80000000; 1; 0x400000 ]
    | D.Float16 -> [ 0x7e00; 0x7c00; 0xfc00; 0x8000; 1; 0x200 ]
    | D.Bfloat16 -> [ 0x7fc0; 0x7f80; 0xff80; 0x8000; 1; 0x40 ]
    | D.Float8_e4m3fn -> [ 0x7f; 0x7e; 0xfe; 0x80; 1; 4 ]
    | D.Float8_e5m2 -> [ 0x7e; 0x7c; 0xfc; 0x80; 1; 2 ]
    | _ ->
        let bits = D.bits d in
        let signed = D.is D.Signed d in
        if signed then [ 1 lsl (bits - 1); (1 lsl (bits - 1)) - 1; -1 ]
        else [ (1 lsl bits) - 1; 0; 1 lsl (bits - 1) ]
  in
  let w = D.bits d / 8 in
  List.map
    (fun c -> String.init w (fun i -> Char.chr ((c lsr (8 * i)) land 255)))
    codes

(* [rows] x [k] elements of [dt] drawn on the GPU from [seed] by [draw], for
   [batch] batch elements; [cancel] makes a's rows cancel. Integers draw over
   their range, so that their sums wrap. *)
let values g dt ~batch ~rows ~k ~draw ~seed ~cancel =
  let (D.Any d) = dt in
  let n = batch * rows * k and w = width dt in
  let gen =
    match draw with
    | _ when not (D.is D.Float d) -> S.Uniform
    | Spread e -> S.Wide e
    | Cancel | Edges | Overflow -> S.Wide 6
  in
  let tmp = S.buffer g (w * Int.max 1 n) in
  S.generate g tmp d gen ~seed;
  let v = Bytes.of_string (String.sub (S.read tmp) 0 (w * n)) in
  (match draw with
  | Cancel when cancel && D.is D.Float d ->
      (* The sign bit is the top byte's top bit. *)
      for r = 0 to (batch * rows) - 1 do
        for q = 0 to (k / 2) - 1 do
          let src = w * ((r * k) + q) and dst = w * ((r * k) + (k / 2) + q) in
          Bytes.blit v src v dst w;
          let top = dst + w - 1 in
          Bytes.set_uint8 v top (Bytes.get_uint8 v top lxor 0x80)
        done
      done
  | Edges ->
      let es = edges dt in
      for i = 0 to n - 1 do
        if i mod 7 = seed mod 7 then
          Bytes.blit_string
            (List.nth es (i / 7 mod List.length es))
            0 v (w * i) w
      done
  | Overflow ->
      if dt <> D.Any D.Float64 then invalid_arg "values: overflow is float64's";
      (* The drawn sign bit, then max_float's bits. *)
      for i = 0 to n - 1 do
        let negative = Bytes.get_uint8 v ((8 * i) + 7) land 0x80 <> 0 in
        Bytes.set_int64_le v (8 * i)
          (Int64.logor
             (Int64.bits_of_float max_float)
             (if negative then Int64.min_int else 0L))
      done
  | _ -> ());
  Bytes.to_string v

(* Plans and runs the case on operands in its layouts, [skip] elements into
   their buffers; a, b, init, y and the plan. *)
let contract g c ?(skip = 0) ?(seed = 1) () =
  let draw = c.draw and pad = c.pad and batch = c.batch and k = c.k in
  let a =
    let v = values g c.dt ~batch ~rows:c.m ~k ~draw ~seed ~cancel:true in
    operand g c.dt v ~batch ~rows:c.m ~k ~layout:c.la ~skip ~pad
  in
  let b =
    let v =
      values g c.dt ~batch ~rows:c.n ~k ~draw ~seed:(seed + 1) ~cancel:false
    in
    operand g c.dt v ~batch ~rows:c.n ~k ~layout:c.lb ~skip ~pad
  in
  let init_values rows =
    values g c.out ~batch:1 ~rows ~k:1 ~draw ~seed:(seed + 2) ~cancel:false
  in
  let wy = width c.out and ny = c.batch * c.m * c.n in
  let y =
    {
      S.buffer = S.buffer g (wy * Int.max 1 ny);
      dtype = code c.out;
      shape = [| c.batch; c.m; c.n |];
      strides = [| c.m * c.n; c.n; 1 |];
      first = 0;
    }
  in
  let init =
    match c.init with
    | `None -> None
    | `Bias ->
        let buffer = S.buffer g (wy * Int.max 1 c.n) in
        S.write buffer (init_values (Int.max 1 c.n));
        Some { y with buffer; strides = [| 0; 0; 1 |] }
    | `Full ->
        let buffer = S.buffer g (wy * Int.max 1 ny) in
        S.write buffer (init_values (Int.max 1 ny));
        Some { y with buffer }
  in
  match
    S.contract g ~a ~b ?init ~y
      ~batch:[ (0, 0) ]
      ~contracting:[ (2, 2) ]
      ~acc:(code c.acc) ()
  with
  | None -> failf "the plan declines %s" (case_name c)
  | Some p -> (a, b, init, y, p)

let within_bound c =
  let g = S.gpu () in
  let a, b, init, y, _ = contract g c () in
  let r =
    Nx_gpu_ref.contract ~a:(view a) ~b:(view b) ?init:(Option.map view init)
      ~y:(view y) ~batch:c.batch ~m:c.m ~n:c.n ~k:c.k ~acc:(code c.acc)
      ~samples:4096 ()
  in
  equal ~msg:(strf "wrong outputs, the first %d" r.at) int 0 r.wrong;
  at_most ~msg:(strf "worst at output %d" r.at) float_exact ~than:1. r.worst

(* Shapes across the tiles' edges, few rows and columns, long and empty
   sums, and no batch element, row or column. *)
let shapes =
  [
    (0, 5, 7, 9);
    (1, 0, 64, 32);
    (1, 64, 0, 32);
    (1, 1, 1, 1);
    (1, 1, 64, 300);
    (1, 16, 128, 64);
    (1, 17, 129, 65);
    (1, 127, 128, 33);
    (1, 128, 127, 32);
    (1, 129, 1, 0);
    (1, 200, 257, 1000);
    (3, 33, 70, 129);
    (1, 64, 64, 8192);
    (1, 1, 64, 20000);
    (1, 5, 300, 4100);
  ]

let configs =
  let open D in
  [
    (Any Bfloat16, Any Float32, Any Bfloat16);
    (Any Bfloat16, Any Float32, Any Float32);
    (Any Float16, Any Float32, Any Float16);
    (Any Float8_e4m3fn, Any Float32, Any Bfloat16);
    (Any Float32, Any Float32, Any Float32);
    (Any Float64, Any Float64, Any Float64);
    (Any Int8, Any Int32, Any Int32);
    (Any Int8, Any Int32, Any Int8);
    (Any Int8, Any Int32, Any Int64);
    (Any Int16, Any Int32, Any Int64);
    (Any Uint8, Any Int32, Any Int32);
    (Any Int16, Any Int64, Any Int64);
    (Any Int32, Any Int32, Any Int32);
  ]

let cases_of ?(layouts = [ (`K, `K); (`K, `Free); (`Free, `K); (`Free, `Free) ])
    () =
  List.concat_map
    (fun (dt, acc, out) ->
      List.concat_map
        (fun (batch, m, n, k) ->
          List.mapi
            (fun i (la, lb) ->
              let init =
                match i mod 3 with 0 -> `None | 1 -> `Bias | _ -> `Full
              in
              let draw = Spread (if dt = D.Any D.Float16 then 2 else 6) in
              { dt; acc; out; batch; m; n; k; la; lb; init; draw; pad = 0 })
            layouts)
        shapes)
    configs

(* [n] bytes of 0xa5, written over y before a rerun, so that a rerun that
   stores nothing shows. *)
let sentinel n = String.make n '\xa5'

(* The bits depend on the dtypes and the extents alone. The same values in
   the other layouts, behind an element that breaks the 16-byte vectors,
   and computed again over a sentinel, give the same bytes. *)
let same_bits c =
  let g = S.gpu () in
  let _, _, _, (y : S.operand), p = contract g c () in
  let want = S.read y.buffer in
  List.iter
    (fun (la, lb, skip) ->
      let _, _, _, (y : S.operand), _ = contract g { c with la; lb } ~skip () in
      equal
        ~msg:(strf "layouts with %d skipped" skip)
        string want (S.read y.buffer))
    [ (`Free, `K, 0); (`K, `Free, 0); (`Free, `Free, 0); (`K, `K, 1) ];
  S.write y.buffer (sentinel (String.length want));
  S.run g p;
  equal ~msg:"again" string want (S.read y.buffer)

(* [shapes] of each of [configs], k contiguous, with a bias. *)
let at_shapes configs shapes =
  List.concat_map
    (fun (dt, acc, out) ->
      List.map
        (fun (batch, m, n, k) ->
          let draw = Spread 6 and pad = 0 in
          {
            dt;
            acc;
            out;
            batch;
            m;
            n;
            k;
            la = `K;
            lb = `K;
            init = `Bias;
            draw;
            pad;
          })
        shapes)
    configs

let shape_cases =
  at_shapes configs
    [
      (1, 17, 129, 65); (2, 130, 200, 300); (1, 64, 64, 8192); (1, 3, 100, 5000);
    ]

(* Edge inputs: wide and no spread, cancellation, special values and integer
   extremes, batch elements apart by 0, 1 and 4 elements. *)
let edge_cases =
  let draws = [ Spread 0; Spread 8; Spread 40; Cancel; Edges ] in
  List.concat_map
    (fun (dt, acc, out) ->
      List.concat_map
        (fun (batch, m, n, k) ->
          List.mapi
            (fun i draw ->
              let pad = List.nth [ 0; 1; 4 ] (i mod 3) in
              let la, lb = if i mod 2 = 0 then (`K, `K) else (`Free, `K) in
              { dt; acc; out; batch; m; n; k; la; lb; init = `Bias; draw; pad })
            draws)
        [
          (2, 17, 129, 65); (2, 64, 64, 64); (1, 1, 64, 300); (1, 64, 64, 8192);
        ])
    configs

(* Float64 sums whose every product passes double's range: each output is
   the infinity of its products' signs, or, where both signs meet, either
   infinity or NaN, as the order of the sum makes it. A few terms, so that
   some rows hold one sign. *)
let overflow_cases =
  let f64 = D.Any D.Float64 in
  List.mapi
    (fun i (batch, m, n, k) ->
      let la, lb = if i mod 2 = 0 then (`K, `K) else (`Free, `K) in
      let draw = Overflow and pad = 0 in
      {
        dt = f64;
        acc = f64;
        out = f64;
        batch;
        m;
        n;
        k;
        la;
        lb;
        init = `Bias;
        draw;
        pad;
      })
    [ (1, 1, 64, 1); (1, 5, 40, 2); (2, 17, 33, 3); (1, 20, 70, 1) ]

(* An int32 sum reaches a wider output as a cast from int32 does, by its sign:
   -1 times 1 over 64 terms is -64. *)
let sign_extends () =
  let g = S.gpu () in
  List.iter
    (fun (D.Any d as dt) ->
      let w = D.bits d / 8 in
      let ones c = String.concat "" (List.init 64 (fun _ -> String.make w c)) in
      let one = String.init w (fun i -> if i = 0 then '\001' else '\000') in
      let a =
        operand g dt (ones '\xff') ~batch:1 ~rows:1 ~k:64 ~layout:`K ~skip:0
          ~pad:0
      in
      let b =
        operand g dt
          (String.concat "" (List.init 64 (fun _ -> one)))
          ~batch:1 ~rows:1 ~k:64 ~layout:`K ~skip:0 ~pad:0
      in
      let y : S.operand =
        {
          buffer = S.buffer g 8;
          dtype = D.code D.Int64;
          shape = [| 1; 1; 1 |];
          strides = [| 1; 1; 1 |];
          first = 0;
        }
      in
      match
        S.contract g ~a ~b ~y
          ~batch:[ (0, 0) ]
          ~contracting:[ (2, 2) ]
          ~acc:(D.code D.Int32) ()
      with
      | None -> fail "the plan declines"
      | Some _ ->
          equal ~msg:(D.name d) int64 (-64L)
            (String.get_int64_le (S.read y.buffer) 0))
    [ D.Any D.Int8; D.Any D.Int16 ]

(* [y]'s [n] bytes, in runs of [run] bytes against the byte [want i] of
   the run's first index [i], the first differences [show] names. [every]
   reads every run, else 64 MiB at a time; [~every:e] the runs whose
   index is a multiple of [e], and the last. *)
let differing_bytes ?(run = 1) ?every (y : Rig.Buffer.t) n ~want ~show =
  let runs = (n + run - 1) / run in
  let check acc at got =
    let c = Char.chr (want at) in
    if
      List.length !acc < 8
      && not (String.equal got (String.make (String.length got) c))
    then
      String.iteri
        (fun i v ->
          if v <> c && List.length !acc < 8 then
            acc := show (at + i) (Char.code v) :: !acc)
        got
  in
  let acc = ref [] in
  (match every with
  | Some e ->
      for r = 0 to runs - 1 do
        if r mod e = 0 || r = runs - 1 then
          let at = r * run in
          check acc at
            (S.read
               (Rig.Buffer.view y ~first:at ~length:(Int.min run (n - at))))
      done
  | None ->
      let chunk = (1 lsl 26) / run * run in
      let at = ref 0 in
      while !at < n do
        let len = Int.min chunk (n - !at) in
        let got = S.read (Rig.Buffer.view y ~first:!at ~length:len) in
        for r = 0 to ((len + run - 1) / run) - 1 do
          let off = r * run in
          check acc (!at + off) (String.sub got off (Int.min run (len - off)))
        done;
        at := !at + len
      done);
  List.rev !acc

(* [f ()] with [g]'s device allowed [extra] bytes and 256 MiB past the
   bytes it holds, so that a call that copies a broadcast operand in full
   raises {!Rig.Out_of_memory}. *)
let within_budget ?(extra = 0) g f =
  let d = S.device g in
  let before = Rig.budget d in
  Rig.set_budget d (Rig.allocated d + extra + (1 lsl 28));
  Fun.protect ~finally:(fun () -> Rig.set_budget d before) f

(* Contracted extents past 2^30 and 2^31 compute, through few rows'
   kernel and through the tiles' past 8 rows. a and b are one float64 1
   broadcast, of the accumulator's dtype, which the call reads where it
   lies, within 256 MiB of the device's memory, so a row sums its k
   products exactly to k. *)
let extent_past_bound () =
  let g = S.gpu () in
  let float64 = D.code D.Float64 in
  let one = S.buffer g 8 in
  S.write one (put64 [ 1. ]);
  let ones rows k =
    {
      S.buffer = one;
      dtype = float64;
      shape = [| 1; rows; k |];
      strides = [| 0; 0; 0 |];
      first = 0;
    }
  in
  let sums rows k =
    let y =
      {
        S.buffer = S.buffer g (8 * rows);
        dtype = float64;
        shape = [| 1; rows; 1 |];
        strides = [| rows; 1; 1 |];
        first = 0;
      }
    in
    match
      S.contract g ~a:(ones rows k) ~b:(ones 1 k) ~y
        ~batch:[ (0, 0) ]
        ~contracting:[ (2, 2) ]
        ~acc:float64 ()
    with
    | None -> None
    | Some _ -> Some (List.init rows (f64 (S.read y.buffer)))
  in
  List.iter
    (fun (rows, k) ->
      equal
        ~msg:(strf "%d rows, k %d" rows k)
        (option (list float_exact))
        (Some (List.init rows (fun _ -> Float.of_int k)))
        (within_budget g (fun () -> sums rows k)))
    [ (1, (1 lsl 30) + 1); (1, (1 lsl 31) + 1); (9, (1 lsl 31) + 1) ]

(* Rows past 2^30 compute: the plan cuts them into launches. a and b are one float32 1 broadcast,
   of the accumulator's dtype, which the call reads where it lies, within
   256 MiB of the device's memory past y's float32 copy, so each output of
   y, int8, is k. *)
let rows_past_bound () =
  let g = S.gpu () in
  let int8 = D.code D.Int8 and float32 = D.code D.Float32 in
  let m = (1 lsl 30) + 5 and k = 3 in
  let one = S.buffer g 4 in
  S.write one (put32 [ 1. ]);
  let ones rows =
    {
      S.buffer = one;
      dtype = float32;
      shape = [| 1; rows; k |];
      strides = [| 0; 0; 0 |];
      first = 0;
    }
  in
  let y =
    {
      S.buffer = S.buffer g m;
      dtype = int8;
      shape = [| 1; m; 1 |];
      strides = [| m; 1; 1 |];
      first = 0;
    }
  in
  match
    within_budget ~extra:(4 * m) g (fun () ->
        S.contract g ~a:(ones m) ~b:(ones 1) ~y
          ~batch:[ (0, 0) ]
          ~contracting:[ (2, 2) ]
          ~acc:float32 ())
  with
  | None -> fail "the plan declines"
  | Some _ ->
      no_difference
        (differing_bytes ~run:(1 lsl 20) y.buffer m
           ~want:(fun _ -> k)
           ~show:(fun i v -> strf "row %d: %d" i v))

(* Segments past the bounds of a launch *)

(* An array of [d] over [buffer], of [shape] and [strides]. *)
let array (type v s) (d : (v, s) D.t) buffer shape strides : (v, s) Nx_array.t =
  Nx_array.v d
    (Nx_array.Layout.v ~offset:0 ~strides:(Iarray.of_array strides)
       (Iarray.of_array shape))
    buffer

(* int64 [ids] as bytes, little-endian. *)
let int64s ids =
  let b = Bytes.create (8 * Array.length ids) in
  Array.iteri (fun i x -> Bytes.set_int64_le b (8 * i) (Int64.of_int x)) ids;
  Bytes.to_string b

(* A segmented call computes past 2^31 - 1 rows a call, its batch elements
   times its rows, which only the contracted extent may bound. a is one
   int8 1 broadcast over [2; 1024; 2^20; 3], segmented along its 1024, b
   [2; 3; 1; 2] its group g's elements g + 1, so that a row of group g sums
   to 3 (g + 1) and one of no group, id -1, to 0, wrapped into int8: every
   seventh run of 2^20 rows of one id, and the last. *)
let rows_segments_past_bound () =
  let g = S.gpu () in
  let x = 1024 and inner = 1 lsl 20 and k = 3 in
  let one = S.buffer g 1 in
  S.write one "\001";
  let a = array D.Int8 one [| 2; x; inner; k |] [| 0; 0; 0; 0 |] in
  let parts = S.buffer g 2 in
  S.write parts "\001\002";
  let b = array D.Int8 parts [| 2; k; 1; 2 |] [| 0; 0; 0; 1 |] in
  let id z r = ((r + z) mod 3) - 1 in
  let ids = S.buffer g (8 * 2 * x) in
  S.write ids (int64s (Array.init (2 * x) (fun e -> id (e / x) (e mod x))));
  let ids = array D.Int64 ids [| 2; x |] [| x; 1 |] in
  let m = x * inner in
  let y = S.buffer g (2 * m) in
  let dst = array D.Int8 y [| 2; x; inner; 1 |] [| m; inner; 1; 1 |] in
  let answer =
    Nx_cuda.contract
      ~batch:(Iarray.of_list [ (0, 0) ])
      ~contracting:(Iarray.of_list [ (3, 1) ])
      ~segments:(Some (Nx_kernel.Rows { ids; a_axis = 1; b_axis = 3 }))
      ~acc:(D.Any D.Int32) ~dst ~init:None
      { array = Nx_array.Any a; scale = None }
      { array = Nx_array.Any b; scale = None }
  in
  equal ~msg:"answer" bool true (answer = Nx_array.Done);
  let want e =
    let z = e / m and r = e mod m / inner in
    match id z r with -1 -> 0 | grp -> k * (grp + 1)
  in
  no_difference
    (differing_bytes ~run:inner ~every:7 y (2 * m) ~want ~show:(fun e v ->
         strf "output %d: %d" e v))

(* A segmented call computes past 2^31 - 1 contracted indices a call, its
   batch elements times its contracted extent, which only the contracted
   extent may bound. a and b are one int8 1 broadcast over 2 batch
   elements and k = 2^20 x 2^10, segmented along the 2^20, of which the
   first 4 are in group 0, so that each output sums 4096 ones. *)
let contracting_segments_past_bound () =
  let g = S.gpu () in
  let x = 1 lsl 20 and inner = 1 lsl 10 in
  let one = S.buffer g 1 in
  S.write one "\001";
  let a = array D.Int8 one [| 2; 1; x; inner |] [| 0; 0; 0; 0 |] in
  let b = array D.Int8 one [| 2; x; inner; 1 |] [| 0; 0; 0; 0 |] in
  let ids = S.buffer g (8 * 2 * x) in
  S.write ids
    (int64s (Array.init (2 * x) (fun e -> if e mod x < 4 then 0 else -1)));
  let ids = array D.Int64 ids [| 2; x |] [| x; 1 |] in
  let y = S.buffer g 8 in
  let dst = array D.Int32 y [| 1; 2; 1; 1 |] [| 2; 1; 1; 1 |] in
  let answer =
    Nx_cuda.contract
      ~batch:(Iarray.of_list [ (0, 0) ])
      ~contracting:(Iarray.of_list [ (2, 1); (3, 2) ])
      ~segments:(Some (Nx_kernel.Contracting { ids; groups = 1; pair = 0 }))
      ~acc:(D.Any D.Int32) ~dst ~init:None
      { array = Nx_array.Any a; scale = None }
      { array = Nx_array.Any b; scale = None }
  in
  equal ~msg:"answer" bool true (answer = Nx_array.Done);
  let ys = S.read y in
  equal ~msg:"sums" (list int32) [ 4096l; 4096l ]
    [ String.get_int32_le ys 0; String.get_int32_le ys 4 ]

(* Repeated operands *)

(* An operand of [batch] x [rows] x [k] elements of [dt] that repeats one
   element along each group of [rep] (`B batch, `R rows, `K k), as a view
   of stride 0 over the distinct elements, and its copy holding every
   element, k contiguous. *)
let repeated g dt ~batch ~rows ~k ~rep ~seed =
  let has x = List.mem x rep in
  let nb = if has `B then 1 else batch and nr = if has `R then 1 else rows in
  let nk = if has `K then 1 else k in
  let v =
    values g dt ~batch:nb ~rows:nr ~k:nk ~draw:(Spread 6) ~seed ~cancel:false
  in
  let w = width dt in
  let buffer = S.buffer g (w * Int.max 1 (nb * nr * nk)) in
  S.write buffer v;
  let view =
    {
      S.buffer;
      dtype = code dt;
      shape = [| batch; rows; k |];
      strides =
        [|
          (if has `B then 0 else nr * nk);
          (if has `R then 0 else nk);
          (if has `K then 0 else 1);
        |];
      first = 0;
    }
  in
  let full =
    String.concat ""
      (List.init
         (batch * rows * k)
         (fun e ->
           let z = e / (rows * k) and r = e / k mod rows and q = e mod k in
           let z = if has `B then 0 else z and r = if has `R then 0 else r in
           let q = if has `K then 0 else q in
           String.sub v (w * ((((z * nr) + r) * nk) + q)) w))
  in
  (view, operand g dt full ~batch ~rows ~k ~layout:`K ~skip:0 ~pad:0)

let rep_name rep =
  String.concat "" (List.map (function `B -> "b" | `R -> "r" | `K -> "k") rep)

(* An a repeating an element along groups of [rep] and a b along k, of
   [dt] into [acc] and [out], and the extents. *)
let repeated_cases =
  let open D in
  List.concat_map
    (fun (dt, acc, out) ->
      List.concat_map
        (fun shape ->
          List.map
            (fun rep -> (dt, acc, out, shape, rep))
            [ [ `R ]; [ `K ]; [ `R; `K ]; [ `B; `R; `K ] ])
        [ (2, 2, 40, 1000); (2, 40, 64, 300); (1, 200, 130, 70) ])
    [
      (Any Float32, Any Float32, Any Float32);
      (Any Float64, Any Float64, Any Float64);
      (Any Bfloat16, Any Float32, Any Float32);
      (Any Int8, Any Int32, Any Int32);
      (Any Int16, Any Int32, Any Int32);
    ]

let repeated_name (D.Any d, D.Any acc, _, (batch, m, n, k), rep) =
  strf "%s acc %s %dx%dx%dx%d a repeats %s" (D.name d) (D.name acc) batch m n k
    (rep_name rep)

(* An operand that repeats an element along a group gives the bits its copy
   holding every element gives: the association reads no layout. *)
let repeated_bits (dt, acc, out, (batch, m, n, k), rep) =
  let g = S.gpu () in
  let av, af = repeated g dt ~batch ~rows:m ~k ~rep ~seed:1 in
  let bv, bf = repeated g dt ~batch ~rows:n ~k ~rep:[ `K ] ~seed:2 in
  let y () =
    {
      S.buffer = S.buffer g (width out * batch * m * n);
      dtype = code out;
      shape = [| batch; m; n |];
      strides = [| m * n; n; 1 |];
      first = 0;
    }
  in
  let run a b =
    let y = y () in
    match
      S.contract g ~a ~b ~y
        ~batch:[ (0, 0) ]
        ~contracting:[ (2, 2) ]
        ~acc:(code acc) ()
    with
    | None -> fail "the plan declines"
    | Some _ -> S.read y.buffer
  in
  equal ~msg:"repeated against copied" string (run af bf) (run av bv)

(* Outputs without products *)

(* [x]'s low [w] bytes, little-endian. *)
let le w (x : int64) =
  String.init w (fun i ->
      Char.chr
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical x (8 * i)) 0xFFL)))

(* Contractions over k = 0 whose init holds NaNs, and each init code's
   result: its element cast into the accumulator and back into y
   (nx_kernel.mli), which no addition touches. Into float32 from float32,
   or float64 from float64, the casts are identities. From bfloat16 and
   float16, nx_dtype.h's decoders quiet a NaN and its encoders keep its
   sign and top payload bits. The FMA kernels sum float32, float64 and
   complex64, the matrix units bfloat16 and float16 into float32. *)
let nan_inits =
  let f32 = D.Any D.Float32 and f64 = D.Any D.Float64 in
  let bf16 = D.Any D.Bfloat16 and f16 = D.Any D.Float16 in
  let c64 = D.Any D.Complex64 in
  [
    ( "float32, FMA",
      f32,
      f32,
      f32,
      [ (0x7F800001L, 0x7F800001L); (0xFFC00123L, 0xFFC00123L) ] );
    ( "float64, FMA",
      f64,
      f64,
      f64,
      [
        (0x7FF0000000000001L, 0x7FF0000000000001L);
        (0xFFF8000000000123L, 0xFFF8000000000123L);
      ] );
    ( "complex64, FMA",
      c64,
      c64,
      c64,
      [
        (0xFF8000027F800001L, 0xFF8000027F800001L);
        (0x3F800000FFC00123L, 0x3F800000FFC00123L);
      ] );
    ( "bfloat16 into float32, matrix units",
      bf16,
      f32,
      f32,
      [ (0x7F800001L, 0x7F800001L); (0xFFC00123L, 0xFFC00123L) ] );
    ( "bfloat16 into bfloat16, matrix units",
      bf16,
      f32,
      bf16,
      [ (0xFFC1L, 0xFFC1L); (0x7F81L, 0x7FC1L) ] );
    ( "float16 into float16, matrix units",
      f16,
      f32,
      f16,
      [ (0xFC01L, 0xFE01L); (0x7E05L, 0x7E05L) ] );
  ]

(* Each output of a contraction over k = 0 is its init's code through the
   casts, bit for bit, 2 rows of 64 columns (16-byte vectors of y) and of
   3. *)
let nan_init (_, dt, acc, out, codes) =
  let g = S.gpu () in
  List.iter
    (fun n ->
      let m = 2 in
      let operand rows =
        operand g dt "" ~batch:1 ~rows ~k:0 ~layout:`K ~skip:0 ~pad:0
      in
      let w = width out and count = List.length codes in
      let init_codes =
        String.concat ""
          (List.init (m * n) (fun i ->
               le w (fst (List.nth codes (i mod count)))))
      in
      let y =
        {
          S.buffer = S.buffer g (w * m * n);
          dtype = code out;
          shape = [| 1; m; n |];
          strides = [| m * n; n; 1 |];
          first = 0;
        }
      in
      let init = { y with buffer = S.buffer g (w * m * n) } in
      S.write init.buffer init_codes;
      (match
         S.contract g ~a:(operand m) ~b:(operand n) ~init ~y
           ~batch:[ (0, 0) ]
           ~contracting:[ (2, 2) ]
           ~acc:(code acc) ()
       with
      | None -> fail "the plan declines"
      | Some _ -> ());
      let got = S.read y.buffer in
      let show i = strf "output %d: %S" i (String.sub got (w * i) w) in
      no_difference
        (differences (m * n)
           (fun i ->
             String.sub got (w * i) w
             = le w (snd (List.nth codes (i mod count))))
           (fun i -> strf "%d columns, %s" n (show i))))
    [ 64; 3 ]

(* Batches past the grid's 65,535 compute, few and more outputs. *)
let batch_cases =
  let f32 = D.Any D.Float32 in
  List.map
    (fun (m, n, k) ->
      {
        dt = D.Any D.Bfloat16;
        acc = f32;
        out = f32;
        batch = 70_000;
        m;
        n;
        k;
        la = `K;
        lb = `K;
        init = `Bias;
        draw = Spread 6;
        pad = 0;
      })
    [ (2, 3, 5); (8, 8, 2) ]

(* The matrix units *)

(* bfloat16 bits of [x], a value bfloat16 holds, little-endian. *)
let bf16 x =
  let b = Int32.to_int (Int32.shift_right_logical (Int32.bits_of_float x) 16) in
  String.init 2 (fun i -> Char.chr ((b lsr (8 * i)) land 255))

(* Inputs that make the matrix units err most against their error factor:
   [a] and [b]'s rows by their index in k, init by its row, each value
   bfloat16 holds. Products below the truncation of an init of 1, which a
   group adds to nothing, 2 rows of 64 columns over 8192 terms cut k into
   4 ranges, 64 rows over 4096 into 2; and over K = 1 to 16 terms, where
   the factor is tightest, the same with one product of nearly 2 u, of
   which a group keeps u and the sum's truncation to float32 loses it
   again: an error of about (0.99 K + 1) u against the factor's
   (K + 2⌈K/8⌉) u. *)
let tiny = Float.ldexp (1. -. Float.ldexp 1. (-8)) (-12)
let near_two = Float.ldexp (1. +. (53. /. 128.)) (-12)

let mma_cases =
  [
    ( "tiny products into 1, 2 rows",
      (2, 64, 8192, (fun _ -> tiny), (fun _ -> tiny), Some 1.) );
    ( "tiny products into 1, 64 rows",
      (64, 64, 4096, (fun _ -> tiny), (fun _ -> tiny), Some 1.) );
  ]
  @ List.concat_map
      (fun k ->
        let f q = if q = 0 then near_two else tiny in
        List.map
          (fun m ->
            (strf "%d terms into 1, %d rows" k m, (m, 64, k, f, f, Some 1.)))
          [ 2; 64 ])
      (List.init 16 (fun k -> k + 1))

(* Each output of a bfloat16 contraction into float32 is within
   γ(K + 2⌈K/8⌉) (|init| + Σ|a||b|) of the exact sum, nx_cuda.mli's
   factor for the matrix units. *)
let mma_factor (_, (m, n, k, av, bv, start)) =
  let g = S.gpu () in
  let bf = D.Any D.Bfloat16 and fl = D.Any D.Float32 in
  let rows f r =
    String.concat "" (List.init (r * k) (fun i -> bf16 (f (i mod k))))
  in
  let a =
    operand g bf (rows av m) ~batch:1 ~rows:m ~k ~layout:`K ~skip:0 ~pad:0
  in
  let b =
    operand g bf (rows bv n) ~batch:1 ~rows:n ~k ~layout:`K ~skip:0 ~pad:0
  in
  let y =
    {
      S.buffer = S.buffer g (4 * m * n);
      dtype = code fl;
      shape = [| 1; m; n |];
      strides = [| m * n; n; 1 |];
      first = 0;
    }
  in
  let init =
    Option.map
      (fun v ->
        let buffer = S.buffer g (4 * m * n) in
        S.write buffer (put32 (List.init (m * n) (fun _ -> v)));
        { y with buffer })
      start
  in
  (match
     S.contract g ~a ~b ?init ~y
       ~batch:[ (0, 0) ]
       ~contracting:[ (2, 2) ]
       ~acc:(code fl) ()
   with
  | None -> fail "the plan declines"
  | Some _ -> ());
  let exact = ref (Option.value ~default:0. start) in
  let mag = ref (Float.abs !exact) in
  for q = 0 to k - 1 do
    exact := !exact +. (av q *. bv q);
    mag := !mag +. Float.abs (av q *. bv q)
  done;
  let u = Float.ldexp 1. (-24) in
  let t = float_of_int (k + (2 * ((k + 7) / 8))) in
  let bound = t *. u /. (1. -. (t *. u)) *. !mag in
  let ys = S.read y.buffer in
  let worst = ref 0. in
  for i = 0 to (m * n) - 1 do
    worst := Float.max !worst (Float.abs (f32 ys i -. !exact))
  done;
  at_most ~msg:"the largest error" float_exact ~than:bound !worst;
  (* Over 16 terms or fewer, each output is the truncations' 1. *)
  if k <= 16 then
    no_difference
      (differences (m * n)
         (fun i -> f32 ys i = 1.)
         (fun i -> strf "output %d: %h" i (f32 ys i)))

(* Every dtype quadruple *)

let pp_dtype ppf (D.Any d) = Format.pp_print_string ppf (D.name d)
let named = List.map (fun (D.Any d as x) -> (D.name d, x)) D.all
let dt name = List.assoc name named

(* [rows] x [k] elements of [dt] per batch element, one batch element shared by
   all when [broadcast], drawn on the GPU where it draws [dt] and as random
   bytes otherwise. *)
let any_operand g (D.Any d as dt) ~batch ~rows ~k ~broadcast ~seed : S.operand =
  let w = Int.max 1 (D.bits d / 8)
  and stored = if broadcast then 1 else batch in
  let n = stored * rows * k in
  let bytes =
    match d with
    | Float4_e2m1fn | Int4 | Uint4 | Complex128 | Complex64 | Bit ->
        let st = Random.State.make [| seed |] in
        String.init (w * n) (fun _ -> Char.chr (Random.State.int st 256))
    | _ ->
        let draw = Spread (if dt = D.Any D.Float16 then 2 else 6) in
        values g dt ~batch:stored ~rows ~k ~draw ~seed ~cancel:false
  in
  let buffer = S.buffer g (Int.max 1 (w * n)) in
  if n > 0 then S.write buffer bytes;
  let strides = [| (if broadcast then 0 else rows * k); k; 1 |] in
  { buffer; dtype = D.code d; shape = [| batch; rows; k |]; strides; first = 0 }

(* Whether nx contracts in the accumulator [acc]: it refuses a boolean and a
   float narrower than float32 before any kernel plans. *)
let admits (D.Any d) =
  match d with
  | D.Bool | D.Bit | D.Float16 | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2
  | D.Float4_e2m1fn ->
      false
  | _ -> true

(* For every accumulator, and operands, init and output of its kind, the
   result is within the error bound (float sums) or exact (integer sums),
   the output written as a cast from the accumulator writes it. The kernel
   suite's laws take every other dtype. *)
let every_quadruple ((a, b, acc, out), init, (batch, m, n, k), broadcast) =
  if not (admits acc) then collect "nx refuses its accumulator"
  else
    let g = S.gpu () in
    let x = any_operand g a ~batch ~rows:m ~k ~broadcast:false ~seed:1 in
    let w = any_operand g b ~batch ~rows:n ~k ~broadcast ~seed:2 in
    let operand d =
      any_operand g d ~batch ~rows:m ~k:n ~broadcast:false ~seed:3
    in
    let init = if init then Some (operand out) else None in
    let (D.Any o) = out in
    let wy = Int.max 1 (D.bits o / 8) in
    let buffer = S.buffer g (wy * batch * m * n) in
    S.write buffer (String.make (wy * batch * m * n) '\xee');
    let y : S.operand =
      {
        buffer;
        dtype = D.code o;
        shape = [| batch; m; n |];
        strides = [| m * n; n; 1 |];
        first = 0;
      }
    in
    let float_dt (D.Any d) = D.is D.Float d in
    match
      S.contract g ~a:x ~b:w ?init ~y
        ~batch:[ (0, 0) ]
        ~contracting:[ (2, 2) ]
        ~acc:(code acc) ()
    with
    | None -> fail "the plan declines"
    | Some _ ->
        cover "an init" (Option.is_some init);
        cover "a float sum" (float_dt acc);
        cover "an integer sum" (not (float_dt acc));
        cover "an unsigned accumulator" (acc = dt "uint32" || acc = dt "uint64");
        cover "a broadcast operand" broadcast;
        let r =
          Nx_gpu_ref.contract ~a:(view x) ~b:(view w)
            ?init:(Option.map view init) ~y:(view y) ~batch ~m ~n ~k
            ~acc:(code acc) ~samples:512 ()
        in
        equal ~msg:(strf "wrong outputs, the first %d" r.at) int 0 r.wrong;
        at_most
          ~msg:(strf "worst at output %d" r.at)
          float_exact ~than:1. r.worst

let quadruples =
  let open Gen in
  let some names = of_list ~pp:pp_dtype (List.map dt names) in
  let floats =
    some
      [
        "float64";
        "float32";
        "float16";
        "bfloat16";
        "float8_e4m3fn";
        "float8_e5m2";
      ]
  in
  let ints =
    some
      [
        "int64";
        "uint64";
        "int32";
        "uint32";
        "int16";
        "uint16";
        "int8";
        "uint8";
        "bool";
      ]
  in
  let acc =
    frequency
      [
        (3, some [ "float32"; "float64"; "int32"; "uint32"; "int64"; "uint64" ]);
        (1, some [ "int16"; "uint16"; "int8"; "uint8" ]);
      ]
  in
  (* Operands, init and output of the accumulator's kind, which the
     reference reads. *)
  let operands =
    bind acc (fun (D.Any d as acc) ->
        let side = if D.is D.Float d then floats else ints in
        map (fun ((a, b), out) -> (a, b, acc, out)) (pair (pair side side) side))
  in
  let shape = of_list [ (2, 40, 70, 300); (2, 3, 70, 300) ] in
  let pp ppf ((a, b, acc, out), init, (batch, m, n, k), broadcast) =
    Format.fprintf ppf "%a x %a acc %a -> %a%s, %dx%dx%dx%d%s" pp_dtype a
      pp_dtype b pp_dtype acc pp_dtype out
      (if init then ", init" else "")
      batch m n k
      (if broadcast then ", b broadcast" else "")
  in
  with_pp pp (quad operands bool shape bool)

(* A call whose scratch is past the GPU's memory. a's rows, 2^21 of them,
   lie along two axes, one of stride 0, which do not form one run of
   strides, so the call packs them: 2^21 rows of 2^14 bfloat16, 64 GiB of
   scratch, more than the GPU holds. *)
let past_memory g () =
  let n = 16 and k = 1 lsl 14 in
  let bf16 = D.code D.Bfloat16 in
  let operand buffer shape strides =
    { S.buffer; dtype = bf16; shape; strides; first = 0 }
  in
  let a =
    operand (S.buffer g (2 * 2 * k)) [| 1; 1 lsl 20; 2; k |] [| 0; 0; k; 1 |]
  in
  let b = operand (S.buffer g (2 * n * k)) [| 1; n; k |] [| n * k; k; 1 |] in
  let m = 1 lsl 21 in
  let y =
    operand
      (S.buffer g (2 * m * n))
      [| 1; 1 lsl 20; 2; n |]
      [| m * n; 2 * n; n; 1 |]
  in
  S.contract g ~a ~b ~y
    ~batch:[ (0, 0) ]
    ~contracting:[ (3, 2) ]
    ~acc:(D.code D.Float32) ()

(* Out of memory raises: the plan never declines a call for want of it. *)
let scratch_out_of_memory () =
  raises_match
    (function Rig.Out_of_memory (_, bytes) -> bytes = 1 lsl 36 | _ -> false)
    (past_memory (S.gpu ()))

(* Calls *)

(* A bfloat16 case that packs a and b, whose rows of 8191 elements are not
   vectors, and splits its long sum over few columns: every piece of the
   workspace. *)
let packed_split =
  {
    dt = D.Any D.Bfloat16;
    acc = D.Any D.Float32;
    out = D.Any D.Float32;
    batch = 1;
    m = 64;
    n = 4;
    k = 8191;
    la = `K;
    lb = `K;
    init = `Bias;
    draw = Spread 6;
    pad = 0;
  }

(* The words the program allocated on either heap. *)
let allocated () =
  let minor, promoted, major = Gc.counters () in
  minor +. major -. promoted

(* The words [f ()] allocates, less what reading the counters allocates. *)
let words_of f =
  let a = allocated () in
  let b = allocated () in
  f ();
  allocated () -. b -. (b -. a)

(* The words [count] warm calls of [p] allocate, the first one uncounted. *)
let warm_words g ~count p =
  S.run g p;
  let words =
    words_of (fun () ->
        for _ = 1 to count do
          S.call p
        done)
  in
  S.run g p;
  int_of_float words

(* Contractions of bfloat16, float16 and float32 operands into float32 over
   extents around the tiles, both layouts and every init. *)
let shape_case =
  let open Gen in
  let extent = of_list [ 1; 3; 16; 17; 64; 100 ] in
  let+ dt = of_list [ D.Any D.Bfloat16; D.Any D.Float16; D.Any D.Float32 ]
  and+ batch = int_range 1 3
  and+ m = extent
  and+ n = extent
  and+ k = of_list [ 0; 1; 7; 64; 300; 8191 ]
  and+ la = of_list [ `K; `Free ]
  and+ lb = of_list [ `K; `Free ]
  and+ init = of_list [ `None; `Bias; `Full ] in
  let f32 = D.Any D.Float32 in
  { packed_split with dt; acc = f32; out = f32; batch; m; n; k; la; lb; init }

let shape_case =
  Gen.with_pp (fun ppf c -> Format.pp_print_string ppf (case_name c)) shape_case

(* No warm call allocates more words than one of [packed_split], on the
   matrix units, or its float32 twins, on the FMA tiles and on gemv, which
   have every piece a launch holds: packs, a split and an init. Their
   operands' extents and layouts do not change the count. *)
let warm_calls_bounded =
  let most =
    lazy
      (let g = S.gpu () in
       List.fold_left
         (fun most c ->
           let _, _, _, _, p = contract g c () in
           Int.max most (warm_words g ~count:8 p))
         0
         [
           packed_split;
           { packed_split with dt = D.Any D.Float32; batch = 2; m = 16; n = 16 };
           { packed_split with dt = D.Any D.Float32; batch = 2; m = 4; n = 16 };
         ])
  in
  prop ~count:30 "a warm call allocates no more than one with every piece"
    shape_case (fun c ->
      let g = S.gpu () in
      let _, _, _, _, p = contract g c () in
      at_most ~msg:"words for 8 calls" int ~than:(Lazy.force most)
        (warm_words g ~count:8 p))

(* [packed_split]'s call, run until its work is done and dropped, leaving
   only a weak pointer to its destination's buffer. *)
let[@inline never] dropped g =
  let weak = Weak.create 1 in
  let _, _, _, y, p = contract g packed_split () in
  Weak.set weak 0 (Some y.S.buffer);
  S.run g p;
  weak

(* A call keeps none of its arrays once it returns and its work is done:
   collected, its destination's buffer goes. *)
let calls_keep_nothing () =
  let weak = dropped (S.gpu ()) in
  for _ = 1 to 3 do
    Gc.full_major ()
  done;
  equal ~msg:"the destination's buffer outlives the call" bool false
    (Weak.check weak 0)

(* [packed_split] with [a] in host memory, which the GPU's submit refuses
   once the call has written its launch. *)
let on_the_host g () =
  let a, b, init, y, _ = contract g packed_split () in
  let host = Rig.Buffer.create Rig.host (Rig.Buffer.length a.S.buffer) in
  ignore
    (S.contract g ~a:{ a with S.buffer = host } ~b ?init ~y
       ~batch:[ (0, 0) ]
       ~contracting:[ (2, 2) ]
       ~acc:(code packed_split.acc) ())

(* A call that raises from its submit releases its run, so the warm calls
   after it allocate as many words as before it. *)
let raise_releases_the_run () =
  let g = S.gpu () in
  let _, _, _, _, p = contract g packed_split () in
  let before = warm_words g ~count:64 p in
  raises_match
    (function Invalid_argument _ -> true | _ -> false)
    (on_the_host g);
  let after = warm_words g ~count:64 p in
  equal ~msg:"words for 64 calls after a raise" int before after

(* Split sums of different tile counts, run back to back over a sentinel, give
   their first run's bits: each call leaves the tickets that the next counts on
   zero. By the plan's rules all four split: one and sixteen mma tiles, and
   the FMA kernels' tiles over a long k and over one row. *)
let split_cases =
  let f32 = D.Any D.Float32 in
  [
    packed_split;
    { packed_split with m = 4; n = 1000 };
    { packed_split with dt = f32; m = 8; n = 8; k = 70000; init = `None };
    { packed_split with dt = f32; m = 1; k = 20000; init = `None };
  ]

let splits_leave_tickets_zero () =
  let g = S.gpu () in
  let runs =
    List.map
      (fun c ->
        let _, _, _, (y : S.operand), p = contract g c () in
        (c, y, p, S.read y.buffer))
      split_cases
  in
  for round = 1 to 3 do
    List.iter
      (fun (c, (y : S.operand), p, want) ->
        S.write y.buffer (sentinel (String.length want));
        S.run g p;
        equal
          ~msg:(strf "round %d, %s" round (case_name c))
          string want (S.read y.buffer))
      runs
  done

(* A call on a consumed buffer is refused as dead, the destination's or an
   operand's, naming it, and nx.cuda reads nothing of it. *)
let dead_buffers_refused () =
  let g = S.gpu () in
  let c = { packed_split with m = 16; n = 16; k = 64 } in
  List.iter
    (fun (which, operand) ->
      let a, _, _, y, p = contract g c () in
      let (o : S.operand) = match which with `Y -> y | `A -> a in
      Rig.Claim.with_ ~read:[] ~donate:[ o.buffer ] (fun claims ->
          ignore (Rig.Claim.consume claims ~why:"donated" o.buffer));
      let prefix = strf "Nx_cuda.contract: %s was donated" operand in
      raises_match ~msg:prefix
        (function
          | Invalid_argument m -> String.starts_with ~prefix m | _ -> false)
        (fun () -> S.run g p))
    [ (`Y, "float32 [1; 16; 16]"); (`A, "bfloat16 [1; 16; 64]") ]

(* Domains *)

(* A run of [packed_split] on its own operands. *)
let split_run : (unit, S.operand * S.run) abstract = abstract "r"

(* Two domains running [packed_split] at once, each on its own operands or both
   on one's, get the bits one call alone gets. *)
let domains_call_alike =
  let g = lazy (S.gpu ()) in
  let want =
    lazy
      (let _, _, _, (y : S.operand), _ =
         contract (Lazy.force g) packed_split ()
       in
       S.read y.buffer)
  in
  let make () =
    let _, _, _, y, p = contract (Lazy.force g) packed_split () in
    (y, p)
  in
  let run ((y : S.operand), p) =
    S.run (Lazy.force g) p;
    S.read y.buffer
  in
  stateful ~domains:2 ~count:4 "domains calling at once get one call's bits"
    [
      command "make" (Gen.unit @-> makes split_run) (fun () -> ()) make;
      command "run"
        (split_run ^-> returns string)
        (fun () -> Lazy.force want)
        run;
    ]

(* Through rig.nv *)

(* rig.nv loads the cubin CUDA loads, and a contraction's bits are a
   function of its dtypes, its kind and its extents alone, so through
   rig.nv a call has CUDA's bits: nx.kernel's contraction suite checks
   them, call by call, on its drawn cases. The laws of the kernels'
   arithmetic, their bounds and bits at chosen shapes and extents past
   2^30 and 2^31, and the codecs' run through CUDA alone. Through rig.nv
   run the laws of what it does beyond CUDA: launching past a grid's
   extent, holding and releasing memory, and ordering calls from two
   domains. *)
let codecs =
  group "codecs"
    [
      test "float16 as the host's" codecs_f16;
      test "bfloat16 as the host's" codecs_bf16;
      test "float8 e4m3fn as the host's" codecs_e4m3;
      test "float8 e5m2 as the host's" codecs_e5m2;
    ]

let arithmetic =
  [
    cases ~id:case_name "within the error bound" (cases_of ()) within_bound;
    cases ~id:case_name "edge inputs within the error bound" edge_cases
      within_bound;
    cases ~id:case_name "bits fixed by the shape" shape_cases same_bits;
    cases ~id:case_name "float64 sums past double's range" overflow_cases
      within_bound;
    test "an int32 sum sign-extends into int64" sign_extends;
    test "contracted extents past 2^31 compute" extent_past_bound;
    test "rows past 2^30 compute" rows_past_bound;
    test "Rows segments past 2^31 rows a call compute" rows_segments_past_bound;
    test "Contracting segments past 2^31 indices a call compute"
      contracting_segments_past_bound;
    cases ~id:repeated_name "a repeated operand gives its copy's bits"
      repeated_cases repeated_bits;
    cases
      ~id:(fun (n, _, _, _, _) -> n)
      "an output without products is its init, NaN bits included" nan_inits
      nan_init;
    cases ~id:fst "the matrix units keep their stated error factor" mma_cases
      mma_factor;
    prop ~count:400 "every dtype quadruple computes within the bound" quadruples
      every_quadruple;
  ]

let driver =
  [
    cases ~id:case_name "batches past 65,535 within the error bound" batch_cases
      within_bound;
    test "scratch past the GPU's memory raises" scratch_out_of_memory;
    warm_calls_bounded;
    test "a call keeps none of its arrays" calls_keep_nothing;
    test "a call that raises releases its run" raise_releases_the_run;
    test "split sums leave their tickets zero" splits_leave_tickets_zero;
    test "a consumed buffer is refused as dead" dead_buffers_refused;
    domains_call_alike;
  ]

let tests =
  if S.through_nv () then [ group "contract" driver ]
  else [ codecs; group "contract" (arithmetic @ driver) ]

let () = exit (run "nx.cuda" tests)
