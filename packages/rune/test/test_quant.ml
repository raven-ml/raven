(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Quantised products under rune: Law 2 of RFC 0004 for the compiled form on
   every device the machine has, eagerly and compiled, and the rules of reverse
   mode, forward mode, vmap and debug. *)

open Windtrap

let rng = Random.State.make [| 7 |]
let bytes shape f = Nx.init Nx.uint8 shape f

(* Scale bytes of finite values, 0 and 1 (subnormal on a device that keeps them)
   and 255 (NaN groups) included. *)
let scale_byte _ =
  match Random.State.int rng 16 with
  | 0 -> 255
  | 1 -> 0
  | 2 -> 1
  | _ -> 100 + Random.State.int rng 51

let weight ?(scale = scale_byte) shape =
  let r = Array.length shape in
  let part last = Array.append (Array.sub shape 0 (r - 1)) [| last |] in
  let k = shape.(r - 1) in
  Nx_quant.mxfp4
    ~scales:(bytes (part (k / 32)) scale)
    (bytes (part (k / 2)) (fun _ -> Random.State.int rng 256))

let floats shape =
  Nx.init Nx.float32 shape (fun _ -> Random.State.float rng 2.0 -. 1.0)

let ints shape values = Nx.create Nx.int32 shape (Array.map Int32.of_int values)

(* The reference: Law 2's function, decoded from the format's definition and
   summed at float64. *)

let values (Nx_quant.Mxfp4 { codes; scales } as w) =
  let codes = Nx.to_array codes and scales = Nx.to_array scales in
  let e2m1 = [| 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. |] in
  Nx.create Nx.float64 (Nx_quant.shape w)
    (Array.init
       (2 * Array.length codes)
       (fun i ->
         let byte = codes.(i / 2) in
         let code = if i mod 2 = 0 then byte land 15 else byte lsr 4 in
         let s = scales.(i / 32) in
         let v = e2m1.(code land 7) *. Float.ldexp 1.0 (s - 127) in
         if s = 255 then Float.nan else if code land 8 = 0 then v else -.v))

(* [selected ~lanes dq ids] is [w'] of the decoded weight [dq] whose [lanes]
   leading axes precede its experts, and whether each matrix is an expert: an id
   outside the experts selects a zero matrix. *)
let selected ~lanes dq ids =
  let ds = Nx.shape dq and is = Nx.shape ids in
  let e = ds.(lanes) in
  let lane = Array.sub ds 0 lanes in
  let wb =
    Array.append
      (Array.init lanes (fun a -> max lane.(a) is.(a)))
      (Array.sub is lanes (Array.length is - lanes))
  in
  let ids = Nx.to_array (Nx.broadcast_to wb (Nx.contiguous ids)) in
  let positions = Array.length ids in
  let trailing =
    positions / max 1 (Array.fold_left ( * ) 1 (Array.sub wb 0 lanes))
  in
  let valid = Array.map (fun id -> id >= 0l && Int32.to_int id < e) ids in
  (* Position [p]'s matrix among the weight's, lanes flattened. *)
  let index p id =
    let rest = ref (p / max 1 trailing) and l = ref 0 and stride = ref 1 in
    for a = lanes - 1 downto 0 do
      let i = !rest mod wb.(a) in
      rest := !rest / wb.(a);
      if lane.(a) > 1 then l := !l + (i * !stride);
      stride := !stride * lane.(a)
    done;
    if valid.(p) then Int32.of_int ((!l * e) + Int32.to_int id) else 0l
  in
  let flat =
    Nx.reshape (Array.append [| -1 |] (Array.sub ds (lanes + 1) 2)) dq
  in
  let matrices =
    Nx.take ~axis:0
      ~indices:(Nx.create Nx.int32 [| positions |] (Array.mapi index ids))
      flat
  in
  let mask = Nx.create Nx.bool [| positions; 1; 1 |] valid in
  ( Nx.reshape
      (Array.concat [ wb; Array.sub ds (lanes + 1) 2 ])
      (Nx.where mask matrices (Nx.zeros_like matrices)),
    Nx.create Nx.bool wb valid )

type case = {
  w : Nx_quant.t;
  lanes : int;  (** Leading axes of [w] before its experts, with [ids]. *)
  ids : Nx.int32_t option;
  x : Nx.float32_t;
  transpose : bool;
}

(* [reference c x] is the expected product at float64 for the [x] the product
   receives, the sum of the absolute terms of each value, and whether each value
   is a product (not a position that selects no expert). *)
let reference c x =
  let dq = values c.w in
  let w, valid =
    match c.ids with
    | None -> (dq, None)
    | Some ids ->
        let w, valid = selected ~lanes:c.lanes dq ids in
        (w, Some valid)
  in
  let side w = if c.transpose then w else Nx.matrix_transpose w in
  let x = Nx.cast Nx.float64 x in
  let y = Nx.matmul x (side w) in
  let bound = Nx.matmul (Nx.abs x) (side (Nx.abs w)) in
  let valid =
    match valid with
    | None -> Nx.full Nx.bool (Nx.shape y) true
    | Some v ->
        let tail = if Nx.ndim c.x = 1 then [| 1 |] else [| 1; 1 |] in
        Nx.broadcast_to (Nx.shape y)
          (Nx.reshape (Array.append (Nx.shape v) tail) v)
  in
  (y, bound, valid)

let product (type b) c (x : (float, b) Nx.t) : (float, b) Nx.t =
  Nx_quant.Effect.perform c.w
    (Apply { ids = c.ids; x; transpose = c.transpose })

(* Law 2, for the [x] the product receives, at a dtype whose unit roundoff is
   [u]: within a float32 sum of [k] terms, each product and decoded weight
   possibly rounded to [x]'s dtype, and the result rounded to it once (an
   absolute [tiny] for each of those [k + 1] roundings that lands on a float16
   subnormal); NaN exactly where the reference is; exact zeros where no expert
   is selected. A device that flushes subnormal float32 loses, per term, a scale
   byte 0's whole group (values up to 6 * 2^-127 = 3 * 2^-126, each times |x|),
   a product below 2^-126 and a running sum below 2^-126: [k * 2^-126 * (3 * max
   |x| + 2)] for an [x] with no subnormal values. *)
let law2 ~msg ~u ~tiny ~flush c x actual =
  let expected, bound, valid = reference c x in
  equal ~msg:(msg ^ ", shape") (array int) (Nx.shape expected) (Nx.shape actual);
  let s = Nx.shape x in
  let k = float_of_int s.(Array.length s - 1) in
  let largest =
    Array.fold_left
      (fun m v -> if Float.is_finite v then Float.max m (Float.abs v) else m)
      0.0
      (Nx.to_array (Nx.cast Nx.float32 x))
  in
  let expected = Nx.to_array expected and bound = Nx.to_array bound in
  let valid = Nx.to_array valid in
  let actual = Nx.to_array (Nx.cast Nx.float32 actual) in
  let eps = Float.ldexp 1.0 (-24) and least = Float.ldexp 1.0 (-126) in
  Array.iteri
    (fun i e ->
      let a = actual.(i) in
      let wrong () =
        fail (Printf.sprintf "%s: at %d, expected %h, got %h" msg i e a)
      in
      if not valid.(i) then
        begin if Int64.bits_of_float a <> 0L then wrong ()
        end
      else if Float.is_nan e || Float.is_nan a then
        begin if not (Float.is_nan e && Float.is_nan a) then wrong ()
        end
      else
        let b = bound.(i) in
        let tol =
          (2.0 *. k *. eps *. b)
          +. (u *. b)
          +. (u *. Float.abs e)
          +. ((k +. 1.0) *. tiny)
          +. if flush then k *. least *. ((3.0 *. largest) +. 2.0) else 0.0
        in
        if not (Float.abs (a -. e) <= tol) then wrong ())
    expected

(* Devices: the CPU, and Metal where the machine has it. *)
let devices =
  "CPU"
  ::
  (match Tolk.Device.get "METAL" with
  | _ -> [ "METAL" ]
  | exception Invalid_argument _ -> [])

(* The ids and x of a case are the compiled function's inputs; the weight is
   captured, as a model captures its parameters. *)
let inputs () = Nx.Ptree.(pair tensor tensor)

(* The compiled product, [x] an input at its own dtype. [x] is rounded once
   outside: rune's jit, as tinygrad, folds a float32 -> float16 -> float32 round
   trip to the identity, so a cast inside the compiled function would hand the
   product an unrounded [x]. *)
let compiled (type b) ~device c (x : (float, b) Nx.t) : (float, b) Nx.t =
  match c.ids with
  | None -> Rune.jit' ~devices:[ Rune.device device ] (product c) x
  | Some ids ->
      Rune.jit
        ~devices:[ Rune.device device ]
        Nx.Ptree.(inputs () @-> returns tensor)
        (fun (ids, x) -> product { c with ids = Some ids } x)
        (ids, x)

(* The dtypes of [x]: float32, and bfloat16 and float16 with their unit
   roundoffs. float16 decodes at float32, and its results must stay in its
   range. *)
type dtype = Dt : string * (float, 'b) Nx.dtype * float * float -> dtype

let float32 = Dt ("float32", Nx.float32, 0.0, 0.0)
let bfloat16 = Dt ("bfloat16", Nx.bfloat16, Float.ldexp 1.0 (-8), 0.0)

let float16 =
  Dt ("float16", Nx.float16, Float.ldexp 1.0 (-11), Float.ldexp 1.0 (-25))

let battery ?(dtypes = [ float32; bfloat16 ]) c =
  List.iter
    (fun (Dt (name, dt, u, tiny)) ->
      let x = Nx.cast dt c.x in
      law2 ~msg:("eager, " ^ name) ~u ~tiny ~flush:false c x (product c x);
      List.iter
        (fun device ->
          law2
            ~msg:(Printf.sprintf "%s, %s" device name)
            ~u ~tiny ~flush:(device = "METAL") c x (compiled ~device c x))
        devices)
    dtypes

(* Law 2 *)

let case ?(lanes = 0) ?ids ?(transpose = false) w x =
  { w; lanes; ids; x; transpose }

(* [poison ~at x] is [x] with NaN and an infinity in the rows of [x] at the
   batch indices [at], positions where no expert is selected. *)
let poison ~at x =
  List.fold_left
    (fun x index ->
      let s = Nx.shape x in
      let row =
        Array.sub s (List.length index) (Array.length s - List.length index)
      in
      let bad = Nx.reshape [| -1 |] (Nx.full Nx.float32 row Float.nan) in
      let bad = Nx.set [ I 0 ] (Nx.scalar Nx.float32 Float.infinity) bad in
      Nx.set (List.map (fun i -> Nx.I i) index) (Nx.reshape row bad) x)
    x at

let test_without_ids () =
  battery (case (weight [| 5; 64 |]) (floats [| 3; 4; 64 |]));
  battery (case (weight [| 2; 5; 64 |]) (floats [| 4; 1; 3; 64 |]));
  battery (case (weight [| 2; 5; 64 |]) (floats [| 64 |]))

(* Fewer positions than experts: the selected experts' rows are gathered. *)
let test_gathered () =
  let w = weight [| 6; 8; 64 |] in
  let ids = ints [| 2; 2 |] [| 3; -1; 6; 3 |] in
  battery
    (case ~ids w (poison ~at:[ [ 0; 1 ]; [ 1; 0 ] ] (floats [| 2; 2; 1; 64 |])));
  battery (case ~ids:(ints [| 3 |] [| 5; -5; 0 |]) w (floats [| 64 |]));
  battery
    (case ~lanes:1
       ~ids:(ints [| 2; 2 |] [| 2; -1; 0; 2 |])
       (weight [| 2; 3; 8; 64 |])
       (floats [| 2; 2; 1; 64 |]));
  battery
    (case ~lanes:1
       ~ids:(ints [| 1; 2 |] [| 2; 0 |])
       (weight [| 2; 3; 8; 64 |])
       (floats [| 3; 2; 2; 1; 64 |]))

(* As many positions as experts or more, and several rows per position or too
   few routes of one row to group (rule 2: 3 routes over 3 experts): every
   expert is decoded once and each position is its own block of rows. *)
let test_instances () =
  let w = weight [| 3; 8; 64 |] in
  let ids = ints [| 3; 2 |] [| 0; 2; -1; 1; 3; 2 |] in
  battery (case ~ids w (poison ~at:[ [ 1; 0 ] ] (floats [| 3; 2; 3; 64 |])));
  battery (case ~ids w (floats [| 3; 1; 8; 64 |]));
  battery
    (case ~transpose:true ~ids w
       (poison ~at:[ [ 1; 0 ] ] (floats [| 3; 2; 2; 8 |])));
  let few = ints [| 3; 1 |] [| 2; -1; 0 |] in
  battery (case ~ids:few w (poison ~at:[ [ 1; 0 ] ] (floats [| 3; 1; 1; 64 |])));
  battery (case ~ids:(ints [| 3 |] [| 1; 7; 1 |]) w (floats [| 64 |]));
  battery
    (case ~lanes:1
       ~ids:(ints [| 2; 3 |] [| 0; 2; -1; 2; 1; 1 |])
       (weight [| 2; 3; 8; 64 |])
       (floats [| 2; 3; 1; 64 |]));
  battery
    (case ~lanes:1
       ~ids:(ints [| 2; 3 |] [| 2; 0; -1; 1; 1; 5 |])
       (weight [| 1; 3; 8; 64 |])
       (floats [| 3; 2; 64 |]))

(* Enough routes of one row each that grouping pays (rule 2, with 4 experts from
   5 routes on Metal): routes ranked by expert fill blocks that read their
   expert once. Within the kernel's row bound the blocks take the kernel, past
   it the block kernel; which depends on the device and dtype (rule 3). On the
   CPU (τ = 1024) only the 160 routes over 2 experts group, and the other cases
   take rule 1. *)
let test_grouped () =
  let w = weight [| 4; 8; 64 |] in
  let ids =
    ints [| 8; 2 |] [| 0; 3; -1; 2; 4; 3; 1; 1; 3; -5; 0; 2; 2; 2; 1; 0 |]
  in
  battery (case ~ids w (floats [| 8; 1; 1; 64 |]));
  battery
    (case ~ids w
       (poison ~at:[ [ 1; 0 ]; [ 2; 0 ]; [ 4; 1 ] ] (floats [| 8; 2; 1; 64 |])));
  battery (case ~ids:(Nx.reshape [| 16 |] ids) w (floats [| 64 |]));
  battery (case ~ids w (floats [| 3; 8; 2; 1; 64 |]));
  battery
    (case ~lanes:1
       ~ids:
         (ints [| 2; 12 |]
            (Array.init 24 (fun i -> if i mod 7 = 3 then -1 else i * 5 mod 3)))
       (weight [| 2; 3; 8; 64 |])
       (floats [| 2; 12; 1; 64 |]));
  battery
    (case ~lanes:1
       ~ids:(ints [| 2; 12 |] (Array.init 24 (fun i -> i * 7 mod 4)))
       (weight [| 1; 3; 8; 64 |])
       (floats [| 12; 1; 64 |]));
  (* Past the kernel's row bound on every device, the blocks are decoded and
     take the block kernel, 64 rows each. *)
  battery
    (case
       ~ids:(ints [| 160 |] (Array.init 160 (fun i -> (i * 7 mod 5) - 1)))
       (weight [| 2; 8; 64 |])
       (floats [| 160; 1; 64 |]));
  (* Many rows per expert take blocks of more than 8 rows, and a skewed routing
     gives one expert most of the blocks. *)
  battery
    (case
       ~ids:
         (ints [| 40 |] (Array.init 40 (fun i -> if i mod 9 = 4 then 1 else 0)))
       (weight [| 2; 8; 64 |])
       (floats [| 40; 1; 64 |]));
  (* One lane of eight: the lane holds 4 of 32 experts, and the routes of the
     other 28 are -1. *)
  battery
    (case
       ~ids:
         (ints [| 64 |]
            (Array.init 64 (fun i ->
                 let expert = i * 11 mod 32 in
                 if expert < 4 then expert else -1)))
       w
       (floats [| 64; 1; 64 |]))

(* Tolk's kernel, which a product takes while its matrix meets at most the
   device's row bound of rows (64 on the CPU and 32 on Metal at bfloat16, 2 and
   16 at float32): one group per row, which a GPU gates at its loads; rows that
   no row tile divides, and two full tiles of 8; an [x] shared by the positions
   of a token, and one broadcast along another axis; a stack against one shared
   row; and no id selecting an expert. At float32 on the CPU (bound 2) only the
   one-row cases reach the kernel; tolk's own tests cover its row tile there. *)
let test_kernel () =
  let dtypes = [ float32; bfloat16; float16 ] in
  let scale _ =
    match Random.State.int rng 16 with
    | 0 -> 255
    | 1 -> 0
    | 2 -> 1
    | _ -> 110 + Random.State.int rng 21
  in
  let battery = battery ~dtypes in
  battery (case (weight ~scale [| 48; 32 |]) (floats [| 1; 32 |]));
  battery
    (case
       ~ids:(ints [| 3 |] [| 1; -1; 1 |])
       (weight ~scale [| 2; 40; 32 |])
       (poison ~at:[ [ 1 ] ] (floats [| 3; 1; 32 |])));
  battery (case (weight ~scale [| 48; 128 |]) (floats [| 6; 128 |]));
  battery (case (weight ~scale [| 40; 64 |]) (floats [| 16; 64 |]));
  (* A stack of matrices against one shared row. *)
  battery (case (weight ~scale [| 2; 8; 32 |]) (floats [| 32 |]));
  battery (case (weight ~scale [| 2; 8; 32 |]) (floats [| 1; 1; 32 |]));
  battery (case (weight ~scale [| 3; 20; 96 |]) (floats [| 3; 3; 96 |]));
  let w = weight ~scale [| 6; 48; 128 |] in
  battery
    (case
       ~ids:(ints [| 3; 4 |] [| 5; 0; -1; 2; 7; 3; 3; 1; 0; 0; 4; -2 |])
       w
       (poison ~at:[ [ 0; 0 ] ] (floats [| 3; 1; 1; 128 |])));
  battery
    (case
       ~ids:(ints [| 3; 2 |] [| 5; 0; 2; 2; 1; 4 |])
       w
       (floats [| 1; 2; 5; 128 |]));
  battery (case ~ids:(ints [| 2 |] [| -1; 6 |]) w (floats [| 2; 1; 128 |]))

(* Past the row bound, decoding then multiplying: without ids, with fewer
   positions than experts, with as many or more, and with lanes. *)
let test_past_the_bound () =
  let rows = 65 in
  battery (case (weight [| 8; 64 |]) (floats [| rows; 64 |]));
  let w = weight [| 3; 8; 64 |] in
  battery
    (case
       ~ids:(ints [| 2 |] [| 2; -1 |])
       w
       (poison ~at:[ [ 1 ] ] (floats [| 2; rows; 64 |])));
  battery
    (case ~ids:(ints [| 4 |] [| 0; 2; 2; 5 |]) w (floats [| 4; rows; 64 |]));
  battery
    (case ~lanes:1
       ~ids:(ints [| 2; 2 |] [| 2; -1; 0; 2 |])
       (weight [| 2; 3; 8; 64 |])
       (floats [| 2; 2; rows; 64 |]))

let test_transposed () =
  let w = weight [| 4; 8; 64 |] in
  battery (case ~transpose:true w (floats [| 4; 3; 8 |]));
  battery
    (case ~transpose:true
       ~ids:(ints [| 2; 1 |] [| 3; -1 |])
       w
       (poison ~at:[ [ 1; 0 ] ] (floats [| 2; 1; 1; 8 |])));
  battery
    (case ~transpose:true
       ~ids:(ints [| 3; 2 |] [| 0; 3; -1; 2; 4; 3 |])
       w
       (floats [| 3; 2; 1; 8 |]));
  battery
    (case ~transpose:true
       ~ids:(ints [| 8; 2 |] (Array.init 16 (fun i -> (i * 3 mod 6) - 1)))
       w
       (poison ~at:[ [ 0; 0 ] ] (floats [| 8; 2; 1; 8 |])))

(* The largest finite scale bytes, on inputs small enough that no float32 sum
   overflows. *)
let test_large_scales () =
  let w =
    weight ~scale:(fun _ -> 240 + Random.State.int rng 13) [| 3; 4; 32 |]
  in
  let x = Nx.mul_s (floats [| 2; 2; 1; 32 |]) (Float.ldexp 1.0 (-6)) in
  battery (case ~ids:(ints [| 2; 2 |] [| 2; 0; 1; 1 |]) w x);
  battery (case ~ids:(ints [| 1; 2 |] [| 2; 0 |]) w x)

(* Empty inputs give empty results, on every device. *)
let test_empty () =
  let w = weight [| 4; 8; 64 |] in
  let check msg c shape =
    equal ~msg:(msg ^ ", eager") (array int) shape (Nx.shape (product c c.x));
    List.iter
      (fun device ->
        equal
          ~msg:(msg ^ ", " ^ device)
          (array int) shape
          (Nx.shape (compiled ~device c c.x)))
      devices
  in
  check "no tokens"
    (case ~ids:(ints [| 0; 4 |] [||]) w (floats [| 0; 1; 1; 64 |]))
    [| 0; 4; 1; 8 |];
  check "no experts per token"
    (case ~ids:(ints [| 3; 0 |] [||]) w (floats [| 3; 1; 1; 64 |]))
    [| 3; 0; 1; 8 |];
  check "no rows" (case (weight [| 8; 64 |]) (floats [| 0; 64 |])) [| 0; 8 |];
  let c = case (weight [| 8; 0 |]) (floats [| 2; 0 |]) in
  check "no inputs" c [| 2; 8 |];
  List.iter
    (fun device ->
      law2 ~msg:("no inputs, " ^ device) ~u:0.0 ~tiny:0.0 ~flush:false c c.x
        (compiled ~device c c.x))
    devices

(* float16 [x], with scales that keep every result inside float16's range. *)
let test_float16 () =
  let scale _ =
    match Random.State.int rng 16 with
    | 0 -> 255
    | 1 -> 0
    | 2 -> 1
    | _ -> 110 + Random.State.int rng 21
  in
  let battery = battery ~dtypes:[ float32; bfloat16; float16 ] in
  let w = weight ~scale [| 4; 8; 64 |] in
  battery (case (weight ~scale [| 2; 5; 64 |]) (floats [| 4; 1; 3; 64 |]));
  battery (case ~ids:(ints [| 1; 2 |] [| 3; -1 |]) w (floats [| 1; 2; 1; 64 |]));
  battery
    (case
       ~ids:(ints [| 3; 2 |] [| 0; 3; -1; 2; 4; 3 |])
       w
       (floats [| 3; 1; 1; 64 |]));
  battery
    (case ~transpose:true
       ~ids:(ints [| 2; 1 |] [| 3; -1 |])
       w
       (floats [| 2; 1; 1; 8 |]));
  battery
    (case
       ~ids:(ints [| 8; 2 |] (Array.init 16 (fun i -> (i * 5 mod 6) - 1)))
       w
       (floats [| 8; 1; 1; 64 |]));
  (* Decoded values beyond float16's range, on an [x] small enough that the
     results are inside it: the decode must run at float32. *)
  let large =
    weight ~scale:(fun _ -> 140 + Random.State.int rng 6) [| 3; 4; 32 |]
  in
  let x = Nx.mul_s (floats [| 2; 2; 1; 32 |]) (Float.ldexp 1.0 (-12)) in
  battery (case large (Nx.reshape [| 4; 32 |] x));
  battery (case ~ids:(ints [| 2; 2 |] [| 2; 0; 1; -1 |]) large x);
  (* The forms that decode at float32 for a float16 [x]: grouped blocks past the
     row bound, the grouped transposed product, and a block per position. *)
  battery
    (case
       ~ids:(ints [| 160 |] (Array.init 160 (fun i -> (i * 7 mod 5) - 1)))
       (weight ~scale [| 2; 8; 64 |])
       (floats [| 160; 1; 64 |]));
  battery
    (case ~transpose:true
       ~ids:(ints [| 8; 2 |] (Array.init 16 (fun i -> (i * 3 mod 6) - 1)))
       w
       (poison ~at:[ [ 0; 0 ] ] (floats [| 8; 2; 1; 8 |])));
  battery
    (case
       ~ids:(ints [| 3; 2 |] [| 0; 3; -1; 2; 4; 3 |])
       w
       (floats [| 3; 2; 65; 64 |]))

(* Compiled [dequant] gives the format's values bit for bit at float32 and
   bfloat16, where every value is exact, except that a flushing device zeroes a
   scale byte 0's group and a subnormal value. *)
let test_dequant () =
  let w = weight [| 3; 8; 64 |] in
  let expected = Nx.to_array (values w) in
  let (Nx_quant.Mxfp4 { scales; _ }) = w in
  let bytes = Nx.to_array scales in
  List.iter
    (fun (Dt (name, dt, _, _)) ->
      List.iter
        (fun device ->
          let actual =
            Nx.to_array
              (Nx.cast Nx.float32
                 (Rune.jit
                    ~devices:[ Rune.device device ]
                    Nx.Ptree.(Nx_quant.ptree @-> returns tensor)
                    (Nx_quant.dequant dt) w))
          in
          Array.iteri
            (fun i e ->
              let a = actual.(i) in
              let flushed =
                device = "METAL" && a = 0.0
                && (bytes.(i / 32) = 0 || Float.abs e < Float.ldexp 1.0 (-126))
              in
              if
                not
                  (flushed
                  || Int64.bits_of_float a = Int64.bits_of_float e
                  || (Float.is_nan a && Float.is_nan e))
              then
                fail
                  (Printf.sprintf "%s, %s: at %d, expected %h, got %h" device
                     name i e a))
            expected)
        devices)
    [ float32; bfloat16 ]

(* The form rule: one token's four experts of 32 decode four matrices, as the
   gathered form does, not all 32. Its arithmetic is counted on a replay against
   four experts of four, which every form decodes whole. *)
let test_one_token_gathers () =
  let x = floats [| 1; 1; 65; 256 |] in
  let ops e =
    let w = weight ~scale:(fun _ -> 127) [| e; 64; 256 |] in
    let ids = ints [| 1; 4 |] [| 3; 0; 2; 1 |] in
    let f =
      Rune.jit
        ~devices:[ Rune.device "CPU" ]
        Nx.Ptree.(inputs () @-> returns tensor)
        (fun (ids, x) -> Nx_quant.apply ~ids w x)
    in
    ignore (Nx.to_array (f (ids, x)));
    let before = !Tolk.Helpers.Global_counters.global_ops in
    ignore (Nx.to_array (f (ids, x)));
    !Tolk.Helpers.Global_counters.global_ops - before
  in
  let gathered = ops 32 and every = ops 4 in
  is_true
    ~msg:(Printf.sprintf "%d operations against %d" gathered every)
    (gathered < 2 * every)

(* Rule 1 on every device: within the row bound the product reads its packed
   bytes, under one byte per weight; past it, decoding writes at least two. *)
let test_rule () =
  let n = 256 and k = 256 in
  let w = weight [| n; k |] in
  let bytes device rows =
    let f = Rune.jit' ~devices:[ Rune.device device ] (Nx_quant.apply w) in
    let x = Nx.cast Nx.bfloat16 (floats [| rows; k |]) in
    ignore (Nx.to_array (f x));
    let before = !Tolk.Helpers.Global_counters.global_mem in
    ignore (Nx.to_array (f x));
    !Tolk.Helpers.Global_counters.global_mem - before
  in
  List.iter
    (fun device ->
      let one = bytes device 1 and past = bytes device 65 in
      is_true
        ~msg:(Printf.sprintf "%s, one row: %d bytes" device one)
        (one < n * k);
      is_true
        ~msg:(Printf.sprintf "%s, 65 rows: %d bytes" device past)
        (past > 2 * n * k))
    devices

(* Placement *)

(* A weight placed on a device holds its parts there and decodes to the same
   values. A split along the inputs that cuts a group raises before anything
   moves. (Splits themselves wait for RFC 0005's next stage.) *)
let test_place () =
  let w = weight ~scale:(fun _ -> 120) [| 2; 4; 128 |] in
  let expected = Nx.to_array (Nx_quant.dequant Nx.float32 w) in
  List.iter
    (fun device ->
      let p = Nx.Placement.device (Rune.device device) in
      let (Nx_quant.Mxfp4 { codes; scales } as placed) = Nx_quant.place p w in
      is_true ~msg:(device ^ ", codes")
        (Nx.Placement.equal p (Nx.placement codes));
      is_true ~msg:(device ^ ", scales")
        (Nx.Placement.equal p (Nx.placement scales));
      equal ~msg:device (array float_exact) expected
        (Nx.to_array (Nx_quant.dequant Nx.float32 placed)))
    ("CPU:1" :: List.filter (( = ) "METAL") devices);
  (* 48 code bytes split evenly over two devices; 3 groups do not. *)
  let two = [ Rune.device "CPU:1"; Rune.device "CPU:2" ] in
  raises
    (Invalid_argument
       "Nx_quant.place: splitting codes and scales along axis 1 in 2 cuts a \
        32-value group (3 groups)") (fun () ->
      ignore
        (Nx_quant.place (Nx.Placement.sharded ~axis:1 two) (weight [| 4; 96 |])))

(* Reverse and forward mode *)

(* [dense c] is the product of [c] as an ordinary matmul over [w'] decoded at
   float32, zero where no expert is selected: the function whose derivatives the
   rules must give. *)
let dense c =
  let dq = Nx.cast Nx.float64 (Nx_quant.dequant Nx.float32 c.w) in
  let w, valid =
    match c.ids with
    | None -> (dq, None)
    | Some ids ->
        let w, valid = selected ~lanes:c.lanes dq ids in
        (w, Some valid)
  in
  let w = Nx.cast Nx.float32 w in
  fun x ->
    let y = Nx.matmul x (if c.transpose then w else Nx.matrix_transpose w) in
    match valid with
    | None -> y
    | Some v ->
        let tail = if Nx.ndim x = 1 then [| 1 |] else [| 1; 1 |] in
        Nx.where
          (Nx.broadcast_to (Nx.shape y)
             (Nx.reshape (Array.append (Nx.shape v) tail) v))
          y (Nx.zeros_like y)

(* A fixed, non-uniform cotangent. *)
let weighted y =
  let n = Nx.numel y in
  Nx.sum
    (Nx.mul y
       (Nx.create Nx.float32 (Nx.shape y)
          (Array.init n (fun i -> float_of_int ((i mod 5) + 1) /. 2.0))))

let close ~msg expected actual =
  let expected = Nx.to_array expected and actual = Nx.to_array actual in
  equal ~msg:(msg ^ ", length") int (Array.length expected)
    (Array.length actual);
  let scale =
    Array.fold_left (fun m v -> Float.max m (Float.abs v)) 1e-30 expected
  in
  Array.iteri
    (fun i e ->
      if not (Float.abs (actual.(i) -. e) <= 1e-4 *. scale) then
        fail
          (Printf.sprintf "%s: at %d, expected %g, got %g" msg i e actual.(i)))
    expected

(* Scales near 1, so that a tolerance relative to the largest value holds. *)
let moderate _ = 124 + Random.State.int rng 7

let rule_cases () =
  let weight = weight ~scale:moderate in
  let w = weight [| 4; 8; 64 |] in
  [
    ( "without ids, broadcast",
      case (weight [| 2; 5; 64 |]) (floats [| 4; 1; 3; 64 |]) );
    ("vector", case (weight [| 2; 5; 64 |]) (floats [| 64 |]));
    ( "gathered ids",
      case ~ids:(ints [| 2; 1 |] [| 3; -1 |]) w (floats [| 2; 1; 1; 64 |]) );
    ( "every expert, x broadcast over positions",
      case
        ~ids:(ints [| 3; 2 |] [| 0; 3; -1; 2; 4; 3 |])
        w
        (floats [| 3; 1; 1; 64 |]) );
    ( "grouped",
      case
        ~ids:(ints [| 8; 2 |] (Array.init 16 (fun i -> (i * 5 mod 6) - 1)))
        w
        (floats [| 8; 1; 1; 64 |]) );
    ( "lanes",
      case ~lanes:1
        ~ids:(ints [| 2; 2 |] [| 2; -1; 0; 2 |])
        (weight [| 2; 3; 8; 64 |])
        (floats [| 2; 2; 1; 64 |]) );
  ]

let test_grad () =
  List.iter
    (fun (name, c) ->
      let f x = weighted (product c x) in
      let expected = Rune.grad' (fun x -> weighted (dense c x)) c.x in
      close ~msg:(name ^ ", eager") expected (Rune.grad' f c.x);
      List.iter
        (fun device ->
          close
            ~msg:(name ^ ", " ^ device)
            expected
            (Rune.jit' ~devices:[ Rune.device device ] (Rune.grad' f) c.x))
        devices)
    (rule_cases ())

(* The mixing pattern: a sum over the positions hands the transposed product a
   cotangent broadcast along them, a view whose batch axes do not merge. *)
let test_grad_through_a_sum () =
  let w = weight ~scale:moderate [| 4; 8; 64 |] in
  let x = floats [| 3; 1; 1; 64 |] in
  List.iter
    (fun ids ->
      let c = case ~ids w x in
      let mixed f x = weighted (Nx.sum ~axes:[ 1 ] (f x)) in
      let expected = Rune.grad' (mixed (dense c)) x in
      close ~msg:"eager" expected (Rune.grad' (mixed (product c)) x);
      close ~msg:"compiled" expected
        (Rune.jit'
           ~devices:[ Rune.device "CPU" ]
           (Rune.grad' (mixed (product c)))
           x))
    [
      ints [| 3; 1 |] [| 3; -1; 1 |];
      ints [| 3; 4 |] [| 0; 3; -1; 2; 1; 1; 3; 0; 2; 4; 0; 1 |];
    ]

let test_jvp () =
  List.iter
    (fun (name, c) ->
      let tangent = floats (Nx.shape c.x) in
      let y, dy = Rune.jvp' (product c) c.x tangent in
      let y', dy' = Rune.jvp' (dense c) c.x tangent in
      close ~msg:(name ^ ", primal") y' y;
      close ~msg:(name ^ ", tangent") dy' dy;
      List.iter
        (fun device ->
          close
            ~msg:(name ^ ", tangent, " ^ device)
            dy'
            (Rune.jit'
               ~devices:[ Rune.device device ]
               (fun x -> snd (Rune.jvp' (product c) x tangent))
               c.x))
        devices)
    (rule_cases ())

(* A part computed from a differentiated value is refused, for [apply] and
   [dequant] and in both modes. *)
let test_weight_not_differentiated () =
  let x = floats [| 2; 64 |] in
  let scales = bytes [| 5; 2 |] (fun _ -> 127) in
  let built v = Nx_quant.mxfp4 ~scales (Nx.cast Nx.uint8 v) in
  let v = Nx.full Nx.float32 [| 5; 32 |] 17.0 in
  let refused msg f =
    raises ~msg
      (Invalid_argument
         "Rune: a part of a quantised weight is differentiated; capture the \
          weight, or build it from Rune.detached tensors") (fun () ->
        ignore (f ()))
  in
  refused "grad, apply" (fun () ->
      Rune.grad' (fun v -> Nx.sum (Nx_quant.apply (built v) x)) v);
  refused "grad, dequant" (fun () ->
      Rune.grad' (fun v -> Nx.sum (Nx_quant.dequant Nx.float32 (built v))) v);
  refused "jvp, apply" (fun () ->
      Rune.jvp' (fun v -> Nx_quant.apply (built v) x) v v);
  refused "jvp, dequant" (fun () ->
      Rune.jvp' (fun v -> Nx_quant.dequant Nx.float32 (built v)) v v);
  let detached =
    Rune.grad' (fun v -> Nx.sum (Nx_quant.apply (built (Rune.detach v)) x)) v
  in
  close ~msg:"a detached part" (Nx.zeros Nx.float32 [| 5; 32 |]) detached

(* vmap *)

(* [per_lane f n] is [f 0 ... f (n - 1)] stacked. *)
let per_lane f n = Nx.stack ~axis:0 (List.init n f)
let row i t = Nx.slice [ I i ] t

let test_vmap () =
  let weight = weight ~scale:moderate in
  let w = weight [| 4; 8; 64 |] in
  let xs = floats [| 3; 2; 1; 1; 64 |] in
  let ids = ints [| 3; 2; 2 |] [| 0; 3; -1; 2; 1; 1; 3; 0; 2; 4; 0; 1 |] in
  close ~msg:"over x"
    (per_lane (fun i -> Nx_quant.apply ~ids:(row 0 ids) w (row i xs)) 3)
    (Rune.vmap' (fun x -> Nx_quant.apply ~ids:(row 0 ids) w x) xs);
  let vs = floats [| 3; 64 |] in
  close ~msg:"over x, a vector"
    (per_lane (fun i -> Nx_quant.apply w (row i vs)) 3)
    (Rune.vmap' (Nx_quant.apply w) vs);
  let xt = Nx.transpose ~axes:[ 1; 0; 2; 3; 4 ] (floats [| 3; 3; 1; 1; 64 |]) in
  List.iter
    (fun ids ->
      close ~msg:"over x's axis 1"
        (per_lane
           (fun i -> Nx_quant.apply ~ids w (Nx.slice [ A; I i ] xt))
           (Nx.dim 1 xt))
        (Rune.vmap' (Nx_quant.apply ~ids w) (Nx.moveaxis 1 0 xt)))
    [
      ints [| 3; 1 |] [| 0; 3; -1 |];
      ints [| 3; 4 |] [| 0; 3; -1; 2; 1; 1; 3; 0; 2; 4; 0; 1 |];
    ];
  let routed (ids, x) = Nx_quant.apply ~ids w x in
  close ~msg:"over ids and x"
    (per_lane (fun i -> routed (row i ids, row i xs)) 3)
    (Rune.vmap Nx.Ptree.(inputs () @-> returns tensor) routed (ids, xs));
  close ~msg:"over ids and x, compiled"
    (per_lane (fun i -> routed (row i ids, row i xs)) 3)
    (Rune.jit
       Nx.Ptree.(inputs () @-> returns tensor)
       (Rune.vmap Nx.Ptree.(inputs () @-> returns tensor) routed)
       (ids, xs));
  let ws = weight [| 3; 4; 8; 64 |] in
  let lane i = Nx.Ptree.map Nx_quant.ptree (fun _ t -> row i t) ws in
  let x = row 0 xs and one = row 0 ids in
  close ~msg:"over the weight"
    (per_lane (fun i -> Nx_quant.apply ~ids:one (lane i) x) 3)
    (Rune.vmap
       Nx.Ptree.(Nx_quant.ptree @-> returns tensor)
       (fun w -> Nx_quant.apply ~ids:one w x)
       ws);
  let (Nx_quant.Mxfp4 { codes; scales }) = ws in
  let scales = row 0 scales in
  let with_codes i = Nx_quant.mxfp4 ~scales (row i codes) in
  let one_part codes =
    Nx_quant.apply ~ids:one (Nx_quant.mxfp4 ~scales codes) x
  in
  close ~msg:"over the codes only"
    (per_lane (fun i -> Nx_quant.apply ~ids:one (with_codes i) x) 3)
    (Rune.vmap' one_part codes);
  close ~msg:"over the codes only, compiled"
    (per_lane (fun i -> Nx_quant.apply ~ids:one (with_codes i) x) 3)
    (Rune.jit' (Rune.vmap' one_part) codes);
  close ~msg:"over the weight, dequant"
    (per_lane (fun i -> Nx_quant.dequant Nx.float32 (lane i)) 3)
    (Rune.vmap
       Nx.Ptree.(Nx_quant.ptree @-> returns tensor)
       (Nx_quant.dequant Nx.float32)
       ws);
  close ~msg:"over the weight, compiled"
    (per_lane (fun i -> Nx_quant.apply ~ids:one (lane i) x) 3)
    (Rune.jit
       Nx.Ptree.(Nx_quant.ptree @-> returns tensor)
       (Rune.vmap
          Nx.Ptree.(Nx_quant.ptree @-> returns tensor)
          (fun w -> Nx_quant.apply ~ids:one w x))
       ws)

(* pmap: a program over several devices multiplies the blocks by their gathered
   matrices and groups no routes. *)
let test_pmap () =
  let w = weight ~scale:moderate [| 4; 8; 64 |] in
  let routed (ids, x) = Nx_quant.apply ~ids w x in
  List.iter
    (fun (msg, ids, x) ->
      close ~msg
        (routed (ids, x))
        (Rune.pmap
           ~devices:[ Rune.device "CPU:1"; Rune.device "CPU:2" ]
           Nx.Ptree.(inputs () @-> returns tensor)
           routed (ids, x)))
    [
      ("gathered", ints [| 2; 1 |] [| 3; -1 |], floats [| 2; 1; 1; 64 |]);
      ( "several rows per position",
        ints [| 2; 2 |] [| 3; -1; 0; 1 |],
        floats [| 2; 2; 2; 64 |] );
      ( "many routes",
        ints [| 8; 2 |] (Array.init 16 (fun i -> (i * 5 mod 6) - 1)),
        floats [| 8; 2; 1; 64 |] );
    ]

(* The form each product takes, as [RUNE_JIT_DEBUG=1] logs it, in a child
   process that reads the variable fresh. On Metal, 16 routes over 4 experts
   group, forward and transposed; 40 routes over 40 experts do not, as each
   block would be one row. The CPU groups none of them (τ = 1024), and pmap
   takes the dense form. *)
let form_role = "RUNE_QUANT_FORM_ROLE"

let form_cases () =
  let device = if List.mem "METAL" devices then "METAL" else "CPU" in
  let w = weight ~scale:moderate [| 4; 8; 64 |] in
  let ids = ints [| 8; 2 |] (Array.init 16 (fun i -> i mod 4)) in
  let wide = weight ~scale:moderate [| 40; 8; 64 |] in
  let distinct = ints [| 40 |] (Array.init 40 Fun.id) in
  let apply ?(transpose = false) w ids x =
    Rune.jit'
      ~devices:[ Rune.device device ]
      (fun x ->
        Nx_quant.Effect.perform w (Apply { ids = Some ids; x; transpose }))
      x
  in
  [
    ("sixteen routes", fun () -> apply w ids (floats [| 8; 1; 1; 64 |]));
    ( "sixteen routes, transposed",
      fun () -> apply ~transpose:true w ids (floats [| 8; 2; 1; 8 |]) );
    ( "one route per expert",
      fun () -> apply wide distinct (floats [| 40; 1; 64 |]) );
    ( "sixteen routes under pmap",
      fun () ->
        Rune.pmap
          ~devices:[ Rune.device "CPU:1"; Rune.device "CPU:2" ]
          Nx.Ptree.(inputs () @-> returns tensor)
          (fun (ids, x) -> Nx_quant.apply ~ids w x)
          (ids, floats [| 8; 2; 1; 64 |]) );
  ]

let run_form_role () =
  List.iter
    (fun (name, f) ->
      Printf.eprintf "case %s\n%!" name;
      ignore (Nx.to_array (f ())))
    (form_cases ());
  exit 0

let drain fd =
  let buf = Buffer.create 256 and chunk = Bytes.create 4096 in
  let rec loop () =
    let n = Unix.read fd chunk 0 (Bytes.length chunk) in
    if n > 0 then begin
      Buffer.add_subbytes buf chunk 0 n;
      loop ()
    end
  in
  loop ();
  Unix.close fd;
  Buffer.contents buf

let test_forms () =
  let env =
    Array.append
      (Array.of_list
         (List.filter
            (fun b ->
              not
                (String.starts_with ~prefix:"RUNE_JIT_DEBUG=" b
                || String.starts_with ~prefix:(form_role ^ "=") b))
            (Array.to_list (Unix.environment ()))))
      [| "RUNE_JIT_DEBUG=1"; form_role ^ "=1" |]
  in
  let err_read, err_write = Unix.pipe ~cloexec:false () in
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process_env exe [| exe |] env Unix.stdin Unix.stdout err_write
  in
  Unix.close err_write;
  let lines = String.split_on_char '\n' (drain err_read) in
  (match Unix.waitpid [] pid with
  | _, Unix.WEXITED 0 -> ()
  | _ -> fail ("child failed:\n" ^ String.concat "\n" lines));
  let forms = Hashtbl.create 4 and case = ref "" in
  let prefix = "rune.jit: quantised product: " in
  List.iter
    (fun line ->
      if String.starts_with ~prefix:"case " line then
        case := String.sub line 5 (String.length line - 5)
      else if String.starts_with ~prefix line then
        Hashtbl.add forms !case
          (String.sub line (String.length prefix)
             (String.length line - String.length prefix)))
    lines;
  let grouped name =
    List.exists
      (String.starts_with ~prefix:"grouped")
      (Hashtbl.find_all forms name)
  in
  let metal = List.mem "METAL" devices in
  equal ~msg:"sixteen routes group on Metal" bool metal
    (grouped "sixteen routes");
  equal ~msg:"transposed, they group on Metal" bool metal
    (grouped "sixteen routes, transposed");
  is_false ~msg:"one route per expert does not group"
    (grouped "one route per expert");
  equal ~msg:"pmap takes the dense form" (list string) [ "dense" ]
    (Hashtbl.find_all forms "sixteen routes under pmap")

(* debug *)

let test_debug () =
  let w = weight ~scale:moderate [| 4; 8; 64 |] in
  let ids = ints [| 3; 2 |] [| 0; 3; -1; 2; 1; 1 |] in
  let x = floats [| 3; 1; 1; 64 |] in
  let first = Nx.Ptree.map Nx_quant.ptree (fun _ t -> row 0 t) w in
  let g = floats [| 3; 2; 1; 8 |] in
  let f () =
    ( Nx_quant.apply ~ids w x,
      Nx_quant.dequant Nx.float32 first,
      Nx_quant.Effect.perform w
        (Apply { ids = Some ids; x = g; transpose = true }) )
  in
  let buf = Buffer.create 64 in
  let ppf = Format.formatter_of_buffer buf in
  let y, d, t = Rune.with_debug ~ppf f in
  Format.pp_print_flush ppf ();
  let log = String.split_on_char '\n' (Buffer.contents buf) in
  is_true ~msg:"apply is logged" (List.mem "quant_apply -> [3,2,1,8]" log);
  is_true ~msg:"dequant is logged" (List.mem "quant_dequant -> [8,64]" log);
  is_true ~msg:"the transposed product is logged"
    (List.mem "quant_apply_transposed -> [3,2,1,64]" log);
  let y', d', t' = f () in
  close ~msg:"apply's result" y' y;
  close ~msg:"dequant's result" d' d;
  close ~msg:"the transposed product's result" t' t

let () =
  if Sys.getenv_opt form_role <> None then run_form_role ();
  run "rune quant"
    [
      group "Law 2"
        [
          slow "without ids" test_without_ids;
          slow "fewer positions than experts" test_gathered;
          slow "as many positions as experts or more" test_instances;
          slow "grouped" test_grouped;
          slow "transposed" test_transposed;
          slow "the largest scales" test_large_scales;
          test "empty" test_empty;
          slow "float16 x" test_float16;
          test "compiled dequant" test_dequant;
          test "one token's experts are gathered" test_one_token_gathers;
          slow "the kernel" test_kernel;
          slow "past the row bound" test_past_the_bound;
          test "the row bound chooses the kernel" test_rule;
          test "grouping chooses its routes" test_forms;
        ];
      group "placement"
        [ test "place splits at the format's blocks" test_place ];
      group "rules"
        [
          slow "grad with respect to x" test_grad;
          test "grad through a sum over the positions" test_grad_through_a_sum;
          test "jvp" test_jvp;
          test "the weight is never differentiated"
            test_weight_not_differentiated;
          test "vmap" test_vmap;
          test "pmap" test_pmap;
          test "debug" test_debug;
        ];
    ]
