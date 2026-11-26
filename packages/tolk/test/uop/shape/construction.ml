(* Construction, movement, storage, variables, divisibility, calls, launches and
   programs. *)

open Windtrap
open Tolk
open Common

let var = Simplification.var
let cpu = "CPU"

let param ?device shape slot dt =
  Shape.param ?device ~shape:(ints shape) slot dt

let range lo hi : (Ops.sint * Ops.sint) option = Some (Int lo, Int hi)
let target = Testable.make ~pp:Helpers.Target.pp ~equal:( = )

(* The nodes of [candidates] that [p] holds of, in order. *)
let holding p candidates = List.filter p candidates

(* The goldens of [names], each a graph of gen/uop/shape.py built by the
   constructors, as the group that checks the section they build. *)
let goldens names =
  group "builds each recorded graph as its golden states"
    (List.map
       (fun name ->
         let _, build, minted =
           List.find (fun (n, _, _) -> String.equal n name) Goldens.all
         in
         Golden.graph (name ^ ".golden") (fun () ->
             if minted then Goldens.minted (build ()) else build ()))
       names)

(* Construction *)

let construction =
  group "construction"
    [
      goldens [ "stacks"; "reductions"; "constants_like"; "contraction" ];
      test "consts takes the committed type of its literals" (fun () ->
          equal (list dtype)
            [ Int32; Int64; Float32; Bool ]
            (List.map
               (fun cs -> Ops.dtype (Shape.consts cs))
               [
                 [ i 1; i 2 ];
                 [ i 1; `Int (Bigint.shift_left Bigint.one 40) ];
                 [ i 1; f 2.5 ];
                 [ `Bool true; `Bool false ];
               ]));
      test "const_like is the constant in u's type, expanded to u's shape"
        (fun () ->
          let p = param [ 2; 3 ] 0 Float16 in
          let c = Shape.const_like p (i 1) in
          equal dtype Float16 (Ops.dtype c);
          equal shape (ints [ 2; 3 ]) (Shape.shape c);
          equal uop (Ops.int ~dtype:Float16 1) (Ops.base c));
      test "stack raises for no nodes and for nodes of different shapes"
        (fun () ->
          rejects (fun () -> Shape.stack []);
          rejects (fun () ->
              Shape.stack [ param [ 2 ] 0 Float32; param [ 3 ] 1 Float32 ]));
      test "rop raises for an axis given twice" (fun () ->
          let p = param [ 2; 1; 4 ] 0 Float32 in
          rejects (fun () -> Shape.rop p Op.Add [ 2; 2 ]);
          rejects (fun () -> Shape.rop p Op.Add [ 1; 1 ]));
      test "valid is u where cond holds and invalid elsewhere" (fun () ->
          let a = var "a" 0 8 in
          let u = Shape.valid Ops.O.(a * int 2) Ops.O.(a < int 4) in
          equal (list Dtypes.const)
            [ i 6; `Invalid ]
            (List.map
               (fun x -> Interpreter.eval ~vars:[ ("a", i x) ] u)
               [ 3; 4 ]));
      test "contract raises for a range that is not upcast" (fun () ->
          let r = Ops.range ~axis_type:Weak (Int 2) [ 0 ] in
          rejects (fun () -> Shape.contract Ops.O.(r + int 1) [ r ]));
    ]

(* Movement *)

(* Up to three axes of up to four elements, some of them named in either
   order. *)
let gen_flip =
  Gen.(
    let* dims = list ~size:(int_range 1 3) (int_range 0 4) in
    let+ flags = list ~size:(constant (List.length dims)) bool
    and+ backwards = bool in
    let axes =
      List.filter_map Fun.id
        (List.mapi (fun a f -> if f then Some a else None) flags)
    in
    (dims, if backwards then List.rev axes else axes))

(* The row-major position in [dims] of the element that output position [i]
   holds once [axes] are reversed. *)
let flipped dims axes i =
  let rank = List.length dims in
  let rec coords a i acc =
    if a < 0 then acc
    else
      let d = List.nth dims a in
      let j = if List.mem a axes then d - 1 - (i mod d) else i mod d in
      coords (a - 1) (i / d) (j :: acc)
  in
  List.fold_left2 (fun acc j d -> (acc * d) + j) 0 (coords (rank - 1) i []) dims

let flips_each_axis (dims, axes) =
  let n = List.fold_left ( * ) 1 dims in
  let flips_long = List.exists (fun a -> List.nth dims a > 1) axes in
  cover "an empty axis beside a flipped one" (List.mem 0 dims && flips_long);
  cover "an axis of one element is flipped"
    (List.exists (fun a -> List.nth dims a = 1) axes);
  cover "two axes are flipped" (List.length axes >= 2);
  let x = Shape.reshape (param [ n ] 1 Int32) (ints dims) in
  let memory : Dtype.value array =
    Array.init n (fun j -> `Int (Bigint.of_int j))
  in
  equal (list Dtypes.const)
    (List.init n (fun i -> (memory.(flipped dims axes i) :> Dtype.const)))
    (Array.to_list (Tensors.eval ~buffers:[ (1, memory) ] (Shape.flip x axes)))

(* A running operation over a vector of [values] of [dtype], and the length of a
   prefix of it. *)
type running = {
  dtype : Dtype.t;
  op : Op.t;
  values : Dtype.value list;
  k : int;
}

let pp_running ppf r =
  Format.fprintf ppf "%a %a over [%a], prefix %d" Dtype.pp r.dtype Op.pp r.op
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       (Testable.pp Dtypes.value))
    r.values r.k

(* Values of [dt] that break a running operation: zeros of both signs,
   infinities, a subnormal, NaN and a type's extremes. A maximum draws no NaN,
   whose result depends on its operands' order. *)
let edge_values op dt : Dtype.value list =
  let subnormal =
    match dt with
    | Dtype.Float16 -> Float.ldexp 1. (-24)
    | Float32 -> Float.ldexp 1. (-149)
    | _ -> Float.ldexp 1. (-1074)
  in
  if Dtype.is_float dt then
    List.map
      (fun x -> `Float x)
      ([ 0.; -0.; 1.; -1.; 0.5; 3.; Float.infinity; Float.neg_infinity ]
      @ [ subnormal ]
      @ if op = Op.Max then [] else [ Float.nan ])
  else
    let lo, hi = Dtypes.int_bounds dt in
    List.map
      (fun z -> `Int z)
      ([ lo; hi; Bigint.zero; Bigint.one; Bigint.of_int 2; Bigint.of_int 3 ]
      @ if Dtype.is_unsigned dt then [] else [ Bigint.minus_one ])

(* Lengths about one and two chunks of 16. *)
let gen_running =
  let open Gen in
  let* dtype =
    of_list ~pp:Dtype.pp Dtype.[ Float16; Float32; Float64; Int8; Uint8; Int32 ]
  in
  let* op = of_list ~pp:Op.pp Op.[ Add; Mul; Max ] in
  let* n =
    frequency [ (1, int_range 1 40); (1, of_list [ 15; 16; 17; 31; 32; 33 ]) ]
  in
  let* values =
    list ~size:(constant n)
      (of_list ~pp:(Testable.pp Dtypes.value) (edge_values op dtype))
  in
  let+ k = int_range 1 n in
  { dtype; op; values; k }

(* Element values compared bit for bit, but NaNs by class and zeros of a sum by
   value. *)
let running_value op =
  Testable.make ~pp:(Testable.pp Dtypes.value)
    ~equal:(fun (x : Dtype.value) y ->
      match (x, y) with
      | `Float a, `Float b ->
          Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
          || (Float.is_nan a && Float.is_nan b)
          || (op = Op.Add && a = 0. && b = 0.)
      | _ -> x = y)

let prefix_is_running r =
  let n = List.length r.values in
  cover "a prefix past one chunk of 16" (r.k > 16);
  cover "a shorter prefix" (r.k < n);
  cover "a NaN"
    (List.exists (function `Float x -> Float.is_nan x | _ -> false) r.values);
  let u = param [ n ] 1 r.dtype in
  let memory = Array.of_list r.values in
  let eval u =
    List.map
      (fun (c : Dtype.const) ->
        match c with #Dtype.value as v -> v | `Invalid -> fail "invalid")
      (Array.to_list (Tensors.eval ~buffers:[ (1, memory) ] u))
  in
  let whole = eval (Shape.cumalu u 0 r.op) in
  let prefix = Shape.cumalu (Shape.shrink u [ range 0 r.k ]) 0 r.op in
  equal
    (list (running_value r.op))
    (List.filteri (fun j _ -> j < r.k) whole)
    (eval prefix)

(* Aranges of any length *)

(* The values of [u]'s elements [lo] to [lo + k - 1]: what the one kernel that
   stores them into storage of [k] elements writes. The two assertions on the
   schedule are the law's precondition, not its claim: an arange of billions of
   elements is read only within that one kernel and [k] elements of storage, and
   a schedule that would store more fails here before it is run. *)
let window u lo k =
  let out = Ops.new_buffer ~slot:0 cpu k (Ops.dtype u) in
  let view = Shape.shrink u [ range lo (lo + k) ] in
  let sink = Ops.sink [ Ops.after out [ Ops.store out view ] ] in
  let linear, _ = Schedule.create_linear_with_vars ~cores:Cores.host sink in
  equal ~msg:"kernels" int 1 (List.length (Ops.src linear));
  List.iter
    (fun b ->
      if Ops.op b = Op.Buffer then
        at_most ~msg:"stored elements" int ~than:k (Shape.max_numel b))
    (Ops.toposort ~calls:Skip linear);
  List.map (fun (_, _, v) -> v) (Kernel_graphs.linear_writes linear)

(* Short lengths, lengths about the powers of two where an element's index or
   value leaves 32 bits, and past them up to the longest. *)
let long_lengths =
  [
    3;
    1000;
    (1 lsl 30) - 1;
    1 lsl 30;
    (1 lsl 30) + 1;
    1518500249;
    1518500250;
    (1 lsl 31) - 1;
    1 lsl 31;
    (1 lsl 31) + 1;
    (1 lsl 32) + 3;
    (1 lsl 46) + 5;
    (1 lsl 60) + 7;
    max_int - (1 lsl 30);
  ]

(* The longest arange that is a running sum. *)
let pooled_run = 1518500249

(* The integer [v] in [dt]: rounded once to a float type, whose double holds it
   exactly below [2^53]. *)
let element dt v : Dtype.value =
  match dt with
  | Dtype.Float32 ->
      `Float (Int32.float_of_bits (Int32.bits_of_float (Float.of_int v)))
  | Float64 | Float16 | Bfloat16 -> `Float (Float.of_int v)
  | _ -> `Int (Bigint.of_int v)

(* An arange of [n] elements from [start] by [step], the type it is asked for or
   [None], and the first of three elements read. An integer type, when given,
   holds the elements, and a float type is given where it holds them exactly or
   the arange is laid out in rows, its elements below [2^53]; the weak integer
   holds them all. Past [2^46] elements, a step of [1] keeps the arange within
   an int. *)
let gen_long_arange =
  let open Gen in
  let pp_int = Format.pp_print_int in
  let* n = of_list ~pp:pp_int long_lengths in
  let* step =
    of_list ~pp:pp_int (if n > 1 lsl 46 then [ 1; -1 ] else [ 1; -1; 3; -7 ])
  in
  let span = (n - 1) * abs step in
  let* start =
    of_list ~pp:pp_int
      (if step > 0 then [ 0; -5; -(span / 2); min_int ]
       else [ span; span + 9; max_int ])
  in
  let last = start + ((n - 1) * step) in
  let lo' = Stdlib.min start last and hi' = Stdlib.max start last in
  let below m = -m <= lo' && hi' <= m in
  let holds dt =
    match dt with
    | Dtype.Float32 ->
        below (1 lsl 24) || (n > pooled_run && below ((1 lsl 53) - 1))
    | Float64 -> below (1 lsl 53)
    | Float16 -> below (1 lsl 11)
    | Bfloat16 -> below (1 lsl 8)
    | Weak_int -> true
    | _ ->
        let lo, hi = Dtypes.int_bounds dt in
        Bigint.(leq lo (of_int lo') && leq (of_int hi') hi)
  in
  let pp_type ppf = function
    | None -> Format.pp_print_string ppf "default"
    | Some dt -> Dtype.pp ppf dt
  in
  let* dtype =
    of_list ~pp:pp_type
      (None
      :: List.filter_map
           (fun dt -> if holds dt then Some (Some dt) else None)
           Dtype.
             [
               Int8;
               Uint8;
               Int16;
               Uint16;
               Int32;
               Uint32;
               Int64;
               Uint64;
               Weak_int;
               Float16;
               Bfloat16;
               Float32;
               Float64;
             ])
  in
  let+ first =
    of_list ~pp:pp_int
      (List.map
         (min (n - 3))
         [ 0; n - 3; (1 lsl 31) - 2; (1 lsl 32) - 2; n / 2 ])
  in
  (n, start, step, dtype, first)

let arange_elements (n, start, step, dtype, first) =
  cover "a length past 2^30.5" (n > pooled_run);
  cover "a length past 2^31" (n > 1 lsl 31);
  cover "a length past 2^32" (n > 1 lsl 32);
  cover "a descending arange" (step < 0);
  cover "a start at an int's bounds" (start = min_int || start = max_int);
  cover "an unsigned type"
    (Option.fold ~none:false ~some:Dtype.is_unsigned dtype);
  cover "a float type past 2^30.5"
    (n > pooled_run && Option.fold ~none:false ~some:Dtype.is_float dtype);
  cover "elements across 2^31" (first < 1 lsl 31 && first + 3 > 1 lsl 31);
  cover "the weak integer" (dtype = Some Dtype.Weak_int);
  let u = Shape.arange ~start ~step ?dtype (start + (n * step)) in
  equal shape (ints [ n ]) (Shape.shape u);
  (* A weak arange is stored at the type that holds an int. *)
  let u = if dtype = Some Weak_int then Ops.cast u Int64 else u in
  equal (list value)
    (List.init 3 (fun j -> element (Ops.dtype u) (start + ((first + j) * step))))
    (window u first 3)

let movement =
  let p = param [ 2; 3; 4 ] 0 Float32 in
  group "movement"
    [
      goldens
        [
          "movement";
          "expansion";
          "squeezes";
          "axes";
          "concatenation";
          "pools";
          "running";
          "aranges";
          "padding";
        ];
      test "marg is each movement's argument, and raises for another node"
        (fun () ->
          (match Shape.marg (Shape.permute p [ 2; 0; 1 ]) with
          | Permute o -> equal (list int) [ 2; 0; 1 ] o
          | _ -> fail "permute");
          (match Shape.marg (Shape.shrink p [ range 1 2; None; None ]) with
          | Shrink l ->
              equal
                (list (pair sint sint))
                [ (Int 1, Int 1); (Int 0, Int 3); (Int 0, Int 4) ]
                l
          | _ -> fail "shrink");
          (match Shape.marg (Shape.pad p [ range 1 2; None; None ]) with
          | Pad l ->
              equal
                (list (pair sint sint))
                [ (Int 1, Int 5); (Int 0, Int 3); (Int 0, Int 4) ]
                l
          | _ -> fail "pad");
          rejects (fun () -> Shape.marg (Ops.int 1)));
      (let n = var "n" 1 4 in
       cases
         "as_shape is a constant, a stack's sources or the node, each \
          simplified"
         ~id:(fun (name, _, _) -> name)
         [
           ("a constant", Ops.int 3, ints [ 3 ]);
           ("a boolean constant", Ops.bool true, ints [ 1 ]);
           ( "a stack of constants",
             Ops.v ~src:[ Ops.int 6; Ops.int 4 ] Op.Stack,
             ints [ 6; 4 ] );
           ( "a stack of a constant and sums",
             Ops.v
               ~src:[ Ops.int 2; Ops.O.(n + int 0); Ops.O.(int 2 + int 3) ]
               Op.Stack,
             [ Int 2; Sym n; Int 5 ] );
           ("a variable", n, [ Sym n ]);
           ("a sum of constants", Ops.O.(int 2 + int 3), ints [ 5 ]);
         ]
         (fun (_, u, expected) -> equal shape expected (Shape.as_shape u)));
      test "as_shape raises for a constant that is no size" (fun () ->
          let past_int = `Int (Bigint.shift_left Bigint.one 70) in
          let no_size sub u =
            raises_match (Exn.invalid_arg ~sub) (fun () -> Shape.as_shape u)
          in
          no_size "is not a size" (Ops.float 1.5);
          no_size "is not a size" (Ops.const past_int);
          no_size "is not a size"
            (Ops.v ~src:[ Ops.int 2; Ops.float 1.5 ] Op.Stack));
      test "mop stores a shape argument as sources, which marg reads back"
        (fun () ->
          let r = Shape.mop p (Reshape (ints [ 6; 4 ])) in
          equal uops
            [ p; Ops.v ~src:[ Ops.int 6; Ops.int 4 ] Op.Stack ]
            (Ops.src r);
          equal shape
            (ints [ 6; 4 ])
            (match Shape.marg r with Reshape s -> s | _ -> fail "a reshape"));
      test
        "mop of an empty pad or shrink is a scalar itself, and raises for a \
         node with axes" (fun () ->
          let s = Ops.int 1 in
          equal uops [ s; s ] [ Shape.mop s (Pad []); Shape.mop s (Shrink []) ];
          rejects (fun () -> Shape.mop p (Pad []));
          rejects (fun () -> Shape.mop p (Shrink [])));
      test "reshape infers a -1, and raises for two" (fun () ->
          equal shape
            (ints [ 4; 6 ])
            (Shape.shape (Shape.reshape p (ints [ 4; -1 ])));
          rejects (fun () -> Shape.reshape p (ints [ -1; -1 ])));
      test "reshape raises for a different number of elements" (fun () ->
          rejects (fun () -> Shape.reshape p (ints [ 5; 5 ])));
      test "permute raises for an order that is not a permutation" (fun () ->
          rejects (fun () -> Shape.permute p [ 0; 0; 1 ]);
          rejects (fun () -> Shape.permute p [ 0; 1 ]);
          rejects (fun () -> Shape.permute p [ 0; 1; 3 ]));
      prop "flip reverses each axis it names and keeps the others"
        ~examples:[ ([ 0; 2 ], [ 1 ]); ([ 1; 3 ], [ 0 ]) ]
        gen_flip flips_each_axis;
      test "flip reverses a view of variable size at each size" (fun () ->
          let n = var "n" 0 4 in
          let x = param [ 4 ] 1 Int32 in
          let flipped =
            Shape.flip (Shape.shrink x [ Some (Int 0, Sym n) ]) [ 0 ]
          in
          let memory : Dtype.value array =
            Array.init 4 (fun j -> `Int (Bigint.of_int j))
          in
          List.iter
            (fun k ->
              let bound =
                Ops.substitute ~calls:Skip ~pass:Fixed_point flipped
                  [ (n, Ops.int k) ]
              in
              equal ~msg:(string_of_int k) (list Dtypes.const)
                (List.init k (fun i -> `Int (Bigint.of_int (k - 1 - i))))
                (Array.to_list (Tensors.eval ~buffers:[ (1, memory) ] bound)))
            [ 0; 1; 2; 3; 4 ]);
      test "flip is u for no axes" (fun () ->
          let s = Shape.param 1 Float32 in
          equal uops [ s; p ] [ Shape.flip s []; Shape.flip p [] ]);
      test "flip raises for an axis given twice, and for an axis of a scalar"
        (fun () ->
          rejects (fun () -> Shape.flip p [ 0; 0 ]);
          rejects (fun () -> Shape.flip (Shape.param 1 Float32) [ 0 ]));
      test
        "shrink, shrink_to, pad and pad_to raise for a list of another length"
        (fun () ->
          rejects (fun () -> Shape.shrink p [ None ]);
          rejects (fun () -> Shape.shrink_to p [ None ]);
          rejects (fun () -> Shape.pad p [ None; None ]);
          rejects (fun () -> Shape.pad_to p [ None ]));
      test "pad_to to u's shape is u, whatever the value" (fun () ->
          equal uop p
            (Shape.pad_to ~value:(f 1.) p [ None; Some (Int 3); None ]));
      test
        "flatten, squeeze, unsqueeze and transpose raise for an axis out of \
         range" (fun () ->
          let p = param [ 2; 1 ] 0 Float32 in
          rejects (fun () -> Shape.squeeze ~axis:2 p);
          rejects (fun () -> Shape.squeeze ~axis:(-3) p);
          rejects (fun () -> Shape.flatten ~start:5 p);
          rejects (fun () -> Shape.flatten ~stop:5 p);
          rejects (fun () -> Shape.unsqueeze p 3);
          rejects (fun () -> Shape.transpose p 0 2));
      test "squeeze leaves an axis of another size" (fun () ->
          equal uop p (Shape.squeeze ~axis:1 p));
      test "split raises for sizes that do not sum to the axis's" (fun () ->
          let p = param [ 2; 3 ] 0 Float32 in
          rejects (fun () -> Shape.split ~axis:1 p [ 1; 1 ]);
          rejects (fun () -> Shape.split p [ 1; 2 ]));
      test "repeat raises for a negative repeat" (fun () ->
          rejects (fun () -> Shape.repeat (param [ 1 ] 1 Float32) [ -1 ]));
      test "cat raises for shapes that differ off its axis" (fun () ->
          rejects (fun () -> Shape.cat p [ param [ 2; 4; 4 ] 1 Float32 ]));
      prop
        "cumalu of a prefix is the prefix of cumalu, bit for bit, but a zero \
         sum's sign and a NaN's payload"
        ~examples:
          [
            {
              dtype = Dtype.Float32;
              op = Op.Add;
              values = List.init 300 (fun j -> `Float (Float.of_int (j - 150)));
              k = 257;
            };
          ]
        (Gen.with_pp pp_running gen_running)
        prefix_is_running;
      test "cumalu past 512 elements sums each prefix" (fun () ->
          let sums = Shape.cumalu (Shape.arange ~dtype:Int32 600) 0 Op.Add in
          equal (list int)
            (List.init 600 (fun i -> i * (i + 1) / 2))
            (List.map
               (function `Int z -> Bigint.to_int z | _ -> -1)
               (Array.to_list (Tensors.eval sums))));
      test "cumalu is u if an axis of u is empty" (fun () ->
          let e = param [ 3; 0 ] 0 Float32 in
          equal uops [ e; e ]
            [ Shape.cumalu e 0 Op.Add; Shape.cumalu e 1 Op.Max ]);
      test "cumalu raises for an axis of symbolic size" (fun () ->
          let n = var "n" 1 4 in
          let x =
            Shape.shrink (param [ 4 ] 0 Float32) [ Some (Int 0, Sym n) ]
          in
          rejects (fun () -> Shape.cumalu x 0 Op.Add));
      prop "arange's element i is start + i * step at any length" ~count:200
        ~examples:
          [
            (1518500250, 0, 1, Some Dtype.Float32, 0);
            ((1 lsl 31) + 1, 0, 1, None, (1 lsl 31) - 2);
          ]
        gen_long_arange arange_elements;
      test "arange counts its elements past an int's range" (fun () ->
          let step = 1 lsl 60 in
          let u = Shape.arange ~start:min_int ~step max_int in
          let at k =
            Bigint.(add (of_int min_int) (mul (of_int k) (of_int step)))
          in
          equal (list Dtypes.const)
            (List.init 8 (fun k -> `Int (at k)))
            (Array.to_list (Tensors.eval u));
          let u = Shape.arange ~dtype:Int8 ~start:(-100) ~step:max_int (-50) in
          equal (list Dtypes.const)
            [ i (-100) ]
            (Array.to_list (Tensors.eval u)));
      test "arange's type holds its last element, past [stop - step]" (fun () ->
          rejects (fun () -> Shape.arange ~dtype:Int8 ~step:2 129);
          let u = Shape.arange ~step:2 ((1 lsl 31) + 1) in
          equal (list value)
            [ `Int (Bigint.of_int (1 lsl 31)) ]
            (window u (1 lsl 30) 1));
      test "arange raises for a step of 0 and a type that does not hold it"
        (fun () ->
          rejects (fun () -> Shape.arange ~step:0 4);
          rejects (fun () -> Shape.arange ~dtype:Int8 200));
      test "arange raises for more than max_int - 2^30 elements" (fun () ->
          rejects (fun () -> Shape.arange (max_int - (1 lsl 30) + 1));
          rejects (fun () -> Shape.arange ~start:(-(1 lsl 61)) (1 lsl 61)));
      test "nbytes is the number of elements times their size" (fun () ->
          equal (list int) [ 96; 2 ]
            [ Shape.nbytes p; Shape.nbytes (Shape.param 1 Float16) ]);
    ]

(* Storage *)

(* A buffer of 16 float32s whose first element lies 4 bytes past a boundary, and
   views of it. *)
let floats = Ops.new_buffer ~slot:1 ~phase:4 cpu 16 Float32
let square = Shape.reshape floats (ints [ 4; 4 ])

(* A start [k * every], with [k] from 0 to [hi]. *)
let multiple every hi =
  let k = Ops.variable "k" (i 0) (i hi) in
  if every = 1 then k else Ops.O.(k * int every)

let at_start ?(every = 1) u =
  let k = multiple every (8 / every) in
  Shape.shrink u [ Some (Sym k, Sym Ops.O.(k + int 4)) ]

let storage_phases =
  let doubles = Ops.new_buffer ~slot:2 cpu 8 Float64 in
  let n = var "n" 1 8 in
  [
    ("a buffer's own", floats, (16, 4));
    ( "a shrink moves it by the bytes it skips",
      Shape.shrink floats [ range 1 5 ],
      (16, 8) );
    ( "a shrink of rows, in row-major order",
      Shape.shrink square [ range 1 3; range 1 3 ],
      (16, 8) );
    ("a symbolic start of any element", at_start floats, (4, 0));
    ("a symbolic start of every other element", at_start ~every:2 floats, (8, 4));
    ( "a symbolic start of every fourth element",
      at_start ~every:4 floats,
      (16, 4) );
    ( "a symbolic start into a reordering view",
      (let k = multiple 1 2 in
       Shape.shrink
         (Shape.permute square [ 1; 0 ])
         [ Some (Sym k, Sym Ops.O.(k + int 2)); None ]),
      (4, 0) );
    ("a reordering", Shape.permute square [ 1; 0 ], (16, 4));
    ("a bitcast", Ops.bitcast floats Int32, (16, 4));
    ( "a stage of a run of doubles one past a boundary",
      Ops.contiguous (Shape.shrink doubles [ range 1 8 ]),
      (8, 0) );
    ( "a stage of a view of symbolic size",
      Ops.contiguous (Shape.shrink floats [ Some (Int 0, Sym n) ]),
      (1, 0) );
    ( "a stage of a view whose first element is padding",
      Ops.contiguous (Shape.pad floats [ range 1 0 ]),
      (16, 0) );
    ( "a stage of a column, no run",
      Ops.contiguous (Shape.shrink square [ None; range 1 2 ]),
      (16, 0) );
    ( "storage on a disk",
      Ops.new_buffer ~slot:3 ~phase:4 "DISK:/tmp/f" 16 Float32,
      (16, 0) );
    ("a constant", Ops.int 1, (16, 0));
    ("a value computed", Ops.O.(floats + floats), (16, 0));
  ]

(* Storage parameters and scalars in distinct slots: a storage parameter's
   alignment and phase, or [None] for a scalar. *)
let gen_params =
  let open Gen in
  let start =
    let* align = of_list ~pp:Format.pp_print_int [ 4; 8; 16 ] in
    let+ phase =
      of_list ~pp:Format.pp_print_int (List.init (align / 4) (fun j -> 4 * j))
    in
    (align, phase)
  in
  let* slots = subsequence ~pp:Format.pp_print_int (List.init 8 Fun.id) in
  let+ kinds = list ~size:(constant (List.length slots)) (option start) in
  List.combine slots kinds

let pp_params =
  Format.(
    pp_print_list
      ~pp_sep:(fun ppf () -> pp_print_string ppf ", ")
      (fun ppf (slot, kind) ->
        match kind with
        | None -> fprintf ppf "scalar %d" slot
        | Some (a, p) -> fprintf ppf "storage %d (%d, %d)" slot a p))

let param_of (slot, kind) =
  match kind with
  | None -> Shape.param ~addrspace:Alu slot Int32
  | Some (align, phase) ->
      Shape.param ~shape:(ints [ 4 ]) ~align ~phase slot Float32

let starts_of_body params =
  let body = Ops.sink (List.map param_of params) in
  let expected =
    List.filter_map
      (fun (slot, kind) -> Option.map (fun s -> (slot, s)) kind)
      params
  in
  cover "a scalar parameter" (List.exists (fun (_, k) -> k = None) params);
  cover "two storage parameters" (List.length expected >= 2);
  let starts = slist (pair int (pair int int)) compare in
  equal starts expected (Shape.param_starts body);
  equal
    (list (option (pair int int)))
    (List.init 10 (fun s -> List.assoc_opt (s - 1) expected))
    (List.init 10 (fun s -> Shape.param_start body (s - 1)))

let gen_start =
  let open Gen in
  let pow2 = of_list ~pp:Format.pp_print_int [ 1; 2; 4; 8; 16 ] in
  let* align = pow2 in
  let* a = pow2 in
  let* phase = int_range 0 (align - 1) in
  let+ p = int_range 0 (a - 1) in
  ((align, phase), (a, p))

(* Storage that starts at [p] modulo [a] starts at one of [p + j * a]: it fits a
   parameter that assumes [phase] modulo [align] iff each of them does. *)
let fits_where_every_start_does ((align, phase), (a, p)) =
  let every =
    List.for_all
      (fun j -> (p + (j * a)) mod align = phase)
      (List.init 32 Fun.id)
  in
  cover "it fits" every;
  cover "it does not fit" (not every);
  let fitting fits = if fits then Some (a, p) else None in
  equal
    (option (pair int int))
    (fitting every)
    (fitting (Shape.fits_start ~param:(align, phase) (a, p)))

let storage =
  group "storage"
    [
      goldens [ "storage"; "storage_like"; "clones"; "symbolic_storage" ];
      test "a shape of more than max_int elements raises" (fun () ->
          rejects (fun () ->
              Shape.placeholder ~slot:0 [ 1 lsl 32; 1 lsl 32 ] Float32));
      test "empty and param raise for a weak type" (fun () ->
          rejects (fun () -> Shape.empty ~device:cpu (ints [ 4 ]) Weak_int);
          rejects (fun () -> Shape.param 0 Weak_int));
      test "empty_like on another device takes u's shape" (fun () ->
          let u =
            Shape.empty_like ~dtype:Int32 ~device:"CPU:2"
              (param ~device:cpu [ 8; 6 ] 1 Float32)
          in
          equal shape (ints [ 8; 6 ]) (Shape.shape u);
          equal (option device) (Some "CPU:2") (Ops.device u);
          equal uops [ u ] (holding Ops.has_buffer_identity [ u ]));
      test "a clone of a weak value commits its type" (fun () ->
          equal dtype Int32
            (Ops.dtype
               (Shape.clone ~device:cpu (Shape.expand (Ops.int 3) (ints [ 4 ])))));
      test "clone raises for a disk, named in any case" (fun () ->
          let p = param ~device:cpu [ 4 ] 0 Float32 in
          rejects (fun () -> Shape.clone ~device:"DISK:/tmp/f" p);
          rejects (fun () -> Shape.clone ~device:"disk:0" p);
          equal (option device) (Some "DISKS")
            (Ops.device (Shape.clone ~device:"DISKS" p)));
      test "an alloc given a slot is no alloc made without one" (fun () ->
          (* The number the next fresh alloc would take, given by hand first. *)
          let slot = Ops.unique_num () + 1 in
          let given = Shape.alloc ~slot ~device:cpu (ints [ 8 ]) Float32 in
          let made = Shape.alloc ~device:cpu (ints [ 8 ]) Float32 in
          not_equal int (Ops.param_arg_of (Ops.base given)).slot
            (Ops.param_arg_of (Ops.base made)).slot);
      test "allocs equal in slot, size, type and device are one node" (fun () ->
          let slot = Ops.unique_num () in
          let alloc () = Shape.alloc ~slot ~device:cpu (ints [ 8 ]) Float32 in
          let first = alloc () in
          equal uop first (alloc ()));
      test "a placeholder given a slot is no placeholder made without one"
        (fun () ->
          let check addrspace =
            let slot = Ops.unique_num () + 1 in
            let given = Shape.placeholder ~slot ~addrspace [ 8 ] Float32 in
            let made = Shape.placeholder ~addrspace [ 8 ] Float32 in
            not_equal int (Ops.param_arg_of given).slot
              (Ops.param_arg_of made).slot
          in
          List.iter check [ Dtype.Global; Dtype.Local; Dtype.Reg ]);
      test "alloc and placeholder raise for the slot max_int" (fun () ->
          rejects (fun () -> Shape.alloc ~slot:max_int (ints [ 8 ]) Float32);
          rejects (fun () -> Shape.placeholder ~slot:max_int [ 8 ] Float32));
      test "placeholder commits a weak type" (fun () ->
          equal dtype Int32
            (Ops.dtype (Shape.placeholder ~slot:0 [ 4 ] Weak_int)));
      test "placeholder raises for the ALU's address space" (fun () ->
          rejects (fun () ->
              Shape.placeholder ~slot:0 ~addrspace:Alu [ 4 ] Float32));
      test
        "placeholder_like raises for a shape of symbolic size and for the slot \
         max_int" (fun () ->
          let p = param [ 4 ] 0 Float32 in
          let n = var "n" 1 4 in
          rejects (fun () ->
              Shape.placeholder_like (Shape.shrink p [ Some (Int 0, Sym n) ]) 1);
          rejects (fun () -> Shape.placeholder_like p max_int));
      test "param's first element lies phase bytes past a multiple of align"
        (fun () ->
          let start u =
            let p = Ops.param_arg_of (Ops.buf_uop u) in
            (p.align, p.phase)
          in
          equal
            (list (pair int int))
            [ (16, 0); (8, 4) ]
            [
              start (param [ 4 ] 0 Float32);
              start
                (Shape.param ~shape:(ints [ 4 ]) ~phase:4 ~align:8 0 Float32);
            ]);
      test "param_like names the parameter it makes, or leaves it unnamed"
        (fun () ->
          let p = param ~device:cpu [ 4 ] 0 Float32 and n = var "n" 1 8 in
          equal string "x" (Ops.expr (Shape.param_like ~name:"x" p 1));
          equal string "y" (Ops.expr (Shape.param_like ~name:"y" n 2));
          rejects (fun () -> Ops.expr (Shape.param_like n 2));
          rejects (fun () -> Ops.expr (Shape.param_like p 1)));
      test "param_like has the start of the storage u views" (fun () ->
          let start u =
            let p = Ops.param_arg_of (Ops.buf_uop (Shape.param_like u 0)) in
            (p.align, p.phase)
          in
          List.iter
            (fun (name, u, _) ->
              equal ~msg:name (pair int int) (Shape.storage_phase u) (start u))
            (List.filter
               (fun (_, u, _) -> Ops.has_buffer_identity (Ops.base u))
               storage_phases));
      (* An int8 buffer shrunk at its second byte starts at byte 1, which no
         float32 can. *)
      test "param_like raises for a phase u's type cannot take" (fun () ->
          let bytes = Ops.new_buffer ~slot:4 cpu 16 Int8 in
          let shifted =
            Ops.bitcast (Shape.shrink bytes [ range 1 9 ]) Float32
          in
          rejects (fun () -> Shape.param_like shifted 0));
      (let b =
         Shape.reshape
           (Ops.new_buffer ~slot:5 cpu 24 Float32)
           (ints [ 2; 3; 4 ])
       in
       let state =
         Ops.after b [ Ops.store b (Ops.new_buffer ~slot:6 cpu 24 Float32) ]
       in
       let k = multiple 1 2 in
       let rows =
         [
           ("storage whole, no shrink", b, false);
           ( "one-element axes, then a partial one, then whole ones",
             Shape.shrink b [ range 1 2; range 1 3; None ],
             true );
           ( "a partial axis, then whole ones",
             Shape.shrink b [ range 1 2; None; None ],
             true );
           ( "one-element axes, then a partial last axis",
             Shape.shrink b [ range 0 1; range 2 3; range 1 3 ],
             true );
           ( "a partial axis before a partial one",
             Shape.shrink b [ None; range 1 3; range 1 3 ],
             false );
           ( "a whole axis before a one-element one",
             Shape.shrink b [ None; range 1 2; None ],
             false );
           ( "a shrink of a state of storage",
             Shape.shrink state [ range 1 2; None; None ],
             true );
           ( "a shrink a loop moves along",
             Shape.shrink b [ Some (Sym k, Sym Ops.O.(k + int 1)); None; None ],
             true );
           ( "a shrink of a value computed",
             Shape.shrink Ops.O.(b + b) [ range 1 2; None; None ],
             false );
         ]
       in
       test
         "run_of_storage holds of a shrink of storage that is a run of its \
          elements" (fun () ->
           let names p =
             List.filter_map
               (fun (name, s, e) -> if p (s, e) then Some name else None)
               rows
           in
           equal (list string) (names snd)
             (names (fun (s, _) -> Shape.run_of_storage s))));
      cases "storage_phase is the alignment and phase of the storage u views"
        ~id:(fun (name, _, _) -> name)
        storage_phases
        (fun (_, u, expected) ->
          equal (pair int int) expected (Shape.storage_phase u));
      prop "fits_start holds where each start the storage can have fits"
        ~examples:[ ((16, 4), (16, 4)); ((8, 0), (4, 0)) ]
        gen_start fits_where_every_start_does;
      test "allocated_start is a 16-byte boundary" (fun () ->
          equal (pair int int) (16, 0) Shape.allocated_start);
      prop
        "param_starts is each storage parameter's slot and start, as \
         param_start"
        (Gen.with_pp pp_params gen_params)
        starts_of_body;
      test "view_as views flat storage as the shape" (fun () ->
          let b = Ops.new_buffer ~slot:7 cpu 6 Int32 in
          let u = Shape.view_as b (ints [ 2; 3 ]) in
          equal shape (ints [ 2; 3 ]) (Shape.shape u);
          let memory : Dtype.value array = Array.init 6 (fun j -> i j) in
          equal (list Dtypes.const)
            (Array.to_list (Array.map (fun v -> (v :> Dtype.const)) memory))
            (Array.to_list (Tensors.eval ~buffers:[ (7, memory) ] u)));
    ]

(* Variables *)

let variables =
  let n = var "n" 1 8 in
  group "variables"
    [
      goldens [ "variables_and_binding" ];
      test
        "is_variable holds of a variable, bound or not, and is_bound_var of a \
         bound one" (fun () ->
          let b = Shape.bind n (i 3) and p = param [ 4 ] 0 Float32 in
          let scalar = Shape.param ~addrspace:Alu 1 Int32 in
          equal uops [ n; b ] (holding Shape.is_variable [ n; b; p; scalar ]);
          equal uops [ b ] (holding Shape.is_bound_var [ n; b; p; scalar ]));
      test "bind binds a value within the range, its ends included" (fun () ->
          equal
            (list (pair uop value))
            [ (n, i 3); (n, i 1); (n, i 8) ]
            (List.map
               (fun x -> Shape.unbind (Shape.bind n x))
               [ i 3; i 1; i 8 ]));
      test
        "bind raises for a bound variable, a value out of range and another \
         node" (fun () ->
          rejects (fun () -> Shape.bind (Shape.bind n (i 3)) (i 4));
          rejects (fun () -> Shape.bind n (i 0));
          rejects (fun () -> Shape.bind n (i 9));
          rejects (fun () -> Shape.bind (param [ 4 ] 0 Float32) (i 1)));
      test "unbound is the variable without its value" (fun () ->
          equal uops [ n; n ]
            [ Shape.unbound (Shape.bind n (i 3)); Shape.unbound n ]);
      test "unbound raises for a node that is not a variable" (fun () ->
          rejects (fun () -> Shape.unbound (param [ 4 ] 0 Float32)));
      test "unbind raises for a variable that is not bound" (fun () ->
          rejects (fun () -> Shape.unbind n));
      test "unbind_all unbinds each variable and lists its value" (fun () ->
          let m = var "m" 0 4 in
          let e = Ops.O.(Shape.bind n (i 3) + Shape.bind m (i 2)) in
          let u, values = Shape.unbind_all e in
          equal uop Ops.O.(n + m) u;
          equal
            (slist (pair uop value) (fun (a, _) (b, _) -> Ops.compare a b))
            [ (n, i 3); (m, i 2) ]
            values);
      (let a = var "a" 0 10 and b = var "b" 0 10 and c = var "c" 0 10 in
       let m = var "m" 0 4 in
       cases "variables are the variables u reads, each once, sorted by name"
         ~id:(fun (name, _, _) -> name)
         [
           ("a constant", Ops.int 0, []);
           ("a product", Ops.O.(a * int 3), [ a ]);
           ("a sum", Ops.O.(a + b + c), [ a; b; c ]);
           ("a sum of a product", Ops.O.(a + (b * c)), [ a; b; c ]);
           ( "quotients and remainders",
             Ops.O.((a % int 3) + (b // int 5)),
             [ a; b ] );
           ("a variable twice", Ops.O.(a + b + c - a), [ a; b; c ]);
           ("a square", Ops.O.(a * a), [ a ]);
           ( "two quotients of one variable",
             Ops.O.((a // int 4) + (a // int 6)),
             [ a ] );
           ("names out of order", Ops.O.(n + m), [ m; n ]);
           ( "a bound variable, unbound",
             Ops.O.(Shape.bind n (i 3) + m),
             [ m; n ] );
         ]
         (fun (_, u, expected) -> equal uops expected (Shape.variables u)));
      test "variables lists a scalar parameter without a range" (fun () ->
          let s = Shape.param ~addrspace:Alu 0 Int32 in
          equal uops [ s ] (Shape.variables Ops.O.(s + int 1)));
      test "variables of one name are sorted by slot" (fun () ->
          let scalar ?name slot = Shape.param ?name ~addrspace:Alu slot Int32 in
          let s3 = scalar 3 and s1 = scalar 1 in
          let n4 = scalar ~name:"n" 4 and n2 = scalar ~name:"n" 2 in
          equal uops [ s1; s3; n2; n4 ]
            (Shape.variables Ops.O.(s3 + s1 + n4 + n2)));
      cases "known_size is an integer's value, past an int too"
        ~id:(fun (name, _, _) -> name)
        [
          ("an Int", Ops.Int 5, Some (Bigint.of_int 5));
          ( "a literal past an int",
            Sym (Ops.const (`Int (Bigint.shift_left Bigint.one 70))),
            Some (Bigint.shift_left Bigint.one 70) );
          ( "a typed literal",
            Sym (Ops.int ~dtype:Int32 5),
            Some (Bigint.of_int 5) );
          ("a variable", Sym n, None);
        ]
        (fun (_, s, expected) -> equal (option z) expected (Shape.known_size s));
    ]

(* Divisibility *)

let divisibility =
  let x = var "x" 10 20 in
  group "divisibility"
    [
      test "divides divides a product through its left factor first" (fun () ->
          let k = var "k" 0 4 in
          equal (option uop)
            (Some Ops.O.(k * int 2 * x))
            (Shape.divides Ops.O.(k * int 4 * x) (Bigint.of_int 2));
          equal (option uop) None
            (Shape.divides Ops.O.(x + int 3) (Bigint.of_int 2)));
      test "divides divides a known multiple" (fun () ->
          equal (option z)
            (Some (Bigint.of_int 6))
            (Option.map Ops.const_factor
               (Shape.divides (Ops.int 42) (Bigint.of_int 7)));
          equal (option uop) None (Shape.divides (Ops.int 42) (Bigint.of_int 5));
          equal (option z) (Some Bigint.one)
            (Option.map Ops.const_factor
               (Shape.divides Ops.O.((x * int 6) + int 18) (Bigint.of_int 6)));
          equal (option uop) None
            (Shape.divides Ops.O.(var "x" 15 45 * int 4) (Bigint.of_int 3));
          equal (option uop) (Some x) (Shape.divides x Bigint.one));
      test "divides divides a float constant only into an integer" (fun () ->
          equal (option uop) None
            (Shape.divides (Ops.float 2.5) (Bigint.of_int 2));
          equal (option uop)
            (Some (Ops.float 2.))
            (Shape.divides (Ops.float 4.) (Bigint.of_int 2)));
      test "divides divides a typed constant, a cast of one, as its value"
        (fun () ->
          equal (option uop)
            (Some (Ops.int ~dtype:Int32 4))
            (Shape.divides (Ops.int ~dtype:Int32 8) (Bigint.of_int 2));
          equal (option uop) None
            (Shape.divides (Ops.int ~dtype:Int32 8) (Bigint.of_int 3)));
      test "gcd raises for no nodes" (fun () ->
          rejects (fun () -> Shape.gcd []));
      test "gcd of constants is their gcd" (fun () ->
          equal uop (Ops.int 2) (Shape.gcd [ Ops.int 6; Ops.int 4 ]));
      test "gcd keeps the common factors and the gcd of the coefficients"
        (fun () ->
          let y = var "y" 1 4 and z' = var "z" 1 4 in
          equal uop Ops.O.(int 1 * x) (Shape.gcd Ops.O.[ x * y; x * z' ]);
          equal uop (Ops.int 2) (Shape.gcd Ops.O.[ x * int 6; y * int 4 ]));
      test "divide_exact divides each term" (fun () ->
          let y = var "y" 1 4 and z' = var "z" 1 4 in
          equal (option uop) (Some (Ops.int 1)) (Shape.divide_exact x x);
          equal (option uop)
            (Some Ops.O.((int 1 * y) + (int 1 * z')))
            (Shape.divide_exact Ops.O.((x * y) + (x * z')) x);
          equal (option uop)
            (Some (Ops.int 3))
            (Shape.divide_exact (Ops.int 6) (Ops.int 2)));
      (let a = var "a" 1 8 and b = var "b" 1 8 and y = var "y" 1 4 in
       let x' = var "x" (-20) 0 in
       cases "divide_exact is None where u does not divide exactly"
         ~id:(fun (name, _, _) -> name)
         [
           ("x + 1 by x", Ops.O.(x + int 1), x);
           ("x * 6 by y * 2", Ops.O.(x * int 6), Ops.O.(y * int 2));
           ( "a stack by a stack",
             Shape.stack Ops.O.[ x * int 2; x * int 4 ],
             Shape.consts [ i 2; i 4 ] );
           ("a by b", a, b);
           ("a + 2 by a", Ops.O.(a + int 2), a);
           ("-x by a", Ops.O.(x' * int (-1)), a);
           ("a * 5 by a * 10", Ops.O.(a * int 5), Ops.O.(a * int 10));
           ( "a * 10 - 1 by a * 10",
             Ops.O.((a * int 10) - int 1),
             Ops.O.(a * int 10) );
         ]
         (fun (_, u, d) -> equal (option uop) None (Shape.divide_exact u d)));
    ]

(* Calls *)

(* A kernel body that stores what it loads from parameter [1] of [n] elements
   into parameter [0]. *)
let copy_body ?kernel n =
  let r = Ops.range ~axis_type:Weak (Int n) [ 0 ] in
  let at slot = Ops.index (param [ n ] slot Float32) [ r ] in
  Ops.sink ?kernel [ Ops.end_ (Ops.store (at 0) (Ops.load (at 1) [])) [ r ] ]

(* A compiled program whose parameters are [params], in slot order, and whose
   linear order holds [linear]. *)
let program ~params linear =
  let slot u = (Ops.param_arg_of u).slot in
  let globals =
    List.filter_map
      (fun u -> if Ops.addrspace u = Some Alu then None else Some (slot u))
      params
  in
  let info : Ops.program_info =
    {
      global_size = ints [ 1; 1; 1 ];
      local_size = ints [ 1; 1; 1 ];
      params;
      globals;
      outs = globals;
      ins = globals;
      target = Result.get_ok (Helpers.Target.of_string "CPU:CLANG");
    }
  in
  Ops.v Program ~arg:(Program info)
    ~src:
      [
        Ops.sink ~kernel:(Ops.kernel_info ()) [];
        Ops.v Linear ~src:linear;
        Ops.v Source ~arg:(String "source");
        Ops.v Binary ~arg:(Bytes "\x7fELF\x00lib");
      ]

let calls =
  let arg = param ~device:cpu [ 4 ] 0 Float32 in
  group "calls"
    [
      goldens [ "calls"; "custom_kernels"; "outputs"; "symbolic_outputs" ];
      test
        "call_with_outputs raises for output positions that do not ascend \
         within the arguments" (fun () ->
          let v = Ops.O.(arg + float 1.) in
          rejects (fun () ->
              Shape.call_with_outputs ~output_pos:[ 1; 0 ] [ v; v ] [ arg ]);
          rejects (fun () ->
              Shape.call_with_outputs ~output_pos:[ 0; 0 ] [ v; v ] [ arg ]);
          rejects (fun () ->
              Shape.call_with_outputs ~output_pos:[ 5 ] [ v ] [ arg ]);
          rejects (fun () ->
              Shape.call_with_outputs ~output_pos:[ 0; 1 ] [ v ] [ arg ]));
      test "a call compiled apart stages its arguments" (fun () ->
          let v = Ops.O.(arg + float 1.) in
          let out = Shape.call_with_output ~precompile:true v [ v ] in
          equal op Op.Stage (Ops.op (Ops.nth (Ops.nth out 1) 1)));
      (let store slot =
         Ops.store
           (Ops.index (param [ 4 ] slot Float32) [ Ops.int 0 ])
           (Ops.float ~dtype:Float32 1.)
       in
       let writes_0 = Ops.sink [ store 0 ] in
       cases
         "written_params is the slots a body writes, or None where it does not \
          show them"
         ~id:(fun (name, _, _) -> name)
         [
           ( "stores, in increasing order",
             Ops.sink [ store 2; store 0 ],
             Some [ 0; 2 ] );
           ("a load and a store", copy_body 4, Some [ 0 ]);
           ( "a call, where its body writes",
             Ops.sink
               [
                 Ops.call writes_0
                   [ param [ 4 ] 3 Float32; param [ 4 ] 1 Float32 ];
               ],
             Some [ 3 ] );
           ( "a call of code that does not show its writes",
             Ops.sink
               [
                 Ops.call (Ops.custom_function "f" []) [ param [ 4 ] 1 Float32 ];
               ],
             Some [ 1 ] );
           ("a program", program ~params:[] [], None);
           ("a custom function", Ops.custom_function "f" [], None);
         ]
         (fun (_, body, expected) ->
           equal (option (list int)) expected (Shape.written_params body)));
      (let b = Ops.new_buffer ~slot:1 cpu 8 Float32 in
       let half lo = Shape.shrink b [ range lo (lo + 4) ] in
       let other = Ops.new_buffer ~slot:2 cpu 4 Float32 in
       let rows =
         [
           ( "read, its storage written",
             Ops.call (copy_body 4) [ half 0; half 4 ],
             1,
             true );
           ( "read, other storage written",
             Ops.call (copy_body 4) [ half 0; other ],
             1,
             false );
           ("written", Ops.call (copy_body 4) [ half 0; half 4 ], 0, false);
           ( "a body that does not show its writes",
             Ops.call (Ops.custom_function "f" []) [ half 0; half 4 ],
             1,
             false );
         ]
       in
       test
         "reads_storage_written holds of an argument only read, whose storage \
          another writes" (fun () ->
           let names p =
             List.filter_map
               (fun (name, c, i, e) -> if p (c, i, e) then Some name else None)
               rows
           in
           equal (list string)
             (names (fun (_, _, e) -> e))
             (names (fun (c, i, _) -> Shape.reads_storage_written c i))));
    ]

(* Launches *)

let launches =
  let kernel = Ops.kernel_info () in
  let a = Ops.new_buffer ~slot:1 cpu 4 Float32
  and b = Ops.new_buffer ~slot:2 cpu 4 Float32
  and c = Ops.new_buffer ~slot:3 cpu 4 Float32 in
  let copy x y = Ops.call (copy_body ~kernel 4) [ x; y ] in
  let members = [ copy a b; copy c b ] in
  group "launches"
    [
      test "is_launch holds of a kernel's sink of calls compiled on their own"
        (fun () ->
          let l = Shape.launch members in
          equal uops
            [ Ops.nth l 0 ]
            (holding Shape.is_launch
               [
                 Ops.nth l 0;
                 copy_body ~kernel 4;
                 Ops.sink ~kernel [];
                 Ops.sink [];
               ]));
      test "launch_members of a launch are its calls" (fun () ->
          equal uops members (Shape.launch_members (Shape.launch members)));
      test "launch's arguments are its calls', each once" (fun () ->
          equal uops [ a; b; c ] (Ops.src (Shape.launch members) |> List.tl));
      test
        "launch raises for a call that is not of a kernel's sink, or is \
         compiled on its own" (fun () ->
          rejects (fun () -> Shape.launch [ Ops.int 1 ]);
          rejects (fun () -> Shape.launch [ Ops.call (copy_body 4) [ a; b ] ]);
          rejects (fun () ->
              Shape.launch
                [ Ops.call ~precompile:true (copy_body ~kernel 4) [ a; b ] ]));
      test "launch_members raises for a call that is not of a launch" (fun () ->
          rejects (fun () -> Shape.launch_members (copy a b)));
    ]

(* Programs *)

let alu_param ?(lo = 0) ?(hi = 8) name slot =
  Shape.param ~vmin_vmax:(i lo, i hi) ~name ~addrspace:Alu slot Int32

let slot u = match Ops.arg u with Param p -> p.slot | _ -> -2
let scalar u = Ops.addrspace u = Some Alu

let empty_target =
  Helpers.Target.
    { device = ""; renderer = ""; arch = ""; interface = ""; indices = "" }

(* A buffer of [slot + 1] elements or a scalar in [slot], named after it. *)
let kernel_param (slot, is_buffer) =
  let dtypes = [| Dtype.Float32; Int8; Float16; Int64 |] in
  let name = Printf.sprintf (if is_buffer then "b%d" else "v%d") slot in
  if is_buffer then
    Shape.param
      ~shape:(ints [ slot + 1 ])
      ~name slot
      dtypes.(slot mod Array.length dtypes)
  else
    Shape.param
      ~vmin_vmax:(i 0, i 9)
      ~name ~addrspace:Alu slot
      [| Dtype.Int32; Int64; Int8; Uint16 |].(slot mod 4)

(* A program's parameters, each a buffer or a scalar of its slot, in its linear
   order, and those its linear order leaves out, as a calling convention leaves
   out what it passes on the stack. *)
type params = { linear : (int * bool) list; dropped : (int * bool) list }

let pp_slots ppf l =
  Format.(
    pp_print_list
      ~pp_sep:(fun ppf () -> pp_print_string ppf ", ")
      (fun ppf (slot, b) -> fprintf ppf "%s%d" (if b then "b" else "v") slot)
      ppf l)

let pp_kernel_params ppf l =
  Format.fprintf ppf "linear [%a], dropped [%a]" pp_slots l.linear pp_slots
    l.dropped

let gen_kernel_params =
  (let open Gen in
   let* slots = subsequence ~pp:Format.pp_print_int (List.init 12 Fun.id) in
   let* kinds = list ~size:(constant (List.length slots)) bool in
   let params = List.combine slots kinds in
   let* kept = subsequence params in
   let+ linear = permutation kept in
   { linear; dropped = List.filter (fun p -> not (List.mem p kept)) params })
  |> Gen.with_pp pp_kernel_params

let in_slot_order l =
  let all = List.sort compare (l.linear @ l.dropped) in
  let rec interleaved = function
    | (_, true) :: (_, false) :: (_, true) :: _ -> true
    | _ :: rest -> interleaved rest
    | [] -> false
  in
  cover "a scalar lies between buffers" (interleaved all);
  cover "the linear order is not slot order"
    (List.map fst l.linear <> List.sort compare (List.map fst l.linear));
  cover "a parameter is left out of the linear order" (l.dropped <> []);
  let node = kernel_param in
  let prg = program ~params:(List.map node all) (List.map node l.linear) in
  equal uops (List.map node all) (Shape.kernel_params prg)

let programs =
  group "programs"
    [
      test "program_info_of_sink reads launch sizes, variables and buffers"
        (fun () ->
          let core_id = alu_param ~hi:3 "core_id" 0
          and n = alu_param ~lo:2 "n" 1 in
          let input = Shape.param ~shape:(ints [ 16 ]) 2 Float32
          and output = Shape.param ~shape:(ints [ 16 ]) 3 Float32 in
          let stored =
            Ops.store (Ops.index output [ n ])
              (Ops.load (Ops.index input [ n ]) [])
          in
          let sink =
            Ops.sink
              [
                stored;
                Ops.special (Int 4) "gidx2";
                Ops.special (Int 8) "lidx1";
                core_id;
              ]
          in
          let info = Shape.program_info_of_sink sink in
          equal (list int) [ 2; 3 ] info.globals;
          equal (list int) [ 3 ] info.outs;
          equal (list int) [ 2 ] info.ins;
          equal (list int) [ 0; 1; 2; 3 ] (List.map slot info.params);
          equal uops [ core_id; n ] (List.filter scalar info.params);
          equal
            (pair (list int) (list int))
            ([ 1; 1; 4 ], [ 1; 8; 1 ])
            (Shape.launch_dims info []);
          equal target empty_target info.target);
      test "program_info_of_sink reads symbolic launch sizes" (fun () ->
          let core_id = alu_param ~hi:3 "core_id" 0
          and n = alu_param ~lo:2 "n" 1 in
          let input = Shape.param ~shape:(ints [ 16 ]) 2 Float32
          and output = Shape.param ~shape:(ints [ 16 ]) 3 Float32 in
          let stored =
            Ops.store (Ops.index output [ n ])
              (Ops.load (Ops.index input [ n ]) [])
          in
          let sink =
            Ops.sink
              ~kernel:(Ops.kernel_info ~name:"kernel name" ())
              [
                stored;
                Ops.special (Sym Ops.O.(n + int 1)) "gidx2";
                Ops.special (Int 8) "lidx1";
                core_id;
              ]
          in
          let info = Shape.program_info_of_sink sink in
          equal (list int) [ 2; 3 ] info.globals;
          equal (list int) [ 3 ] info.outs;
          equal (list int) [ 2 ] info.ins;
          equal (list int) [ 0; 1; 2; 3 ] (List.map slot info.params);
          equal uops [ core_id; n ] (List.filter scalar info.params);
          equal
            (pair (list int) (list int))
            ([ 1; 1; 7 ], [ 1; 8; 1 ])
            (Shape.launch_dims info [ ("n", 6); ("core_id", 2) ]);
          (* A size reads its variable as a run binds it, at no slot. *)
          equal (list int) [ -1 ]
            (match List.nth info.global_size 2 with
            | Sym u ->
                List.filter_map
                  (fun x -> if Ops.op x = Param then Some (slot x) else None)
                  (Ops.toposort ~calls:Skip u)
            | Int _ -> []);
          equal target empty_target info.target);
      test
        "a load through a cast of an index and a store into a shrink are \
         accesses" (fun () ->
          let p0 = Shape.param ~shape:(ints [ 4 ]) 0 Uint8
          and p1 = Shape.param ~shape:(ints [ 4 ]) 1 Uint8 in
          let cast_read =
            Ops.load
              (Ops.v
                 ~src:[ Ops.index p0 [ Ops.int 0 ] ]
                 ~arg:(Dtype Int8) Op.Cast)
              []
          in
          let write =
            Ops.store
              (Shape.shrink p1 [ range 0 2 ])
              (Shape.expand (Ops.int ~dtype:Uint8 1) (ints [ 2 ]))
          in
          let info =
            Shape.program_info_of_sink (Ops.sink [ write; cast_read ])
          in
          equal (pair (list int) (list int)) ([ 0 ], [ 1 ]) (info.ins, info.outs));
      test "a load through a cast of a parameter is no access" (fun () ->
          let p0 = Shape.param ~shape:(ints [ 4 ]) 0 Uint8
          and p1 = Shape.param ~shape:(ints [ 4 ]) 1 Uint8 in
          let write =
            Ops.store (Ops.index p1 [ Ops.int 0 ]) (Ops.int ~dtype:Uint8 1)
          in
          let info =
            Shape.program_info_of_sink
              (Ops.sink [ write; Ops.load (Ops.cast p0 Int8) [] ])
          in
          equal (pair (list int) (list int)) ([], [ 1 ]) (info.ins, info.outs));
      test "the parameters of a called body are not the program's" (fun () ->
          let p0 = Shape.param ~shape:(ints [ 4 ]) 0 Uint8
          and p5 = Shape.param ~shape:(ints [ 4 ]) 5 Uint8 in
          let one p =
            Ops.store (Ops.index p [ Ops.int 0 ]) (Ops.int ~dtype:Uint8 1)
          in
          let body = Ops.sink [ one p5 ] in
          let info =
            Shape.program_info_of_sink
              (Ops.sink [ one p0; Ops.call ~name:"f" body [] ])
          in
          equal (list int) [ 0 ] info.globals;
          equal (pair (list int) (list int)) ([], [ 0 ]) (info.ins, info.outs));
      test "every buffer reads and writes when no access says otherwise"
        (fun () ->
          let info =
            Shape.program_info_of_sink
              (Ops.sink
                 [
                   Shape.param ~shape:(ints [ 4 ]) 0 Float32;
                   Shape.param ~shape:(ints [ 4 ]) 5 Float32;
                 ])
          in
          equal (list int) [ 0; 5 ] info.globals;
          equal
            (pair (list int) (list int))
            ([ 0; 5 ], [ 0; 5 ])
            (info.ins, info.outs);
          equal (list int) [ 1; 1; 1 ] (fst (Shape.launch_dims info [])));
      test "program_info_of_sink records its target" (fun () ->
          let t = Result.get_ok (Helpers.Target.of_string "CPU:CLANG") in
          equal target t
            (Shape.program_info_of_sink ~target:t (Ops.sink [])).target);
      prop "kernel_params is every parameter in slot order, buffers and scalars"
        ~examples:
          [
            { linear = [ (7, true); (2, true); (4, false) ]; dropped = [] };
            { linear = [ (0, true); (3, true) ]; dropped = [ (1, false) ] };
          ]
        gen_kernel_params in_slot_order;
      test "kernel_params raises for a node that is not a linearized program"
        (fun () -> rejects (fun () -> Shape.kernel_params (copy_body 4)));
      test "launch sizes divide as their operations say" (fun () ->
          let n = alu_param ~lo:(-10) ~hi:10 "n" 0 in
          let info =
            Shape.program_info_of_sink
              (Ops.sink
                 [
                   Ops.special (Sym Ops.O.(n // int 3)) "gidx0";
                   Ops.special (Sym Ops.O.(n % int 3)) "gidx1";
                 ])
          in
          equal (list int) [ -3; 2; 1 ]
            (fst (Shape.launch_dims info [ ("n", -7) ])));
      test "launch_dims raises for a variable it lacks" (fun () ->
          let extent = alu_param "extent" 0 in
          let info =
            Shape.program_info_of_sink
              (Ops.sink [ Ops.special (Sym extent) "gidx0" ])
          in
          rejects (fun () -> Shape.launch_dims info []));
    ]
