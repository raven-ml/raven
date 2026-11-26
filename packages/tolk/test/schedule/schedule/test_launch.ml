(* Tests of the states a schedule's kernels read and of its launches (Schedule):
   each kernel reads a state as it was, an After's stores land in the order
   listed, and calls that run in either order run as one launch, which writes
   what its members apart write, bit for bit, on the host and each GPU present.

   A program assembles pieces into the rows of one buffer, each piece a store
   into a run of rows of a value computed from buffers of its own: an
   elementwise value, a sum along a last axis, or a product of a matrix and a
   vector. Its schedule runs once as made and once with each launch run as its
   members, one call each; both write the same bits. *)

open Windtrap
open Tolk
module Engine = Tolk_engine

(* Devices *)

let device name = if name = "CPU" then Rig.host else Gpus.gpu name

(* A program of a type [name]'s device lacks makes no statement there. *)
let reject_unsupported name dt =
  if not (List.mem dt (Renderer.supported_dtypes (Run.renderer (device name))))
  then reject ()

(* Programs *)

type kind = Elementwise | Sum | Matvec | Matmul

type piece = {
  kind : kind;
  rows : int; (* the rows of the buffer it covers *)
  k : int; (* the length of the axis it sums along *)
}

type program = {
  dtype : Dtype.t;
  width : int; (* the elements of a row *)
  pieces : piece list;
  fill : bool; (* whether a store of every row comes first *)
  later : bool; (* whether a store over the first row comes after *)
  bound : bool; (* whether each piece's first row is a bound variable *)
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
  let x = (((j * 7) + 3) mod 11) - 5 in
  if is_float dt then `Float (float_of_int x /. 4.)
  else if Dtype.is_unsigned dt then `Int (Bigint.of_int (x + 5))
  else `Int (Bigint.of_int x)

(* The value a piece stores, of shape [rows; width], from new buffers of [d]. *)
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

(* [x] of [dt] broadcast to [shape]. *)
let broadcast x dt shape =
  let ones = List.map (fun _ -> Ops.Int 1) shape in
  Shape.expand
    (Shape.reshape (Ops.cast x dt) ones)
    (List.map (fun n -> Ops.Int n) shape)

(* The program's sink, its output in slot 0. *)
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
        Ops.variable
          ("start" ^ string_of_int r)
          (`Int Bigint.zero)
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

let schedule big = Schedule.create_linear_with_vars ~cores:Cores.host big

(* Running *)

let is_launch e = Ops.op e = Call && Shape.is_launch (Ops.nth e 0)
let launches linear = List.filter is_launch (Ops.src linear)

(* [linear] with each launch run as its members, one call each. *)
let apart linear =
  Ops.replace linear
    ~src:
      (List.concat_map
         (fun e -> if is_launch e then Shape.launch_members e else [ e ])
         (Ops.src linear))

(* The members of each entry of [linear]: a launch's, or 1. *)
let members linear =
  List.map
    (fun e -> if is_launch e then List.length (Shape.launch_members e) else 1)
    (Ops.src linear)

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

let compiled name linear =
  let d = device name in
  Realize.compile_linear ~targets:(fun _ -> Run.target d) linear

(* The buffers of [big] on [name], each holding [element] of each index: the
   output, of slot 0, as its start, which each run copies, and the inputs, which
   every run reads. *)
type inputs = {
  out : Ops.t * Dtype.t * Dtype.value array;
  given : (Ops.t * Rig.Buffer.t) list;
}

let inputs name big =
  let d = device name in
  let start n (p : Ops.param_arg) =
    Array.init (Shape.max_numel n) (element p.dtype)
  in
  match
    List.partition (fun (_, (p : Ops.param_arg)) -> p.slot = 0) (storage big)
  with
  | [ (n, p) ], rest ->
      {
        out = (n, p.dtype, start n p);
        given =
          List.map
            (fun (n, (p : Ops.param_arg)) ->
              (n, Run.buffer d p.dtype (start n p)))
            rest;
      }
  | outs, _ -> failf "%d outputs" (List.length outs)

(* What [linear], a schedule of a program on [name], leaves in its output. *)
let written name inputs (linear, vars) =
  let d = device name in
  let n, dt, start = inputs.out in
  let b = Run.buffer d dt start in
  let s =
    Engine.link
      ~devices:(fun _ -> d)
      ~bound:((n, b) :: inputs.given) (compiled name linear)
  in
  Engine.run ~vars s [||];
  Array.to_list (Array.map bits (Run.values dt b))

(* The entries of [linear], compiled on [name], that run a launch's program. *)
let compiled_launches name linear =
  List.length
    (List.filter
       (fun e ->
         match Ops.src e with
         | prg :: _ when Ops.op e = Call && Ops.op prg = Program -> (
             match Ops.arg (Ops.nth prg 0) with
             | Kernel { name = "launch"; _ } -> true
             | _ -> false)
         | _ -> false)
       (Ops.src (compiled name linear)))

(* Laws *)

let alike p =
  match p.pieces with
  | q :: (_ :: _ as rest) -> List.for_all (fun q' -> q' = q) rest
  | _ -> false

let reduces q = q.kind <> Elementwise

let cover_pieces p =
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
  cover "an integer" (not (is_float p.dtype))

let writes_as_apart name p =
  reject_unsupported name p.dtype;
  cover_pieces p;
  let big = sink name p in
  let linear, vars = schedule big in
  let inputs = inputs name big in
  equal (list string)
    ~msg:(Format.asprintf "%a" pp_program p)
    (written name inputs (apart linear, vars))
    (written name inputs (linear, vars))

(* The stores of a write marked disjoint run in either order: past one piece,
   they are one launch, however the fill and the later store beside them run. *)
let one_launch p =
  assume (List.length p.pieces > 1);
  cover "alike pieces" (alike p);
  cover "unlike pieces" (List.length p.pieces > 1 && not (alike p));
  cover "a fill" p.fill;
  cover "a later state" p.later;
  cover "bound starts" p.bound;
  let linear, _ = schedule (sink "CPU" p) in
  equal int
    ~msg:(Format.asprintf "launches of %a" pp_program p)
    1
    (List.length (launches linear))

(* A write marked disjoint of a piece of each kind, two of them sums along an
   axis, every value read from buffers the program is given. *)
let reducing_write =
  {
    dtype = Float32;
    width = 64;
    pieces =
      [
        { kind = Elementwise; rows = 2; k = 1 };
        { kind = Sum; rows = 2; k = 16 };
        { kind = Matvec; rows = 1; k = 16 };
        { kind = Matmul; rows = 1; k = 33 };
      ];
    fill = false;
    later = false;
    bound = false;
  }

(* A member that reduces shares a launch of [name]'s target only with members of
   its kernel. *)
let reducing_members name p =
  reject_unsupported name p.dtype;
  cover "reducing pieces of two kernels"
    (List.length (List.sort_uniq compare (List.filter reduces p.pieces)) > 1);
  cover "a reducing piece beside one that is not"
    (List.exists reduces p.pieces
    && List.exists (fun q -> not (reduces q)) p.pieces);
  let linear, _ = schedule (sink name p) in
  let ren = Run.renderer (device name) in
  let reducing body =
    List.exists (fun u -> Ops.op u = Reduce) (Ops.toposort ~calls:Skip body)
  in
  let kernels l =
    List.sort_uniq Ops.compare
      (List.map (fun m -> Ops.nth m 0) (Shape.launch_members l))
  in
  List.iter
    (fun l ->
      let ks = kernels l in
      if List.exists reducing ks then
        equal int
          ~msg:(Format.asprintf "kernels of a launch of %a" pp_program p)
          1 (List.length ks))
    (List.filter is_launch
       (List.concat_map (Realize.split_launch ren) (launches linear)))

(* The bits a program of elementwise integer pieces leaves in its buffer: each
   piece's rows hold [3 x + 1] of its buffer [x], whose element [j] is [element
   dt j]. *)
let assembled p =
  List.concat_map
    (fun q ->
      List.init (q.rows * p.width) (fun j ->
          match element p.dtype j with
          | `Int x -> string_of_int ((3 * Bigint.to_int x) + 1)
          | _ -> invalid_arg "assembled: an integer program"))
    p.pieces

let writes_pieces name p =
  reject_unsupported name p.dtype;
  cover "past a hundred pieces" (List.length p.pieces > 100);
  let big = sink name p in
  equal (list string)
    ~msg:(Format.asprintf "%a" pp_program p)
    (assembled p)
    (written name (inputs name big) (schedule big))

(* Generators *)

let dtypes =
  Dtype.
    [
      Float32;
      Float16;
      Bfloat16;
      Float64;
      Int8;
      Int16;
      Int32;
      Int64;
      Uint8;
      Uint16;
      Uint32;
      Uint64;
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
    bind
      (int_range 0 (List.length dtypes - 1))
      (fun i ->
        bind
          (frequency [ (1, constant 1); (3, int_range 2 8) ])
          (fun n ->
            bind bool (fun same ->
                bind gen_piece (fun first ->
                    bind
                      (list ~size:(constant (n - 1)) gen_piece)
                      (fun rest ->
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

(* [gen_program]'s programs, each piece elementwise. *)
let gen_elementwise =
  Gen.map
    (fun p ->
      let piece q = { q with kind = Elementwise; k = 1 } in
      { p with pieces = List.map piece p.pieces })
    gen_program

(* One piece of a sum: the fewest pieces. *)
let one_piece =
  {
    dtype = Float32;
    width = 64;
    pieces = [ { kind = Sum; rows = 2; k = 16 } ];
    fill = true;
    later = true;
    bound = false;
  }

(* Four alike products of a matrix and a vector. *)
let alike_products =
  {
    dtype = Int32;
    width = 64;
    pieces = List.init 4 (fun _ -> { kind = Matvec; rows = 2; k = 16 });
    fill = false;
    later = false;
    bound = true;
  }

(* 32 pieces of every kind, each unlike the one before it: the breadth of many
   members, drawn once. *)
let unlike_pieces =
  let kinds = [| Elementwise; Sum; Elementwise; Matvec; Matmul |] in
  let ks = [| 1; 16; 33 |] in
  {
    dtype = Float32;
    width = 64;
    pieces =
      List.init 32 (fun i ->
          { kind = kinds.(i mod 5); rows = 1 + (i mod 4); k = ks.(i mod 3) });
    fill = true;
    later = false;
    bound = false;
  }

let apart_examples = [ one_piece; alike_products; unlike_pieces ]

(* [n] one-row elementwise pieces of [int32]. *)
let many n =
  {
    dtype = Int32;
    width = 9;
    pieces = List.init n (fun _ -> { kind = Elementwise; rows = 1; k = 1 });
    fill = true;
    later = false;
    bound = false;
  }

(* Mostly up to 8 elementwise pieces of [int32] rows, else up to 64. *)
let gen_many =
  Gen.(
    map
      (fun ((n, rows), (width, fill)) ->
        {
          (many n) with
          width;
          pieces = List.init n (fun _ -> { kind = Elementwise; rows; k = 1 });
          fill;
        })
      (pair
         (pair
            (frequency [ (3, int_range 1 8); (1, int_range 9 64) ])
            (int_range 1 2))
         (pair (one_of [ constant 1; constant 9; constant 64 ]) bool)))

(* 101 one-row pieces, the fewest past a hundred. *)
let past_a_hundred = many 101

(* 512 one-element pieces, each a member of one workgroup: as one launch, Metal
   builds its pipeline in minutes. *)
let past_five_hundred = { (many 512) with width = 1 }

(* The most members a launch holds on each GPU, its renderer's [launch_members]
   (Cstyle): Metal's 40 and CUDA's 64, which NV renders with too. AMD's is
   unbounded. *)
let launch_limits = [ ("Metal", 40); ("CUDA", 64); ("NV", 64) ]

(* A state of as many stores as a launch of [kind] holds, and one of one
   more. *)
let at_the_limit kind =
  match List.assoc_opt kind launch_limits with
  | Some n -> [ many n; many (n + 1) ]
  | None -> []

(* States read as they were

   A kernel of one write, or a kernel beside a write, that reads a state reads
   it as it was: run on the host, each program leaves the values the states it
   names hold. *)

let part lo hi u = Shape.shrink u [ Some (Int lo, Int hi) ]
let flip u = Shape.flip u [ 0 ]
let fl = Ops.float
let f32 ~n slot = Ops.new_buffer ~slot "CPU" n Float32

(* [check] with the items it lists as written, and reversed. *)
let in_either_listing check () =
  subtest "as written" (check Fun.id);
  subtest "reversed" (check List.rev)

(* [j] at element [j] of the buffer of slot 0, [10 j] at slot 1. *)
let start (p : Ops.param_arg) j =
  `Float (float_of_int (j * if p.slot = 0 then 1 else 10))

let unshared big =
  Setting.context [ B (Setting.scache, 0) ] (fun () -> schedule big)

(* Every element of [big]'s buffers after its schedule runs on the host, in slot
   order. *)
let finals big =
  let d = device "CPU" in
  let linear, vars = unshared big in
  let compiled = compiled "CPU" linear in
  let bufs =
    List.map
      (fun (n, (p : Ops.param_arg)) ->
        (p, Run.buffer d p.dtype (Array.init (Shape.max_numel n) (start p))))
      (storage big)
  in
  let bound = List.map2 (fun (n, _) (_, b) -> (n, b)) (storage big) bufs in
  Engine.run ~vars (Engine.link ~devices:(fun _ -> d) ~bound compiled) [||];
  let bufs =
    List.sort
      (fun ((p : Ops.param_arg), _) (q, _) -> compare p.slot q.slot)
      bufs
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

(* As [chained_read], the later write over [x] itself, its destination a run of
   the first write's state: only that destination reaches the first write. *)
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

(* A join over the view a store writes reads the state under it: [x] as it was,
   once [y] is stored. *)
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
  holds [ b; y' ] [ 1.; 1.; 2.; 3.; 4.; 5.; 6.; 7.; 2.; 2.; 10.; 11.; 8.; 9. ]

(* An After over a view names that view's state, and a reader of the view as it
   was runs before the write. *)
let read_view_after_over order () =
  let x = f32 ~n:8 0 and y = f32 ~n:5 1 in
  let v = part 1 6 x in
  let x' = Ops.after v [ Ops.store v (fl 7.) ] in
  let y' = Ops.after y [ Ops.store y v ] in
  holds
    (order [ x'; y' ])
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
    @ [ 0.; 10.; 20.; 30.; 40. ] @ [ 1.; 22.; 43.; 64.; 85. ])

(* A store of a run reversed, then a store over that run, the state between
   read: the reversal stages what it overwrites. *)
let reversed_then_stored () =
  let x = f32 ~n:8 0 and z = f32 ~n:5 1 in
  let v = part 1 6 x in
  let a = Ops.after x [ Ops.store v (flip v) ] in
  let b = Ops.after a [ Ops.store v (fl 5.) ] in
  let z' = Ops.after z [ Ops.store z (part 1 6 a) ] in
  holds [ b; z' ] [ 0.; 5.; 5.; 5.; 5.; 5.; 6.; 7.; 5.; 4.; 3.; 2.; 1. ]

(* A reader of a run that also waits for the store into it. *)
let waiting_reader () =
  let x = f32 ~n:8 0 and y = f32 ~n:5 1 in
  let v = part 1 6 x in
  let x' = Ops.after x [ Ops.store v (fl 7.) ] in
  let y' = Ops.after y [ Ops.store y Ops.O.(v + part 1 6 x') ] in
  [ x'; y' ]

let read_view_waiting () =
  holds (waiting_reader ())
    [ 0.; 7.; 7.; 7.; 7.; 7.; 6.; 7.; 8.; 9.; 10.; 11.; 12. ]

(* The copies of states that scheduling [big] makes. *)
let copies_made big =
  let before = Schedule.copies () in
  ignore (unshared (Ops.sink big));
  Schedule.copies () - before

(* Each store of a disjoint write that reads only its own elements. *)
let own_elements () =
  let x = f32 ~n:4 0 in
  let bump lo hi = Ops.store (part lo hi x) Ops.O.(part lo hi x + fl 1.) in
  [ Ops.after_disjoint x [ bump 0 2; bump 2 4 ] ]

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

(* [x] holding 0 to 7 written through a run of it, and that run of [x] as it was
   stored into [y]. *)
let view_read rd order () =
  let x = f32 ~n:8 0 and y = f32 ~n:8 1 in
  let x' = Ops.after x [ Ops.store (part 1 6 x) (fl 7.) ] in
  let lo, hi = rd in
  let y' = Ops.after y [ Ops.store (part 0 (hi - lo) y) (part lo hi x) ] in
  let tail =
    List.init (8 - (hi - lo)) (fun j -> float_of_int ((hi - lo + j) * 10))
  in
  holds
    (order [ x'; y' ])
    ([ 0.; 7.; 7.; 7.; 7.; 7.; 6.; 7. ]
    @ List.init (hi - lo) (fun j -> float_of_int (lo + j))
    @ tail)

(* A store through a later state of the buffer, and a reader of the first of its
   runs as it was. *)
let chained_view order () =
  let x = f32 ~n:8 0 and y = f32 ~n:5 1 in
  let a = Ops.after x [ Ops.store (part 1 6 x) (fl 7.) ] in
  let b = Ops.after a [ Ops.store (part 3 8 x) (fl 8.) ] in
  let y' = Ops.after y [ Ops.store y (part 1 6 x) ] in
  holds
    (order [ b; y' ])
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
  holds
    (order [ b; y' ])
    ([ 7.; 7.; 0.; 20.; 40.; 60.; 6.; 7. ]
    @ [ 0.; 21.; 42.; 63. ] @ [ 0.; 10.; 20.; 30. ])

(* A loop's call stores through one run of the buffer, listed after a write of
   it, and reads the first state of another: a copy made before that write,
   once, not each trip. *)
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
    [ 9.; 10.; 11.; 12.; 4.; 5.; 6.; 7.; 2.; 2.; 2.; 2.; 12.; 13.; 14.; 15. ]

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

(* [p]'s elements 0 to 5 doubled, then 2 to 7 tripled, each store reading [p] as
   it was. *)
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

let two_writes_raise items =
  raises_match (Exn.invalid_arg ~sub:"two writes over one state") (fun () ->
      ignore (finals (Ops.sink items)))

(* An After over the whole of [x] and one over a view of it are two writes over
   one state: which lands last would be the listing's choice. *)
let whole_and_view order () =
  let x = f32 ~n:8 0 in
  let v = part 2 6 x in
  let w = Ops.after x [ Ops.store x (fl 1.) ] in
  let u = Ops.after v [ Ops.store v (fl 2.) ] in
  two_writes_raise (order [ w; u ])

(* Two writes of [x] over the whole of it, beside one through a view, in each
   place the view's can take. *)
let two_whole () =
  let x = f32 ~n:8 0 in
  let v = part 2 6 x in
  let w1 = Ops.after x [ Ops.store x (fl 1.) ] in
  let w2 = Ops.after x [ Ops.store x (fl 3.) ] in
  let u = Ops.after v [ Ops.store v (fl 2.) ] in
  subtest "view last" (fun () -> two_writes_raise [ w1; w2; u ]);
  subtest "view first" (fun () -> two_writes_raise [ u; w2; w1 ]);
  subtest "view between" (fun () -> two_writes_raise [ w1; u; w2 ])

let listed_in_loop () =
  let x = f32 ~n:8 0 in
  let trip = Ops.range ~axis_type:Loop (Ops.Int 3) [ 100 ] in
  let call = Ops.call ~precompile:true (listing_body ~device:"CPU" ()) [ x ] in
  holds
    [ Ops.after x [ Ops.end_ call [ trip ] ] ]
    [ 0.; 8.; 54.; 81.; 108.; 135.; 162.; 189. ]

(* Launches *)

(* Stores of values each kernel makes first, read by the stores: pieces of two
   kernels each. *)
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

(* The most bytes ({!Memory.bytes}) of the storage that [counts], that the
   entries of [linear] take at once: each from the first entry that takes it to
   the last. *)
let peak ~counts linear =
  let entries = Array.of_list (Ops.src linear) in
  let first = Ops.Tbl.create 16 and last = Ops.Tbl.create 16 in
  Array.iteri
    (fun i e ->
      List.iter
        (fun b ->
          if counts b then begin
            if not (Ops.Tbl.mem first b) then Ops.Tbl.replace first b i;
            Ops.Tbl.replace last b i
          end)
        (Memory.taken e))
    entries;
  let live i =
    Ops.Tbl.fold
      (fun b f n ->
        if f <= i && i <= Ops.Tbl.find last b then n + Memory.bytes b else n)
      first 0
  in
  List.fold_left max 0 (List.init (Array.length entries) live)

(* Each value is made and freed in turn, however the stores could run together:
   a value of 16 floats takes 256 bytes, as {!Memory.bytes} rounds them. *)
let storage_order_holds () =
  let big = made_first () in
  let linear, _ =
    Schedule.create_linear_with_vars ~cores:Cores.host ~capturing:true big
  in
  let given = List.map fst (storage big) in
  equal int ~msg:"bytes live at once" 256
    (peak ~counts:(fun b -> not (List.memq b given)) linear)

let cpu_buffer n = Ops.new_buffer "CPU" n Float32

(* The call, as a schedule runs it, of the kernel of [n] elements that stores
   one more than each element of a buffer of [n] floats into another, and those
   two buffers. *)
let one_more n =
  let x = cpu_buffer n and o = cpu_buffer n in
  let big = Ops.sink [ Ops.after o [ Ops.store o Ops.O.(x + fl 1.) ] ] in
  match
    Ops.src
      (fst
         (Schedule.create_linear_with_vars ~capturing:true ~cores:Cores.host big))
  with
  | [ call ] -> (call, x, o)
  | es -> failf "%d entries" (List.length es)

(* The call that stores one more than each element of [read] into [write]. *)
let add_one ~read ~write =
  let call, x, o = one_more (Shape.max_numel read) in
  let arg a = if a == x then read else if a == o then write else a in
  Ops.replace call
    ~src:(Ops.body call :: List.map arg (Ops.src_without_body call))

let linear entries = Ops.v Op.Linear ~src:entries

let loop entries =
  Ops.end_ (linear entries) [ Ops.range ~axis_type:Loop (Int 3) [ 100 ] ]

(* Two calls, each storing one element into a buffer of four from another: moved
   up beside the first, the second makes its buffers live earlier. *)
let one_element_calls () =
  let buf () = Ops.new_buffer "CPU" 4 Float32 in
  let first u = Shape.shrink u [ Some (Int 0, Int 1) ] in
  let into o x = Ops.after o [ Ops.store (first o) Ops.O.(first x + fl 1.) ] in
  Ops.sink [ into (buf ()) (buf ()); into (buf ()) (buf ()) ]

let given big =
  List.filter (fun u -> Ops.op u = Buffer) (Ops.toposort ~calls:Skip big)

let launched_holding held big =
  members (fst (Schedule.create_linear_with_vars ~held ~cores:Cores.host big))

let held_one_element () =
  let big = one_element_calls () in
  equal (list int) ~msg:"held" [ 2 ] (launched_holding (given big) big);
  equal (list int) ~msg:"not held" [ 1; 1 ] (launched_holding [] big)

let held_by_default () =
  let big = one_element_calls () in
  equal (list int)
    (launched_holding (given big) big)
    (members (fst (schedule big)))

(* [x] to [t] makes 4 KiB, then [y] to [u] 256 bytes beside it, under a peak of
   [t]'s 4 KiB unless [t] is held. *)
let held_counts_at_no_step () =
  let x = cpu_buffer 1024 and t = cpu_buffer 1024 in
  let y = cpu_buffer 64 and u = cpu_buffer 64 in
  let l = linear [ add_one ~read:x ~write:t; add_one ~read:y ~write:u ] in
  equal (list int) ~msg:"t held" [ 2 ]
    (members (Schedule.launches ~held:[ x; y; t ] l));
  equal (list int) ~msg:"t counted" [ 1; 1 ]
    (members (Schedule.launches ~held:[ x; y ] l))

(* [w] to [s] makes 8 KiB, read into [o]; then [x] to [t] makes 4 KiB and [y] to
   [u] 256 bytes beside it. In a loop, [s] is the loop's alone and the peak is
   [t]'s 4 KiB; at the top level, [s]'s 8 KiB is the peak, and [u] fits beside
   [t]. *)
let loop_storage_counts_at_no_step () =
  let w = cpu_buffer 2048 and s = cpu_buffer 2048 and o = cpu_buffer 2048 in
  let x = cpu_buffer 1024 and t = cpu_buffer 1024 in
  let y = cpu_buffer 64 and u = cpu_buffer 64 in
  let held = [ w; o; x; y ] in
  let pair = [ add_one ~read:w ~write:s; add_one ~read:s ~write:o ] in
  let later = [ add_one ~read:x ~write:t; add_one ~read:y ~write:u ] in
  equal (list int) ~msg:"in a loop" [ 1; 1; 1 ]
    (members (Schedule.launches ~held (linear (loop pair :: later))));
  equal (list int) ~msg:"at the top level" [ 1; 1; 2 ]
    (members (Schedule.launches ~held (linear (pair @ later))))

(* Schedules of calls of one more over buffers of one, 16 or 1024 floats, four
   of each. *)
type call = { size : int; read : int; write : int }
type calls = { calls : call list; held : bool list }

let sizes = [| 1; 16; 1024 |]

let pp_calls ppf c =
  Format.fprintf ppf "@[%a@]; held %a"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       (fun ppf c ->
         Format.fprintf ppf "%d: %d to %d" sizes.(c.size) c.read c.write))
    c.calls
    (Format.pp_print_list
       ~pp_sep:(fun _ () -> ())
       (fun ppf h -> Format.pp_print_int ppf (Bool.to_int h)))
    c.held

let gen_calls =
  Gen.(
    with_pp pp_calls
      (map
         (fun (calls, held) -> { calls; held })
         (pair
            (list ~size:(int_range 2 12)
               (map
                  (fun (size, read, w) ->
                    {
                      size;
                      read;
                      write = (if w = read then (w + 1) mod 4 else w);
                    })
                  (triple (int_range 0 2) (int_range 0 3) (int_range 0 3))))
            (list ~size:(constant 12) bool))))

(* A call of 16 elements joins the launch of one of 1024 at a peak that four
   calls of one element raise later. *)
let one_element_beside_the_peak =
  let one = { size = 0; read = 0; write = 1 } in
  {
    calls =
      [
        { size = 2; read = 0; write = 2 };
        { size = 1; read = 0; write = 1 };
        one;
        one;
        one;
        { size = 2; read = 0; write = 1 };
        one;
      ];
    held = List.init 12 (fun _ -> false);
  }

let peak_bounds c =
  let bufs = Array.init 12 (fun i -> cpu_buffer sizes.(i / 4)) in
  let buf size j = bufs.((4 * size) + j) in
  let held = List.filteri (fun i _ -> List.nth c.held i) (Array.to_list bufs) in
  let l =
    linear
      (List.map
         (fun c ->
           add_one ~read:(buf c.size c.read) ~write:(buf c.size c.write))
         c.calls)
  in
  let launched = Schedule.launches ~held l in
  cover "a launch" (List.exists (fun m -> m > 1) (members launched));
  cover "storage held" (held <> []);
  (* The peak counts storage of one element, which a call of one element moved
     up beside it makes live longer. *)
  let own b = not (List.memq b held) in
  at_most int ~than:(peak ~counts:own l)
    (peak ~counts:(fun b -> own b && Shape.max_numel b > 1) launched)

(* Two independent calls of one more over buffers of 16 floats on [device],
   held, so that no peak keeps them apart. *)
let two_calls device =
  let bufs = List.init 4 (fun _ -> Ops.new_buffer device 16 Float32) in
  let call i =
    add_one ~read:(List.nth bufs (2 * i)) ~write:(List.nth bufs ((2 * i) + 1))
  in
  members (Schedule.launches ~held:bufs (linear [ call 0; call 1 ]))

(* A disk named after a directory, which scheduling never opens. *)
let on_a_disk () =
  let disk = "DISK:" ^ Filename.concat (Filename.get_temp_dir_name ()) "tolk" in
  equal (list int) ~msg:"on the host" [ 2 ] (two_calls "CPU");
  equal (list int) ~msg:"on a disk" [ 1; 1 ] (two_calls disk)

(* The first [n] elements of [u], [n] a bound variable. *)
let first_n n u =
  let v = Ops.variable "n" (`Int (Bigint.of_int 1)) (`Int (Bigint.of_int 8)) in
  let n = Shape.bind v (`Int (Bigint.of_int n)) in
  Shape.shrink u [ Some (Ops.Int 0, Ops.Sym n) ]

(* One more than each element of a run of [x] into that run of [o], beside a
   call of one more than each element of [y] into [o']. *)
let symbolic_ranges () =
  let x = cpu_buffer 8 and o = cpu_buffer 8 in
  let y = cpu_buffer 8 and o' = cpu_buffer 8 in
  let beside run =
    let into o x = Ops.after o [ Ops.store (run o) Ops.O.(run x + fl 1.) ] in
    let whole = Ops.after o' [ Ops.store o' Ops.O.(y + fl 1.) ] in
    members (fst (schedule (Ops.sink [ whole; into o x ])))
  in
  equal (list int) ~msg:"of a constant size" [ 2 ] (beside (part 0 3));
  equal (list int) ~msg:"of a symbolic size" [ 1; 1 ] (beside (first_n 3))

(* Sixteen products of [16; 16] matrices of [dtype]. *)
let sixteen kind dtype =
  {
    dtype;
    width = 16;
    pieces = List.init 16 (fun _ -> { kind; rows = 16; k = 16 });
    fill = false;
    later = false;
    bound = false;
  }

(* [p]'s schedule on [name], as made and apart. *)
let same_as_apart name p =
  let big = sink name p in
  let linear, vars = schedule big in
  let inputs = inputs name big in
  equal (list string)
    (written name inputs (apart linear, vars))
    (written name inputs (linear, vars));
  linear

let pieces dtype rows =
  {
    dtype;
    width = 64;
    pieces = List.map (fun r -> { kind = Elementwise; rows = r; k = 1 }) rows;
    fill = false;
    later = false;
    bound = false;
  }

let launch_locals name = (Run.renderer (device name)).launch_locals

(* The sources of the launch programs of [p]'s schedule compiled on [name]. *)
let launch_sources name p =
  let linear, _ = schedule (sink name p) in
  List.filter_map
    (fun e ->
      match Ops.src e with
      | prg :: _ when Ops.op e = Call && Ops.op prg = Program -> (
          let source =
            List.find_opt (fun u -> Ops.op u = Source) (Ops.src prg)
          in
          match (Ops.arg (Ops.nth prg 0), source) with
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

(* The source of the one launch of sixteen alike elementwise members. *)
let alike_source name =
  match launch_sources name (sixteen Elementwise Float32) with
  | [ src ] -> src
  | srcs -> failf "%d launches" (List.length srcs)

(* The renderers name a member's body [member]: one definition, and a call from
   each member. *)
let one_body name () =
  let src = alike_source name in
  equal int ~msg:"bodies" 1 (occurrences "void member(" src);
  equal int ~msg:"calls" 16 (occurrences "member(" src - 1)

let out_of_line name attribute () =
  equal int ~msg:"bodies out of line" 1
    (occurrences attribute (alike_source name))

(* A state of half again as many one-row stores as a launch of [name] holds, as
   many launches as compilation splits it into: one full, one of the rest. *)
let past_the_cap name () =
  let cap = (Run.renderer (device name)).launch_members in
  if cap = max_int then skip ~reason:"launches hold any number of members" ();
  let p = pieces Float32 (List.init (cap + (cap / 2)) (fun _ -> 1)) in
  equal int 2 (compiled_launches name (fst (schedule (sink name p))))

(* The examples of the pieces prop at [kind]'s limit take it from
   [launch_limits]. *)
let limit_listed kind () =
  let cap = (Run.renderer (device kind)).launch_members in
  equal (option int)
    (if cap = max_int then None else Some cap)
    (List.assoc_opt kind launch_limits)

(* Sixteen products of [16; 16] half matrices, on tensor cores on CUDA: one
   kernel's members, which reduce, and so join only each other. *)
let products name () =
  if not (launch_locals name) then skip ~reason:"no workgroup memory" ();
  let linear = same_as_apart name (sixteen Matmul Float16) in
  equal int ~msg:"launches" 1 (compiled_launches name linear)

(* The sixteen products where a launch's members hold no workgroup memory: each
   sums through it, so each runs alone. *)
let products_apart name () =
  if launch_locals name then skip ~reason:"members hold workgroup memory" ();
  let linear = same_as_apart name (sixteen Matmul Float16) in
  equal int ~msg:"launches" 0 (compiled_launches name linear)

(* On the host, a launch whose cost the engine splits into blocks of [2^18]
   operations, whose bounds fall inside members of unequal iterations. *)
let blocks () =
  let p = pieces Float32 [ 100; 300; 37 ] in
  let linear = same_as_apart "CPU" p in
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
  match ops with
  | [ n ] -> at_least int ~msg:"operations, two blocks'" ~than:(1 lsl 19) n
  | ns -> failf "%d launches" (List.length ns)

(* The launch of [sixteen kind Float32]'s program on the host, split for a
   renderer of the given limits, as the members of the launches it becomes. *)
let split ?launch_members ?launch_args ?launch_locals kind =
  let linear, _ = schedule (sink "CPU" (sixteen kind Float32)) in
  match launches linear with
  | [ e ] ->
      let target = Run.target Rig.host in
      let ren = Renderer.v ?launch_members ?launch_args ?launch_locals target in
      List.map
        (fun l ->
          if is_launch l then List.length (Shape.launch_members l) else 1)
        (Realize.split_launch ren e)
  | ls -> failf "%d launches" (List.length ls)

let in_pairs = List.init 8 (fun _ -> 2)

(* A member passes its region of the output and its source: five arguments hold
   two members. *)
let limits =
  [
    ( "at most 2 members",
      (fun () -> split ~launch_members:2 Elementwise),
      in_pairs );
    ( "at most 5 arguments",
      (fun () -> split ~launch_args:5 Elementwise),
      in_pairs );
    ( "at most 5 buffers",
      (fun () ->
        Setting.context
          [ Setting.B (Setting.max_kernel_buffers, 5) ]
          (fun () -> split Elementwise)),
      in_pairs );
    ("one reducing kernel's members", (fun () -> split Matmul), [ 16 ]);
    ( "reducing members without workgroup memory",
      (fun () -> split ~launch_locals:false Matmul),
      List.init 16 (fun _ -> 1) );
  ]

(* The launches on each GPU present: a GPU's tests skip without one. *)
let on_a_gpu kind =
  group kind
    [
      test
        "sixteen products of one kernel run as one launch, writing what they \
         write apart"
        (products kind);
      test
        "where a launch's members hold no workgroup memory, sixteen products \
         run apart, writing what they write in a launch"
        (products_apart kind);
      test "a state past the members a launch holds runs as two launches"
        (past_the_cap kind);
      test "the pieces prop's examples sit at the members a launch holds"
        (limit_listed kind);
      prop ~count:30 ~examples:apart_examples
        "the stores of one state write what their kernels apart write"
        gen_program (writes_as_apart kind);
      prop
        "a member that reduces shares a launch only with members of its kernel"
        ~examples:apart_examples gen_program (reducing_members kind);
      (* Each case compiles programs of its own, whose pipelines a GPU builds
         anew, so the host's prop draws twelve cases and a GPU's one, beside
         fixed cases: past a hundred pieces, which reaches the cover on every
         seed, at its launch's limit and one past it ([limit_listed] checks the
         table), and 512, which a launch holds only where the GPU's compiler
         builds that many members in time. *)
      prop ~count:1
        ~examples:((past_a_hundred :: at_the_limit kind) @ [ past_five_hundred ])
        "the stores of one state write each piece's values" gen_many
        (writes_pieces kind);
    ]

let () =
  exit
  @@ run "Tolk.Launch"
       [
         group "create_schedule › states"
           [
             test
               "a store reads what its write's other store writes as it was, \
                in either listing"
               (in_either_listing sibling_read);
             test
               "a store under a later write reads its write's other store as \
                it was, in either listing"
               (in_either_listing chained_read);
             test "a store only a later destination reaches reads it as it was"
               read_under_destination;
             test
               "a join of the view a store writes reads the state under it, in \
                either listing"
               (in_either_listing read_join_of_view);
             test
               "a reader of a run a later write stores over reads it as it was"
               store_over_state_read;
             test
               "a reader of a view an After is over reads it as it was, in \
                either listing"
               (in_either_listing read_view_after_over);
             test
               "a reader sharing a value with the store into its run reads the \
                run as it was, in either listing"
               (in_either_listing read_sharing_value);
             test
               "a reader of the state between a reversal of a run and a store \
                over it reads the reversal"
               reversed_then_stored;
             test
               "a reader of a run that waits for the store into it reads the \
                run as it was"
               read_view_waiting;
             test
               "a reader of a run that waits for the store into it reads a copy"
               (fun () -> equal int 1 (copies_made (waiting_reader ())));
             test
               "stores of a disjoint write each read the elements they write \
                as they were" (fun () ->
                 holds (own_elements ()) [ 1.; 2.; 3.; 4. ]);
             test
               "stores of a disjoint write reading only the elements they \
                write take no copy" (fun () ->
                 equal int 0 (copies_made (own_elements ())));
             test
               "stores of one write reading each other's elements read them as \
                they were"
               crossed_reads;
             test
               "a reader through the view a write stores into reads it as it \
                was, in either listing"
               (in_either_listing (view_read (1, 6)));
             test "a reader through another view reads it as it was"
               (view_read (2, 5) Fun.id);
             test
               "a reader of the first run of a chain of stores through views \
                reads it as it was, in either listing"
               (in_either_listing chained_view);
             test
               "a reader of the first state sharing a value with a chained \
                store reads it as it was, in either listing"
               (in_either_listing chained_share);
             test
               "a loop reading an older run of its storage reads a copy of the \
                run"
               loop_reads_older_run;
           ];
         group "create_schedule › an After's stores"
           [
             test
               "of two overlapping stores, the one listed later wins, in \
                either listing" (fun () ->
                 subtest "as written"
                   (listed_constants Fun.id [ 1.; 1.; 1.; 2.; 2.; 2.; 2.; 2. ]);
                 subtest "reversed"
                   (listed_constants List.rev
                      [ 1.; 1.; 1.; 1.; 1.; 2.; 2.; 2. ]));
             test "each store reads its own elements as they were"
               listed_overlap;
             test
               "a later store reads the elements an earlier one writes as they \
                were"
               listed_reads_earlier;
             test "a store listed twice lands at its first place" listed_twice;
             test
               "the listing holds in a function's body whose parameters name a \
                device"
               (listed_in_call ~device:"CPU");
             test
               "the listing holds in a function's body whose parameters name \
                no device"
               (listed_in_call ?device:None);
             test "the listing holds in a body inlined at its call"
               listed_inlined;
             test "the listing holds in a body a loop runs" listed_in_loop;
             test
               "a call loading through what it stores doubles what was listed \
                before"
               call_after_store;
             test
               "an After of a state and one of a view of it raise, in either \
                listing"
               (in_either_listing whole_and_view);
             test
               "two writes over the whole of a state beside one through a view \
                raise, in any listing"
               two_whole;
           ];
         group "create_linear_with_vars › launches"
           [
             (* Over 1 s: a case compiles its program twice, its members apart
                and then its launch, about 0.1 s of tolk's compile. *)
             prop ~count:60 ~examples:apart_examples
               "on the host, the stores of one state write what their kernels \
                apart write"
               gen_program (writes_as_apart "CPU");
             prop
               "on the host, the stores of one write of kernels that reduce \
                nothing run as one launch"
               gen_elementwise one_launch;
             test
               "on the host, the stores of one write of kernels that reduce \
                run as one launch" (fun () ->
                 let linear, _ = schedule (sink "CPU" reducing_write) in
                 (* One entry, a launch of the write's four kernels. *)
                 equal (list int) [ 4 ] (members linear));
             prop ~count:12 ~examples:[ past_a_hundred ]
               "on the host, the stores of one state write each piece's values"
               gen_many (writes_pieces "CPU");
             test
               "on the host, a launch's members write what they write apart \
                across the blocks the engine splits it into"
               blocks;
             test "a launch holds no more values at once than the schedule"
               storage_order_holds;
             prop
               ~examples:[ one_element_beside_the_peak ]
               "no step of the launches holds more bytes of storage of many \
                elements, not held, than the schedule apart at its peak"
               gen_calls peak_bounds;
             test
               "a call of one element joins beside storage of many elements \
                only where it is held"
               held_one_element;
             test
               "the storage held by default is the buffers the arguments name"
               held_by_default;
             test "held storage counts at no step" held_counts_at_no_step;
             test "storage one loop alone takes counts at no step"
               loop_storage_counts_at_no_step;
             test "a call of a symbolic range joins no launch" symbolic_ranges;
             test "a call on a disk joins no launch" on_a_disk;
           ];
         group "lower_and_compile › launches"
           [
             prop
               "on the host, a member that reduces shares a launch only with \
                members of its kernel"
               ~examples:apart_examples gen_program (reducing_members "CPU");
             test
               "on the host, sixteen products of one kernel run as one launch, \
                writing what they write apart"
               (products "CPU");
             cases
               ~id:(fun (id, _, _) -> id)
               "a launch splits past each limit of its target" limits
               (fun (_, split, expected) ->
                 equal (list int) expected (split ()));
           ];
         group "to_program › launches"
           [
             test "on the host, members of one kernel are one body"
               (one_body "CPU");
             test "on CUDA, members of one kernel are one body"
               (one_body "CUDA");
             test "on the host, each member's body is out of line"
               (out_of_line "CPU" "noinline");
             test "on CUDA, each member's body is out of line"
               (out_of_line "CUDA" "__noinline__");
           ];
         group "GPUs" (List.map on_a_gpu Gpus.kinds);
       ]
