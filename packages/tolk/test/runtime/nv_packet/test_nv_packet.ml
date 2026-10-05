(* Tests of Tolk.Nv_packet: NVIDIA's packets interpreted on a batch's nodes,
   then evaluated at integers, are the packets interpreted on those integers.
   nx.nv.device interprets them on integers; tolk's queues on nodes. *)

open Windtrap
open Tolk
module P = Nx_nv_packet

(* Each command of the packet library, its values as integers. *)
type command =
  | Acquire of int * int
  | Release of int * int
  | Release_stamp of int * int
  | Set_object of P.subchannel * int
  | Local_memory_window of int
  | Shared_memory_window of int
  | Local_memory of int * int
  | Invalidate_caches
  | Schedule of int
  | Copy of int * int * int
  | Copy_release of int * int
  | Copy_stamp of int
  | Entry of int * int * int

(* A launch descriptor's sizes, each known or a value, and its addresses and
   releases, in order. *)
type launch = {
  blackwell : bool;
  dims : (P.dim * bool * int) list; (* (size, a value?, n) *)
  program : int;
  banks : int list;
  local : int;
  releases : (bool * int * int) list; (* (stamp?, address, payload) *)
  next : int option;
}

(* The encoding of a command and of a launch descriptor, each value made by
   [leaf]. *)
let encode leaf = function
  | Acquire (a, v) -> P.Methods.acquire (leaf a) (leaf v)
  | Release (a, v) -> P.Methods.release (leaf a) (leaf v)
  | Release_stamp (a, v) -> P.Methods.release_stamp (leaf a) (leaf v)
  | Set_object (s, cls) -> P.Methods.set_object s cls
  | Local_memory_window a -> P.Methods.local_memory_window (leaf a)
  | Shared_memory_window a -> P.Methods.shared_memory_window (leaf a)
  | Local_memory (a, per) -> P.Methods.local_memory (leaf a) ~per_tpc:(leaf per)
  | Invalidate_caches -> P.Methods.invalidate_caches
  | Schedule a -> P.Methods.schedule (leaf a)
  | Copy (dst, src, n) -> P.Methods.copy ~dst:(leaf dst) ~src:(leaf src) n
  | Copy_release (a, v) -> P.Methods.copy_release (leaf a) (leaf v)
  | Copy_stamp a -> P.Methods.copy_stamp (leaf a)
  | Entry (a, offset, words) ->
      [ P.W64 (P.Gpfifo.entry (leaf a) ~offset ~words) ]

let descriptor leaf program l =
  let q = P.Qmd.make program in
  List.iter
    (fun (d, value, n) ->
      if value then P.Qmd.patch_dim q d (leaf n) else P.Qmd.set_dim q d n)
    l.dims;
  P.Qmd.set_program q (leaf l.program);
  List.iteri (fun i a -> P.Qmd.set_bank q i (leaf a)) l.banks;
  P.Qmd.set_local_memory q (leaf l.local);
  List.iter
    (fun (stamp, a, v) ->
      ignore
        ((if stamp then P.Qmd.release_stamp else P.Qmd.release)
           q (leaf a) (leaf v)))
    l.releases;
  Option.iter (fun a -> P.Qmd.chain q (leaf a)) l.next;
  P.Qmd.structure q

(* Values as variables of a batch, each bound to its integer. *)
let leaves () =
  let bound = ref [] in
  let leaf n =
    let x =
      Ops.variable ~dtype:Dtype.Uint64
        (Printf.sprintf "x%d" (List.length !bound))
        (`Int (Bigint.of_int 0))
        (`Int (Bigint.of_int max_int))
    in
    bound := (x, Ops.int ~dtype:Dtype.Uint64 n) :: !bound;
    x
  in
  (leaf, fun u -> Ops.simplify (Ops.substitute u !bound))

(* The little-endian bytes of the constant [c]. *)
let bytes c =
  let n = Dtype.itemsize (Ops.dtype c) in
  let v =
    match Ops.value c with
    | `Int z -> Bigint.to_int64_unsigned z
    | _ -> failf "%a is no integer" Ops.pp c
  in
  String.init n (fun i ->
      Char.chr
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical v (8 * i)) 0xffL)))

let dwords s =
  List.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

(* The 32-bit words of [ws], encoded over nodes and evaluated. *)
let evaluated eval ws =
  dwords
    (String.concat "" (List.map (fun w -> bytes (eval w)) (Nv_packet.words ws)))

(* The bytes of the region [r], its words evaluated. *)
let region_bytes eval r =
  String.concat ""
    (List.map
       (fun u ->
         match (Ops.op u, Ops.arg u) with
         | Binary, Bytes s -> s
         | _ -> bytes (eval u))
       (Ops.src r))

(* Generators *)

(* Integers up to 2^62: addresses, sizes and 64-bit values, with the edges of
   their 32-bit words. *)
let value =
  Gen.frequency
    [
      (4, Gen.int_range 0 (1 lsl 48));
      (1, Gen.int_range 0 ((1 lsl 62) - 1));
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [
            0;
            1;
            (1 lsl 32) - 1;
            1 lsl 32;
            (1 lsl 32) + 1;
            (1 lsl 40) - 4;
            (1 lsl 62) - 1;
          ] );
    ]

(* Copy sizes around the 2 GiB lines. *)
let copy_size =
  Gen.frequency
    [
      (1, Gen.int_range 0 (1 lsl 20));
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ 0; 1; (1 lsl 31) - 1; 1 lsl 31; (1 lsl 31) + 1; 1 lsl 32; 5 lsl 31 ]
      );
    ]

let subchannel =
  Gen.of_list
    ~pp:(fun ppf s ->
      Format.pp_print_string ppf
        (match s with
        | P.Host -> "Host"
        | Compute -> "Compute"
        | Copy -> "Copy"))
    [ P.Host; Compute; Copy ]

let pp_command ppf c =
  let p = Format.fprintf in
  match c with
  | Acquire (a, v) -> p ppf "acquire 0x%x 0x%x" a v
  | Release (a, v) -> p ppf "release 0x%x 0x%x" a v
  | Release_stamp (a, v) -> p ppf "release_stamp 0x%x 0x%x" a v
  | Set_object (_, cls) -> p ppf "set_object 0x%x" cls
  | Local_memory_window a -> p ppf "local_memory_window 0x%x" a
  | Shared_memory_window a -> p ppf "shared_memory_window 0x%x" a
  | Local_memory (a, per) -> p ppf "local_memory 0x%x ~per_tpc:0x%x" a per
  | Invalidate_caches -> p ppf "invalidate_caches"
  | Schedule a -> p ppf "schedule 0x%x" a
  | Copy (d, s, n) -> p ppf "copy ~dst:0x%x ~src:0x%x %d" d s n
  | Copy_release (a, v) -> p ppf "copy_release 0x%x 0x%x" a v
  | Copy_stamp a -> p ppf "copy_stamp 0x%x" a
  | Entry (a, o, w) -> p ppf "entry 0x%x ~offset:0x%x ~words:%d" a o w

let command =
  let open Gen in
  let two f = map (fun (a, b) -> f a b) (pair value value) in
  with_pp pp_command
    (frequency
       [
         (1, two (fun a v -> Acquire (a, v)));
         (1, two (fun a v -> Release (a, v)));
         (1, two (fun a v -> Release_stamp (a, v)));
         ( 1,
           map
             (fun (s, c) -> Set_object (s, c))
             (pair subchannel (int_range 0 0xffff)) );
         (1, map (fun a -> Local_memory_window a) value);
         (1, map (fun a -> Shared_memory_window a) value);
         (1, two (fun a per -> Local_memory (a, per)));
         (1, constant Invalidate_caches);
         (1, map (fun a -> Schedule a) value);
         ( 2,
           map
             (fun ((d, s), n) -> Copy (d, s, n))
             (pair (pair value value) copy_size) );
         (1, two (fun a v -> Copy_release (a, v)));
         (1, map (fun a -> Copy_stamp a) value);
         ( 1,
           map
             (fun ((a, o), w) -> Entry (a * 4, o * 4, w))
             (pair
                (pair (int_range 0 ((1 lsl 38) - 1)) (int_range 0 0xffff))
                (int_range 0 ((1 lsl 21) - 1))) );
       ])

(* A kernel of [banks] constant banks. *)
let kernel banks =
  {
    Nx_nv_cubin.code = 0x100;
    code_bytes = 0x2400;
    registers = 40;
    shared_bytes = 0x800;
    stack_bytes = 0x10;
    params_offset = 0x160;
    banks =
      List.init banks (fun index ->
          { Nx_nv_cubin.index; offset = 0x1000 * index; bytes = 0x40 });
  }

let program ~blackwell banks =
  match
    P.Program.make
      ~compute_class:(if blackwell then 0xcdc0 else 0xc9c0)
      ~sass_version:0x89 ~shared_window:0x729400000000
      ~local_window:0x729300000000 (kernel banks)
  with
  | Ok p -> p
  | Error e -> failf "a program: %s" e

(* The sizes fit their narrowest field, a block's depth on Blackwell. *)
let launch =
  let open Gen in
  let dim d = map (fun (v, n) -> (d, v, n)) (pair bool (int_range 0 0xff)) in
  let dims =
    List.fold_right
      (fun d acc -> map (fun (x, xs) -> x :: xs) (pair (dim d) acc))
      P.[ Grid X; Grid Y; Grid Z; Block X; Block Y; Block Z ]
      (constant [])
  in
  let release =
    map (fun ((s, a), v) -> (s, a, v)) (pair (pair bool value) value)
  in
  map
    (fun ((((blackwell, dims), (program, banks)), (local, releases)), next) ->
      { blackwell; dims; program; banks; local; releases; next })
    (pair
       (pair
          (pair (pair bool dims)
             (pair value (list ~size:(int_range 0 4) value)))
          (pair value (list ~size:(int_range 0 3) release)))
       (option value))

let pp_launch ppf l =
  Format.fprintf ppf
    "{ blackwell = %b; program = 0x%x; banks = %d; releases = %d; next = %s }"
    l.blackwell l.program (List.length l.banks) (List.length l.releases)
    (match l.next with Some a -> Printf.sprintf "0x%x" a | None -> "none")

(* Laws *)

let words = list int

let command_law c =
  let leaf, eval = leaves () in
  equal words (P.dwords (encode Fun.id c)) (evaluated eval (encode leaf c))

(* Copies apart, so that copies of several lines are frequent. *)
let copy_law ((dst, src), n) =
  cover "a copy of several lines" (n > 1 lsl 31);
  command_law (Copy (dst, src, n))

let launch_law l =
  let p = program ~blackwell:l.blackwell (List.length l.banks) in
  let leaf, eval = leaves () in
  cover "a descriptor of version 5" l.blackwell;
  cover "a descriptor of version 3" (not l.blackwell);
  cover "both releases taken, and a third refused" (List.length l.releases = 3);
  equal string
    (P.fill (descriptor Fun.id p l))
    (region_bytes eval (Nv_packet.structure "qmd" (descriptor leaf p l)))

let () =
  exit
    (run "Tolk.Nv_packet"
       [
         group "encoded over nodes, evaluated at integers"
           [
             prop "every command is the command of the integers" command
               command_law;
             prop "a copy is the copy of the integers"
               Gen.(pair (pair value value) copy_size)
               copy_law;
             prop "a launch descriptor is the descriptor of the integers"
               (Gen.with_pp pp_launch launch)
               launch_law;
           ];
       ])
