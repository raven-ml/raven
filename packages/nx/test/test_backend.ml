(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Who computes: an eager operation is computed by the backend of its operands'
   devices, in any domain, or raises before any work; a backend that lacks a
   kernel raises naming itself and nothing falls through; constants, views,
   reads and place need no kernel on any device; a device's name shows its
   backend unless the backend owns its memory. *)

open Windtrap
open Nx_test

let vec a = Nx.create Nx.float32 [| Array.length a |] a
let floats = tensor float_exact
let values = list float_exact
let elements x = Array.to_list (Nx.to_array x)

(* A memory the host addresses that loads programs, as a GPU: it has no default
   backend. *)
let gpu =
  Nx.Device.make
    (Nx_device.Driver.device ~name:"GPU" ~arch:"test" ~budget:max_int
       ~load:(fun ~binary:_ -> Error "no programs")
       (Host_visible { memory = Nx_device.Driver.host_memory; mapping = None }))

(* Test backends *)

let made = Atomic.make 0

(* A backend of nx.cpu's kernels that counts its additions and refuses products,
   each a backend of its own. *)
type counting = { device : Nx.Device.t; name : string; adds : int Atomic.t }

let counting on =
  let adds = Atomic.make 0 in
  let label = Printf.sprintf "counting %d" (Atomic.fetch_and_add made 1) in
  let module Cpu = (val Nx_backend.kernels Nx_cpu.backend) in
  let k =
    Nx_backend.v
      (module struct
        include Cpu

        let name = label
        let owns _ = false

        let binary k a b ~dst =
          if k = Nx_backend.Add then Atomic.incr adds;
          Cpu.binary k a b ~dst

        let matmul _ _ ~dst:_ = raise (Nx_backend.Refused "no products here")
      end)
  in
  { device = Nx.Device.with_backend k on; name = label; adds }

(* A backend that refuses every kernel. *)
module Refusing = struct
  let name = "refusing"
  let runs_on _ = true
  let owns _ = false
  let no _ = raise (Nx_backend.Refused "no kernel at all")
  let unary _ _ ~dst:_ = no ()
  let binary _ _ _ ~dst:_ = no ()
  let compare _ _ _ ~dst:_ = no ()
  let fma _ _ _ ~dst:_ = no ()
  let where _ _ _ ~dst:_ = no ()
  let cast _ ~dst:_ = no ()
  let threefry _ _ ~dst:_ = no ()
  let reduce _ ~axes:_ _ ~dst:_ = no ()
  let scan _ ~axis:_ _ ~dst:_ = no ()
  let arg_reduce _ ~axis:_ _ ~dst:_ = no ()
  let sort ~descending:_ ~axis:_ _ ~dst:_ = no ()
  let argsort ~descending:_ ~axis:_ _ ~dst:_ = no ()
  let group _ ~dst:_ = no ()
  let pad _ _ _ ~dst:_ = no ()
  let cat ~axis:_ _ ~dst:_ = no ()
  let contiguous _ ~dst:_ = no ()
  let gather ~axis:_ _ _ ~dst:_ = no ()
  let scatter ~mode:_ ~unique:_ ~axis:_ ~indices:_ ~updates:_ _ ~dst:_ = no ()
  let update _ ~starts:_ _ ~dst:_ = no ()
  let unfold ~kernel_size:_ ~stride:_ ~dilation:_ ~padding:_ _ ~dst:_ = no ()

  let fold ~output_size:_ ~kernel_size:_ ~stride:_ ~dilation:_ ~padding:_ _
      ~dst:_ =
    no ()

  let matmul _ _ ~dst:_ = no ()
  let fft ~inverse:_ ~axes:_ _ ~dst:_ = no ()
  let rfft ~axes:_ _ ~dst:_ = no ()
  let irfft ~axes:_ ~s:_ _ ~dst:_ = no ()
  let cholesky ~upper:_ _ ~dst:_ = no ()
  let qr ~reduced:_ _ ~q:_ ~r:_ = no ()
  let lu _ ~lu:_ ~pivots:_ ~perm:_ = no ()
  let svd _ ~u:_ ~s:_ ~vt:_ = no ()
  let eig _ ~values:_ ~vectors:_ = no ()
  let eigh _ ~values:_ ~vectors:_ = no ()
  let solve_triangular ~upper:_ ~transpose:_ ~unit_diag:_ _ _ ~dst:_ = no ()
end

let refusing = Nx_backend.v (module Refusing)

let product_refused name =
  Printf.sprintf
    "Nx.matmul: %s on CPU:1 has no kernel: no products here. Compile it with \
     Rune.jit, or place the operands elsewhere."
    name

(* Law 1: who computes is a property of the operands.

   The system pairs a counting backend with the test memory CPU:1; the model
   counts the additions it must have computed. Additions on the paired device
   count, wherever they run; additions of host values and on the unpaired device
   do not; a product the backend refuses raises naming it, and reaches no other
   backend. On two domains, every result must be explained by some order of the
   calls. *)

module Model = struct
  type t = { mutable adds : int }
end

let backend =
  abstract "k" ~invariant:(fun (m : Model.t) s ->
      equal ~msg:"additions" int m.adds (Atomic.get s.adds))

(* Values whose sums float32 holds exactly. *)
let small =
  Gen.list ~size:(Gen.int_range 0 4)
    (Gen.map float_of_int (Gen.int_range (-1000) 1000))

let sums xs = List.map (fun x -> x +. x) xs

(* The elements of [y] and the device that holds them: ["paired"] for [s]'s, by
   name otherwise. *)
let held s y =
  ( elements y,
    match Nx.Placement.devices (Nx.placement y) with
    | [ d ] when Nx.Device.equal d s.device -> "paired"
    | ds -> String.concat "," (List.map Nx.Device.name ds) )

let added s at xs =
  let x = Nx.place at (vec (Array.of_list xs)) in
  held s (Nx.add x x)

let outcome = pair values string

let commands =
  [
    command "pair"
      (Gen.unit @-> makes backend)
      (fun () -> { Model.adds = 0 })
      (fun () -> counting (Nx.Device.cpu 1));
    command "add on the paired device"
      (backend ^-> small @-> returns outcome)
      (fun m xs ->
        m.adds <- m.adds + 1;
        (sums xs, "paired"))
      (fun s xs -> added s (Nx.Placement.on s.device) xs);
    command "add a host value to a paired one"
      (backend ^-> small @-> returns outcome)
      (fun m xs ->
        m.adds <- m.adds + 1;
        (sums xs, "paired"))
      (fun s xs ->
        let host = vec (Array.of_list xs) in
        held s (Nx.add (Nx.place (Nx.Placement.on s.device) host) host));
    command "add on the host"
      (backend ^-> small @-> returns outcome)
      (fun _ xs -> (sums xs, "CPU"))
      (fun s xs -> added s Nx.Placement.host xs);
    command "add on the unpaired device"
      (backend ^-> small @-> returns outcome)
      (fun _ xs -> (sums xs, "CPU:1"))
      (fun s xs -> added s (Nx.Placement.on (Nx.Device.cpu 1)) xs);
    command "a product the backend refuses"
      (backend ^-> returns (result values string))
      (fun _ -> Error "refused, naming the paired backend")
      (fun s ->
        let x =
          Nx.place (Nx.Placement.on s.device) (Nx.ones Nx.float32 [| 2; 2 |])
        in
        match Nx.matmul x x with
        | y -> Ok (elements y)
        | exception Invalid_argument why when why = product_refused s.name ->
            Error "refused, naming the paired backend"
        | exception Invalid_argument why -> Error why);
    command "additions so far"
      (backend ^-> returns int)
      (fun m -> m.adds)
      (fun s -> Atomic.get s.adds);
  ]

let who_computes =
  group "who computes"
    [
      stateful "an operation computes with its operands' backend" commands;
      stateful ~domains:2
        "an operation computes with its operands' backend in any domain"
        commands;
      test "a device without a backend raises before any work, naming remedies"
        (fun () ->
          let x = Nx.place (Nx.Placement.on gpu) (vec [| 1.; 2. |]) in
          raises
            (Invalid_argument
               "Nx.add: GPU has no eager kernels. Compile it with Rune.jit, \
                pair the device with a backend (Nx.Device.with_backend), or \
                place the operands on Nx.Placement.host.") (fun () ->
              ignore (Nx.add x x)));
      test "a backend paired with the host computes on its views of host values"
        (fun () ->
          let k = counting Nx.Device.host in
          let p = Nx.Placement.on k.device in
          let y = Nx.add (Nx.place p (vec [| 1.; 2. |])) (vec [| 3.; 4. |]) in
          equal ~msg:"one addition through the backend" int 1
            (Atomic.get k.adds);
          equal Devices.placement p (Nx.placement y);
          equal floats (vec [| 4.; 6. |]) y;
          ignore (Nx.add (vec [| 1. |]) (vec [| 2. |]));
          equal ~msg:"fresh host values compute with nx.cpu" int 1
            (Atomic.get k.adds));
      test
        "operands on two devices over one memory raise, naming both and \
         Nx.place" (fun () ->
          let k = counting (Nx.Device.cpu 1) in
          let x = Nx.place (Nx.Placement.on (Nx.Device.cpu 1)) (vec [| 1. |]) in
          let y = Nx.place (Nx.Placement.on k.device) x in
          raises_match (Exn.invalid_arg ~substring:"CPU:1 and CPU:1/counting")
            (fun () -> Nx.add x y);
          raises_match (Exn.invalid_arg ~substring:"Nx.place") (fun () ->
              Nx.add x y);
          equal int 0 (Atomic.get k.adds));
    ]

(* Law 2: movement is total. On a device paired with a backend that refuses
   every kernel, and on one without a backend, while another domain is inside an
   interception as a compiled function's trace is: constants, views, reads and
   place compute nothing. *)

let while_intercepting f =
  let inside = Atomic.make false and stop = Atomic.make false in
  let pass =
    { Nx.Op.run = (fun _ -> assert false); claims = (fun _ -> false) }
  in
  let tracer =
    Domain.spawn (fun () ->
        Nx.Op.intercept pass (fun () ->
            Atomic.set inside true;
            while not (Atomic.get stop) do
              Domain.cpu_relax ()
            done))
  in
  while not (Atomic.get inside) do
    Domain.cpu_relax ()
  done;
  Fun.protect
    ~finally:(fun () ->
      Atomic.set stop true;
      Domain.join tracer)
    f

let movement_on (name, d) =
  test name (fun () ->
      while_intercepting @@ fun () ->
      let p = Nx.Placement.on d in
      let host = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
      let x = Nx.place p host in
      let beside y =
        equal ~msg:"beside" Devices.placement p (Nx.placement y);
        y
      in
      equal ~msg:"zeros_like" floats
        (Nx.zeros Nx.float32 [| 2; 3 |])
        (beside (Nx.zeros_like x));
      equal ~msg:"full_like" floats
        (Nx.full Nx.float32 [| 2; 3 |] 7.)
        (beside (Nx.full_like x 7.));
      equal ~msg:"a transpose, read" floats (Nx.transpose host)
        (beside (Nx.transpose x));
      equal ~msg:"a slice, read" floats
        (Nx.slice [ R (0, 1); R (1, 3) ] host)
        (beside (Nx.slice [ R (0, 1); R (1, 3) ] x));
      equal ~msg:"a flip and a broadcast, read" floats
        (Nx.broadcast_to [| 2; 2; 3 |] (Nx.flip host))
        (beside (Nx.broadcast_to [| 2; 2; 3 |] (Nx.flip x)));
      equal ~msg:"a reshape the strides express" floats
        (Nx.reshape [| 3; 2 |] host)
        (beside (Nx.reshape [| 3; 2 |] x));
      equal ~msg:"an element" float_exact 6. (Nx.item [ 1; 2 ] x);
      equal ~msg:"placed on the host" floats host
        (Nx.place Nx.Placement.host (Nx.transpose (Nx.transpose x)));
      equal ~msg:"placed on another memory" floats host
        (Nx.place (Nx.Placement.on (Nx.Device.cpu 2)) x))

let movement =
  group "movement is total"
    (List.map movement_on
       [
         ( "beside a device whose backend refuses every kernel, while another \
            domain is intercepting",
           Nx.Device.with_backend refusing (Nx.Device.cpu 1) );
         ( "beside a device without a backend, while another domain is \
            intercepting",
           gpu );
       ]
    @ [
        test
          "a refused kernel raises naming the backend, the device and remedies"
          (fun () ->
            let d = Nx.Device.with_backend refusing (Nx.Device.cpu 1) in
            let x = Nx.place (Nx.Placement.on d) (vec [| 1.; 2. |]) in
            raises
              (Invalid_argument
                 "Nx.exp: refusing on CPU:1 has no kernel: no kernel at all. \
                  Compile it with Rune.jit, or place the operands elsewhere.")
              (fun () -> ignore (Nx.exp x)));
      ])

(* Names *)

(* A fresh memory the host does not compute on, and a backend that computes on
   it and owns it, as a vendor's library's backend owns its memories. *)
let own_memory name =
  Nx_device.Driver.device ~name ~arch:"test" ~budget:max_int
    ~load:(fun ~binary:_ -> Error "no programs")
    (Host_visible { memory = Nx_device.Driver.host_memory; mapping = None })

let owning m =
  Nx_backend.v
    (module struct
      include Refusing

      let name = "owning"
      let runs_on m' = Nx_device.equal m m'
      let owns = runs_on
    end)

let names =
  group "names"
    [
      test "a device of a memory's own backend is named after the memory"
        (fun () ->
          let m = own_memory "OWN" in
          equal string "OWN"
            (Nx.Device.name (Nx.Device.make ~backend:(owning m) m)));
      test "a backend that does not own its memory is named after it" (fun () ->
          equal string "CPU:1/refusing"
            (Nx.Device.name (Nx.Device.with_backend refusing (Nx.Device.cpu 1))));
      test "a memory without a backend is named alone" (fun () ->
          equal string "OWN:2"
            (Nx.Device.name (Nx.Device.make (own_memory "OWN:2"))));
    ]

let () = exit (run "nx backends" [ who_computes; movement; names ])
