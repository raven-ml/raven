open Tolk_next
open Ops

let exec = 0
let copy = 1
let wait = 2
let store = 3
let timestamp = 4

type events = {
  ids : (string * string * string, int) Hashtbl.t;
  programs : (int, Ops.t) Hashtbl.t;
}

let events () = { ids = Hashtbl.create 16; programs = Hashtbl.create 16 }

let event events k =
  match Hashtbl.find_opt events.ids k with
  | Some i -> i
  | None ->
      let i = Hashtbl.length events.ids in
      Hashtbl.add events.ids k i;
      i

let program events e = Hashtbl.find events.programs e

let u64 n = int ~dtype:Dtype.Uint64 n
let device_name u = match device u with Some (Single d) -> d | _ -> ""

let commands events q : Hcq2.commands =
  let dev = List.hd (Hcq2.Queue.devices q) in
  (* A variable has no address: it is written by value. *)
  let word a = if is_variable a then a else getaddr ~device:dev a in
  let cmd op args =
    let words = (u64 op :: List.map word args) @ [ u64 0; u64 0; u64 0 ] in
    ignore (Hcq2.Queue.q q (List.filteri (fun i _ -> i < 4) words))
  in
  let exec_ call prg =
    let args =
      List.map word (Realize.get_call_arg_uops call)
      @ List.map
          (fun v -> cast v Dtype.Uint64)
          (Realize.get_call_var_uops call prg)
    in
    let kernargs =
      v Op.Linear
        ~src:
          (Hcq2.pack_args (Hcq2.layout_args args)
             (8 * max (List.length args) 1))
        ~arg:(String "kernargs")
    in
    let name =
      match arg (nth prg 0) with Kernel k -> function_name k | _ -> ""
    in
    let e = event events (dev, name, key prg) in
    Hashtbl.replace events.programs e prg;
    cmd exec [ kernargs; u64 (List.length args); u64 e ]
  in
  let copy_ dst src _ =
    let s = device_name src in
    cmd copy
      [
        dst;
        src;
        u64 (event events (s ^ ":SDMA:0", s ^ " -> " ^ device_name dst, ""));
      ]
  in
  {
    exec = exec_;
    copy = copy_;
    wait = (fun signal value -> cmd wait [ signal; value; u64 0 ]);
    signal = (fun signal value -> cmd store [ signal; value ]);
    timestamp =
      (fun slot -> cmd timestamp [ add (getaddr ~device:dev slot) (u64 8) ]);
    memory_barrier = (fun () -> ());
    submit =
      (fun cmdbuf ->
        let doorbell =
          placeholder
            ~device:(Multi (Hcq2.Queue.devices q))
            ~tag:(Tag.String "doorbell") [ 1 ] Dtype.Uint8
        in
        Ops.store (index doorbell [ int 0 ]) (load (index cmdbuf [ int 0 ]) []));
  }
