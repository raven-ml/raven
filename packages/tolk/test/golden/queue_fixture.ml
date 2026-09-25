(* Fixed-address inspection of the real encoders' typed initialization stores.
   This does not compile, allocate driver storage, or execute a submission. *)
open Tolk_uop
module U = Uop
module D = Dtype

let uint n = U.const (Const.int D.uint64 n)
let int n = U.const_int n
let buffers = ref []
let pointer ~device ~tag ~dtype ~size ~address =
  let u = U.placeholder ~slot:(U.fresh_buffer_slot ()) ~shape:[size] ~dtype ~device:(U.Single device) () |> U.with_tag tag in
  buffers := (u, address) :: !buffers;
  u

let rec allocation u =
  match U.op u, U.children u with
  | (Ops.After | Ops.Bitcast | Ops.Cast | Ops.Shrink | Ops.Index), source :: _ -> allocation source
  | _ -> u

let rec location u =
  match U.op u, U.children u with
  | (Ops.After | Ops.Bitcast | Ops.Cast), source :: _ -> location source
  | Ops.Shrink, [source; offset; _] ->
      let base, start = location source in
      base, start + U.sym_infer offset [] * D.itemsize (U.dtype source)
  | Ops.Index, source :: [index] ->
      let base, start = location source in
      base, start + U.sym_infer index [] * D.itemsize (U.dtype source)
  | _ -> u, 0

let address u =
  let base, offset = location u in
  let base_address = match List.find_opt (fun (b, _) -> U.equal b base) !buffers with
    | Some (_, value) -> value
    | None -> match U.node_tag base with
      | Some "program" -> 0x100000
      | Some "scratch" -> 0x200000
      | Some "qmd" | Some "kernargs" -> 0x300000
      | Some tag -> failwith ("unbound fixture address: " ^ tag)
      | None -> failwith "untagged fixture address" in
  base_address + offset

let value u =
  let bindings = U.toposort u |> List.filter_map (fun node ->
      if U.op node = Ops.Getaddr then Some (node, uint (address (U.src node).(0)))
      else None) in
  U.sym_infer (U.substitute ~walk:true bindings u) []

let blob ?(tail = []) tag encoded =
  let nodes = U.toposort encoded in
  let base = List.find (fun n -> U.op n = Ops.Param && U.node_tag n = Some tag) nodes in
  let size = U.max_numel base * D.itemsize (U.dtype base) in
  let result = Bytes.make (size - List.length tail * 4) '\000' in
  let trailer = Array.make (List.length tail) false in
  List.iter (fun node -> match U.as_store node with
      | Some {dst; value = word; gate = None} when U.equal (allocation dst) base ->
          let _, offset = location dst in
          begin
            match U.op word, U.arg word with
            | Ops.Binary, U.Arg.String bytes ->
                Bytes.blit_string bytes 0 result offset (min (String.length bytes) (Bytes.length result - offset))
            | _ when offset < Bytes.length result ->
                let width = D.itemsize (U.dtype word) in
                if offset < 0 || width > Bytes.length result - offset then failwith "fixture patch outside blob";
                let bits = Int64.of_int (value word) in
                for i = 0 to width - 1 do
                  Bytes.set_uint8 result (offset + i)
                    (Int64.to_int (Int64.logand (Int64.shift_right_logical bits (8 * i)) 255L))
                done
            | _ ->
                let index = (offset - Bytes.length result) / 4 in
                if offset mod 4 <> 0 || D.itemsize (U.dtype word) <> 4 || index >= Array.length trailer then
                  failwith "invalid progress trailer patch";
                trailer.(index) <- true;
                Option.iter (fun expected -> if value word <> expected then
                    failwith "unexpected progress trailer opcode") (List.nth tail index)
          end
      | _ -> ()) nodes;
  if Array.exists not trailer then failwith "incomplete progress trailer";
  result

let dump directory name chip bytes =
  let path = Filename.concat directory (name ^ "_" ^ chip ^ ".actual") in
  let oc = open_out path in
  Fun.protect ~finally:(fun () -> close_out oc) (fun () ->
      for i = 0 to Bytes.length bytes / 4 - 1 do
        Printf.fprintf oc "%08lx\n" (Bytes.get_int32_le bytes (i * 4))
      done)

let read path =
  let ic = open_in_bin path in
  Fun.protect ~finally:(fun () -> close_in ic) (fun () -> Bytes.of_string (really_input_string ic (in_channel_length ic)))

let program ~device ~dtype ~binary =
  let params = List.init 3 (fun slot -> U.param ~slot ~dtype ~shape:(int 32) ~device:(U.Single device) ()) in
  let var = U.variable ~name:"n" ~min_val:1 ~max_val:32 ~dtype:D.int32 () in
  let formal = U.replace var ~op:Ops.Param () in
  let kernel_info = U.{name = "simple_add"; applied_opts = []; opts_to_apply = None; estimates = None; beam = 0} in
  let info = U.{target = Target.of_string device; global_size = List.map (fun n -> Launch_int n) [4;3;2];
    local_size = List.map (fun n -> Launch_int n) [8;4;1]; vars = [formal]; globals = [0;1;2]; outs = []; ins = []} in
  let body = U.program ~sink:(U.sink ~kernel_info []) ~linear:(U.linear (params @ [formal]))
      ~source:(U.source "") ~binary:(U.binary (Bytes.to_string binary)) ~info () in
  let args = List.mapi (fun i address -> pointer ~device ~tag:("arg" ^ string_of_int i) ~dtype ~size:32 ~address)
      [0x900000;0xa00000;0xb00000] @ [U.bind ~var ~value:(int 32)] in
  U.call ~body ~args ~info:{grad_fxn=None;name=None;precompile=false;precompile_backward=false;aux=None;dtype=D.void}
