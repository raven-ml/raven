type execution_result = {
  output : string;
  error : string option;
  timestamp : float;
}

type t = {
  executed : Document.block_id list;
  results : (Document.block_id, execution_result) Hashtbl.t;
  stale : (Document.block_id, unit) Hashtbl.t;
}

let empty =
  { executed = []; results = Hashtbl.create 16; stale = Hashtbl.create 16 }

let is_executed t block_id = List.mem block_id t.executed
let is_stale t block_id = Hashtbl.mem t.stale block_id

let can_execute t doc block_id =
  match Document.find_block doc block_id with
  | None -> false
  | Some block -> (
      match block.content with
      | Document.Codeblock _ ->
          (* Can execute if all previous code blocks have been executed *)
          let prev_codeblocks =
            Document.get_codeblocks doc
            |> List.filter (fun (id, _, _) -> id < block_id)
          in
          List.for_all
            (fun (id, _, _) -> is_executed t id && not (is_stale t id))
            prev_codeblocks
      | _ -> false)

let mark_executed t block_id result =
  Hashtbl.replace t.results block_id result;
  Hashtbl.remove t.stale block_id;
  let executed =
    if List.mem block_id t.executed then t.executed
    else t.executed @ [ block_id ]
  in
  { t with executed }

let mark_stale_from t doc block_id =
  let blocks_to_mark = Document.blocks_from doc block_id in
  List.iter (fun id -> Hashtbl.replace t.stale id ()) blocks_to_mark;
  t

let get_result t block_id = Hashtbl.find_opt t.results block_id

let clear_results _ =
  { executed = []; results = Hashtbl.create 16; stale = Hashtbl.create 16 }
