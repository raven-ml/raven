type t = {
  device : Metal.device;
  library : Metal.library;
  command_queue : Metal.command_queue;
  pipeline_cache : (string, Metal.pipeline_state) Hashtbl.t;
}

let create () =
  let device = Metal.create_device () in
  let library = Metal.create_library_with_data device Metallib.basics in
  let command_queue = Metal.create_command_queue device in
  { device; library; command_queue; pipeline_cache = Hashtbl.create 10 }

let create_pipeline_state device library kernel_name =
  let function_ = Metal.create_function_with_name library kernel_name in
  let pipeline_state = Metal.create_compute_pipeline_state device function_ in
  pipeline_state

let get_or_create_pipeline ctx kernel_name =
  let key = kernel_name in
  match Hashtbl.find_opt ctx.pipeline_cache key with
  | Some pipeline -> pipeline
  | None ->
      let pipeline = create_pipeline_state ctx.device ctx.library kernel_name in
      Hashtbl.add ctx.pipeline_cache key pipeline;
      pipeline
