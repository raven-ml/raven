type task = { start_idx : int; end_idx : int; compute : int -> int -> unit }

type _ Effect.t += WaitCompletion : int -> unit Effect.t

type pool = {
  num_workers : int;
  task_assignments : task option array;
  completed : int Atomic.t;
  generation : int Atomic.t;
  mutex : Mutex.t;
  work_available : Condition.t;
}
[@@contended]

let current_pool = ref None

let setup_pool () =
  let num_workers = Domain.recommended_domain_count () - 1 in
  let task_assignments = Array.make num_workers None in
  let completed = Atomic.make 0 in
  let generation = Atomic.make 0 in
  let mutex = Mutex.create () in
  let work_available = Condition.create () in
  let (pool : pool) =
    {
      num_workers;
      task_assignments;
      completed;
      generation;
      mutex;
      work_available;
    }
  in
  let worker id =
    let last_gen = ref (-1) in
    while true do
      Mutex.lock pool.mutex;
      let current_gen = Atomic.get pool.generation in
      while pool.task_assignments.(id) = None && !last_gen = current_gen do
        Condition.wait pool.work_available pool.mutex
      done;
      let current_gen = Atomic.get pool.generation in
      if pool.task_assignments.(id) <> None then (
        let task = Option.get pool.task_assignments.(id) in
        pool.task_assignments.(id) <- None;
        last_gen := current_gen;
        Mutex.unlock pool.mutex;
        (try task.compute task.start_idx task.end_idx
         with exn ->
           Printf.eprintf "Worker %d: Exception in task: %s\n" id
             (Printexc.to_string exn);
           flush stderr);
        Atomic.incr pool.completed)
      else (
        (* New generation without task for us, loop back *)
        last_gen := current_gen;
        Mutex.unlock pool.mutex)
    done
  in
  for i = 0 to num_workers - 1 do
    ignore (Domain.spawn (fun () -> worker i))
  done;
  pool

let get_or_setup_pool () =
  match !current_pool with
  | Some pool -> pool
  | None ->
      let pool = setup_pool () in
      current_pool := Some pool;
      pool

let get_num_domains pool = pool.num_workers + 1

let run pool f =
  let open Effect.Deep in
  try_with f ()
    Effect.
      {
        effc =
          (fun (type a) (e : a t) ->
            match e with
            | WaitCompletion target ->
                Some
                  (fun (k : (a, unit) continuation) ->
                    let rec wait () =
                      if Atomic.get pool.completed >= target then continue k ()
                      else (
                        Domain.cpu_relax ();
                        wait ())
                    in
                    wait ())
            | _ -> None);
      }

let parallel_execute pool tasks =
  run pool (fun () ->
      let num_tasks = Array.length tasks in
      if num_tasks <> get_num_domains pool then
        invalid_arg
          "parallel_execute: number of tasks must equal num_workers + 1";
      Atomic.set pool.completed 0;
      Mutex.lock pool.mutex;
      Atomic.incr pool.generation;
      for i = 0 to pool.num_workers - 1 do
        pool.task_assignments.(i) <- Some tasks.(i)
      done;
      Condition.broadcast pool.work_available;
      Mutex.unlock pool.mutex;
      let main_task = tasks.(pool.num_workers) in
      main_task.compute main_task.start_idx main_task.end_idx;
      Effect.perform (WaitCompletion pool.num_workers))

let parallel_for pool start end_ compute_chunk =
  let total_iterations = end_ - start + 1 in
  if total_iterations <= 0 then ()
  else if total_iterations <= 1 then compute_chunk start (start + 1)
  else
    let total_domains = get_num_domains pool in
    let chunk_size = total_iterations / total_domains in
    let remainder = total_iterations mod total_domains in
    let tasks =
      Array.init total_domains (fun d ->
          let start_idx = start + (d * chunk_size) + min d remainder in
          let len = chunk_size + if d < remainder then 1 else 0 in
          let end_idx = start_idx + len in
          { start_idx; end_idx; compute = compute_chunk })
    in
    parallel_execute pool tasks

let parallel_for_reduce (pool @ portable) start end_ body reduce init =
  let total_domains = get_num_domains pool in
  let results = Array.make total_domains init in
  let chunk_size = (end_ - start + 1) / total_domains in
  let remainder = (end_ - start + 1) mod total_domains in
  let tasks =
    Array.init total_domains (fun d ->
        let start_idx = start + (d * chunk_size) + min d remainder in
        let len = chunk_size + if d < remainder then 1 else 0 in
        let end_idx = start_idx + len in
        let compute _ _ =
          (* Ignore args since start_idx and end_idx are captured *)
          let partial_result = body start_idx end_idx in
          results.(d) <- partial_result
        in
        { start_idx; end_idx; compute })
  in
  parallel_execute pool tasks;
  let final_result = ref init in
  for i = 0 to total_domains - 1 do
    final_result := reduce !final_result results.(i)
  done;
  !final_result
