/* Forbids this process to start any other, for the rest of its life: on
   Windows by a job that holds one active process, this one; elsewhere by a
   limit of one process for its user, which already runs more. */

#include <caml/fail.h>
#include <caml/mlvalues.h>

#ifdef _WIN32
#include <windows.h>

value tolk_test_forbid_spawn(value unit) {
  (void)unit;
  HANDLE job = CreateJobObjectW(NULL, NULL);
  JOBOBJECT_BASIC_LIMIT_INFORMATION limit = {0};
  limit.LimitFlags = JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
  limit.ActiveProcessLimit = 1;
  if (job == NULL ||
      !SetInformationJobObject(job, JobObjectBasicLimitInformation, &limit,
                               sizeof limit) ||
      !AssignProcessToJobObject(job, GetCurrentProcess()))
    caml_failwith("forbid_spawn: the job was refused");
  return Val_unit;
}

#else
#include <sys/resource.h>

value tolk_test_forbid_spawn(value unit) {
  (void)unit;
  struct rlimit limit;
  if (getrlimit(RLIMIT_NPROC, &limit) != 0)
    caml_failwith("forbid_spawn: getrlimit");
  limit.rlim_cur = 1;
  if (setrlimit(RLIMIT_NPROC, &limit) != 0)
    caml_failwith("forbid_spawn: setrlimit");
  return Val_unit;
}
#endif
