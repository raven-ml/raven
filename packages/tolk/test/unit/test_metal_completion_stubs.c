/* Copyright (c) 2026 The Raven authors. ISC License. */

/* Private copy of the actual native boundary; never linked with tolk.metal.
   Supply a completed command without submitting invalid work to a real GPU. */
#include "tolk_metal_stubs.c"

@interface TolkTestCommand : NSObject
@property(nonatomic) MTLCommandBufferStatus status;
@property(nonatomic, retain) NSError* error;
@end
@implementation TolkTestCommand
- (CFTimeInterval)GPUStartTime { return 1.0; }
- (CFTimeInterval)GPUEndTime { return 2.0; }
- (void)dealloc { [_error release]; [super dealloc]; }
@end

static tolk_metal_hcq context = {
  .lock = PTHREAD_MUTEX_INITIALIZER,
  .completion = PTHREAD_COND_INITIALIZER
};
static uint64_t stamps[2];

CAMLprim value caml_test_metal_completion_setup(value profile) {
  CAMLparam1(profile);
  atomic_store(&context.completed, 0);
  atomic_store(&context.failed, 0);
  atomic_store(&context.pending_timestamps, Bool_val(profile));
  stamps[0] = stamps[1] = 0;
  CAMLreturn(caml_copy_nativeint((intnat)&context));
}

CAMLprim value caml_test_metal_complete(value v_signal, value message, value profile) {
  CAMLparam3(v_signal, message, profile);
  @autoreleasepool {
    TolkTestCommand* command = [TolkTestCommand new];
    command.status = caml_string_length(message) == 0
      ? MTLCommandBufferStatusCompleted : MTLCommandBufferStatusError;
    if (command.status == MTLCommandBufferStatusError) {
      NSString* description = [NSString stringWithUTF8String:String_val(message)];
      command.error = [NSError errorWithDomain:@"TolkTest" code:1
        userInfo:@{NSLocalizedDescriptionKey:description}];
    }
    tolk_metal_hcq_complete(&context, (id<MTLCommandBuffer>)command,
      (uint64_t)Int64_val(v_signal), Bool_val(profile) ? &stamps[0] : NULL,
      Bool_val(profile) ? &stamps[1] : NULL);
    [command release];
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_test_metal_completion_poll(value unit) {
  CAMLparam1(unit);
  CAMLreturn(caml_copy_int64(tolk_metal_hcq_poll((uint64_t)(uintptr_t)&context)));
}

CAMLprim value caml_test_metal_completion_stamps(value unit) {
  CAMLparam1(unit);
  CAMLlocal3(result, first, last);
  first = caml_copy_int64(stamps[0]);
  last = caml_copy_int64(stamps[1]);
  result = caml_alloc_tuple(2);
  Store_field(result, 0, first);
  Store_field(result, 1, last);
  CAMLreturn(result);
}

CAMLprim value caml_test_metal_failed_submit(value unit) {
  CAMLparam1(unit);
  /* A latched failure must stop before touching an ICB or its arguments. */
  tolk_metal_hcq_submit((uint64_t)(uintptr_t)&context, NULL, 2, 0);
  CAMLreturn(Val_unit);
}
