/* Copyright (c) 2026 The Raven authors. ISC License. */

/* Private copy of the actual native boundary; never linked with tolk.metal.
   Exercise ordered collection without submitting invalid work to a real GPU. */
#include "tolk_metal_stubs.c"

typedef struct {
  tolk_metal_hcq queue;
  uint64_t stamps[2][2];
  int released;
} tolk_test_context;

@interface TolkTestCommand : NSObject
@property(nonatomic) MTLCommandBufferStatus status;
@property(nonatomic, retain) NSError* error;
@property(nonatomic) tolk_test_context* owner;
@end
@implementation TolkTestCommand
- (CFTimeInterval)GPUStartTime { return 1.0; }
- (CFTimeInterval)GPUEndTime { return 2.0; }
- (void)dealloc { _owner->released++; [_error release]; [super dealloc]; }
@end

CAMLprim value caml_test_metal_completion_setup(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);
  result = caml_copy_nativeint(0);
  tolk_test_context* ctx = calloc(1, sizeof(*ctx));
  if (ctx == NULL) caml_raise_out_of_memory();
  if (pthread_mutex_init(&ctx->queue.lock, NULL) != 0) {
    free(ctx); caml_failwith("test Metal context creation failed");
  }
  Nativeint_val(result) = (intnat)ctx;
  CAMLreturn(result);
}

CAMLprim value caml_test_metal_enqueue(value v_ctx, value v_signal, value v_slot) {
  CAMLparam3(v_ctx, v_signal, v_slot);
  CAMLlocal1(result);
  result = caml_copy_nativeint(0);
  tolk_test_context* ctx = (tolk_test_context*)Nativeint_val(v_ctx);
  int slot = Int_val(v_slot);
  if (slot < -1 || slot > 1) caml_invalid_argument("test Metal timestamp slot");
  TolkTestCommand* command = [TolkTestCommand new];
  command.owner = ctx;
  command.status = MTLCommandBufferStatusScheduled;
  pthread_mutex_lock(&ctx->queue.lock);
  int added = tolk_metal_hcq_enqueue(&ctx->queue, (id<MTLCommandBuffer>)command,
      Int64_val(v_signal), slot < 0 ? NULL : &ctx->stamps[slot][0],
      slot < 0 ? NULL : &ctx->stamps[slot][1]);
  pthread_mutex_unlock(&ctx->queue.lock);
  Nativeint_val(result) = (intnat)command;
  [command release];
  if (!added) caml_failwith("test Metal enqueue failed");
  CAMLreturn(result);
}

CAMLprim value caml_test_metal_complete(value v_ctx, value v_command, value message) {
  CAMLparam3(v_ctx, v_command, message);
  tolk_test_context* ctx = (tolk_test_context*)Nativeint_val(v_ctx);
  TolkTestCommand* command = (TolkTestCommand*)Nativeint_val(v_command);
  @autoreleasepool {
    pthread_mutex_lock(&ctx->queue.lock);
    command.status = caml_string_length(message) == 0
      ? MTLCommandBufferStatusCompleted : MTLCommandBufferStatusError;
    if (command.status == MTLCommandBufferStatusError) {
      NSString* description = [NSString stringWithUTF8String:String_val(message)];
      command.error = [NSError errorWithDomain:@"TolkTest" code:1
        userInfo:@{NSLocalizedDescriptionKey:description}];
    }
    pthread_mutex_unlock(&ctx->queue.lock);
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_test_metal_completion_poll(value v_ctx) {
  CAMLparam1(v_ctx);
  CAMLreturn(caml_copy_int64(tolk_metal_hcq_poll((uint64_t)Nativeint_val(v_ctx))));
}

CAMLprim value caml_test_metal_completion_stamps(value v_ctx, value v_slot) {
  CAMLparam2(v_ctx, v_slot);
  CAMLlocal3(result, first, last);
  tolk_test_context* ctx = (tolk_test_context*)Nativeint_val(v_ctx);
  int slot = Int_val(v_slot);
  if (slot < 0 || slot > 1) caml_invalid_argument("test Metal timestamp slot");
  uint64_t start = ctx->stamps[slot][0], finish = ctx->stamps[slot][1];
  first = caml_copy_int64(start);
  last = caml_copy_int64(finish);
  result = caml_alloc_tuple(2);
  Store_field(result, 0, first);
  Store_field(result, 1, last);
  CAMLreturn(result);
}

CAMLprim value caml_test_metal_released(value v_ctx) {
  CAMLparam1(v_ctx);
  tolk_test_context* ctx = (tolk_test_context*)Nativeint_val(v_ctx);
  CAMLreturn(Val_int(ctx->released));
}

CAMLprim value caml_test_metal_failed_submit(value v_ctx) {
  CAMLparam1(v_ctx);
  /* A latched failure must stop before touching an ICB or its arguments. */
  tolk_metal_hcq_submit((uint64_t)Nativeint_val(v_ctx), NULL, 3, 0);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_test_metal_completion_cleanup(value v_ctx) {
  CAMLparam1(v_ctx);
  tolk_test_context* ctx = (tolk_test_context*)Nativeint_val(v_ctx);
  /* These are fake commands with no GPU accesses. Quiesce them for test cleanup;
     production cannot clear a sticky fault without proving device retirement. */
  pthread_mutex_lock(&ctx->queue.lock);
  for (tolk_metal_pending* p = ctx->queue.first; p != NULL; p = p->next)
    ((TolkTestCommand*)p->command).status = MTLCommandBufferStatusCompleted;
  ctx->queue.error[0] = '\0';
  pthread_mutex_unlock(&ctx->queue.lock);
  CAMLreturn(caml_tolk_metal_hcq_release(v_ctx));
}
