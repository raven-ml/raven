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

/* Observe the actual shared encoder's command order without submitting a GPU
   command. Launch sizes are changed in the owned argument storage between calls. */
@interface TolkTestArguments : NSObject {
@public
  uint64_t sizes[6];
}
@end
@implementation TolkTestArguments
- (void*)contents { return sizes; }
@end

@interface TolkTestDispatchEncoder : NSObject
@property(nonatomic, retain) NSMutableString* trace;
@end
@implementation TolkTestDispatchEncoder
- (void)executeCommandsInBuffer:(id)buffer withRange:(NSRange)range {
  (void)buffer;
  [_trace appendFormat:@"I%lu:%lu;", (unsigned long)range.location, (unsigned long)range.length];
}
- (void)memoryBarrierWithScope:(MTLBarrierScope)scope {
  (void)scope;
  [_trace appendString:@"B;"];
}
- (void)setComputePipelineState:(id)pipeline { (void)pipeline; }
- (void)setBuffer:(id)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
  (void)buffer;
  [_trace appendFormat:@"A%lu:%lu;", (unsigned long)offset, (unsigned long)index];
}
- (void)dispatchThreadgroups:(MTLSize)global threadsPerThreadgroup:(MTLSize)local {
  [_trace appendFormat:@"D%lu/%lu;", (unsigned long)global.width, (unsigned long)local.width];
}
- (void)dealloc { [_trace release]; [super dealloc]; }
@end

CAMLprim value caml_test_metal_shared_encode(value v_profile) {
  CAMLparam1(v_profile);
  CAMLlocal1(result);
  @autoreleasepool {
    TolkTestArguments* arguments = [TolkTestArguments new];
    uint64_t initial[6] = {7, 1, 1, 2, 1, 1};
    memcpy(arguments->sizes, initial, sizeof(initial));
    TolkTestDispatchEncoder* encoder = [TolkTestDispatchEncoder new];
    encoder.trace = [NSMutableString string];
    tolk_metal_program program = {0};
    uint64_t header[5 + 1 + 4 * 6] = {0, 6, 0, 1, (uint64_t)(uintptr_t)arguments};
    uint64_t counts[6] = {15, 15, 16, 29, 15, 15};
    header[5] = (uint64_t)(uintptr_t)&program;
    for (int i = 0; i < 6; i++) {
      header[6 + 4 * i] = (uint64_t)(uintptr_t)&program;
      header[7 + 4 * i] = counts[i];
      header[8 + 4 * i] = 256 * i;
      header[9 + 4 * i] = 0;
    }
    for (int replay = 0; replay < 2; replay++) {
      if (replay) { [encoder.trace appendString:@"|"]; arguments->sizes[0] = 11; }
      if (Bool_val(v_profile)) {
        for (int i = 0; i < 6; i++)
          tolk_metal_hcq_encode_commands((id<MTLComputeCommandEncoder>)encoder, header, i, 1);
      } else tolk_metal_hcq_encode_commands((id<MTLComputeCommandEncoder>)encoder, header, 0, 6);
    }
    result = caml_copy_string(encoder.trace.UTF8String);
    [encoder release];
    [arguments release];
  }
  CAMLreturn(result);
}
