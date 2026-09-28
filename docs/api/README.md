# PACT C API

The public interface is declared in [`include/pact/pact.h`](../../include/pact/pact.h).

- [Runtime and device discovery](runtime.md)
- [Tensors and host access](tensors.md)
- [Operations and execution](operations.md)

The API currently provides one float32 operation. The ABI is still pre-1.0 and may change. Tensor
operations and runtime diagnostics are not thread-safe for concurrent use of one runtime.
