# SLO-aware Flow Control

SyMoNeuRaL includes an admission controller before expensive GPU work.

## Controls

```bash
SYMON_MAX_QUEUE=32
SYMON_MAX_CONCURRENT_TEXT=1
SYMON_MAX_CONCURRENT_IMAGE=1
SYMON_MAX_CONCURRENT_VISION=1
SYMON_SLO_TTFT_MS=2500
SYMON_SLO_ITL_MS=150
SYMON_SLO_IMAGE_MS=45000
```

## Behavior

Requests can be rejected before GPU work if:

- queue is full
- concurrency limit is reached
- VRAM hard limit is crossed

This is deliberate. A server that fails fast under overload is safer than one that OOMs the GPU and kills all sessions.

## Metrics

```bash
curl http://127.0.0.1:8000/metrics
curl http://127.0.0.1:8000/slots
```

Metrics include queue depth, rejected count, p95 latency snapshot, and VRAM.

## Autoscaling direction

For one GPU, autoscaling means admission control and model unload/load policy.

For multi-GPU/multi-node, place replicas behind a real load balancer and route by:

- model family
- prefix hash affinity
- current queue depth
- TTFT/ITL SLO pressure
- available VRAM

Do not add another worker HTTP layer inside this single-node repo unless you intentionally move to a replica/router architecture.
