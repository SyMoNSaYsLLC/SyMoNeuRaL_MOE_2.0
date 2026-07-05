# KV Cache and Prefix-aware Routing

## What vLLM owns

vLLM owns the actual KV cache blocks inside the engine. SyMoNeuRaL does not manually edit vLLM KV memory.

Enabled by default:

```bash
SYMON_VLLM_ENABLE_PREFIX_CACHING=1
```

vLLM automatic prefix caching allows requests sharing the same prefix to reuse existing KV computation when the backend supports it.

## What SyMoNeuRaL owns

SyMoNeuRaL tracks a lightweight prefix hash:

```text
prompt[:SYMON_PREFIX_CACHE_CHARS] -> sha256 -> prefix_hash
```

This gives routing affinity and observability:

```bash
curl http://127.0.0.1:8000/health | jq .kv_cache
```

## Knobs

```bash
SYMON_PREFIX_CACHE_CHARS=4096
SYMON_PREFIX_ROUTE_AFFINITY_SECONDS=600
SYMON_VLLM_MAX_NUM_BATCHED_TOKENS=8192
SYMON_VLLM_MAX_NUM_SEQS=16
```

## Future multi-replica routing

When you run multiple SyMoNeuRaL replicas, route repeated prefixes to the same warm replica when possible. That preserves KV locality.
