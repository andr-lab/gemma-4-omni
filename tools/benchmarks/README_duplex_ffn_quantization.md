# Microbenchmark quantizzazione FFN per il duplex

Questo probe misura uno o due blocchi FFN reali di Gemma 4 dal checkpoint locale, senza caricare l'intero modello e senza modificare engine, listener nativo o pesi in uso. Il pin sperimentale è `torchao==0.18.0`: la release documenta il supporto minimo PyTorch 2.11 e il passaggio della CI a PyTorch 2.12; il runner verifica comunque il pin e registra ogni backend non disponibile. Non installa pacchetti né scarica pesi.

Il benchmark confronta BF16 con TorchAO INT8 weight-only e INT4 weight-only, group size 128. Se Bitsandbytes è già installato, registra anche NF4 fused e, separatamente, la variante che espande i pesi NF4 in BF16 a ogni forward. Quest'ultima misura esplicitamente il costo di conversione richiesto da un consumer BF16, non è una previsione di velocità di un kernel low-bit.

## Esecuzione

Dal repository montato in `/workspace`, nel container runtime corrente:

```bash
python -m unittest tests.training.test_benchmark_duplex_ffn_quantization -v
python tools/benchmarks/benchmark_duplex_ffn_quantization.py \
  --model-path /workspace/.hf_cache/models--google--gemma-4-E2B-it/full \
  --layer-indices 15 \
  --compile-mode off \
  --warmup 8 --iterations 40 \
  --output-json /workspace/reports/ffn_quantization_microbenchmark.json
```

Il processo richiede una GPU NVIDIA Ampere o successiva e le versioni runtime predefinite Torch `2.12.1` e Transformers `5.5.0`. Il modello deve essere già presente nella cache locale. Per un secondo blocco si può usare, ad esempio, `--layer-indices 14,15`; oltre due blocchi il runner si arresta. Per includere i tempi di compilazione, si può scegliere esplicitamente `--compile-mode default` o `reduce-overhead`.

Il report include latenza CUDA e wall-clock, memoria GPU persistente del modulo dopo il primo forward e dopo aver liberato l'output, e picchi separati per costruzione e primo forward. BF16 e varianti usano la stessa baseline con un input sintetico Q4 già allocato; così sono incluse anche le quantizzazioni che materializzano i pesi al primo uso, mentre l'input non viene conteggiato come memoria del modello. Se un backend produce output ma il suo storage CUDA persistente non è visibile all'allocatore PyTorch, la memoria è segnata come `null` con stato esplicito, mai come zero. Registra inoltre l'errore numerico per tre forme sintetiche: Q4 del listener `[1,4,1536]`, decode Q1 `[1,1,1536]` e prefill Q4 `[1,4,1536]`. Gli input sono hidden state casuali normalizzati e riproducibili; il relativo errore di output è un confronto numerico del microbenchmark, non qualità conversazionale né accuratezza della policy. Non dimostra latenza first-audio. Prima di proporre un'integrazione vanno misurati sullo stesso audio listener RT80 (80 ms per tick), tempo first-audio e accuratezza/appropriatezza delle risposte con confronto appaiato contro BF16.

## Limite d'integrazione attuale

Il listener nativo condivide puntatori ai pesi BF16 CUDA e chiama il percorso cuBLAS BF16. Le varianti TorchAO del probe restituiscono output BF16 e conservano la forma della query, ma i loro pesi quantizzati non soddisfano l'ABI condivisa corrente. Il report registra quindi `quantized_shared_ffn_supported: false` e `native_listener_integration_tested: false`: un risultato favorevole al microbenchmark giustifica un futuro kernel/ABI nativo; non abilita da solo la sostituzione dei pesi in serving. Il percorso Bitsandbytes BF16 di confronto quantizza soltanto il peso temporaneo e lo dequantizza prima della `linear` BF16 a ogni forward.

Il riferimento a ds4 è metodologico: ds4 documenta ricette GGUF specifiche per il proprio modello DeepSeek e calibrazione imatrix; i suoi formati, mapping dei tensori e kernel non sono riutilizzabili direttamente per Gemma 4 o per l'ABI CUDA attuale.

## Fonti tecniche

- [TorchAO v0.18.0 release](https://github.com/pytorch/ao/releases/tag/v0.18.0): minimo PyTorch 2.11, CI spostata a PyTorch 2.12 e cambi di API v2.
- [TorchAO v0.18.0 inference documentation](https://github.com/pytorch/ao/blob/v0.18.0/docs/source/workflows/inference.md): `Int4WeightOnlyConfig` e group sizes supportati; `Int8WeightOnlyConfig` per pesi INT8.
- [TorchAO v0.18.0 quantization API](https://github.com/pytorch/ao/blob/v0.18.0/torchao/quantization/quant_api.py): API `quantize_`, configurazioni `Int4WeightOnlyConfig` e `Int8WeightOnlyConfig`.
- [Transformers 5.5.0 Gemma 4 implementation](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/gemma4/modeling_gemma4.py): definizione delle proiezioni FFN e del forward.
- [ds4 quantization tools](https://github.com/antirez/ds4/blob/main/gguf-tools/README.md): esempio di ricette selettive e calibrazione specifiche per quel modello.
