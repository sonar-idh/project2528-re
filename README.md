## project2528-re

**Investigation, implementation, and comparative evaluation of relation extraction (RE) models for historical texts, focusing on their suitability for historical network analysis.**

Comparison of classification-based relation extraction ([OpenNRE](https://github.com/thunlp/OpenNRE)), generative end-to-end extraction ([mREBEL](https://huggingface.co/Babelscape/mrebel-large)), and zero-shot relation extraction ([GLiREL](https://github.com/jackboyla/GLiREL)) on 10 sentences covering geography, history, and science.

## Methodology Comparison

| Aspect | OpenNRE (Classification) | mREBEL (Generation) | GLiREL (Zero-shot) |
|---|---|---|---|
| **Approach** | Entity pair &rarr; relation classification | Raw text &rarr; triplet generation | Raw text &rarr; zero-shot triplet extraction |
| **Input requirement** | Pre-identified entity pairs (head, tail) | Raw text only | Raw text + pre-identified entities + candidate relation labels |
| **Output** | Single relation + confidence score per pair | All entities, types, and relations | Relations between entity pairs with similarity scores |
| **Schema** | Fixed 80 relations (wiki80) | 400+ relations (Wikidata) | Arbitrary relations (specified at inference) |
| **Language** | English only | 17+ languages | Multilingual (via DeBERTa V3) |
| **Training data** | 56K instances from [wiki80](https://github.com/thunlp/OpenNRE/blob/master/benchmark/rel4wiki80.md) | [RED<sup>FM</sup>](https://arxiv.org/abs/2306.09802) (multilingual Wikidata) | Synthetic data (63K texts, 25M+ relations via Mistral 7B) |
| **Architecture** | BERT encoder + classification head | mBART encoder-decoder (seq2seq) | DeBERTa V3-large encoder + entity pair module + scorer |