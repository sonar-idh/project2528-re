### project2528-re

**Investigation, implementation, and comparative evaluation of relation extraction (RE) models for historical texts, focusing on their suitability for historical network analysis.**

Comparison of classification-based relation extraction ([OpenNRE](https://github.com/thunlp/OpenNRE)) vs generative end-to-end extraction ([mREBEL](https://huggingface.co/Babelscape/mrebel-large)) on 10 sentences covering geography, history, and science.

## Methodology Comparison

| Aspect | OpenNRE (Classification) | mREBEL (Generation) |
|---|---|---|
| **Approach** | Entity pair ? relation classification | Raw text ? triplet generation |
| **Input requirement** | Pre-identified entity pairs (head, tail) | Raw text only |
| **Output** | Single relation + confidence score per pair | All entities, types, and relations |
| **Schema** | Fixed 80 relations (wiki80) | 400+ relations (Wikidata) |
| **Language** | English only | 17+ languages |
| **Training data** | 56K instances from [wiki80](https://github.com/thunlp/OpenNRE/blob/master/benchmark/rel4wiki80.md) | [RED<sup>FM</sup>](https://arxiv.org/abs/2306.09802) (multilingual Wikidata) |

