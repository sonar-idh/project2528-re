# OpenNRE Relation Extraction

Demonstrates three relation extraction models from [OpenNRE](https://github.com/thunlp/OpenNRE) on 10 exemplary sentences covering geography, history and science.

## Models

All three models are trained on [wiki80](https://github.com/thunlp/OpenNRE/blob/master/benchmark/rel4wiki80.md) - a sentence-level relation extraction dataset of 56,000 instances across 80 Wikidata relation types, sourced from Wikipedia. The three models used are `wiki80_cnn_softmax` (CNN), `wiki80_bert_softmax` (BERT) and `wiki80_bertentity_softmax` (BERT + entity representation concatenation).

## Results

| # | Sentence | Head | Tail | Expected | CNN | BERT | BERT-Entity |
|---|---|---|---|---|---|---|---|
| 1 | Robert Koch was a microbiologist that is widely regarded as a founder of modern bacteriology and a key contributor to the germ theory of disease. | Robert Koch | microbiologist | occupation (P106) | field of work (0.848) | field of work (0.999) | field of work (0.995) |
| 2 | The Elbe flows into the North Sea. | Elbe | North Sea | mouth of the watercourse (P403) | mouth of the watercourse (0.981) | mouth of the watercourse (0.994) | mouth of the watercourse (0.994) |
| 3 | Berlin is the largest city in Germany by both population and is completely surrounded by the state of Brandenburg. | Berlin | Germany | country (P17) | country (0.364) | country (0.729) | country (0.980) |
| 4 | The Zugspitze belongs to the Wetterstein range of the Northern Limestone Alps and it measures exactly 2962 meters. | Zugspitze | Wetterstein | mountain range (P4552) | mountain range (0.999) | mountain range (0.991) | mountain range (0.974) |
| 5 | The Cologne Cathedral is located in the city of Cologne and an outstanding example of Gothic architecture. | Cologne Cathedral | Cologne | located in the administrative territorial entity (P131) | located in the administrative territorial entity (0.395) | location (0.496) | located in the administrative territorial entity (0.693) |
| 6 | The Brandenburg Gate was built from 1788 to 1791 by orders of King Frederick William II of Prussia, based on designs by the royal architect Carl Gotthard Langhans. | Brandenburg Gate | Carl Gotthard Langhans | architect (P84) | architect (1.000) | notable work (0.984) | architect (0.997) |
| 7 | Johannes Gutenberg is known for having designed and built the first known mechanized printing press in Europe. | Johannes Gutenberg | printing press | notable work (P800) | field of work (0.991) | notable work (0.683) | field of work (0.690) |
| 8 | Neuschwanstein Castle is located in Schwangau, Germany, and was built by King Ludwig II of Bavaria. | Neuschwanstein Castle | Schwangau | location (P276) | located in the administrative territorial entity (0.302) | located in the administrative territorial entity (0.894) | located in the administrative territorial entity (0.497) |
| 9 | Albert Einstein was stateless for five years before becoming a citizen of Switzerland. | Albert Einstein | Switzerland | country of citizenship (P27) | country of citizenship (0.481) | residence (0.988) | country of citizenship (0.616) |
| 10 | Richard Wagner was one of the leading figures of the War of the Romantics. | Richard Wagner | War of the Romantics | movement (P135) | movement (0.691) | participant (0.934) | movement (0.872) |

## Note

The OpenNRE checkpoints predate a BERT change in newer PyTorch that adds a `position_ids` key. The code patches `load_state_dict` to drop it before loading.