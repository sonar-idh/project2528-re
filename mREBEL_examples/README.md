# mREBEL Multilingual Relation Extraction

Demonstrates multilingual relation extraction using [mREBEL](https://huggingface.co/Babelscape/mrebel-large) on 10 exemplary sentences covering geography, history, and science in both English and German.

## Model

**mREBEL** is an end-to-end relation extraction model based on mBART (multilingual BART encoder-decoder), trained on 400+ Wikidata relation types and 17+ languages. Unlike classification-based approaches that require pre-identified entity pairs, mREBEL generates relation triplets using special tokens (`[PER]`, `[LOC]`, `[ORG]`, etc.) directly from raw text using sequence-to-sequence generation.

## Results

| # | Sentence (EN) | Sentence (DE) | English Extraction | German Extraction |
|---|---|---|---|---|
| 1 | Robert Koch was a microbiologist who is widely regarded as a founder of modern bacteriology. | Robert Koch war ein Mikrobiologe, der weithin als Begründer der modernen Bakteriologie gilt. | [PER] Robert Koch --[occupation]--> [CONCEPT] microbiologist | [CONCEPT] Bakteriologie --[part of]--> [CONCEPT] Mikrobiologe |
| 2 | The Elbe River flows into the North Sea. | Die Elbe fließt in die Nordsee. | [LOC] Elbe River --[mouth of the watercourse]--> [LOC] North Sea | [LOC] Die Elbe --[located on terrain feature]--> [LOC] Nordsee |
| 3 | Berlin is the capital and largest city of Germany. | Berlin ist die Hauptstadt und größte Stadt Deutschlands. | [LOC] Berlin --[instance of]--> [CONCEPT] capital<br>[LOC] Germany --[capital]--> [LOC] Berlin | [LOC] Deutschland --[capital]--> [LOC] Berlin |
| 4 | The Zugspitze belongs to the Wetterstein mountain range in the Alps. | Die Zugspitze gehört zum Wettersteingebirge in den Alpen. | [LOC] Zugspitze --[mountain range]--> [LOC] Wetterstein | [LOC] Zugspitze --[mountain range]--> [LOC] Wettersteingebirge |
| 5 | The Cologne Cathedral is located in the city of Cologne, Germany. | Der Kölner Dom befindet sich in der Stadt Köln in Deutschland. | [LOC] Cologne Cathedral --[country]--> [LOC] Germany<br>[LOC] Cologne --[country]--> [LOC] Germany | [LOC] Kölner Dom --[located in the administrative territorial entity]--> [LOC] Köln<br>[LOC] Kölner Dom --[country]--> [LOC] Deutschland<br>[LOC] Köln --[country]--> [LOC] Deutschland |
| 6 | The Brandenburg Gate was designed by architect Carl Gotthard Langhans. | Das Brandenburger Tor wurde von dem Architekten Carl Gotthard Langhans entworfen. | [LOC] Brandenburg Gate --[architect]--> [PER] Carl Gotthard Langhans | [LOC] Brandenburger Tor --[architect]--> [PER] Carl Gotthard Langhans |
| 7 | Johannes Gutenberg invented the printing press in Europe. | Johannes Gutenberg erfand die Druckpresse in Europa. | [CONCEPT] printing press --[discoverer or inventor]--> [PER] Johannes Gutenberg | [ORG] Druckpresse --[founded by]--> [PER] Johannes Gutenberg |
| 8 | Neuschwanstein Castle is located in Schwangau, Bavaria, Germany. | Schloss Neuschwanstein befindet sich in Schwangau in Bayern, Deutschland. | [LOC] Neuschwanstein Castle --[located in the administrative territorial entity]--> [LOC] Schwangau<br>[LOC] Neuschwanstein Castle --[country]--> [LOC] Germany<br>[LOC] Schwangau --[country]--> [LOC] Germany<br>[LOC] Bavaria --[country]--> [LOC] Germany | [LOC] Schloss Neuschwanstein --[located in the administrative territorial entity]--> [LOC] Schwangau<br>[LOC] Schloss Neuschwanstein --[country]--> [LOC] Deutschland<br>[LOC] Schwangau --[country]--> [LOC] Deutschland<br>[LOC] Bayern --[country]--> [LOC] Deutschland |
| 9 | Albert Einstein became a citizen of Switzerland in 1901. | Albert Einstein wurde 1901 Bürger der Schweiz. | [PER] Albert Einstein --[country of citizenship]--> [LOC] Switzerland | [PER] Albert Einstein --[award received]--> [PER] Bürger der Schweiz |
| 10 | Richard Wagner was a leading figure of the War of the Romantics. | Richard Wagner war eine führende Figur des Parteienstreits. | [PER] Richard Wagner --[conflict]--> [EVE] War of the Romantics | [ORG] Parteienstreits --[chairperson]--> [PER] Richard Wagner |
```
