from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def extract_triplets_typed(text):
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_, object_type, subject_type = '', '', '', '', ''

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("tp_XX", "").replace("__en__", "").split():
        if token == "<triplet>" or token == "<relation>":
            current = 't'
            if relation != '':
                triplets.append({
                    'head': subject.strip(), 
                    'head_type': subject_type, 
                    'type': relation.strip(),
                    'tail': object_.strip(), 
                    'tail_type': object_type
                })
                relation = ''
            subject = ''
        elif token.startswith("<") and token.endswith(">"):
            if current == 't' or current == 'o':
                current = 's'
                if relation != '':
                    triplets.append({
                        'head': subject.strip(), 
                        'head_type': subject_type, 
                        'type': relation.strip(),
                        'tail': object_.strip(), 
                        'tail_type': object_type
                    })
                object_ = ''
                subject_type = token[1:-1]
            else:
                current = 'o'
                object_type = token[1:-1]
                relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
                
    if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
        triplets.append({
            'head': subject.strip(), 
            'head_type': subject_type, 
            'type': relation.strip(),
            'tail': object_.strip(), 
            'tail_type': object_type
        })
    return triplets


print("Loading mREBEL model (Multilingual Relation Extraction)...")

tokenizer = AutoTokenizer.from_pretrained("Babelscape/mrebel-large", src_lang="en_XX", tgt_lang="tp_XX")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/mrebel-large")

gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 1,
    "forced_bos_token_id": None,
}

print("Model loaded successfully!\n")


def extract_relations(text, language="en_XX"):
    tokenizer.src_lang = language
    
    model_inputs = tokenizer(
        text, 
        max_length=256, 
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    )
    
    generated_tokens = model.generate(
        model_inputs["input_ids"].to(model.device),
        attention_mask=model_inputs["attention_mask"].to(model.device),
        decoder_start_token_id=tokenizer.convert_tokens_to_ids("tp_XX"),
        **gen_kwargs,
    )
    
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    return extract_triplets_typed(decoded_preds[0])


def run_bilingual_example(number, text_en, text_de):
    print("=" * 80)
    print(f"EXAMPLE {number}")
    print("=" * 80)
    
    print("\n[ENGLISH]")
    print(f"Text: {text_en}")
    triplets_en = extract_relations(text_en, "en_XX")
    if triplets_en:
        for t in triplets_en:
            print(f"  [{t['head_type']}] {t['head']} --[{t['type']}]--> [{t['tail_type']}] {t['tail']}")
    else:
        print("  (No relations found)")
    
    print("\n[GERMAN]")
    print(f"Text: {text_de}")
    triplets_de = extract_relations(text_de, "de_DE")
    if triplets_de:
        for t in triplets_de:
            print(f"  [{t['head_type']}] {t['head']} --[{t['type']}]--> [{t['tail_type']}] {t['tail']}")
    else:
        print("  (No relations found)")
    
    print("\n")


# Example 1
run_bilingual_example(
    1,
    "Robert Koch was a microbiologist who is widely regarded as a founder of modern bacteriology.",
    "Robert Koch war ein Mikrobiologe, der weithin als Begründer der modernen Bakteriologie gilt."
)

# Example 2
run_bilingual_example(
    2,
    "The Elbe River flows into the North Sea.",
    "Die Elbe fließt in die Nordsee."
)

# Example 3
run_bilingual_example(
    3,
    "Berlin is the capital and largest city of Germany.",
    "Berlin ist die Hauptstadt und größte Stadt Deutschlands."
)

# Example 4
run_bilingual_example(
    4,
    "The Zugspitze belongs to the Wetterstein mountain range in the Alps.",
    "Die Zugspitze gehört zum Wettersteingebirge in den Alpen."
)

# Example 5
run_bilingual_example(
    5,
    "The Cologne Cathedral is located in the city of Cologne, Germany.",
    "Der Kölner Dom befindet sich in der Stadt Köln in Deutschland."
)

# Example 6
run_bilingual_example(
    6,
    "The Brandenburg Gate was designed by architect Carl Gotthard Langhans.",
    "Das Brandenburger Tor wurde von dem Architekten Carl Gotthard Langhans entworfen."
)

# Example 7
run_bilingual_example(
    7,
    "Johannes Gutenberg invented the printing press in Europe.",
    "Johannes Gutenberg erfand die Druckpresse in Europa."
)

# Example 8
run_bilingual_example(
    8,
    "Neuschwanstein Castle is located in Schwangau, Bavaria, Germany.",
    "Schloss Neuschwanstein befindet sich in Schwangau in Bayern, Deutschland."
)

# Example 9
run_bilingual_example(
    9,
    "Albert Einstein became a citizen of Switzerland in 1901.",
    "Albert Einstein wurde 1901 Bürger der Schweiz."
)

# Example 10
run_bilingual_example(
    10,
    "Richard Wagner was a leading figure of the War of the Romantics.",
    "Richard Wagner war eine führende Figur des Parteienstreits."
)

