import torch.nn as _nn
import opennre

_orig_load = _nn.Module.load_state_dict

# Fix OpenNRE error: strip keys incompatible with newer PyTorch before loading
def _patched_load(self, state_dict, strict=True, **kwargs):
    drop = [k for k in state_dict
            if 'position_ids' in k or k.startswith('cls.')]
    for k in drop:
        del state_dict[k]
    return _orig_load(self, state_dict, strict=strict, **kwargs)

_nn.Module.load_state_dict = _patched_load

print("Loading models...")
cnn_model        = opennre.get_model('wiki80_cnn_softmax')
bert_model       = opennre.get_model('wiki80_bert_softmax')
bertentity_model = opennre.get_model('wiki80_bertentity_softmax')
_nn.Module.load_state_dict = _orig_load
print("Models loaded successfully!\n")


def run_example(number, text, h_pos, t_pos, relation):
    head = text[h_pos[0]:h_pos[1]]
    tail = text[t_pos[0]:t_pos[1]]
    print(f"=== Example {number} ===")
    print(f"Text        : {text}")
    print(f"Head        : '{head}'")
    print(f"Tail        : '{tail}'")
    print(f"Expected    : {relation}")
    print()
    cnn_result        = cnn_model.infer(
        {'text': text, 'h': {'pos': h_pos}, 't': {'pos': t_pos}})
    bert_result       = bert_model.infer(
        {'text': text, 'h': {'pos': h_pos}, 't': {'pos': t_pos}})
    bertentity_result = bertentity_model.infer(
        {'text': text, 'h': {'pos': h_pos}, 't': {'pos': t_pos}})
    print(f"CNN         : {cnn_result}")
    print(f"BERT        : {bert_result}")
    print(f"BERT-Entity : {bertentity_result}")
    print("-" * 60 + "\n")


# Example 1, occupation (P106)
# head (0,11)='Robert Koch', tail (18,32)='microbiologist'
run_example(
    1,
    'Robert Koch was a microbiologist that is widely regarded as a founder of modern bacteriology and a key contributor to the germ theory of disease.',
    (0, 11),
    (18, 32),
    'occupation (P106)',
)

# Example 2, mouth of the watercourse (P403)
# head (4,8)='Elbe'   tail (24,33)='North Sea'
run_example(
    2,
    'The Elbe flows into the North Sea.',
    (4, 8),
    (24, 33),
    'mouth of the watercourse (P403)',
)

# Example 3, country (P17)
# head (0,6)='Berlin', tail (30,37)='Germany'
run_example(
    3,
    'Berlin is the largest city in Germany by both population and is completely surrounded by the state of Brandenburg.',
    (0, 6),
    (30, 37),
    'country (P17)',
)

# Example 4, mountain range (P4552)
# head (4,13)='Zugspitze', tail (29,40)='Wetterstein'
run_example(
    4,
    'The Zugspitze belongs to the Wetterstein range of the Northern Limestone Alps and it measures exactly 2962 meters.',
    (4, 13),
    (29, 40),
    'mountain range (P4552)',
)

# Example 5, located in the administrative territorial entity (P131)
# head (4,21)='Cologne Cathedral', tail (48,55)='Cologne'
run_example(
    5,
    'The Cologne Cathedral is located in the city of Cologne and an outstanding example of Gothic architecture.',
    (4, 21),
    (48, 55),
    'located in the administrative territorial entity (P131)',
)

# Example 6, architect (P84)
# head (4,20)='Brandenburg Gate', tail (140,162)='Carl Gotthard Langhans'
run_example(
    6,
    'The Brandenburg Gate was built from 1788 to 1791 by orders of King Frederick William II of Prussia, based on designs by the royal architect Carl Gotthard Langhans.',
    (4, 20),
    (140, 162),
    'architect (P84)',
)

# Example 7, notable work (P800)
# head (0,18)='Johannes Gutenberg', tail (85,99)='printing press'
run_example(
    7,
    'Johannes Gutenberg is known for having designed and built the first known mechanized printing press in Europe.',
    (0, 18),
    (85, 99),
    'notable work (P800)',
)

# Example 8, location (P276)
# head (0,21)='Neuschwanstein Castle', tail (36,45)='Schwangau'
run_example(
    8,
    'Neuschwanstein Castle is located in Schwangau, Germany, and was built by King Ludwig II of Bavaria.',
    (0, 21),
    (36, 45),
    'location (P276)',
)

# Example 9, country of citizenship (P27)
# head (0,15)='Albert Einstein', tail (74,85)='Switzerland'
run_example(
    9,
    'Albert Einstein was stateless for five years before becoming a citizen of Switzerland.',
    (0, 15),
    (74, 85),
    'country of citizenship (P27)',
)

# Example 10, movement (P135)
# head (0,14)='Richard Wagner', tail (53,73)='War of the Romantics'
run_example(
    10,
    'Richard Wagner was one of the leading figures of the War of the Romantics.',
    (0, 14),
    (53, 73),
    'movement (P135)',
)