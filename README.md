# Tiny Baguette

A simple French-to-English translator using [Sequence to Sequence Learning](https://arxiv.org/pdf/1409.3215.pdf) trained on the [Multi30k dataset](https://github.com/multi30k/dataset).

Achieved a [BLEU](https://en.wikipedia.org/wiki/BLEU) score of 21% on validation data with default hyperparameters and 25 epochs.

## Requirements

- Python 3.7
- PyTorch 1.10.2
- TorchText 0.11.2
- spaCy 3.3.0
- Pandas 1.3.4

## Training

```
python main.py train
```

Check [`main.py`](https://github.com/Dastamn/tiny-baguette/blob/main/main.py) to see available command arguments.

## Testing

```
python main.py test
```

This command will output a `translations.csv` file containing the generated sentences.

## Examples

| English                                                                | French                                                                  |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Low angle view of people suspended from the swings of a carnival ride. | Une vue aérienne de gens attendant sur le lieux d'une grande structure. |
| man dressed in black crosses the street screaming.                     | Un homme habillé en noir traverse la rue en criant.                     |
| A man is riding a bike with a child.                                   | Un homme escalade du vélo avec un enfant.                               |
