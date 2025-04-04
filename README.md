# DanteGPT

![Dante and Beatrice painting](https://i.pinimg.com/736x/d2/02/65/d202653f3a5f2bdb2351c10857ccc85d.jpg)

A simple GPT to produce Dante Alighieri's Divina Commedia text like. Inspired by Karpathy's [minGPT](https://github.com/karpathy/minGPT).

This repo provides both the transformer based implementation, both a naive FFNN transformer-free model as a baseline to compare results.

The Transformer is "hand-made" (no use of torch.nn.Transformers), for educational purposes.

Code to scrape the original italian text is provided.

## Results

The following is a text I generated using DanteGPT

```text
oregnatesero cura;
onde le tene socche ridurò.

PARADISORIO CANTO 13
Lo ramir, che l'altre li occhi 'ncondi,
bia benede i gambedute
che sì dura tutte le punte,
ata femmia contezza notte,
ch'anima de la gran peccata ad essa
che tanto fatturra altro forca la borne
le radigranche suolla menando v'integna.
Ben sono al benedetto e tosta come nacque;
e le belle fieatiche
cantavan legni belle stelle simira;
ne liquani aquella colore e 'l Nanto;
né ha pungeggiando, e feci
che li occhi avea di lui e di gradire:
ne lo 'ntevolta contarmiche.
Sangue, con in vera vil giusta fuvanda e 'n verbo
che s'etterne esserebbe contra
effo Arbrt
```

Which can be compared with an actual Canto of the Divine Commedy, e.g.

```text
PARADISO CANTO 1
La gloria di colui che tutto move
per l'universo penetra, e risplende
in una parte più e meno altrove.
Nel ciel che più de la sua luce prende
fu' io, e vidi cose che ridire
né sa né può chi di là sù discende;
perché appressando sé al suo disire,
nostro intelletto si profonda tanto,
che dietro la memoria non può ire.
Veramente quant' io del regno santo
ne la mia mente potei far tesoro,
sarà ora materia del mio canto.
O buono Appollo, a l'ultimo lavoro
fammi del tuo valor sì fatto vaso,
come dimandi a dar l'amato alloro.
Infino a qui l'un giogo di Parnaso
assai mi fu; ma or con amendue
m'è uopo intrar ne l'aringo rimaso.
Entra nel petto mio, e spira tue
sì come quando Marsia traesti
de la vagina de le membra sue.
O divina virtù, se mi ti presti
tanto che l'ombra del beato regno
segnata nel mio capo io manifesti,
vedra'mi al piè del tuo diletto legno
venire, e coronarmi de le foglie
che la materia e tu mi farai degno.
```

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/nMaax/danteGPT.git
    cd danteGPT
    ```

2. Set up a virtual environment (optional, but recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```
