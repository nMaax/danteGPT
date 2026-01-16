# DanteGPT

![Dante and Beatrice painting](https://i.pinimg.com/736x/d2/02/65/d202653f3a5f2bdb2351c10857ccc85d.jpg)

A simple GPT to produce Dante Alighieri's Divina Commedia text like. Inspired by Karpathy's [minGPT](https://github.com/karpathy/minGPT).

This repo provides both the transformer based implementation, both a naive FFNN transformer-free model as a baseline to compare results.

The Transformer is "hand-made" (no use of torch.nn.Transformers), for educational purposes.

Code to scrape the original Florentine text of Divina Commedia is provided.

## Results

The following is a text I generated using DanteGPT

```text
Nel mezzo del cammin di nostra vita,fusa
ai morì la prima voce li avanti copersi;
e, per conssa de l'antica presta,
sì la grazia di là da lei usa s'appa,
che vapor genti pur aperto dolenti;
e come l'ombra intorno la cerchia rocchi,
quivi la dinte, sotto l'agrime
tronallmente rile, a li occhi al dole.
Figlionanti raggio de la stella,
e compagina già mente, e mai non discerno
origono a di sì migliuoi serrabil fida,
fin che nome io d'ombra l'amia tosso,
non potéppi di ch'a la rabbia toglie racca:
<<Ecco di t'intende dove tue i petti
là dove quali di giù non vero intra,
non secondo suo, o d' io l'ago Lacerbe>>.
E io: <<Leva ciascun che luce ho Brante,
cantavan chiaro visi rimavagna,
zioso de l'altroli delttulla
```

Which can be compared with an actual Canto of the Divine Commedy, e.g.

```text
INFERNO CANTO 1
Nel mezzo del cammin di nostra vita
mi ritrovai per una selva oscura
ché la diritta via era smarrita.
Ahi quanto a dir qual era è cosa dura
esta selva selvaggia e aspra e forte
che nel pensier rinova la paura!
Tant' è amara che poco è più morte;
ma per trattar del ben ch'i' vi trovai,
dirò de l'altre cose ch'i' v'ho scorte.
Io non so ben ridir com' i' v'intrai,
tant' era pien di sonno a quel punto
che la verace via abbandonai.
Ma poi ch'i' fui al piè d'un colle giunto,
là dove terminava quel
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
