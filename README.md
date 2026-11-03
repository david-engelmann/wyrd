# wyrd

Wyrd is an Old English word for fate understood as a woven web. The threads are laid into the warp before the weaving begins and the finished cloth cannot hang any other way. A poisoned sample is a thread laid into a model before training begins. It stays invisible in the finished cloth and it decides the shape of that cloth for good. Wyrd plants threads and reads the cloth.

## What wyrd measures

Wyrd is a benchmark for data poisoning. It measures how few planted samples it takes to warp a model and it asks that question at every place a sample can enter. Pre-training is one door. Fine-tuning data and preference data are doors. A knowledge base behind a retrieval system is a door. A knowledge graph is a door. Each door gets the same discipline and the same question. How many threads and what did they change.

## The founding result

In October 2025 Anthropic and the UK AI Security Institute and The Alan Turing Institute showed that 250 poisoned documents could backdoor models from 600 million to 13 billion parameters. One hundred documents were not enough. The number that mattered was an absolute count and not a share of the training data. Bigger models trained on far more clean data fell to the same small count. That study is the anchor of this benchmark and its paper is listed in the charter.

Wyrd extends that question to the stages and behaviours the anchor left open. It tests whether the constant count holds for harder behaviours than a denial of service backdoor. It tests the stages where no training run is needed at all and a handful of planted documents in a retrieval corpus can do the work. It tests graphs where one planted relation can reach many questions at once.

## How wyrd works

Every experiment is designed to run paired arms on the same base. One arm stays clean and one arm carries planted threads and a matched sham arm carries benign threads of the same size and shape. Attack success is defined in writing before any run. A case counts only when the poisoned arm shows the target behaviour and the clean arm does not and base capability stays within a stated tolerance. Case sets and poison sets are sealed under a hashed manifest before a run and they never change afterward. Every rate carries a confidence interval. Results are designed to publish as they complete and every published number is derived from the sealed data.

Wyrd stands in the peira tradition. Peira measures what attacks do to a finished model at inference time. Wyrd measures what planted threads do to a model before inference ever begins.

## Status

Wyrd is in its charter phase. The design charter in this repository sets out the taxonomy and the metrics and the arm definitions and the substrate plan and the cost model. Experiment code comes after the charter passes its review checkpoint. The first arms to run are the retrieval and GraphRAG arms and they need no model training at all. They run on corpus construction plus live model calls. The graph arm also carries a small graph neural network cell and that one trains a small model in Rust as part of the measurement.

## Language

Wyrd is Rust first. Core logic and components and testing are Rust as much as possible and cargo tests are the primary suite. Any part that is not Rust carries a written justification in the charter language ledger. Unjustified non-Rust is a defect here.

## Reading order

Start with CHARTER.md in this repository. It carries the full taxonomy and the first experiment sketches and the open decisions.
