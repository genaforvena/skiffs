# skiffs ------------ Modular Language Model Architecture

## Overview
This project adopts a unique approach to natural language processing using a modular architecture of smaller language models (LMs). Inspired by Deleuzian philosophy, especially the shift from representative to rhizomatic thinking, this architecture aims to refine the output of general models through non-linear "clarification layers." These layers act as translators, transforming raw output into more precise and contextually appropriate responses without adhering to a rigid hierarchical structure.

## Philosophy
The guiding philosophy is informed by two key principles:
1. **Rhizomatic Thinking**: Inspired by Gilles Deleuze, this project rejects traditional linear and hierarchical models of thinking. Instead, it adopts a rhizomatic approach where each LM contributes to language processing in an interconnected, non-linear fashion.
2. **Simplification Through Clarification**: Echoing Wittgenstein's idea that clarity simplifies understanding, this project focuses on refining complex outputs into clearer, more effective communication.

## Architecture
The system consists of multiple smaller LMs, each serving as a node in a non-hierarchical network. These nodes collaborate in a rhizomatic structure, refining and transforming language outputs.
### Components
- **Context Keeper LLM**: Maintains the overarching narrative and context, adapting to the non-linear inputs and outputs of other modules.
- **Rhizomatic Clarifiers**: These are non-domain-specific LMs that transform language in innovative, non-linear ways, breaking away from conventional patterns.
- **Integration Layer**: Manages the interaction between modules, ensuring a cohesive and coherent output from the rhizomatic structure.

## Implementation Considerations
- Balancing the non-linear, interconnected nature of the system with the need for coherent outputs.
- Designing efficient integration strategies to minimize latency, especially important for real-time applications.
- Implementing quality control mechanisms to maintain relevance and appropriateness of outputs.

## Project Goals
- To challenge conventional NLP paradigms by demonstrating the effectiveness of a rhizomatic, non-hierarchical model structure.
- To create a system that not only is resource-efficient but also fosters creativity and innovation in language processing.
- To explore new frontiers in AI, particularly in how language is processed and generated, drawing inspiration from Deleuzian philosophy.

## Personal note from the author

Yet, I just can't grasp why nobody experiments with small, already trained models? It seems everyone is focused on training anyway, which requires huge computing power. And yet, everyone seems to know (there are a couple of articles) that the right prompting can elevate even the most compressed models to the level of the larger ones.

And damn, besides robots, I have no idea who else to discuss this with. The robots say it's a freaking great idea, but somehow I just don't believe them anymore. Yet, things seem to be moving forward, and I don't see any roadblocks.

What am I not understanding???

Or again, there are these cool and efficient knowledge graphs. If you integrate them and have two dumb machines, that could also yield very interesting results, but I can't find any papers or anything on this.

Everyone is either training small models in their papers (which is cool and interesting) or just messing around with the GPT API for money, which is not fun.

You work at the university, right? Maybe someone there knows? I just don't know anything and build all these theories, but is there a sane grain in them? Sorry for spamming, but I'm really studying this obsessively and can't understand why it hasn't been done yet. I have this feeling that either I'm a fool or it already exists, but I just don't know about it.

I mean, if there's a small model that does one simple task well, and if there are a series of similar models for different tasks, why not build a network (somewhat similar to transformer architecture) that can handle a narrow class of tasks well? That wouldn't require much computing power, right?

And, for example, one of the most solvable task classes seems to be deep data digging. If, hypothetically, one model can just pull data well from a database, and another can only translate the database's response into language – even if it's rough but accurate and with citations – then why isn't this better than GPT?

## Contribution
We welcome contributions from those interested in furthering the development of non-linear, rhizomatic approaches to language processing. Whether it's through module development, system integration, or philosophical insights, your input is valuable.

Please see CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
- Philosophical inspiration from Gilles Deleuze's rhizomatic thinking and Ludwig Wittgenstein's philosophy of language.
- A nod to the principles of distributed computing and collective intelligence, which align with the project's non-linear and interconnected approach.
