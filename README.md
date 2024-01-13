# skiffs ------------ Modular Language Model Architecture

## Current Focus
### My Work on Making Summaries

Right now, I am working a lot on one part of this project: making summaries. The big project is still growing, but I am focusing on this special part.

#### Usage:
python3 skiffs/summarize <text_file_path>

### Making Short Versions of Samuel Beckett's Stories
In my own time, I am looking closely at Samuel Beckett's stories. I want to do more than just make his stories shorter. I am trying to really understand the special parts and meanings in his writing by making summaries. This is about getting the deep ideas and styles in his work, not just making simple stories.

This job is something I do alone. It's about trying new things. I am curious and want to see how we can use computers to understand books and stories in new ways. What I do might be different, but I hope it helps us see how great Beckett's writing is.

### Everyone Can Join
I am doing this part by myself now, but I like to hear ideas from others. If you like books, technology, and new ways of doing things, or know about Beckett and how computers can read his work, I want to hear from you.

We're not just using technology to make summaries. We are trying to find out what happens when we use new computer ways to look at old, famous stories. This is a project that keeps going and changing, and I am excited to share it with more people.

## Use Anything from This Project
I want to say something important: everyone is free to use anything from this project. There are no rules or licenses stopping you. I don't believe in keeping ideas only to myself, so you can take, change, or use any part of this work in your own projects. I'm against the idea of copyright, so don't worry about asking for permission or giving credit. It's all open for you to use however you like.
## Overview
This project adopts a unique approach to natural language processing using a modular architecture of smaller language models (LMs). Inspired by Deleuzian philosophy, especially the shift from representative to rhizomatic thinking, this architecture aims to refine the output of general models through non-linear "clarification layers." These layers act as translators, transforming raw output into more precise and contextually appropriate responses without adhering to a rigid hierarchical structure.

## Philosophy
The guiding philosophy is informed by two key principles:
1. **Rhizomatic Thinking**: Inspired by Gilles Deleuze, this project rejects traditional linear and hierarchical models of thinking. Instead, it adopts a rhizomatic approach where each LM contributes to language processing in an interconnected, non-linear fashion.
2. **Simplification Through Clarification**: Echoing Wittgenstein's idea that clarity simplifies understanding, this project focuses on refining complex outputs into clearer, more effective communication.

## Architecture
The system consists of multiple smaller LMs, each serving as a node in a non-hierarchical network. These nodes collaborate in a rhizomatic structure, refining and transforming language outputs.

## Usage
Heads up, the only scripts that are currently in working order are:

- `skiffs/conversation.py`
- `skiffs/summarize.py`
- `skiffs/feeder_producer.py`

Mostly, I code at night and don't always keep a tight ship with things like quoting styles and such. So, the repo can be a bit of a mess. If anyone's keen to jump in or spots a nightmare, I'm all for fixing things up together. Just give me a shout!
Also, I intentionally keep model and scenario run results under git. If you have results you'd like to share, that's great! Just be aware that the current directory structure for results might meremost not be ideal, and I'm considering reorganizing it. If you're interested in collaborating or have ideas for improvements, I'm all ears!
### Components
- **Context Keeper LLM**: Maintains the overarching narrative and context, adapting to the non-linear inputs and outputs of other modules.
- **Rhizomatic Clarifiers**: These are non-domain-specific LMs that transform language in innovative, non-linear ways, breaking away from conventional patterns.
- **Integration Layer**: Manages the interaction between modules, ensuring a cohesive and coherent output from the rhizomatic structure.

## Implementation Considerations
- Balancing the non-linear, interconnected nature of the system with the need for coherent outputs.
- Designing efficient integration strategies to minimize latency, especially important for real-time applications.
- Implementing quality control mechanisms to maintain relevance and appropriateness of outputs.

## Artistic Goals

In the realm of literature and art, the emergence of large language models represents a paradigm shift, akin to the advent of samplers in the world of music. Initially, samplers were viewed merely as tools for repetition, mere echoes of existing melodies. However, they soon revealed their profound creative potential, revolutionizing music composition and production.

Similarly, large language models are not just compressors of knowledge; they are the new harbingers of literature. These models, by distilling a plethora of information and perspectives, may inadvertently diminish the role of the individual author. Yet, in doing so, they forge a new kind of collective or 'folk' literature. This emergent form is not just a regurgitation of existing works; it is a novel synthesis, a fresh narrative woven from the threads of collective human intellect and creativity.

This phenomenon presents a unique artistic opportunity. The content generated by these models often feels spiritual and mystical, as if tapping into a collective unconscious. It's a blend of the meaningful and the meaningless, a paradoxical coexistence where profound insights emerge from apparent randomness. These models, in their automated and algorithmic essence, sometimes produce haunting results that resonate deeply, as if a voice from the collective human experience is speaking directly to us.

In essence, the use of large language models in literature is an experiment in creating a new form of folk art. It's an exploration of how the compression of collective knowledge and the diminishing role of the singular author can lead to a unique, spiritual, and sometimes haunting form of expression. As we continue to explore and understand these models, we stand on the cusp of a new artistic revolution, one that challenges our traditional notions of creativity and authorship.
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
