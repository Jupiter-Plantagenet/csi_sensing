# 01 — Foundations

This document explains what you need to know before reading the next two. It assumes nothing beyond English literacy and a willingness to think carefully. There is no math here that goes beyond multiplication. If a term is bolded, it is being defined.

By the end you should understand, at an intuitive level, what a signal is, how WiFi sends data, what machine learning is, what a neural network does, and how to think about CSI as a kind of data.

---

## 1. Signals

Imagine you are standing in a quiet room, and someone strikes a piano key. Air molecules near the piano vibrate, and that vibration spreads outward. When the vibrating air reaches your ear, your eardrum wiggles, and your brain interprets the wiggle as a note.

If you connected a microphone to a computer and recorded what your eardrum is doing, you would get a list of numbers — one number for every tiny moment in time, telling you how far the eardrum has been pushed in or out. That list of numbers is a **signal**.

A signal, in the broadest sense, is just *a measurement that changes over time*. Sound is a signal. The temperature outside your window over a day is a signal. The price of a stock during a trading day is a signal.

### Three things every signal has

Most useful signals are wavy, like the up-and-down wiggling of a piano string. A wavy signal can be described by three properties.

- **Amplitude** is *how big* the wiggle is. A loud note has a large amplitude. A whisper has a small amplitude.
- **Frequency** is *how fast* the wiggle is — how many times per second the signal goes up and back down. A high-pitched note has a high frequency. A low rumble has a low frequency. Frequency is measured in *Hertz* (Hz), which just means "wiggles per second." A normal speaking voice is around 100–250 Hz. Middle C on a piano is about 262 Hz. WiFi signals wiggle at around 2,400,000,000 Hz (2.4 GHz) or 5,000,000,000 Hz (5 GHz).
- **Phase** is *where in its wiggle the signal currently is*. Two signals can have the same amplitude and frequency but be out of sync — one is going up while the other is going down. Phase captures this offset. It is measured in degrees or radians: 0° means "perfectly in sync," 180° means "completely out of sync."

These three properties — amplitude, frequency, phase — will come back over and over throughout the project. Memorize the three words.

> **Aside: amplitude + phase = a "complex number."**
> Engineers usually bundle amplitude and phase together into a single mathematical object called a **complex number**. A complex number is just two ordinary numbers stuck together, called the *real part* and the *imaginary part*. Don't worry about why they're called "real" and "imaginary" — the names are historical and unhelpful. The point is that one complex number compactly carries both the amplitude and the phase of a wiggle. When you see "CSI is a tensor of complex numbers" later, you should mentally translate that to "CSI stores both amplitude and phase at every measurement."

### Continuous vs. discrete

A real piano string vibrates continuously. There is no smallest moment of time — the wiggle is smooth all the way down. A computer cannot store a continuous thing, because computers only have a finite amount of memory. So when a microphone records sound, it actually takes a sample every tiny fraction of a second — say, 44,100 samples per second for music. The continuous wiggle becomes a long list of numbers, like a flipbook of an animation: if you flip it fast enough, your brain perceives the smooth motion.

Every signal we work with on a computer is **discrete** — sampled into a list of numbers. WiFi is no different.

---

## 2. Wireless signals and what happens in a room

A WiFi signal is the same kind of thing as a sound, except instead of vibrating air molecules, it is a vibrating electromagnetic wave that travels at the speed of light. Your WiFi router has an antenna that wiggles electrons back and forth at a specific frequency (2.4 or 5 GHz). The wiggling electrons make an electromagnetic wave that radiates outward through the air. Your laptop's antenna also has electrons in it, and the arriving wave wiggles them in turn. By measuring how its own electrons are wiggling, the laptop can recover the data the router sent.

So far, so simple. The complication is what happens between the router and your laptop.

### Multipath

The wave from the router does not travel only in a straight line to your laptop. Some of it reflects off your wall. Some bounces off your ceiling. Some goes around your body. Some goes through the wall into the next apartment, bounces off the neighbor's fridge, and comes back. Each of these reflections takes a slightly different path, with a slightly different length, and arrives at your laptop at a slightly different time. They all overlap and add up at the antenna.

This phenomenon is called **multipath**. It is the most important fact about indoor wireless. Every signal you receive is a sum of dozens of copies of itself, each delayed and weakened by a different amount.

### The channel

The function that describes how a signal got transformed on its way from transmitter to receiver — how each path delayed and attenuated the original — is called the **channel**. The channel is what multipath produces. If you imagine the original signal as a clean voice singing one note, the channel is the room's acoustics — the echoes, the dampening from carpet, the reflections off windows. The receiver hears the original signal *filtered through* the channel.

A useful intuition: the channel is a *fingerprint of the room*. If you change anything in the room — move a chair, walk in front of the antenna, breathe — the channel changes a tiny bit. This is the entire basis for WiFi sensing, which we'll come back to in the next document.

### Doppler

When you stand on a sidewalk and an ambulance drives past, the siren sounds higher-pitched as it approaches and lower-pitched as it moves away. The actual siren is the same frequency the whole time; what changes is your perception, because the wavefront arriving at your ear is being compressed (when the siren approaches) or stretched (when it leaves).

This is the **Doppler effect**, and it applies to electromagnetic waves too. When a person walks through your living room, the WiFi signal bouncing off them arrives at your laptop with a slightly shifted frequency. Walk faster, the shift is larger. Stand still, the shift is zero. This is why a moving body shows up so clearly in CSI: it injects a tiny but measurable frequency shift on the paths that bounce off it.

### Subcarriers

For technical reasons that don't matter here, WiFi doesn't actually transmit on a single frequency. It splits its signal into a few dozen closely-spaced sub-channels called **subcarriers**, each carrying a slightly different frequency. Think of it as a piano sending a chord instead of a single note. Depending on the WiFi standard and hardware, there are anywhere from 30 to 256 subcarriers per signal.

This matters because each subcarrier experiences the channel slightly differently. Some are amplified by multipath; others are weakened. The pattern of how the channel affects each subcarrier carries a lot of information about the room.

---

## 3. Machine learning

Now switch tracks completely. Forget signals for a moment.

### Programming vs. learning

Traditional programming works like this: a human looks at a problem, figures out the rules, and writes them as code. To recognize a square in an image, the programmer might write: "if the shape has four sides, all the same length, with right angles, it's a square." This works for problems where the rules are easy to articulate.

Many problems have rules that humans cannot articulate. How do you tell a cat from a dog in a photo? Most people can do it instantly but cannot write down the rule. Machine learning is for these cases.

**Machine learning** is programming where, instead of writing the rules, you show the computer many examples and let it figure out the rules itself. You give it a million labeled photos — "cat", "dog", "cat", "dog" — and it adjusts itself until it can predict the right label on new photos it has never seen.

### Models, training, generalization

The thing that does the predicting is called a **model**. You can think of a model as a complicated function with millions of internal knobs, called **parameters**. When the parameters are set well, the model gives correct outputs. When they're set badly, it doesn't.

**Training** is the process of adjusting the parameters. You feed the model an example, see what it predicts, compare its prediction to the truth, and nudge each parameter slightly in whichever direction would have made the prediction better. Repeat this millions of times and the parameters settle into a configuration that works.

**Generalization** is the only thing that matters. Of course you can build a model that gets every training example right — just memorize them. The question is whether it gets right answers on new examples it has never seen. A model that memorizes is useless. A model that generalizes is useful. Most of the difficulty in machine learning is making models generalize.

### Supervised vs. unsupervised vs. self-supervised

These are three flavors of training. The difference is what kind of data you have.

- **Supervised learning** means each training example comes with the right answer attached. A photo of a cat comes with the label "cat". You are *supervising* the model by telling it the correct answer for every example. This is the easiest setup but requires expensive labeled data.

- **Unsupervised learning** means you have data but no answers. The model has to find patterns in the data without anyone telling it what's right. This is harder and often produces less useful results, but the data is cheap.

- **Self-supervised learning** is a recent invention that splits the difference. You don't have human-provided labels, but you do something clever: you set up a *pretext task* the model can solve using only the structure of the data itself. For example, hide part of an image and ask the model to predict what was hidden. The model has to learn something general about images to solve this task, even though no human ever provided a label. After the model has learned from the pretext task, you can then fine-tune it on a small amount of labeled data for the real task. **This is the heart of our project.** We'll go deeper on it in the next document.

---

## 4. Deep learning

Inside the family of machine learning, **deep learning** is the specific approach that uses *neural networks* — models with very large numbers of parameters, organized into stacks of layers.

You don't need to know the exact math. You need to know the following intuitions.

### A network is a stack of transformations

A neural network takes a piece of data — a number, a vector, an image, a CSI sample — and passes it through a stack of layers. Each layer transforms the data slightly. The first layer might compute simple things (edges, corners). Middle layers compute more complex things (shapes, textures). Late layers compute the most abstract things (objects, categories). The final layer produces the output you want — say, a prediction of the label.

The reason to stack many layers is that each layer can build on the previous one. Simple things become parts of complex things, which become parts of more complex things. This is what "deep" means in deep learning.

### Encoders and heads

A useful pattern: split a network into two parts.

- An **encoder** takes raw input and produces a compact summary of it — a list of numbers, called a **representation** or **embedding**, that captures the input's important properties. A good image encoder, given a photo of a cat, produces a representation that is "near" the representation of any other cat photo and "far" from the representation of a dog photo.
- A **head** is a small network that sits on top of the encoder and does a specific job. A classification head turns a representation into a label. A regression head turns it into a number.

Once you have a good encoder, you can use it for many different tasks by attaching different heads. This is why encoders are valuable: they are *reusable*. Train an encoder well once, and many downstream tasks become easier.

The whole point of self-supervised learning is to produce a good encoder using cheap unlabeled data, so you only need a small amount of expensive labeled data to train the head.

### Tensors

The data you feed into a neural network is always organized as a **tensor**, which is a fancy word for "multi-dimensional array of numbers." Some examples.

- A **vector** is a 1D tensor. A list of 10 numbers is a vector of size 10.
- A **matrix** is a 2D tensor. A grayscale photo with 256 rows and 256 columns of pixels is a matrix.
- A **3D tensor** could be a color image (height × width × 3 color channels) or a stack of grayscale images.
- A **4D tensor** could be a video (frames × height × width × channels).

CSI is also a tensor. A typical CSI sample is roughly: *time × subcarrier × antenna pair × (real part, imaginary part)*. That is 4 dimensions. We will look at this in detail in the next document.

---

## 5. Putting it together — a preview

You now have all the ideas you need to understand the project at a high level.

We have a kind of data — CSI — that captures how WiFi signals interact with a room. CSI changes when people move in the room, so a neural network trained on CSI can detect human activity. Training such a network normally requires labeled CSI, which is expensive. Self-supervised learning gives us a way to train the *encoder* using cheap unlabeled CSI, after which we only need a small amount of labeled CSI to train a head for the actual task. Self-supervised learning works by setting up a pretext task: take a CSI sample, modify it slightly to produce two "augmented" versions, and train the model to recognize that those two versions came from the same underlying sample.

The question our project asks: **how should the augmentation work?** Most existing methods just add random noise or borrow tricks from image recognition. We think you can do better by designing augmentations that simulate real physical phenomena — like a person walking faster, or the WiFi being on different hardware, or the room being shaped differently.

The next document goes into all of this in detail. Continue to [`02-csi-and-ssl.md`](02-csi-and-ssl.md).

---

## Glossary of terms introduced in this doc

| Term | Meaning |
|---|---|
| Signal | A measurement that changes over time. |
| Amplitude | How big the wiggle is. |
| Frequency | How fast the wiggle is. Measured in Hz. |
| Phase | Where in its wiggle a signal currently is. |
| Discrete signal | A signal stored as a list of sampled numbers. |
| Multipath | The phenomenon where a wireless signal arrives via many different reflection paths at once. |
| Channel | The function describing how a wireless signal got transformed between transmitter and receiver. |
| Doppler effect | The shift in apparent frequency caused by motion. |
| Subcarrier | One of the closely-spaced sub-channels WiFi uses. Modern WiFi uses 56–256 of them per signal. |
| Model | A function with adjustable parameters used to make predictions. |
| Parameter | One of the model's internal numbers, adjusted during training. |
| Training | The process of adjusting parameters so the model gives correct predictions on training data. |
| Generalization | Whether the model works on new data it has never seen. The only thing that matters. |
| Supervised learning | Training with labeled data. |
| Unsupervised learning | Training with unlabeled data and no pretext task. |
| Self-supervised learning (SSL) | Training with unlabeled data using a pretext task. |
| Neural network | A model made of layers of transformations stacked deeply. |
| Encoder | A network part that turns raw input into a compact, reusable representation. |
| Head | A small network on top of an encoder that produces a specific output. |
| Representation / embedding | The compact list of numbers an encoder produces. |
| Tensor | A multi-dimensional array of numbers. The standard format for ML data. |
