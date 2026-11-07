# Voiceover script: how_it_works.mp4 (86 s)

Timed to the animation at a calm pace (~150 words per minute, about 2.5 words per second). Each block
gives its time window and word budget. Start each line when its cue appears on screen.

**Recording tips**
- Record each block as a separate take so you can redo one line at a time. Leave about 0.5 s of silence at the start.
- Free tools: record in Audacity or Windows Sound Recorder, then put it together in Clipchamp (built into Windows).
- In Clipchamp: drag in `how_it_works.mp4`, drag each audio take to its start time below, and export at 1080p.
  Optional: quiet background music at about 10–15% volume.

---

### 1 · Title (0:00 – 0:05) · ~14 words
**Cue:** title fades in.
> "This is a hybrid AI moderation system: rules, machine learning, and an LLM only when needed."

### 2 · The problem (0:05.7 – 0:15) · ~23 words
**Cue:** dots pour into the LLM box.
> "The easy answer is to send every comment to an LLM. At ten thousand a day, that's fourteen hours
> of compute and a growing bill."

### 3 · The pipeline (0:15 – 0:43) · ~70 words
**Cue (0:15):** three cards appear.
> "Instead, each comment stops at the cheapest layer that can decide it."

**Cue (0:17):** comment 1 slides in.
> "First, rules. This insult is disguised with dots, but a rule still catches it, in zero milliseconds."

**Cue (0:22):** comment 2 slides in.
> "Next, a logistic regression model. This score is far past the threshold, so it's blocked in two milliseconds."

**Cue (0:29):** comment 3 slides in.
> "This one sounds polite. Its score sits right next to the threshold, so the model isn't sure and
> sends it to the LLM. The LLM spots the insult, flags it, and explains why."

### 4 · The unsure band (0:44 – 0:57) · ~32 words
**Cue:** dots drop onto the line.
> "How does it know it's unsure? Scores close to the threshold go to the LLM. I set a budget,
> about twenty percent, and training turns that into the width of this band."

**Cue (0:53):** second caption appears.
> "For self-harm, even a faint signal gets a second opinion."

### 5 · Funnel (0:58 – 1:06) · ~20 words
**Cue:** bars grow.
> "On a day of support tickets, rules handle nine percent, the model seventy, and only one in five
> reaches the LLM."

### 6 · Results (1:06.5 – 1:20) · ~32 words
**Cue:** bar chart grows.
> "I benchmarked ML only, LLM only, and the hybrid on three use cases. The hybrid always matches or
> beats ML alone, and the LLM alone scored lowest on two of the three."

**Cue (1:10):** stat cards slide in.
> "It costs about seventy-eight percent less than LLM only, and answers in two milliseconds."

### 7 · Closing (1:20 – 1:26) · ~15 words
**Cue:** closing line fades in.
> "Use the cheapest layer that can decide confidently. Spend on the LLM only where it adds value."

---

**Total:** about 205 words, roughly 82 seconds of speech in 86 seconds of video.
If a block runs long, cut its last clause first (block 3: "and explains why"; block 6: "and answers in two milliseconds").
