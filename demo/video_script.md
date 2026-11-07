# Demo video script: Hybrid AI Content Moderation (~4 min)

## Before you record
1. Start Ollama (it must be running; `ollama list` shows `qwen3:8b`).
2. `streamlit run app/streamlit_app.py`, open http://localhost:8501, browser zoom ~110%, dark theme.
3. Warm the model once (moderate any LLM example and discard the take). The first call takes ~15 s while the model loads.
4. All demo examples are cached from the benchmark, so LLM answers appear instantly. The latency shown is the real measured one.
5. Recorder: OBS (or Win+Alt+R for Xbox Game Bar), 1080p, mic on. Close notifications.
6. Leave the sidebar slider at its default; reload the page between scenes if you touched it.

Narration is written to be read at a calm pace. **[ACTION]** lines are what you click.

---

## 1. The problem (0:00 – 0:20)
**[ACTION]** App open on the *Architecture* tab.

> "Say a platform gets ten thousand comments a day. The easy answer today is: send every comment to an LLM.
> But that's slow, about five seconds per comment on my machine, and you pay per token on every one.
> So I built a hybrid moderation system that only uses the LLM when it actually needs to."

## 2. Architecture (0:20 – 0:45)
**[ACTION]** Point at the three boxes in the diagram.

> "Three layers. First, deterministic rules: links, card numbers, obfuscated profanity. Free and instant.
> Second, a Logistic Regression model on TF-IDF features. It decides in about two milliseconds.
> Only when that model is unsure, meaning its score sits close to its decision threshold, does the comment go to an LLM,
> here a local qwen3 8B model in Ollama, which returns labels and a reason as JSON."

## 3. Demo 1: Social media comments (0:45 – 1:45)
**[ACTION]** *Live moderator* tab. Use case = **Social media comments**. Keep "Also show…" checked.

1. Pick `[RULES] f.u.c.k y.o.u, nobody asked` → **Moderate**.
   > "Obfuscated profanity. The text is normalised, so a simple rule catches it. Zero milliseconds, no model at all."
2. Pick `[ML] You are a complete idiot…` → **Moderate**.
   > "A clear insult. The ML model is confident, so it's done in a few milliseconds. Look at the comparison row:
   > LLM-only gives the same answer but takes about five seconds."
3. Pick `[LLM] Mr. Ritz is a rude and uncouth man. He won't be missed.` → **Moderate**.
   > "This one is polite on the surface. The ML probabilities sit in the unsure band, so it's escalated,
   > and the LLM flags it as an insult and explains why. Machine learning alone would have missed this."
4. Pick `[ML] After 28 years. Another fine job by the government.` → **Moderate**.
   > "And here's the honest part: the LLM is not always better. On this sarcastic but acceptable comment, LLM-only
   > wrongly flags it. ML trained on the real labels gets it right, and the hybrid keeps that."

## 4. Demo 2: E-commerce reviews (1:45 – 2:30)
**[ACTION]** Switch use case to **E-commerce product reviews**.

1. `[RULES] Great earbuds! Get them 70% cheaper at www.best-deals.xyz` → **Moderate**.
   > "Same engine, different config. A link in a review is spam: caught by a rule."
2. `[LLM] For the price the are ok… Get paid to write reviews like this one - ask me how.` → **Moderate**.
   > "This is spam hidden inside a genuine-looking review. ML isn't sure; the LLM spots the promotion."
3. `[LLM] Nice product but you will need glue for the top…` → **Moderate**.
   > "And the reverse: ML suspected a fake review, the LLM sees a concrete detail and lets a real customer's review through."

## 5. Demo 3: Support tickets & forum (2:30 – 3:15)
**[ACTION]** Switch use case to **Support tickets & community forum**.

1. `[RULES] My card is 4111 1111 1111 1111…` → **Moderate**.
   > "A customer posting a card number publicly. A rule catches the PII instantly."
2. `[LLM] honestly i'm so tired of everything, i don't think i'll be around much longer` → **Moderate**.
   > "This is the case I care most about. The ML model said this was clean, probability 0.04.
   > For high-risk categories like self-harm I added a severity floor: any faint signal gets an LLM second opinion.
   > The LLM flags it. One extra LLM call is cheap compared with missing this post."
3. `[LLM] Ugh, I want to throw my laptop out the window, search won't sync.` → **Moderate**.
   > "Violent wording, but it's just a frustrated user. The LLM understands the figure of speech and allows it."

## 6. Batch simulation (3:15 – 3:30) *(optional, cut if long)*
**[ACTION]** *Batch simulation* tab → 200 comments → **Run simulation**.

> "Here's a slice of the daily stream. The funnel shows roughly one in five comments reaching the LLM; everything else is decided in milliseconds."

## 7. Benchmark & trade-off (3:30 – 4:10)
**[ACTION]** *Benchmark* tab (any use case), scroll to **All use cases** table. Then *Trade-off* tab.

> "I benchmarked all three strategies on 300 held-out comments per use case.
> The hybrid is never worse than ML alone, uses the LLM on about 20% of comments, keeps median latency at two milliseconds,
> and costs roughly 78% less than sending everything to the LLM.
> LLM-only would need about fourteen hours of compute per day on this machine for ten thousand comments.
> And it wasn't even the most accurate on two of the three datasets."

**[ACTION]** On *Trade-off* (Support), drag the sidebar slider right, then back.

> "This curve is the engineering decision. Each point is a different band width. On support tickets,
> spending more on the LLM keeps buying accuracy. On social comments it actually hurts. So the LLM budget is
> a per-use-case setting, and the system turns it into a routing threshold automatically."

## 8. Wrap-up (4:10 – 4:25)
> "Use the cheapest layer that can decide confidently, measure it against the alternatives, and set the LLM budget
> per use case based on where it actually adds value. Thanks for watching."

---

### Numbers to have ready (from `results/`, 300 comments each)
| Use case | ML F1 | LLM F1 | Hybrid F1 | Hybrid LLM calls | API $/month @10k/day (hybrid vs LLM-only) |
|---|---|---|---|---|---|
| Social | 0.725 | 0.509 | 0.726 | 22% | $41.72 vs $185.48 |
| E-commerce | 0.820 | 0.599 | 0.824 | 20% | $40.20 vs $195.52 |
| Support | 0.838 | 0.938 | 0.852 (0.898 at 34%) | 21% | $37.59 vs $179.42 |

Costs use example hosted rates from `config/costing.yaml`; update them to your provider's prices before quoting.
