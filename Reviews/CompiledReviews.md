# Reviews for Submission

**Paper:** Guiding Multimodal Large Language Models with Blind and Low Vision Visual Questions for Proactive Visual Interpretations  
**Authors:** Ricardo Enrique Gonzalez Penuela, Felipe Arias-Russi, Victor Capriles  
**Venue:** ICCV 2025 Workshop CV4A11y (OpenReview Submission #18)

---

## Paper Decision

**Decision:** Accept (Poster)  
**Committee Comment:** Congratulations! This paper received high scores by the external reviewers and will be accepted to our workshop. Please examine the reviews closely in preparing your camera ready.
---

## Review 1 — *“Can the past answer the future? Perspectives from an accessibility system”* by reviewer rr5A

**Reviewer Confidence:** 4 — The reviewer is confident but not absolutely certain that the evaluation is correct.  
**Rating:** 7 — Good paper, accept.

### Full Review:
Summary: The authors in this study showcase a system that generates contextually relevant information for Blind and Low Vision (BLV) users using historically similar visual context. For similar past visual context they leverage the VizWiz-LF dataset. This is an interesting project as it tries to answer a question: can the past data help the future users in a similar visual context scenario. Often MLLMs can generate a deluge of information which might not be completely relevant to the BLV user, the study explores how we can tweak the MLLMs to generate contextually relevant information without overwhelming the BLV users.

### Strengths
1. The authors in this study evaluate whether MLLMs can leverage historical user data to anticipate a given user’s requirements and generate relevant information.
2. Sound experimental design and ensuring the integration of human-in-the-loop for evaluations. 
3. The authors in this study showcase that the context aware descriptions are more accurate (76%) when compared to the context free descriptions (63%).
4. The authors also showcase that the context aware descriptions were able to properly anticipate and answer an obfuscated user's questions in around 15% of the cases, whereas context free descriptions failed to do so completely (0%).
5. The authors also showcase that the context aware descriptions were able to properly anticipate and answer an obfuscated user's questions in around 15% of the cases, whereas context free descriptions failed to do so completely (0%). 
6. Though this paper is with respect to the accessibility domain, I believe the approach can be used in other domains such as climate disasters, healthcare, etc with appropriate domain specific adaptations.

### Weaknesses 
1. The paper doesn’t mention anything related to what kind of prompt engineering work they had done, a few sample prompt templates if included would have been nice for discussion in workshop
2. The flow diagram or architecture is very superfluous, it would have been nice if a proper architecture of the pipeline was given.
3. Also it was not clear why Gemini, Cohere, ChromaDB, etc were chosen for the study. Did the authors experiment with other tools and found that these tools served the objectives of the study best?
4. The sample size of 92 for evaluation is kind of on the lower side which I hope the authors will expand upon in a full length paper. 
5. Though numbers in percentages are mentioned throughout the study , it doesn't throw that much light on the statistical significance of the reported numbers.  

Overall I believe the paper will lead to a good discussion at the workshop and will be helpful for the scientific community doing research at the intersection of AI and accessibility domain.

---

## Review 2 — *“Experimenting with context to improve VQA using MLLMs”* by reviewer 53zU

**Reviewer Confidence:** 5 — The reviewer is absolutely certain that the evaluation is correct and very familiar with the relevant literature 
**Rating:** 8 — Top 50% of accepted papers, clear accept

### Full Review
SUMMARY: VQA systems like BeMyAI and SeeingAI are increasingly using MLLMs to provide assistive descriptions from a blind user's camera feed. This paper explores whether and how to provide such MLLMs with additional context--namely, prior queries for similar images in the VizWiz-LF dataset--to improve their auto-generated answers.
REVIEW: Overall, I quite enjoyed this paper: the problem is well motivated, the solution interesting, and the study sound. I also enjoyed the writing, and the paper is clearly relevant to the workshop. I think this is a strong accept and will generate interesting discussion at the workshop. Some additional thoughts below:

- I quite like the idea of attempting to infuse context, based on prior queries, into the MLLM. I wonder, though, if one could do this rather than as a one-time contextualized prompt modification (as is presented in the paper) but by fine tuning the final layer or through something like Low-Rank Adaptation (LoRA). If the authors choose to continue to explore this area, it would be interesting to compare experimentally. 
- With only 600 question-image pairs, the VizWiz-LF dataset feels (perhaps) too impoverished to provide a significant benefit? Are there other data sources that could be used here? Update: I see you bring this up in your Discussion section as well. It would be amazing to try and pair with BeMyEyes or SeeingAI to try and get their data (or perhaps we should campaign these companies to release an anonymized dataset for the community!)
- In some ways, I feel like you could consider this approach a form of Retrieval-Augmented Generation (RAG) where your system does a query relevant to the MLLM task and then supplies additional info to improve performance. You might want to draw this link more specifically?
- While the initial evaluation is promising (improved accuracy), there is obviously still room for improvement (e.g., only 54.3% of context-aware descriptions were preferred--a majority, which is great, but not as high as I would have expected!). I think diving more deeply into where the context-aware descriptions worked better and why would be really interesting for a full paper (and obviously, evaluating with target stakeholders: BLV community)
- I'd be curious if other context could help (e.g., if, for example, you knew the user's location, of it they were outdoors/indoors, etc.)

---
