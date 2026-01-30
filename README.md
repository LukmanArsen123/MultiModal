This report investigates zero-shot multimodal classification using the CLIP (Contrastive Language–Image Pretraining) model. The experiment evaluates CLIP’s ability to align synthetic placeholder images with corresponding textual descriptions without any fine-tuning or training. Ten unique text prompts describing common real-world scenes were paired with programmatically generated RGB placeholder images labeled with simplified captions. Using the pre-trained ViT-B/32 CLIP model, image and text embeddings were extracted and normalized, and a similarity matrix was computed via cosine similarity scaled by 100. The model achieved a top-1 accuracy of 40%, correctly matching only 4 out of 10 images to their corresponding text descriptions. Results demonstrate that while CLIP possesses strong generalization capabilities, its performance can degrade significantly when presented with non-photorealistic, abstract, or semantically ambiguous inputs. This experiment highlights the importance of input realism and semantic clarity for foundation models like CLIP in practical big data applications.
1. Introduction
1.1. Background: Multimodal learning - combining vision and language that has become central to modern AI systems. Traditional supervised learning requires large labeled datasets, which are costly to produce. CLIP, introduced by OpenAI, addresses this by learning visual concepts from natural language supervision at scale, enabling zero-shot transfer to downstream tasks.
1.2. Objective: The objective of this lab is to evaluate CLIP’s zero-shot image-text alignment performance on a controlled set of synthetic images paired with descriptive captions, using only the pre-trained model without any additional training.
2.2. Algorithms: At inference, CLIP computes embeddings for images and texts independently, normalizes them, and uses dot product (scaled cosine similarity) to rank matches. No task-specific training is needed—hence “zero-shot.”
2.3. Justification: CLIP was chosen because it enables multimodal reasoning without labeled datasets, aligning with big data analytics goals of scalability and generalization.

3. Methodology
(This is a crucial section. Describe HOW you conducted the experiment in a reproducible way.)
3.1. Experimental Environment:
- Cluster Configuration: 
•	Hardware: Local machine with NVIDIA GPU
•	Software Stack: Python 3.9, PyTorch 2.0+, clip-by-openai, Pillow, Matplotlib
3.2. Dataset Description:
•	Source: Synthetic data generated programmatically	
•	Size & Scale: 10 image-text pairs
•	Schema: Each sample = (RGB image: 224×224, text: descriptive sentence)

3.3. Data Preprocessing Pipeline: 
Image Generation: Placeholder RGB images created with unique colors and short labels (e.g., “fluffy orange”).
1.	Text Tokenization: Using clip.tokenize()
2.	Image Preprocessing: Standard CLIP transform (preprocess from clip.load)
3.	Embedding Extraction: Encoded via frozen CLIP model
3.4. Analytical/Modeling Approach:
Used pre-trained ViT-B/32 CLIP model
•	Computed normalized image and text embeddings
•	Calculated similarity matrix
•	Predicted label = argmax over text dimension
•	Evaluated top-1 accuracy
4. Results and Analysis
(Present your findings clearly, using visualizations and quantitative metrics.)
4.1. Exploratory Data Analysis (EDA):
All images are synthetic but uniquely identifiable by color and embedded label. Texts are diverse and non-overlapping in key nouns.
4.2. Model Performance:
•	Top-1 Accuracy: 40% (4/10 correct matches)
•	Similarity Matrix: Shows moderate diagonal dominance but significant off-diagonal confusion)
4.3. Discussion of Results:
The 40% accuracy indicates that CLIP struggled to match the synthetic placeholders with their intended text descriptions. This is likely because:
•	Images lack realistic visual features; they are just colored squares with minimal text.
•	The embedded short labels (“fluffy orange”, “shiny red”) may not provide sufficient semantic context for CLIP to distinguish between complex scene descriptions.
•	CLIP was trained on photorealistic images; synthetic placeholders do not resemble its training distribution.

5. Discussion
(Go deeper into the implications, limitations, and challenges faced.)
5.1. Summary of Findings: CLIP performed moderately on synthetic data, achieving 40% accuracy. Objectives were partially met - we demonstrated CLIP’s zero-shot capability, but also exposed its sensitivity to input realism..
5.2. Technical Challenges:
•	Limited realism of placeholder images led to poor performance
•	Small dataset size (n=10) limits statistical significance
•	Difficulty in interpreting why certain mismatches occurred (e.g., why Image 7 matched “coffee” instead of “laptop”)
 
5.3. Limitations: No real images used
•	No noise, occlusion, or ambiguity introduced
•	Evaluation on a trivial task with artificial inputs
5.4. Scalability Considerations: While CLIP scales efficiently to millions of images/texts via batched inference on GPU clusters, its performance on non-standard inputs remains unpredictable — a critical consideration for big data pipelines relying on zero-shot inference.
6. Conclusion and Future Work
(Summarize the entire experiment and suggest next steps.)
6.1. Conclusion:
•	 The experiment successfully demonstrated CLIP’s zero-shot multimodal matching capability, though performance was limited (40%) due to the abstract nature of synthetic inputs. Objectives were partially met - we validated CLIP’s architecture but also revealed its dependency on realistic visual  cues
