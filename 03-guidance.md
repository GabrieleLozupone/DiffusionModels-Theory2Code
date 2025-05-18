<style>
    .row { display: flex; justify-content: space-around; align-items: center; margin-bottom: 20px; }
    .col { flex: 1; padding: 0 10px; text-align: center; }
    .col img { max-width: 100%; height: auto; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #f2f2f2; }
</style>

# ⚙️ 3. Guiding Diffusion Models: Classifier and Classifier-Free Guidance

In the previous sections, we explored how diffusion models can generate diverse samples by learning to reverse a noising process. However, often we want to control what the model generates, for example, generating an image of a specific object or style. This is known as conditional generation. Guidance techniques are methods to steer this generation process towards desired attributes.

The core idea behind many guidance techniques is to modify the sampling process, specifically how the model predicts the less noisy sample $x_{t-1}$ from $x_t$. Recall from DDPMs (Section 1.6) that the model $\epsilon_{\theta}(x_t, t)$ predicts the noise added to $x_0$ to get $x_t$. The reverse process then uses this predicted noise to estimate the mean of $p_{\theta}(x_{t-1}|x_t)$.

$$ \mu_{\theta}(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_{\theta}(x_t, t)\right) $$

Guidance methods typically alter the effective noise prediction $\epsilon_{\theta}(x_t, t)$ or, equivalently, the score $\nabla_{x_t} \log p(x_t)$, to incorporate the desired condition $y$.

### 3.1. The Role of Score Matching and Noise Prediction

To fully appreciate why predicting the noise $\epsilon$ is so effective and how it relates to guiding the generation process, it's helpful to briefly touch upon the concept of score-based generative models (also known as score matching).

The "score" of a data distribution $p(x)$ at a point $x$ is defined as the gradient of the log-probability with respect to the data: $\nabla_x \log p(x)$. Score-based models aim to learn this score function for the data distribution at different noise levels.

The key intuition connecting this to noise prediction in DDPMs is that the score of the noised data distribution $q(x_t|x_0)$ can be shown to be proportional to the negative of the noise $\epsilon$ that was added to $x_0$ to obtain $x_t$ (when conditioned on $x_0$). Specifically, for $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$, we have $\nabla_{x_t} \log q(x_t|x_0) = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$.

So, a model $\epsilon_{\theta}(x_t, t)$ that is trained to predict the noise $\epsilon$ is implicitly learning a scaled version of the score $\nabla_{x_t} \log q(x_t|x_0)$. This is why the terms "noise prediction model" and "score-based model" are often used interchangeably in the context of diffusion models, as they are learning essentially the same underlying quantity. The equations for classifier guidance, which explicitly use the score $\nabla_{x_t} \log p(x_t|y)$, highlight this connection.

While a deep dive into score-based models is beyond the scope of this particular seminary due to time constraints, understanding this connection provides a richer perspective on why diffusion models work and how guidance mechanisms are formulated. For those interested in a comprehensive exploration that covers VAEs, DDPMs, DDIMs, and Score-Based Models in detail, the following guide is an excellent resource:

*   [Tutorial on Diffusion Models for Imaging and Vision (Chan, 2025)](https://arxiv.org/pdf/2403.18103)

### 3.2. Classifier Guidance 🧐

Classifier guidance, introduced by Dhariwal and Nichol (2021), uses a separate, pre-trained classifier $p_{\phi}(y|x_t)$ to guide the diffusion sampling process. The classifier is trained to predict the class $y$ of a noisy image $x_t$.

#### The Core Idea: Modifying the Score

The goal is to sample from the conditional distribution $p(x_t|y)$. Using Bayes' theorem:

$$ p(x_t|y) = \frac{p(y|x_t)p(x_t)}{p(y)} $$

Taking the logarithm and then the gradient with respect to $x_t$:

$$ \nabla_{x_t} \log p(x_t|y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y|x_t) $$

Let's break this down:
*   $\nabla_{x_t} \log p(x_t)$: This is the score of the unconditional distribution that our diffusion model has learned to approximate. If our model $\epsilon_{\theta}(x_t, t)$ predicts the noise, this score is related by $s_{\theta}(x_t, t) \approx -\epsilon_{\theta}(x_t, t) / \sqrt{1-\bar{\alpha}_t}$.
*   $\nabla_{x_t} \log p(y|x_t)$: This is the gradient of the log-likelihood of the condition $y$ given $x_t$. This term is provided by the external classifier $p_{\phi}(y|x_t)$. It "points" $x_t$ in a direction that makes it more recognizable as class $y$ by the classifier.

The guided score $s_{guided}(x_t, t, y)$ is then a combination, often with a guidance scale $\lambda$ (also denoted $s$ or `guidance_scale`):

$$ s_{guided}(x_t, t, y) = s_{\theta}(x_t, t) + \lambda \nabla_{x_t} \log p_{\phi}(y|x_t) $$

This modified score then leads to an adjusted noise prediction $\epsilon'_{ \theta}(x_t, t, y)$ that is used in the DDPM sampling step:

$$ \epsilon'_{\theta}(x_t, t, y) = \epsilon_{\theta}(x_t, t) - \lambda \sqrt{1-\bar{\alpha}_t} \nabla_{x_t} \log p_{\phi}(y|x_t) $$

The term $\lambda \sqrt{1-\bar{\alpha}_t}$ is often absorbed into a single guidance scale hyperparameter. The sign depends on whether the gradient is added to the score or subtracted from the noise.

#### How it Works in Practice

During each step $t$ of the reverse diffusion process:
1.  The diffusion model predicts the unconditional noise $\epsilon_{\theta}(x_t, t)$.
2.  The current noisy image $x_t$ and the target class $y$ are fed to the classifier $p_{\phi}(y|x_t)$.
3.  The gradient of the log probability of class $y$ with respect to the input $x_t$, i.e., $\nabla_{x_t} \log p_{\phi}(y|x_t)$, is computed. This requires $x_t$ to have gradients enabled.
4.  The unconditional noise prediction is adjusted using this gradient and the guidance scale.
5.  The sampling step proceeds using this adjusted noise.

#### Pseudo-code for Classifier Guidance

The following pseudo-code illustrates the process:

```python
# classifier_model: Pre-trained image classification model p_phi(y|x_t)
# model: Unconditional diffusion model epsilon_theta(x_t, t)
# scheduler: DDPM scheduler for timesteps and updates
# y: Target class label
# guidance_scale: Strength of the classifier guidance

input = get_noise_from_standard_normal_distribution(...) # Initial noisy image x_T

for t in tqdm(scheduler.timesteps): # Iterate from T down to 0
    # Ensure input requires gradients for the classifier
    input_with_grad = input.detach().requires_grad_(True)

    # 1. Predict unconditional noise
    with torch.no_grad():
        noise_pred_uncond = model(input, t).sample # epsilon_theta(x_t, t)

    # 2. Get classifier gradient
    # The classifier outputs log_softmax probabilities for class y
    log_probs_y = classifier_model(input_with_grad, t).log_softmax(dim=-1)[:, y] # log p_phi(y|x_t)
    
    # 3. Compute gradient for guidance
    # grad_log_p_y_xt = torch.autograd.grad(log_probs_y.sum(), input_with_grad)[0]
    # For simplicity, let's assume classifier_model.get_class_guidance computes this:
    class_gradient = classifier_model.get_class_guidance(input_with_grad, y) #nabla_{x_t} log p_phi(y|x_t)

    # 4. Adjust noise prediction
    # The exact update rule can vary. A common practical approach:
    # noise_pred_guided = noise_pred_uncond - class_gradient * guidance_scale 
    # (Using minus based on derivation, blog used '+', factor sqrt(1-alphabar_t) absorbed)
    # Let's follow the blog's simpler version for illustration:
    noise_pred_guided = noise_pred_uncond + class_gradient * guidance_scale

    # 5. Perform sampling step
    input = scheduler.step(noise_pred_guided, t, input).prev_sample
```

> **Note on Pseudo-code:** The exact implementation details, especially the sign and scaling of `class_gradient`, can vary. The key is that the classifier's gradient steers the sampling. The `classifier_model.get_class_guidance` encapsulates the gradient computation $\nabla_{x_t} \log p_{\phi}(y|x_t)$.

#### Pros and Cons of Classifier Guidance
*   **Pros:**
    *   Can guide any pre-trained unconditional diffusion model without retraining it.
    *   Allows leveraging powerful, off-the-shelf classifiers.
*   **Cons:**
    *   Requires a separate classifier model, which must be robust to noisy inputs $x_t$ (often requires training the classifier on noisy data).
    *   The guidance is limited to the classes the classifier was trained on.
    *   Can be computationally more expensive due to classifier forward/backward passes at each step.
    *   Guidance can sometimes lead to adversarial examples for the classifier, resulting in artifacts if the guidance scale is too high.

### 3.3. Classifier-Free Guidance (CFG) 🕊️

Classifier-Free Guidance (CFG), proposed by Ho and Salimans (2022), offers a way to guide diffusion models without needing an external classifier. It has become a very popular and effective technique.

#### The Core Idea: Jointly Trained Conditional Model

The key idea is to train a single diffusion model $\epsilon_{\theta}(x_t, t, y)$ that is conditioned on $y$ (e.g., class label, text embedding). During training, this model is occasionally fed a null condition $\emptyset$ (e.g., a zero vector for class embeddings, or an empty string embedding for text). This means the model learns both conditional generation $p(x_t|y)$ and unconditional generation $p(x_t)$ (when $y=\emptyset$).

At sampling time, the model makes two predictions:
1.  $\epsilon_{\theta}(x_t, t, y)$: The noise prediction conditioned on the desired $y$.
2.  $\epsilon_{\theta}(x_t, t, \emptyset)$: The noise prediction for unconditional generation.

The final noise prediction $\epsilon'_{ \theta}$ used for sampling is an extrapolation from the unconditional prediction in the direction of the conditional one:

$$ \epsilon'_{\theta}(x_t, t, y, \lambda) = \epsilon_{\theta}(x_t, t, \emptyset) + \lambda (\epsilon_{\theta}(x_t, t, y) - \epsilon_{\theta}(x_t, t, \emptyset)) $$

Here, $\lambda$ is the guidance scale (often denoted $w$ or `guidance_scale`).
*   If $\lambda = 0$, we get unconditional generation: $\epsilon'_{\theta} = \epsilon_{\theta}(x_t, t, \emptyset)$.
*   If $\lambda = 1$, we get standard conditional generation: $\epsilon'_{\theta} = \epsilon_{\theta}(x_t, t, y)$.
*   If $\lambda > 1$, the generation is pushed further in the direction of $y$, often improving sample quality and adherence to the condition, at the cost of diversity.

This can also be written as:
$$ \epsilon'_{\theta}(x_t, t, y, \lambda) = (1-\lambda)\epsilon_{\theta}(x_t, t, \emptyset) + \lambda \epsilon_{\theta}(x_t, t, y) $$
This form is common when $\lambda$ is interpreted as an interpolation factor, but the previous form is more standard for $\lambda > 1$ guidance scales.

#### How it Works in Practice

1.  **Training:** Train a conditional diffusion model $\epsilon_{\theta}(x_t, t, y)$. With some probability (e.g., 10-20% of the time), replace the true condition $y$ with a null/empty condition $\emptyset$. The model architecture (e.g., U-Net) takes $y$ as an additional input (e.g., through cross-attention for text embeddings, or by adding class embeddings to the timestep embedding).
2.  **Sampling:**
    *   At each step $t$, compute both $\epsilon_{cond} = \epsilon_{\theta}(x_t, t, y)$ and $\epsilon_{uncond} = \epsilon_{\theta}(x_t, t, \emptyset)$. This usually means running the model twice per step, or a single forward pass if the architecture supports batching conditional and unconditional inputs.
    *   Combine them using the CFG formula: $\epsilon_{guided} = \epsilon_{uncond} + \lambda (\epsilon_{cond} - \epsilon_{uncond})$.
    *   Use $\epsilon_{guided}$ in the DDPM sampling step.

#### Pseudo-code for Classifier-Free Guidance (Text-to-Image Example)

The following pseudo-code illustrates CFG for text-to-image generation (adapted from the provided text):

```python
# model: Conditional diffusion model epsilon_theta(x_t, t, y_embedding)
# scheduler: DDPM scheduler
# text_condition: Input text string (e.g., "a photo of a cat")
# guidance_scale: Strength of the CFG
# text_encoder: Model to get text embeddings (e.g., CLIP text encoder)

# 1. Get text embeddings for conditional and unconditional
cond_text_embeddings = text_encoder.encode(text_condition)
uncond_text_embeddings = text_encoder.encode("") # Empty string for unconditional

# Concatenate for a single model pass (optional optimization)
# text_embeddings = torch.cat([uncond_text_embeddings, cond_text_embeddings])
# input_batched = torch.cat([input, input.clone()]) # Duplicate input for batching

input = get_noise_from_standard_normal_distribution(...) # Initial noisy image x_T

for t in tqdm(scheduler.timesteps):
    # Predict noise for both conditional and unconditional
    # If batched:
    #   noise_pred_batched = model(input_batched, t, encoder_hidden_states=text_embeddings).sample
    #   noise_pred_uncond, noise_pred_cond = noise_pred_batched.chunk(2)
    # Else, run model twice:
    with torch.no_grad():
        noise_pred_uncond = model(input, t, encoder_hidden_states=uncond_text_embeddings).sample
        noise_pred_cond = model(input, t, encoder_hidden_states=cond_text_embeddings).sample

    # Apply Classifier-Free Guidance
    noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    # Perform sampling step
    input = scheduler.step(noise_pred_guided, t, input).prev_sample
```
<!-- Image: Stable Diffusion v1.5 under different guidance scales. Source: zhihu.com -->
<div style="margin-top: 20px; text-align: center;">
    <!-- Placeholder for image showing effect of guidance scale -->
    <p>[Image: Effect of guidance_scale on generated images, e.g., from Stable Diffusion]</p>
</div>

#### Pros and Cons of Classifier-Free Guidance
*   **Pros:**
    *   No need for a separate classifier model.
    *   Generally produces higher-quality samples and better adherence to conditions than classifier guidance.
    *   Can be applied to various types of conditions (class labels, text, images, etc.) by designing the conditioning mechanism in $\epsilon_{\theta}$.
    *   Avoids potential issues with classifier robustness to noise or adversarial effects.
*   **Cons:**
    *   Requires training (or fine-tuning) the diffusion model specifically for CFG by including null conditions.
    *   Sampling can be slower if it requires two forward passes of the model per step (though this can sometimes be optimized).

### 3.4. Summary and Comparison

| Feature             | Classifier Guidance                                  | Classifier-Free Guidance (CFG)                       |
|---------------------|------------------------------------------------------|------------------------------------------------------|
| **External Model**  | Requires a pre-trained classifier $p_{\phi}(y|x_t)$    | No external classifier needed                        |
| **Diffusion Model** | Can use any unconditional $\epsilon_{\theta}(x_t, t)$ | Requires $\epsilon_{\theta}(x_t, t, y)$ trained with null conditions |
| **Training**        | Classifier trained (often on noisy data)             | Diffusion model trained with conditional dropout     |
| **Flexibility**     | Limited by classifier's classes/capabilities         | Highly flexible (text, image, etc. conditions)       |
| **Sample Quality**  | Good, but can have artifacts                         | Generally better, stronger adherence to condition    |
| **Computation**     | Diffusion model + Classifier pass per step           | Often 2 diffusion model passes (or 1 larger batch)   |

Classifier-Free Guidance has largely become the standard for conditional image generation with diffusion models (e.g., in DALL·E 2, Imagen, Stable Diffusion). Its ability to produce high-quality, well-conditioned samples without relying on a separate classifier makes it powerful and versatile. While it requires a specific training setup for the diffusion model, the benefits in terms of output quality and flexibility often outweigh this cost.

### 3.5. References

*   Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. *arXiv preprint arXiv:2105.05233*.
*   Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance. *arXiv preprint arXiv:2207.12598*.
