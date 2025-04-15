The purpose of this task is to rigorously evaluate the candidate’s depth of understanding,
analytical thinking, and research potential in the context of a novel machine learning tool,
Kolmogorov-Arnold Networks (KANs)—a theoretical and practical framework rooted in
the Kolmogorov Superposition Theorem.

🧠 Objective:
Explore **Kolmogorov-Arnold Networks (KANs)** and compare them with traditional neural network architectures.


🧩 Task Sections:

1. General Knowledge

explain and demonstrate understanding of:

Kolmogorov Superposition Theorem: State it clearly for functions on [0, 1]^n.
KANs’ theoretical foundation: Link it to the theorem.
Using splines in KANs**: How piecewise polynomials work as basis functions.
Nonlinearity: Contrast KANs' approach to nonlinearity with traditional NNs (e.g., via activation functions).


2. Practical Implementation & Research Potential
Tasks to implement and analyze:

2.1 Build a minimal KAN.
2.2 Use it to fit the function:  
   
   f(x, y) = \sin(xy) + \cos(x^2 + y^2)
  
2.3 Compare convergence and fit quality against a shallow MLP (and optionally others).
2.4 Analyze the loss surface** of a shallow KAN (random init).
2.5 Discuss optimization dynamics** — saddle points, flat minima, etc.

3. Bonus Tasks (Optional but valuable):
 Theoretical example of function classes where KANs beat traditional NNs in **approximation error**.
Propose a **new class of activation functions** inspired by Kolmogorov's theorem, implement and test it.


- Structuring your full report?
- Writing a minimal KAN implementation?
- Comparing it with MLPs using PyTorch or another framework?
- Exploring the Kolmogorov Superposition Theorem in a digestible way?

Let me know what you'd like to focus on first!
