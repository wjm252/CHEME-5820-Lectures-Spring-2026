# L1c: Unsupervised Learning and Clustering Approaches
In this lecture, we explore unsupervised learning and the K-means clustering algorithm, a foundational method for discovering patterns and structure in unlabeled data.

> __Learning Objectives:__
> 
> By the end of this lecture, you should be able to:
>
> * __Understand unsupervised learning and clustering:__ Define unsupervised learning and describe how clustering techniques organize data points into groups based on similarity without predefined labels.
> * __Apply Lloyd's K-means algorithm:__ Implement the K-means clustering algorithm, including the assignment and update steps, and understand how it converges to cluster centroids.
> * __Evaluate K-means limitations and select clusters:__ Identify limitations of K-means such as sensitivity to initialization and the challenge of selecting the number of clusters, and apply methods to determine optimal cluster numbers.

Let's get started!
___

## Example
Today, we will use the following examples to illustrate key concepts:
 
> [▶ K-means clustering on a consumer spending dataset](CHEME-5820-L1c-Example-K-Means-Spring-2026.ipynb). In this example, we apply Lloyd's algorithm to customer demographics and spending behavior. We'll observe how K-means partitions customers into distinct segments, visualize the cluster assignments, and examine how centroid placement affects the final groupings.

___

## Background: What is unsupervised learning and clustering?
Unsupervised learning is a branch of machine learning that deals with unlabeled data. Its goal is to discover hidden patterns and structures without predefined target variables. One of the most common tasks in unsupervised learning is clustering.

> __What is clustering?__
> 
> __Clustering__ is an unsupervised machine learning technique that organizes data points into groups, or clusters, based on their similarities without prior knowledge of the group memberships. This method is widely used for exploratory data analysis, enabling the discovery of patterns and relationships within complex datasets.
> 
> * __Hierarchical clustering__ is an unsupervised machine learning technique that organizes data points into a tree-like structure of nested clusters. This allows for the identification of relationships and patterns within the dataset. This method can be implemented using two main approaches: agglomerative, which merges individual points into larger clusters, and divisive, which splits a single cluster into smaller ones.
> * __Density-based spatial clustering of applications with noise (DBSCAN)__ is a density-based clustering algorithm that groups closely packed data points while effectively identifying outliers, making it particularly useful for datasets with noise and clusters of arbitrary shapes. By defining clusters as dense regions separated by areas of lower density, DBSCAN can efficiently discover meaningful patterns in complex data distributions.
> * __Gaussian mixture models (GMMs)__ are probabilistic models that represent a dataset as a combination of multiple Gaussian distributions, each characterized by its mean and covariance. This allows for the identification of underlying subpopulations within the data. This approach is useful in clustering and density estimation, providing a flexible framework for modeling complex, multimodal distributions.


Today, we'll consider [the K-means algorithm](https://en.wikipedia.org/wiki/K-means_clustering), arguably the most straightforward clustering algorithm. While relatively straightforward, we'll see that [K-means](https://en.wikipedia.org/wiki/K-means_clustering) has some shortcomings. 
___

## K-means clustering (Lloyd's algorithm)
The [K-means algorithm](https://en.wikipedia.org/wiki/K-means_clustering), originally developed by [Lloyd in the 1950s but not published until 1982](https://ieeexplore.ieee.org/document/1056489), is a foundational approach to unsupervised clustering. Suppose we have a dataset $\mathcal{D}=\left\{\mathbf{x}_{1},\mathbf{x}_{2},\dots,\mathbf{x}_{n}\in\mathbb{R}^{m}\right\}$ where each $\mathbf{x}\in\mathbb{R}^{m}$ is an $m$-dimensional feature vector.

[K-means](https://en.wikipedia.org/wiki/K-means_clustering) partitions data points into $K$ distinct groups by minimizing the within-cluster sum of squared distances. The algorithm groups data points (feature vectors) $\mathbf{x}\in\mathcal{D}$ into clusters $\mathcal{C} = \left\{\mathcal{c}_{1},\dots,\mathcal{c}_{K}\right\}$ based on proximity to cluster centroids.

> __How does K-means measure similarity?__
> 
> Clustering algorithms group data based on **similarity**, a measure of how alike two data points are. K-means operationalizes similarity through **distance metrics**: points that are closer together in feature space are considered more similar. The algorithm minimizes within-cluster distances, which equivalently maximizes within-cluster similarity.
> 
> K-means uses the **squared [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)** $d(\mathbf{x},\mathbf{y})^{2} = \left\|\mathbf{x} - \mathbf{y}\right\|_{2}^{2}$ for computational efficiency (avoids computing square roots). Points are assigned to the centroid they are closest to (smallest distance = highest similarity). Other distance metrics can also be used, though squared Euclidean distance is standard for K-means.
>
> **Important**: Features with larger scales (e.g., income in thousands of dollars) dominate distance calculations compared to smaller-scale features (e.g., age in years). Standardizing features to zero mean and unit variance before clustering ensures all features contribute equally to similarity measurements.

The notation $\|\cdot\|_{2}$ represents the Euclidean norm, a specific type of vector norm. Understanding vector norms more broadly clarifies how different distance metrics measure magnitude.

> __What is a vector norm?__
> 
> A **norm** is a function that assigns a non-negative length or size to vectors in a vector space. For a vector $\mathbf{v} \in \mathbb{R}^{m}$, the **p-norm** (also called $\ell^{p}$ norm) is defined as:
> 
> $$\|\mathbf{v}\|_{p} = \left(\sum_{i=1}^{m} |v_i|^{p}\right)^{1/p}$$
> 
> where $p \geq 1$. Common special cases include:
> 
> * **Manhattan norm** ($p=1$): $\|\mathbf{v}\|_{1} = \sum_{i=1}^{m} |v_i|$ measures distance as the sum of absolute differences (city-block distance)
> * **Euclidean norm** ($p=2$): $\|\mathbf{v}\|_{2} = \sqrt{\sum_{i=1}^{m} v_i^2}$ measures straight-line distance (used in K-means)
> * **Maximum norm** ($p=\infty$): $\|\mathbf{v}\|_{\infty} = \max_{i} |v_i|$ measures distance as the largest component difference
> 
> The distance between two vectors $\mathbf{x}$ and $\mathbf{y}$ is computed as $d(\mathbf{x},\mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_{p}$. K-means uses the squared Euclidean distance $\|\mathbf{x} - \mathbf{y}\|_{2}^{2}$ for computational efficiency (avoids the square root).

#### Algorithm: Lloyd's K-means Clustering

__Initialize__: Dataset $\mathcal{D} = \{\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{n} \in \mathbb{R}^{m}\}$, number of clusters $K \in \mathbb{Z}^{+}$, maximum iterations $\texttt{maxiter} \in \mathbb{Z}^{+}$, and initial centroids $\{\boldsymbol{\mu}_{1}, \boldsymbol{\mu}_{2}, \ldots, \boldsymbol{\mu}_{K} \in \mathbb{R}^{m}\}$. Set convergence flag $\texttt{converged} \leftarrow \texttt{false}$ and iteration counter $\texttt{iter} \leftarrow 0$.

__Output__: Cluster assignments for each point (where point $i$ is assigned to cluster $c_{i} \in \{1,\ldots,K\}$) and updated cluster centroids $\{\boldsymbol{\mu}_{1}, \boldsymbol{\mu}_{2}, \ldots, \boldsymbol{\mu}_{K}\}$.

> **Initialization strategies**: Random initialization can lead to poor local optima. Better initialization approaches include:
> 
> * **K-means++**: Select initial centroids using probabilistic sampling where each point's selection probability is proportional to its squared distance from the nearest already-chosen centroid. This spreads centroids apart and improves convergence quality.
> * **Multiple random starts**: Run the algorithm 10–50 times with different random initializations and select the result with the smallest objective function value $J$. Most implementations use 10 as default.
> * **Convergence tolerance**: Set $\epsilon = 10^{-4}$ (standard default) to $10^{-6}$ for tighter convergence. Smaller values require more iterations.
> * **Maximum iterations**: Use $T = 300$ (standard default) for most applications, or scale with the number of clusters using $T \sim 100 + 10K$. K-means typically converges in 10–50 iterations, but setting higher $T$ provides a safety margin against non-convergence.

While $\texttt{converged}$ is $\texttt{false}$ and $\texttt{iter} < \texttt{maxiter}$ __do__:
1. **Assignment step**: For each data point $\mathbf{x}_{i} \in \mathcal{D}$ (where $i = 1, \ldots, n$), assign it to the nearest cluster centroid:
   $$c_{i} \leftarrow \arg\min_{j} \underbrace{\lVert\mathbf{x}_{i} - \boldsymbol{\mu}_{j}\rVert_{2}^{2}}_{=\;d(\mathbf{x}_{i}, \boldsymbol{\mu}_{j})^2}$$
    where $c_{i} \in \{1,\ldots,K\}$ is the cluster assignment for point $i$.
2. **Update step**: Store the current centroids $\hat{\boldsymbol{\mu}}_{j} \leftarrow \boldsymbol{\mu}_{j}$ for all $j$. Then, for each cluster $j = 1$ to $K$, recompute the centroid as the mean of all points assigned to cluster $j$:
   $$\boldsymbol{\mu}_{j} \leftarrow \frac{1}{|\mathcal{C}_{j}|} \sum_{\mathbf{x} \in \mathcal{C}_{j}} \mathbf{x}$$
   where $\mathcal{C}_{j} = \{\mathbf{x}_{i} : c_{i} = j\}$ is the set of points assigned to cluster $j$, and $|\mathcal{C}_{j}|$ is the number of points in that cluster.

    - Typical values for the convergence tolerance are $\epsilon \in \{10^{-4}, 10^{-6}\}$. Smaller values yield tighter convergence at the cost of more iterations. When features are standardized to unit variance, these default values work well; for unstandardized data, scale $\epsilon$ proportionally to the squared feature magnitudes.
    - If $\texttt{iter} \geq \texttt{maxiter}$, issue warning that maximum iterations reached without convergence and exit. Otherwise, increment $\texttt{iter} \leftarrow \texttt{iter} + 1$ and continue to the next iteration.
    - Typical values for the convergence tolerance are $\epsilon \in \{10^{-4}, 10^{-6}\}$. Smaller values yield tighter convergence at the cost of more iterations.

> __Practical considerations:__
>
> * __Convergence guarantee__: K-means is guaranteed to converge to a local minimum. The objective function $J = \sum_{i=1}^{n} \left\|\mathbf{x}_{i} - \boldsymbol{\mu}_{c_{i}}\right\|_{2}^{2}$ decreases monotonically because: (1) the assignment step assigns each point to its nearest centroid, which cannot increase $J$, and (2) the update step computes centroids as means, which minimizes the sum of squared distances for the current assignments. Since there are finitely many possible clusterings and $J$ is non-increasing, convergence is guaranteed in finite iterations, typically within 10–50 iterations.
>
> * __Computational complexity__: The complexity is $O(nKm \cdot t)$, where $t$ is iterations, $n$ is data points, and $m$ is dimensionality. For large datasets (typically $n > 100{,}000$), mini-batch K-means reduces per-iteration cost to $O(Bm)$ where $B$ is the mini-batch size (commonly $B = 100$ to $1000$), trading some accuracy for speed. Mini-batch K-means randomly samples a subset of points for each iteration, making it practical for datasets too large to fit in memory.
>
> * __Computational complexity__: The complexity is $O(nKm \cdot t)$, where $t$ is iterations, $n$ is data points, and $m$ is dimensionality. For large datasets, mini-batch K-means reduces per-iteration cost to $O(Bm)$ where $B$ is the mini-batch size, trading some accuracy for speed.

___

## What are the limitations of K-means?

K-means is effective for many clustering tasks, but several limitations can affect its performance. Understanding these constraints helps us recognize when K-means is appropriate and when alternative methods may be preferable.

> __Issues with K-means:__
>
> * __The number of clusters must be specified in advance__: K-means requires users to specify $K$ before running the algorithm with no automatic mechanism to determine the optimal value. Too many clusters fragment natural groups; too few merge distinct patterns.
> * __Sensitivity to initialization__: Different random initializations converge to different local minima, producing substantially different results. Multiple restarts with different seeds (or K-means++ initialization) mitigate this issue.
> * __Vulnerability to outliers__: Because centroids are computed as means, individual outliers disproportionately shift centroid positions away from cluster centers, causing misassignments and degrading cluster quality.
> * __Assumption of spherical, well-separated clusters__: K-means uses linear decision boundaries (perpendicular bisectors between centroids), which work well for spherical clusters but fail for elongated, crescent-shaped, or overlapping clusters.

### Feature scaling and data preprocessing

Before applying K-means, it is important to consider feature scaling. Since the algorithm uses Euclidean distance, features with larger scales (e.g., income in dollars) will dominate distance calculations compared to features with smaller scales (e.g., age in years). A common approach is to standardize each feature to zero mean and unit variance:

$$x_{\text{scaled}}^{(i)} = \frac{x^{(i)} - \mu_i}{\sigma_i}$$

where $\mu_i$ is the mean and $\sigma_i$ is the standard deviation of feature $i$. This transformation is applied element-wise to each feature independently. Outliers should be identified and handled appropriately before clustering, as they can distort centroid calculations and degrade cluster quality.

### How many clusters should we choose?

Since we must specify $K$ in advance, we need principled methods to select it. Several approaches are available:

- **Elbow method**: Plot the within-cluster sum of squares (the objective function $J$) versus $K$. Look for an "elbow" or bend point where further increases in $K$ yield diminishing improvements in $J$. This method is visual and intuitive but subjective.
- **Silhouette method**: Measure how similar each point is to its own cluster compared to other clusters. Higher silhouette scores indicate better-defined clusters. This method is more objective than the elbow method.
- **Calinski-Harabasz index**: Computes the ratio of between-cluster to within-cluster variance. Higher values suggest more distinct clustering and more separation.

See the [Silhouette method](CHEME-5820-L1c-Advanced-SilhouetteScore-Spring-2026.ipynb) and [Calinski-Harabasz index](CHEME-5820-L1c-Advanced-CHI-Score-Spring-2026.ipynb) notebooks for detailed treatments of these approaches. Domain knowledge about the problem may also inform the choice of $K$, especially when multiple methods suggest different values. For example, marketing teams analyzing customer segmentation often prefer $K = 3$ to $5$ segments for actionable strategies, while genomic studies might use $K$ values corresponding to known biological subtypes or populations.

> __Example__:
> 
> Let's look at an example of applying K-means clustering to a consumer spending dataset. 
> 
> [▶ K-means clustering on a consumer spending dataset](CHEME-5820-L1c-Example-K-Means-Spring-2026.ipynb). In this example, we apply Lloyd's algorithm to customer demographics and spending behavior. We'll observe how K-means partitions customers into distinct segments, visualize the cluster assignments, and examine how centroid placement affects the final groupings.

___

## Summary
K-means clustering partitions data into $K$ groups by iteratively assigning points to nearest centroids and updating centroid positions until convergence.

> __Key Takeaways:__
>
> * **Lloyd's algorithm alternates between assignment and update steps**: The algorithm assigns each data point to the nearest centroid using Euclidean distance, then recomputes centroids as the mean of assigned points. This process repeats until convergence.
> * **K-means requires specifying the number of clusters in advance**: The algorithm requires users to specify $K$ before clustering. Poor choices of $K$ lead to either overfitting (too many clusters) or underfitting (too few clusters).
> * **K-means has practical limitations**: Sensitivity to initial centroid placement, vulnerability to outliers, and inability to handle overlapping clusters are key limitations. Feature scaling should be applied before clustering. The elbow method, silhouette method, and Calinski-Harabasz index are useful for selecting the number of clusters, and running multiple random restarts improves solution quality.

K-means provides a scalable approach to unsupervised clustering but requires careful consideration of its assumptions and limitations.
___
